"""QM9 training with a global optimizer-step stopping condition.

This keeps the original Equiformer QM9 model and optimizer recipe, but replaces
the epoch stopping condition with ``--max-steps``.  The learning-rate schedule,
validation, and checkpoint cadence are expressed on the reference batch-128
step axis (859 optimizer steps per original epoch).
"""

import argparse
import hashlib
import json
import os
import sys
import time
from contextlib import suppress
from pathlib import Path

import numpy as np
import torch
from timm.scheduler import create_scheduler
from timm.utils import ModelEmaV2, NativeScaler, dispatch_clip_grad
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

EQUIFORMER_ROOT = os.environ.get(
    "EQUIFORMER_ROOT", "/home/20262202788/equiformer"
)
if EQUIFORMER_ROOT not in sys.path:
    sys.path.insert(0, EQUIFORMER_ROOT)

import main_qm9 as base
import utils
from engine import AverageMeter, evaluate
from logger import FileLogger
from optim_factory import create_optimizer

from equivariant_nas.training.data_protocol import (
    is_epoch_validation_step,
    next_epoch_validation_step,
    steps_per_data_epoch,
)


ModelEma = ModelEmaV2


def get_parser():
    parser = argparse.ArgumentParser(
        "Equiformer QM9 fixed-step training",
        parents=[base.get_args_parser()],
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=257700,
        help="Maximum cumulative optimizer updates (300 * 859 by default).",
    )
    parser.add_argument(
        "--reference-steps-per-epoch",
        type=int,
        default=859,
        help="Batch-128 baseline steps used to preserve the original LR schedule.",
    )
    parser.add_argument(
        "--eval-interval-steps",
        type=int,
        default=859,
        help="Legacy step-based validation interval; ignored when --eval-interval-epochs > 0.",
    )
    parser.add_argument(
        "--eval-interval-epochs",
        type=int,
        default=0,
        help="Validate after every N completed epochs of the current training dataset.",
    )
    parser.add_argument(
        "--train-subset-file",
        type=str,
        default="",
        help="NPZ containing fixed train_local_indices; empty means the full train split.",
    )
    parser.add_argument(
        "--data-epoch-origin-step",
        type=int,
        default=0,
        help="Global step at which the current dataset phase began; resets data-epoch counting.",
    )
    parser.add_argument(
        "--allow-data-transition",
        action="store_true",
        default=False,
        help="Allow --resume-step to continue after an intentional training-dataset change.",
    )
    parser.add_argument(
        "--resume-model-only",
        action="store_true",
        default=False,
        help="Load model/global step but reset optimizer and LR scheduler for transition warm-up.",
    )
    parser.add_argument(
        "--lr-schedule-origin-step",
        type=int,
        default=0,
        help="Global step treated as LR-schedule epoch zero (used when re-warming after transition).",
    )
    parser.add_argument(
        "--checkpoint-interval-steps",
        type=int,
        default=859,
        help="Save resumable checkpoint_last.pth every N global steps.",
    )
    parser.add_argument(
        "--resume-step",
        type=str,
        default=None,
        help="Resume from checkpoint_last.pth produced by this script.",
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        default=False,
        help="Evaluate the locked test set. Forbidden during search; final runs only.",
    )
    parser.add_argument(
        "--evaluation-only",
        action="store_true",
        default=False,
        help="Load a completed checkpoint and evaluate validation/test without optimizer steps.",
    )
    parser.add_argument(
        "--dsl-program",
        type=str,
        default=None,
        help="EvoEquiLang JSON program. If omitted, use the trusted model registry.",
    )
    parser.add_argument(
        "--dsl-task-contract",
        type=str,
        default="",
        help="Immutable task-contract JSON used to derive the same semantic ID across search and training.",
    )
    parser.add_argument(
        "--equiformer-root",
        type=str,
        default=EQUIFORMER_ROOT,
        help="Path to the trusted Equiformer source tree.",
    )
    parser.add_argument(
        "--equiformer-v2-root",
        type=str,
        default=os.environ.get("EQUIFORMER_V2_ROOT", ""),
        help="Optional pinned Equiformer V2 source used by DSL V2 graph fusions.",
    )
    parser.add_argument(
        "--equiformer-v3-root",
        type=str,
        default=os.environ.get("EQUIFORMER_V3_ROOT", ""),
        help="Pinned Equiformer V3 source used by generic V3 primitive Lowering.",
    )
    parser.add_argument(
        "--allow-experimental-generic-lowering",
        action="store_true",
        default=False,
        help=(
            "Explicitly enable audit-only generic DSL graph lowering. "
            "This does not certify the run for formal V1 ranking."
        ),
    )
    return parser


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fixed_training_subset(dataset, subset_file):
    if not subset_file:
        return dataset, "qm9_train_full", None
    path = Path(subset_file).resolve()
    with np.load(path) as payload:
        indices = np.asarray(payload["train_local_indices"], dtype=np.int64)
    if indices.ndim != 1 or len(indices) == 0:
        raise ValueError("fixed training subset indices must be a non-empty vector")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("fixed training subset contains duplicate indices")
    if int(indices.min()) < 0 or int(indices.max()) >= len(dataset):
        raise ValueError("fixed training subset contains out-of-range indices")
    fingerprint = sha256_file(path)
    metadata = {
        "path": str(path),
        "sha256": fingerprint,
        "size": int(len(indices)),
        "source_train_size": int(len(dataset)),
    }
    return Subset(dataset, indices.tolist()), "qm9_train_subset:" + fingerprint, metadata


def target_mean_std(dataset, target):
    if isinstance(dataset, Subset):
        indices = torch.as_tensor(dataset.indices, dtype=torch.long)
        values = dataset.dataset.data.y[indices, target].float()
        return float(values.mean()), float(values.std())
    return dataset.mean(target), dataset.std(target)


def cycling_batches(dataset, args, start_global_step=0, data_epoch_origin_step=0):
    """Yield deterministic shuffled batches with exact resume semantics.

    Each data cycle uses a seed derived only from ``args.seed`` and the cycle
    number. Therefore ``global_step`` uniquely determines both the permutation
    and the next batch; no opaque DataLoader iterator state is required.
    """
    steps_per_cycle = steps_per_data_epoch(len(dataset), args.batch_size)
    if steps_per_cycle <= 0:
        raise ValueError("training dataset is smaller than one batch")
    relative_step = start_global_step - data_epoch_origin_step
    if relative_step < 0:
        raise ValueError("data epoch origin cannot be greater than the resume step")
    data_cycle = relative_step // steps_per_cycle
    start_batch = relative_step % steps_per_cycle
    while True:
        generator = torch.Generator()
        generator.manual_seed(int(args.seed) + int(data_cycle))
        data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=args.workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )
        for batch_index, data in enumerate(data_loader):
            if batch_index < start_batch:
                continue
            yield data, data_cycle, batch_index
        data_cycle += 1
        start_batch = 0


def train_until_step(
    model,
    criterion,
    norm_factor,
    target,
    data_stream,
    optimizer,
    lr_scheduler,
    reference_steps_per_epoch,
    lr_schedule_origin_step,
    device,
    global_step,
    target_global_step,
    max_steps,
    model_ema=None,
    amp_autocast=None,
    loss_scaler=None,
    clip_grad=None,
    print_freq=100,
    logger=None,
):
    model.train()
    criterion.train()

    loss_metric = AverageMeter()
    mae_metric = AverageMeter()
    segment_start_step = global_step
    segment_start_time = time.perf_counter()
    task_mean, task_std = norm_factor

    while global_step < target_global_step:
        # Match the original batch-128 recipe: scheduler.step(epoch) was called
        # once before each block of 859 optimizer updates.
        schedule_step = global_step - lr_schedule_origin_step
        if schedule_step < 0:
            raise ValueError("LR schedule origin cannot be greater than global step")
        if schedule_step % reference_steps_per_epoch == 0:
            reference_epoch = schedule_step // reference_steps_per_epoch
            lr_scheduler.step(reference_epoch)

        data, data_cycle, batch_index = next(data_stream)
        data = data.to(device)

        with amp_autocast():
            pred = model(
                f_in=data.x,
                pos=data.pos,
                batch=data.batch,
                node_atom=data.z,
                edge_d_index=data.edge_d_index,
                edge_d_attr=data.edge_d_attr,
            ).squeeze()
            loss = criterion(pred, (data.y[:, target] - task_mean) / task_std)

        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(loss, optimizer, parameters=model.parameters())
        else:
            loss.backward()
            if clip_grad is not None:
                dispatch_clip_grad(model.parameters(), value=clip_grad, mode="norm")
            optimizer.step()

        if model_ema is not None:
            model_ema.update(model)

        batch_items = pred.shape[0]
        loss_metric.update(loss.item(), n=batch_items)
        err = pred.detach() * task_std + task_mean - data.y[:, target]
        mae_metric.update(torch.mean(torch.abs(err)).item(), n=batch_items)

        global_step += 1
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        if (
            global_step % print_freq == 0
            or global_step == target_global_step
            or global_step == max_steps
        ):
            elapsed = time.perf_counter() - segment_start_time
            completed = global_step - segment_start_step
            logger.info(
                "Global step: [{}/{}] data_cycle={} batch_index={} "
                "loss: {:.5f}, MAE: {:.5f}, time/step={:.0f}ms, lr={:.2e}".format(
                    global_step,
                    max_steps,
                    data_cycle,
                    batch_index,
                    loss_metric.avg,
                    mae_metric.avg,
                    1000.0 * elapsed / completed,
                    optimizer.param_groups[0]["lr"],
                )
            )

    return mae_metric.avg, global_step, time.perf_counter() - segment_start_time


def save_checkpoint(
    path,
    args,
    model,
    optimizer,
    lr_scheduler,
    loss_scaler,
    model_ema,
    global_step,
    training_time_sec,
    best_step,
    best_train_err,
    best_val_err,
    best_test_err=None,
):
    model_without_ddp = model.module if args.distributed else model
    checkpoint = {
        "global_step": global_step,
        "model": model_without_ddp.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "loss_scaler": loss_scaler.state_dict() if loss_scaler is not None else None,
        "model_ema": model_ema.module.state_dict() if model_ema is not None else None,
        "training_time_sec": training_time_sec,
        "best_step": best_step,
        "best_train_err": best_train_err,
        "best_val_err": best_val_err,
        "best_test_err": best_test_err,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy_rng_state": np.random.get_state(),
        "args": vars(args),
    }
    tmp_path = str(path) + ".tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, path)


def main(args):
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if args.resume_model_only and not args.resume_step:
        raise ValueError("--resume-model-only requires --resume-step")
    if args.reference_steps_per_epoch < 0 or args.checkpoint_interval_steps < 0:
        raise ValueError("reference/checkpoint step intervals must be non-negative")
    if args.eval_interval_epochs <= 0 and args.eval_interval_steps <= 0:
        raise ValueError("one validation interval must be positive")
    if args.data_epoch_origin_step < 0 or args.lr_schedule_origin_step < 0:
        raise ValueError("step origins must be non-negative")
    # ``args.epochs`` is retained only to construct the original 300-epoch
    # timm cosine scheduler. It is not used as a stopping condition.
    schedule_span_steps = args.max_steps - args.lr_schedule_origin_step
    if schedule_span_steps <= 0:
        raise ValueError("LR schedule origin must be smaller than --max-steps")
    required_reference_epochs = (
        schedule_span_steps + args.reference_steps_per_epoch - 1
    ) // args.reference_steps_per_epoch
    if args.resume_model_only and args.lr_schedule_origin_step > 0:
        args.epochs = required_reference_epochs
    elif args.epochs < required_reference_epochs:
        args.epochs = required_reference_epochs

    utils.init_distributed_mode(args)
    if args.distributed:
        raise NotImplementedError(
            "stage-one exact-resume trainer is intentionally single-GPU only"
        )
    is_main_process = args.rank == 0
    log = FileLogger(
        is_master=is_main_process,
        is_rank0=is_main_process,
        output_dir=args.output_dir,
    )
    log.info(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    full_train_dataset = base.QM9(
        args.data_path, "train", feature_type=args.feature_type
    )
    train_dataset, training_dataset_id, subset_metadata = fixed_training_subset(
        full_train_dataset, args.train_subset_file
    )
    args.training_dataset_id = training_dataset_id
    data_steps_per_epoch = steps_per_data_epoch(len(train_dataset), args.batch_size)
    if args.reference_steps_per_epoch <= 0:
        args.reference_steps_per_epoch = data_steps_per_epoch
    if args.checkpoint_interval_steps <= 0:
        args.checkpoint_interval_steps = data_steps_per_epoch
    if args.reference_steps_per_epoch <= 0 or args.checkpoint_interval_steps <= 0:
        raise ValueError("resolved reference/checkpoint step intervals must be positive")
    val_dataset = base.QM9(args.data_path, "valid", feature_type=args.feature_type)
    test_dataset = (
        base.QM9(args.data_path, "test", feature_type=args.feature_type)
        if args.evaluate_test
        else None
    )
    observed_mean, observed_std = target_mean_std(train_dataset, args.target)
    log.info("Training set mean: {}, std:{}".format(observed_mean, observed_std))
    log.info(
        "Training dataset id: {}; size={}; steps_per_data_epoch={}; subset={}".format(
            training_dataset_id,
            len(train_dataset),
            data_steps_per_epoch,
            json.dumps(subset_metadata, sort_keys=True),
        )
    )
    task_mean, task_std = 0, 1
    if args.standardize:
        task_mean, task_std = observed_mean, observed_std
    norm_factor = [task_mean, task_std]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dsl_program:
        from equivariant_nas.dsl import Compiler, core_registry, reference_motif_registry
        from equivariant_nas.dsl.backends.qm9_model import build_qm9_dsl_model
        from equivariant_nas.dsl.serialization import load_program, load_task_contract

        dsl_program = load_program(args.dsl_program)
        dsl_task = load_task_contract(args.dsl_task_contract) if args.dsl_task_contract else None
        dsl_compiler = Compiler(core_registry(), reference_motif_registry())
        if "equiformer_v3_spec" in dsl_program.parameters:
            if not args.equiformer_v3_root:
                raise ValueError("V3 DSL training requires --equiformer-v3-root")
            from equivariant_nas.training.v3_qm9_runtime import build_lowered_v3_qm9_model

            model = build_lowered_v3_qm9_model(
                dsl_program,
                equiformer_v3_root=args.equiformer_v3_root,
            ).to(device)
        else:
            model = build_qm9_dsl_model(
                dsl_program,
                dsl_compiler,
                radius=args.radius,
                equiformer_root=args.equiformer_root,
                equiformer_v2_root=args.equiformer_v2_root or None,
                task_mean=task_mean,
                task_std=task_std,
                atomref=None,
                task=dsl_task,
                allow_experimental_generic_lowering=args.allow_experimental_generic_lowering,
            ).to(device)
        log.info("DSL architecture ID: {}".format(model.dsl_architecture_id))
        log.info("DSL language version: {}".format(model.dsl_language_version))
        log.info("DSL lowering plan: {}".format(json.dumps(model.lowering_plan, sort_keys=True)))
        if hasattr(model, "generic_lowering_admission"):
            log.info(
                "DSL generic Lowering admission: {}".format(
                    json.dumps(model.generic_lowering_admission, sort_keys=True)
                )
            )
    else:
        create_model = base.model_entrypoint(args.model_name)
        model = create_model(
            irreps_in=args.input_irreps,
            radius=args.radius,
            num_basis=args.num_basis,
            out_channels=args.output_channels,
            task_mean=task_mean,
            task_std=task_std,
            atomref=None,
            drop_path=args.drop_path,
        ).to(device)
    log.info(model)

    model_ema = None
    if args.model_ema:
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device="cpu" if args.model_ema_force_cpu else None,
        )
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank]
        )
    log.info(
        "Number of params: {}".format(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        )
    )

    optimizer = create_optimizer(args, model)
    lr_scheduler, _ = create_scheduler(args, optimizer)
    if args.loss == "l1":
        criterion = torch.nn.L1Loss()
    elif args.loss == "l2":
        criterion = torch.nn.MSELoss()
    else:
        raise ValueError("Unsupported loss: {}".format(args.loss))

    amp_autocast = suppress
    loss_scaler = None
    if args.amp:
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = (
        DataLoader(test_dataset, batch_size=args.batch_size)
        if test_dataset is not None
        else None
    )

    global_step = 0
    training_time_sec = 0.0
    best_step = 0
    best_train_err = float("inf")
    best_val_err = float("inf")
    best_test_err = None

    if args.resume_step:
        checkpoint = torch.load(args.resume_step, map_location="cpu")
        checkpoint_args = checkpoint.get("args", {})
        previous_dataset_id = checkpoint_args.get("training_dataset_id", "")
        dataset_changed = bool(previous_dataset_id) and previous_dataset_id != training_dataset_id
        if dataset_changed and not args.allow_data_transition:
            raise ValueError(
                "resume checkpoint dataset differs from current dataset; "
                "pass --allow-data-transition for the intentional quarter-to-full transition"
            )
        model_without_ddp = model.module if args.distributed else model
        model_without_ddp.load_state_dict(checkpoint["model"])
        global_step = checkpoint["global_step"]
        training_time_sec = checkpoint.get("training_time_sec", 0.0)
        best_step = checkpoint.get("best_step", 0)
        best_train_err = checkpoint.get("best_train_err", float("inf"))
        best_val_err = checkpoint.get("best_val_err", float("inf"))
        best_test_err = checkpoint.get("best_test_err")
        if args.resume_model_only:
            if not dataset_changed:
                log.info("Model-only resume requested without a dataset transition")
            log.info(
                "Loaded model/global step from {} and reset optimizer/scheduler for warm-up".format(
                    args.resume_step
                )
            )
        else:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            torch.set_rng_state(checkpoint["torch_rng_state"])
            if torch.cuda.is_available() and checkpoint.get("cuda_rng_state_all") is not None:
                torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
            np.random.set_state(checkpoint["numpy_rng_state"])
            if loss_scaler is not None and checkpoint.get("loss_scaler") is not None:
                loss_scaler.load_state_dict(checkpoint["loss_scaler"])
            if model_ema is not None and checkpoint.get("model_ema") is not None:
                model_ema.module.load_state_dict(checkpoint["model_ema"])
            log.info("Resumed from {} at global_step={}".format(args.resume_step, global_step))
        if dataset_changed:
            log.info(
                "Intentional data transition: {} -> {} at global_step={}".format(
                    previous_dataset_id, training_dataset_id, global_step
                )
            )

    if args.data_epoch_origin_step > global_step:
        raise ValueError("data epoch origin cannot be greater than the current global step")
    if args.lr_schedule_origin_step > global_step:
        raise ValueError("LR schedule origin cannot be greater than the current global step")
    if args.evaluation_only:
        if not args.evaluate_test:
            raise ValueError("evaluation-only mode requires --evaluate-test")
        if not args.resume_step:
            raise ValueError("evaluation-only mode requires --resume-step")
        if global_step != args.max_steps:
            raise ValueError(
                "evaluation-only checkpoint step {} must equal max_steps {}".format(
                    global_step, args.max_steps
                )
            )
    job_start_global_step = global_step
    data_stream = cycling_batches(
        train_dataset,
        args,
        start_global_step=global_step,
        data_epoch_origin_step=args.data_epoch_origin_step,
    )
    metrics_path = Path(args.output_dir) / "metrics.jsonl"
    job_start_time = time.perf_counter()

    while global_step < args.max_steps:
        if args.eval_interval_epochs > 0:
            next_evaluation = next_epoch_validation_step(
                global_step,
                args.data_epoch_origin_step,
                data_steps_per_epoch,
                args.eval_interval_epochs,
            )
        else:
            next_evaluation = (
                ((global_step // args.eval_interval_steps) + 1)
                * args.eval_interval_steps
            )
        next_checkpoint = (
            ((global_step // args.checkpoint_interval_steps) + 1)
            * args.checkpoint_interval_steps
        )
        next_boundary = min(next_evaluation, next_checkpoint, args.max_steps)
        train_err, global_step, segment_training_sec = train_until_step(
            model=model,
            criterion=criterion,
            norm_factor=norm_factor,
            target=args.target,
            data_stream=data_stream,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            reference_steps_per_epoch=args.reference_steps_per_epoch,
            lr_schedule_origin_step=args.lr_schedule_origin_step,
            device=device,
            global_step=global_step,
            target_global_step=next_boundary,
            max_steps=args.max_steps,
            model_ema=model_ema,
            amp_autocast=amp_autocast,
            loss_scaler=loss_scaler,
            clip_grad=args.clip_grad,
            print_freq=args.print_freq,
            logger=log,
        )
        training_time_sec += segment_training_sec

        if args.eval_interval_epochs > 0:
            evaluate_now = (
                is_epoch_validation_step(
                    global_step,
                    args.data_epoch_origin_step,
                    data_steps_per_epoch,
                    args.eval_interval_epochs,
                )
                or global_step == args.max_steps
            )
        else:
            evaluate_now = (
                global_step % args.eval_interval_steps == 0
                or global_step == args.max_steps
            )
        val_err = None
        test_err = None
        if evaluate_now:
            val_err, _ = evaluate(
                model,
                norm_factor,
                args.target,
                val_loader,
                device,
                amp_autocast=amp_autocast,
                print_freq=args.print_freq,
                logger=log,
            )
            if args.evaluate_test:
                test_err, _ = evaluate(
                    model,
                    norm_factor,
                    args.target,
                    test_loader,
                    device,
                    amp_autocast=amp_autocast,
                    print_freq=args.print_freq,
                    logger=log,
                )

        improved = val_err is not None and val_err < best_val_err
        if improved:
            best_step = global_step
            best_train_err = train_err
            best_val_err = val_err
            best_test_err = test_err

        wall_time_sec = time.perf_counter() - job_start_time
        progress = {
            "global_step": global_step,
            "reference_epoch": global_step / args.reference_steps_per_epoch,
            "data_epoch": (global_step - args.data_epoch_origin_step) / data_steps_per_epoch,
            "steps_per_data_epoch": data_steps_per_epoch,
            "training_dataset_id": training_dataset_id,
            "train_mae": train_err,
            "lr": optimizer.param_groups[0]["lr"],
            "training_time_sec": training_time_sec,
            "wall_time_sec_current_job": wall_time_sec,
            "start_global_step": job_start_global_step,
            "steps_executed_current_job": global_step - job_start_global_step,
            "best_step": best_step,
            "best_val_mae": best_val_err,
        }
        if val_err is not None:
            progress["val_mae"] = val_err
        if args.evaluate_test and test_err is not None:
            progress["test_mae"] = test_err
            progress["best_test_mae"] = best_test_err
        log.info("Step summary: {}".format(json.dumps(progress, sort_keys=True)))
        if is_main_process:
            (Path(args.output_dir) / "progress.json").write_text(
                json.dumps(progress, indent=2, sort_keys=True), encoding="utf-8"
            )
            if val_err is not None:
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(progress, sort_keys=True) + "\n")
            checkpoint_path = Path(args.output_dir) / "checkpoint_last.pth"
            save_checkpoint(
                checkpoint_path,
                args,
                model,
                optimizer,
                lr_scheduler,
                loss_scaler,
                model_ema,
                global_step,
                training_time_sec,
                best_step,
                best_train_err,
                best_val_err,
                best_test_err,
            )
            if improved:
                save_checkpoint(
                    Path(args.output_dir) / "checkpoint_best.pth",
                    args,
                    model,
                    optimizer,
                    lr_scheduler,
                    loss_scaler,
                    model_ema,
                    global_step,
                    training_time_sec,
                    best_step,
                    best_train_err,
                    best_val_err,
                    best_test_err,
                )

    endpoint_val_err = None
    endpoint_test_err = None
    if args.evaluation_only:
        endpoint_val_err, _ = evaluate(
            model,
            norm_factor,
            args.target,
            val_loader,
            device,
            amp_autocast=amp_autocast,
            print_freq=args.print_freq,
            logger=log,
        )
        endpoint_test_err, _ = evaluate(
            model,
            norm_factor,
            args.target,
            test_loader,
            device,
            amp_autocast=amp_autocast,
            print_freq=args.print_freq,
            logger=log,
        )
        best_test_err = endpoint_test_err
        evaluation_progress = {
            "global_step": global_step,
            "start_global_step": job_start_global_step,
            "steps_executed_current_job": 0,
            "val_mae": endpoint_val_err,
            "test_mae": endpoint_test_err,
            "test_evaluated": True,
            "evaluation_only": True,
        }
        if is_main_process:
            (Path(args.output_dir) / "progress.json").write_text(
                json.dumps(evaluation_progress, indent=2, sort_keys=True), encoding="utf-8"
            )
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(evaluation_progress, sort_keys=True) + "\n")

    formal_ranking_admitted = not (
        args.dsl_program
        and getattr(model, "lowering_mode", "") == "experimental_node_graph"
    )
    final_summary = {
        "status": "completed",
        "batch_size": args.batch_size,
        "global_step": global_step,
        "max_steps": args.max_steps,
        "reference_steps_per_epoch": args.reference_steps_per_epoch,
        "checkpoint_interval_steps": args.checkpoint_interval_steps,
        "eval_interval_epochs": args.eval_interval_epochs,
        "training_dataset_id": training_dataset_id,
        "training_dataset_size": len(train_dataset),
        "train_subset": subset_metadata,
        "steps_per_data_epoch": data_steps_per_epoch,
        "data_epoch_origin_step": args.data_epoch_origin_step,
        "completed_data_epochs": (global_step - args.data_epoch_origin_step) / data_steps_per_epoch,
        "effective_data_cycles": (global_step - args.data_epoch_origin_step) / data_steps_per_epoch,
        "training_time_sec": training_time_sec,
        "wall_time_sec_current_job": time.perf_counter() - job_start_time,
        "start_global_step": job_start_global_step,
        "steps_executed_current_job": global_step - job_start_global_step,
        "best_step": best_step,
        "best_val_mae": best_val_err,
        "test_evaluated": bool(args.evaluate_test),
        "evaluation_only": bool(args.evaluation_only),
        "inherited_initialization": False,
        "selection_eligible": formal_ranking_admitted,
        "final_training_allowed": formal_ranking_admitted,
    }
    if args.dsl_program:
        final_summary.update(
            {
                "dsl_architecture_id": getattr(model, "dsl_architecture_id", ""),
                "dsl_language_version": getattr(model, "dsl_language_version", ""),
                "backend_family": getattr(model, "backend_family", ""),
                "backend_semantics_version": getattr(model, "backend_semantics_version", ""),
                "lowering_mode": getattr(model, "lowering_mode", ""),
                "lowering_plan": getattr(model, "lowering_plan", {}),
                "generic_lowering_admission": getattr(model, "generic_lowering_admission", {}),
            }
        )
    if args.evaluate_test:
        final_summary["best_test_mae"] = best_test_err
    if args.evaluation_only:
        final_summary["endpoint_validation_mae"] = endpoint_val_err
        final_summary["endpoint_test_mae"] = endpoint_test_err
    if is_main_process:
        with (Path(args.output_dir) / "training_summary.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(final_summary, handle, indent=2, sort_keys=True)
    log.info("Training completed: {}".format(json.dumps(final_summary, sort_keys=True)))


if __name__ == "__main__":
    args = get_parser().parse_args()
    if not args.output_dir:
        raise ValueError("--output-dir is required for fixed-step training")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
