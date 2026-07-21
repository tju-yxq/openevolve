"""QM9 training with a global optimizer-step stopping condition.

This keeps the original Equiformer QM9 model and optimizer recipe, but replaces
the epoch stopping condition with ``--max-steps``.  The learning-rate schedule,
validation, and checkpoint cadence are expressed on the reference batch-128
step axis (859 optimizer steps per original epoch).
"""

import argparse
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
        help="Run validation/test and save a checkpoint every N global steps.",
    )
    parser.add_argument(
        "--resume-step",
        type=str,
        default=None,
        help="Resume from checkpoint_last.pth produced by this script.",
    )
    parser.add_argument(
        "--inherit-checkpoint",
        type=str,
        default=None,
        help="Calibration-only parent model weights; optimizer/scheduler are reset.",
    )
    parser.add_argument(
        "--inherit-parent-spec",
        type=str,
        default=None,
        help="ArchitectureSpec for --inherit-checkpoint semantic transfer policy.",
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        default=False,
        help="Evaluate the locked test set. Forbidden during search; final runs only.",
    )
    parser.add_argument(
        "--architecture-spec",
        type=str,
        default=None,
        help="JSON ArchitectureSpec. If omitted, use the legacy model registry.",
    )
    parser.add_argument(
        "--equiformer-root",
        type=str,
        default=EQUIFORMER_ROOT,
        help="Path to the trusted Equiformer source tree.",
    )
    return parser


def cycling_batches(dataset, args, start_global_step=0):
    """Yield deterministic shuffled batches with exact resume semantics.

    Each data cycle uses a seed derived only from ``args.seed`` and the cycle
    number. Therefore ``global_step`` uniquely determines both the permutation
    and the next batch; no opaque DataLoader iterator state is required.
    """
    steps_per_cycle = len(dataset) // args.batch_size
    if steps_per_cycle <= 0:
        raise ValueError("training dataset is smaller than one batch")
    data_cycle = start_global_step // steps_per_cycle
    start_batch = start_global_step % steps_per_cycle
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
        if global_step % reference_steps_per_epoch == 0:
            reference_epoch = global_step // reference_steps_per_epoch
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
    if args.reference_steps_per_epoch <= 0 or args.eval_interval_steps <= 0:
        raise ValueError("step intervals must be positive")

    # ``args.epochs`` is retained only to construct the original 300-epoch
    # timm cosine scheduler. It is not used as a stopping condition.
    required_reference_epochs = (
        args.max_steps + args.reference_steps_per_epoch - 1
    ) // args.reference_steps_per_epoch
    if args.epochs < required_reference_epochs:
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

    train_dataset = base.QM9(args.data_path, "train", feature_type=args.feature_type)
    val_dataset = base.QM9(args.data_path, "valid", feature_type=args.feature_type)
    test_dataset = (
        base.QM9(args.data_path, "test", feature_type=args.feature_type)
        if args.evaluate_test
        else None
    )
    log.info(
        "Training set mean: {}, std:{}".format(
            train_dataset.mean(args.target), train_dataset.std(args.target)
        )
    )
    task_mean, task_std = 0, 1
    if args.standardize:
        task_mean = train_dataset.mean(args.target)
        task_std = train_dataset.std(args.target)
    norm_factor = [task_mean, task_std]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.architecture_spec:
        from equivariant_nas.builder import build_equiformer
        from equivariant_nas.spec import ArchitectureSpec

        spec_text = Path(args.architecture_spec).read_text(encoding="utf-8")
        architecture_spec = ArchitectureSpec.from_json(spec_text)
        model = build_equiformer(
            architecture_spec,
            equiformer_root=args.equiformer_root,
            task_mean=task_mean,
            task_std=task_std,
            atomref=None,
        ).to(device)
        log.info("Architecture ID: {}".format(architecture_spec.architecture_id()))
        log.info("Architecture spec: {}".format(architecture_spec.canonical_json()))
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
    if bool(args.inherit_checkpoint) != bool(args.inherit_parent_spec):
        raise ValueError(
            "--inherit-checkpoint and --inherit-parent-spec must be provided together"
        )
    if args.resume_step and args.inherit_checkpoint:
        raise ValueError("exact resume and inherited initialization are mutually exclusive")
    if args.inherit_checkpoint:
        if args.evaluate_test:
            raise ValueError(
                "inherited initialization is calibration-only and cannot evaluate the test split"
            )
        if not args.architecture_spec:
            raise ValueError("inherited initialization requires --architecture-spec")
        from equivariant_nas.inheritance import apply_transfer
        from equivariant_nas.spec import ArchitectureSpec

        parent_spec = ArchitectureSpec.from_json(
            Path(args.inherit_parent_spec).read_text(encoding="utf-8")
        )
        checkpoint = torch.load(args.inherit_checkpoint, map_location="cpu")
        parent_state = checkpoint["model"] if "model" in checkpoint else checkpoint
        inherited_state, inheritance_report = apply_transfer(
            parent_spec,
            architecture_spec,
            parent_state,
            model.state_dict(),
        )
        model.load_state_dict(inherited_state)
        report_path = Path(args.output_dir) / "inheritance_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(inheritance_report.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        log.info(
            "Calibration-only inherited initialization: element_coverage={:.6f}; "
            "optimizer and scheduler reset; selection_eligible=false".format(
                inheritance_report.element_coverage
            )
        )
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
        model_without_ddp = model.module if args.distributed else model
        model_without_ddp.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        global_step = checkpoint["global_step"]
        training_time_sec = checkpoint.get("training_time_sec", 0.0)
        best_step = checkpoint.get("best_step", 0)
        best_train_err = checkpoint.get("best_train_err", float("inf"))
        best_val_err = checkpoint.get("best_val_err", float("inf"))
        best_test_err = checkpoint.get("best_test_err")
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if torch.cuda.is_available() and checkpoint.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
        np.random.set_state(checkpoint["numpy_rng_state"])
        if loss_scaler is not None and checkpoint.get("loss_scaler") is not None:
            loss_scaler.load_state_dict(checkpoint["loss_scaler"])
        if model_ema is not None and checkpoint.get("model_ema") is not None:
            model_ema.module.load_state_dict(checkpoint["model_ema"])
        log.info("Resumed from {} at global_step={}".format(args.resume_step, global_step))

    job_start_global_step = global_step
    data_stream = cycling_batches(train_dataset, args, start_global_step=global_step)
    metrics_path = Path(args.output_dir) / "metrics.jsonl"
    job_start_time = time.perf_counter()

    while global_step < args.max_steps:
        next_boundary = min(
            ((global_step // args.eval_interval_steps) + 1)
            * args.eval_interval_steps,
            args.max_steps,
        )
        train_err, global_step, segment_training_sec = train_until_step(
            model=model,
            criterion=criterion,
            norm_factor=norm_factor,
            target=args.target,
            data_stream=data_stream,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            reference_steps_per_epoch=args.reference_steps_per_epoch,
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
        test_err = None
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

        improved = val_err < best_val_err
        if improved:
            best_step = global_step
            best_train_err = train_err
            best_val_err = val_err
            best_test_err = test_err

        wall_time_sec = time.perf_counter() - job_start_time
        record = {
            "global_step": global_step,
            "reference_epoch": global_step / args.reference_steps_per_epoch,
            "train_mae": train_err,
            "val_mae": val_err,
            "lr": optimizer.param_groups[0]["lr"],
            "training_time_sec": training_time_sec,
            "wall_time_sec_current_job": wall_time_sec,
            "start_global_step": job_start_global_step,
            "steps_executed_current_job": global_step - job_start_global_step,
            "best_step": best_step,
            "best_val_mae": best_val_err,
        }
        if args.evaluate_test:
            record["test_mae"] = test_err
            record["best_test_mae"] = best_test_err
        log.info("Step summary: {}".format(json.dumps(record, sort_keys=True)))
        if is_main_process:
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
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

    final_summary = {
        "status": "completed",
        "batch_size": args.batch_size,
        "global_step": global_step,
        "max_steps": args.max_steps,
        "reference_steps_per_epoch": args.reference_steps_per_epoch,
        "effective_data_cycles": global_step / (len(train_dataset) // args.batch_size),
        "training_time_sec": training_time_sec,
        "wall_time_sec_current_job": time.perf_counter() - job_start_time,
        "start_global_step": job_start_global_step,
        "steps_executed_current_job": global_step - job_start_global_step,
        "best_step": best_step,
        "best_val_mae": best_val_err,
        "test_evaluated": bool(args.evaluate_test),
        "inherited_initialization": bool(args.inherit_checkpoint),
        "selection_eligible": False if args.inherit_checkpoint else True,
        "final_training_allowed": False if args.inherit_checkpoint else True,
    }
    if args.evaluate_test:
        final_summary["best_test_mae"] = best_test_err
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
