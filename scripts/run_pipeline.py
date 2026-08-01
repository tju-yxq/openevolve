#!/usr/bin/env python
import argparse
import json

from equivariant_nas.dsl.pipeline import evaluate_dsl_candidate_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--program", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--equiformer-root", required=True)
    parser.add_argument("--equiformer-v2-root", default="")
    parser.add_argument("--dsl-task-contract", default="")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-symmetry", action="store_true")
    parser.add_argument("--resume-checkpoint", default="")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-subset-file", default="")
    parser.add_argument("--eval-interval-epochs", type=int, default=0)
    parser.add_argument("--data-epoch-origin-step", type=int, default=0)
    parser.add_argument("--allow-data-transition", action="store_true")
    parser.add_argument("--resume-model-only", action="store_true")
    parser.add_argument("--lr-schedule-origin-step", type=int, default=0)
    parser.add_argument(
        "--allow-experimental-generic-lowering",
        action="store_true",
        help="Enable audit-only generic graph Lowering; never implied for formal ranking.",
    )
    args = parser.parse_args()
    result = evaluate_dsl_candidate_pipeline(
        program_path=args.program,
        project_root=args.project_root,
        equiformer_root=args.equiformer_root,
        data_path=args.data_path,
        max_steps=args.max_steps,
        seed=args.seed,
        run_symmetry=not args.skip_symmetry,
        resume_checkpoint=args.resume_checkpoint,
        batch_size=args.batch_size,
        train_subset_file=args.train_subset_file,
        eval_interval_epochs=args.eval_interval_epochs,
        data_epoch_origin_step=args.data_epoch_origin_step,
        allow_data_transition=args.allow_data_transition,
        resume_model_only=args.resume_model_only,
        lr_schedule_origin_step=args.lr_schedule_origin_step,
        equiformer_v2_root=args.equiformer_v2_root,
        task_contract_path=args.dsl_task_contract,
        allow_experimental_generic_lowering=args.allow_experimental_generic_lowering,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
