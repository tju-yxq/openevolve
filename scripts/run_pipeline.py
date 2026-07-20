#!/usr/bin/env python
import argparse
import json

from equivariant_nas.pipeline import evaluate_candidate_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--program", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--equiformer-root", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-symmetry", action="store_true")
    parser.add_argument("--resume-checkpoint", default="")
    args = parser.parse_args()
    result = evaluate_candidate_pipeline(
        program_path=args.program,
        project_root=args.project_root,
        equiformer_root=args.equiformer_root,
        data_path=args.data_path,
        max_steps=args.max_steps,
        seed=args.seed,
        run_symmetry=not args.skip_symmetry,
        resume_checkpoint=args.resume_checkpoint,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
