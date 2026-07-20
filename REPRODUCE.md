# Stage-one reproduction protocol

All commands run on `volcano-equiformer`. API keys are sourced from the
permission-600 file `/home/20262202788/.config/openevolve/apis.env`; they are
never committed or copied into reports. Network access uses the SSH reverse
proxy at `127.0.0.1:12356`, not the Volcengine paid proxy.

```bash
cd /home/20262202788/equivariant-nas
export PYTHONPATH=.
export EQUIVARIANT_NAS_ROOT=/home/20262202788/equivariant-nas
```

## Verification

```bash
/home/20262202788/conda-envs/openevolve/bin/python -m pytest -q
/home/20262202788/conda-envs/openevolve/bin/python -m py_compile \
  equivariant_nas/*.py scripts/*.py reporting/generate_stage1_report.py
```

## Zero-GPU schema / quality-diversity smoke

```bash
scripts/run_schema_smoke.sh 48 8
```

The smoke uses `--max-steps 0 --skip-symmetry`: it validates LLM RC/SAR,
compiler/duplicate repair, typed construction and the equivariance-aware
MAP-Elites descriptors without spending training GPU budget.

## Warmup-end factorized cohort

```bash
source /home/20262202788/.config/openevolve/apis.env
export http_proxy=http://127.0.0.1:12356
export https_proxy="$http_proxy"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$http_proxy"

/home/20262202788/conda-envs/openevolve/bin/python \
  scripts/run_factorized_evolution.py \
  --initial-program openevolve_adapter/initial_program.py \
  --evaluator-file openevolve_adapter/evaluator.py \
  --config /home/20262202788/openevolve/configs/local_glm_5_2.yaml \
  --output runs/stable_factorized_seed45 \
  --iterations 2 \
  --max-steps 5000 \
  --seed 45 \
  --repair-attempts 1 \
  --initial-metrics runs/candidates/8639c8a64dad5d25/seed0_steps5000_87020e2c61/result.json
```

## Matched random cohort

```bash
/home/20262202788/conda-envs/equiformer/bin/python \
  scripts/run_random_search.py \
  --output runs/stable_random_seed45 \
  --max-steps 5000 \
  --seed 45 \
  --valid-target 2 \
  --max-proposals 10
```

Both methods use seed 0 inside the trusted trainer, the same shuffled data
stream, validation-only selection, batch size 64 and a 5,000-step endpoint.

## Interaction counterfactual

The observed rescue chain triggers IACC. Reconstruct the request from the
audited endpoints and evaluate the sibling under the same protocol:

```bash
/home/20262202788/conda-envs/openevolve/bin/python \
  scripts/make_counterfactual_request.py \
  --ancestor-program openevolve_adapter/initial_program.py \
  --child-program runs/stable_factorized_seed45/candidates/iteration_0002.py \
  --selected-factor OPERATOR \
  --ancestor-mae 0.7535404392242432 \
  --parent-mae 1.2996022186279297 \
  --child-mae 0.6398895030975342 \
  --output runs/stable_factorized_seed45/counterfactual_requests.jsonl

/home/20262202788/conda-envs/equiformer/bin/python \
  scripts/run_counterfactual_requests.py \
  --requests runs/stable_factorized_seed45/counterfactual_requests.jsonl \
  --output runs/stage1_interaction_ablation \
  --max-steps 5000 \
  --seed 0
```

## Fidelity trust evidence

```bash
/home/20262202788/conda-envs/openevolve/bin/python \
  scripts/assess_fidelity_trust.py \
  --pairs configs/stage1_fidelity_pairs.json \
  --output runs/fidelity_trust_300_to_5000.json \
  --top-k 1 \
  --minimum-cohort 8
```

## Report

```bash
/home/20262202788/conda-envs/equiformer/bin/python \
  reporting/generate_stage1_report.py
```

Search and calibration commands do not pass `--evaluate-test`. The 257,700-step
three-seed final protocol is intentionally absent from all automatic scripts.
