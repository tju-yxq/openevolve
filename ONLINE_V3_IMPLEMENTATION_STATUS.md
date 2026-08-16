# Online V3 20k/60 实施状态

更新时间：2026-08-16

实施分支：`codex/v3-online-20k-60`

协议哈希：`8de5571b11dacfffe09818e8738bee6b3302450b187588dc17e7971699435e94`

## 已完成

- 新增冻结机器协议 `configs/dsl_v3_online_20k_60_protocol.json`；
- 新增 60×20k → 15×80k → 10×250k 的崩溃可恢复状态机；
- 每轮仅生成并训练一个 child，20k 完成后才进入后续父代采样池；
- 新增五路 fidelity-aware parent sampling，并使用同 fidelity rank 权重避免负 MAE 权重退化；
- 新增 Top 10 + novelty 3 + lower-half random 2 的原子选择快照；
- 新增 80k Top 10、跨保真 Spearman 报告、250k Validation 唯一赢家冻结；
- 新增 Test 单次门禁和 TOS 最终归档门禁；
- 新增通用在线控制器、V3 单 child 结构变异 adapter、正式 pipeline 训练 adapter；
- 新增 `/mlplatform`、数据哈希、CUDA、环境变量和 `gpu_scheduler` 前置检查；
- 历史 `8→4→2` 协议和入口未修改。

## 已执行测试

- `python -m compileall -q equivariant_nas scripts tests`：通过；
- 新增 focused tests：`14 passed`；
- 60→15→10 完整 mock controller：通过，包括重启后不重复训练；
- 单个真实 V3 DSL structural child 离线生成：通过；
- `git diff --check`：通过；
- 全仓测试：`393 passed, 48 skipped, 28 failed`。失败项来自本地缺少仓库外的 sibling Equiformer/Equiformer-V3 checkout，以及基线分支中一个旧 Compiler API 测试不一致；新增在线测试全部通过。

## 新实例正式启动门结果

- `115.190.90.101:56144` 已于 2026-08-16 14:58（北京时间）恢复；
- 新实例 `di-20260612175839-qv4pn`、Conda `equiNAS`、A100 80GB 检查通过；
- `tmux gpu_scheduler` 当前不存在，preflight 确认未运行；
- `/mlplatform` 可写且可用空间约 2.1 TiB；
- Linux checkout 下 dataset、quarter subset 和 equivariance contract SHA-256 全部通过；
- 新实例 focused tests：`16 passed`；
- 真实 84,961,153 参数 V3 child 已在 A100 上完成 step 0→1 和 checkpoint 1→2 恢复；
- 两个 checkpoint 均写入 `/mlplatform`，恢复结果证明 `start_global_step=1`、`endpoint_step=2`、本 job 仅执行 1 step；
- 两次 smoke 均为 `valid=true`、`test_evaluated=false`。

正式 60 轮长训练仍未自动启动。最终 Test evaluation-only adapter 应在 250k Validation 赢家冻结后单独审核，不在搜索入口中隐式执行。
