# Online V3 20k/60 实施状态

更新时间：2026-08-16

实施分支：`codex/v3-online-20k-60`

协议哈希：`2ad771d6ca161bd9e83d6c4cbceeeaa796910177348730f6a9089c565600acac`

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

## 尚未完成的正式启动门

- 目标 A100 服务器 `115.190.90.101:56144` 当前 TCP 超时；
- 因服务器不可达，尚未停止远端 `tmux gpu_scheduler`；
- 尚未在 `/mlplatform` 执行 1～10 step checkpoint 保存与恢复 smoke；
- 正式 60 轮长训练未启动；
- 最终 Test evaluation-only adapter 应在 250k Validation 赢家冻结后单独审核，不在搜索入口中隐式执行。

服务器恢复后必须先完成上述前三项，再允许正式启动。
