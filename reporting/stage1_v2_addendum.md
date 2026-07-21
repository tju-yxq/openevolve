## 14. 第一阶段框架补强：从双阶段提示升级为有记忆的自进化闭环

在第一版报告之后完成的代码审计发现：如果 ECFR 每次从空统计开始，低预算实验的前几次提案主要用于轮询尚未尝试的因子；同时，IACC 虽然能够计算 sibling 反事实，但反事实主效应尚未自动回写路由器。这两个问题会让“证据驱动的自进化”弱化成一次性的提示工程。当前版本新增以下两个机制：

1. **Frozen Evidence Memory（FEM）**：把第一阶段因子统计冻结为机器可读先验，并记录两份原始证据的 SHA-256。每个因子都有一个中性虚拟观测，避免未观测因子产生无穷 UCB；ACTION 使用实测有害增益 `-0.546062`，OPERATOR 使用 IACC 隔离后的 standalone gain `0.177094`，而不是受 ACTION 交互污染的 rescue gain `0.659713`。该先验仅供 full evidence-router 使用，uniform-router 与 typed-random 对照不读取。
2. **Trajectory-Conditioned Reflect–Edit（TCRE）**：在 SPARK 式 RC/SAR 之前，由可信代码生成有界谱系摘要，包括 lineage depth、因子历史、同 fidelity 验证 MAE 历史、近期最优改善和 plateau status。证据不足时 plateau 必须为 `unknown`；因此零步可行性检查不能被 LLM 错称为收敛或平台期。只有出现实测平台期时，SAR 才被允许在已选因子内部采用更探索性的修改。
3. **IACC credit ingestion**：已经解决的反事实主效应会以唯一 key 回写路由器一次；interaction-specific rescue 不再被当作一般因子收益重复强化。
4. **Bounded scientific context**：父代和 inspiration 只向 LLM 暴露白名单指标；大型逐层对称性张量被压缩为每层最大值，减少 API token 成本，并防止无关细节淹没科学证据。

### 14.1 零 GPU 真实闭环证据

在服务器运行了 `seed=117`、2 次提案的端到端 smoke，强制设置 `NAS_GPU_BUDGET_HOURS=0`、`max_steps=0` 且跳过 CUDA 对称性诊断。该测试真实调用 OpenEvolve `ProgramDatabase`、LLM ensemble、FEM/ECFR、TCRE、SPAG compiler、可信 Equiformer builder 和 MAP-Elites。

| 指标 | 结果 |
|---|---:|
| 新提案数 | 2 |
| 合法且唯一候选 | 2/2 |
| 总 archive 程序数 | 3 |
| 占据的等变容量 cells | 2 |
| 第一代修改因子 | REPRESENTATION |
| 第二代修改因子 | MACRO |
| 第二代观测 lineage depth | 1 |
| 第二代观测因子历史 | `[REPRESENTATION]` |
| 零证据 plateau 判定 | `unknown` |
| 三次构建累计 GPU 秒数 | `0.0` |

两代候选分别达到参数比 `1.142023`，均低于 `1.2×` 硬上限。第二代记录证明谱系信息确实穿过 OpenEvolve parent-child 链进入 RC/SAR，而不是仅存在于独立单元测试。router state 同时保存先验名称、冻结标志和证据哈希。

证据文件：

- `reports/stage1/evidence/memory_smoke/evolution.jsonl`
- `reports/stage1/evidence/memory_smoke/summary.json`
- `reports/stage1/evidence/memory_smoke/router_state.json`
- `reports/stage1/evidence/memory_smoke/database_metadata.json`
- `reports/stage1/evidence/memory_smoke/budget_ledger.jsonl`

### 14.2 更新后的严谨结论

该补强证明 OpenEvolve、SPARK 的历史条件化 reviewer/editor 思路和类型化 Equiformer 搜索已经形成带持久证据记忆、谱系上下文和反事实信用回写的可执行闭环。它显著强于“两个 LLM 顺序调用”的表面融合。但零 GPU smoke 仍只证明工程语义和控制逻辑；FEM/ECFR/TCRE 是否提高 5,000-step 搜索效率，仍必须由预注册的 matched controls 和独立 search seeds 验证，不能提前声称达到 CCF-A oral 证据标准。

## 15. ISWT：等变语义约束权重继承

为了降低架构搜索的单候选成本，本阶段新增 Irrep-Semantic Weight Transfer（ISWT）。它不是按同名同形状盲目复制 checkpoint，而是将 state-dict 兼容性与改变因子的语义约束求交：表示阶数变化时重置全部表示相关状态；Gaussian/Bessel、basis 数量或 radial hidden 改变时重置径向基和径向网络；归一化语义变化时重置 normalization 参数；optimizer 和 scheduler 永不继承。

现有 baseline → Bessel/64 checkpoint 的零训练审计：

| 项目 | 结果 |
|---|---:|
| 子模型 state tensors | 413 |
| 安全继承 tensors | 342 |
| tensor coverage | 82.81% |
| 子模型 state elements | 3,590,210 |
| 安全继承 elements | 3,093,314 |
| element coverage | 86.16% |
| blocked radial-semantic tensors | 63 |
| blocked shape-changed tensors | 7 |
| selection eligible | false |
| final training allowed | false |

这项覆盖率只证明“有一部分状态在语义上可以安全复用”，不证明继承短跑可以作为搜索代理。`configs/inheritance_calibration.json` 预注册至少 8 个跨因子 paired candidates，对 inherited 与 scratch 的 5,000-step 排名比较 Spearman、Kendall、top-1 recall 和 normalized selection regret；所有阈值通过前才可能解锁代理选择权。继承模式禁止 test split，且最终 257,700-step × 3 seeds 永远从头训练。

证据文件：`reports/stage1/evidence/state_transfer/baseline_to_bessel64.json`。
