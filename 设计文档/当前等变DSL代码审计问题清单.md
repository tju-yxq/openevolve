# 当前等变DSL代码审计问题清单

## 一、文档目的

本文档用于向后续接手Agent说明当前`equivariant-nas`代码的真实完成度、关键缺口和建议修复顺序。

审计结论不是“代码完全不可用”。当前系统已经完成可信的Equiformer V1父代和readout级DSL子代展示闭环，但尚未完成通用等变架构自进化系统。尤其不能把“类型检查通过、能够训练、输出近似不变”直接等同于“数值语义忠实且能够发现V2类核心结构”。

本次审计为只读审计，没有修改源码或启动服务器训练。

## 二、总体结论

当前代码可以可信支持以下有限结论：

> 系统能够从真实官方Equiformer V1父代出发，通过Router、Critic和Synthesizer三阶段LLM调用，在编译器强制的`v1_readout`区域内生成多层readout子代，并使用`exact_hybrid`后端完成真实训练、验证和对称性审计。

当前代码不能支持以下强结论：

> 系统已经完成多因子、多区域、多修改半径的等变架构自进化，能够从Equiformer V1自动发现Equiformer V2类核心算子，并让DSL语言随成功候选持续进化。

当前完成度可以概括为：

| 模块 | 当前状态 |
|---|---|
| 等变类型系统、AST和JSON Schema | 已完成 |
| Typed Patch和不可变Scope | 已完成 |
| 区域外结构Hash审计 | 已完成 |
| 官方Equiformer V1父代精确执行 | 已完成 |
| V1 readout混合子代 | 已完成 |
| Router→Critic→Synthesizer | 已完成，但Router目前只有一个可选区域 |
| 通用e3nn计算图后端 | 部分完成，存在数值语义缺失 |
| Equiformer V2 SO(2)局部适配器 | 原型完成，未形成完整V2搜索 |
| 多Region搜索 | 未完成 |
| 因子×修改半径搜索 | 未完成 |
| DSL多保真`8→4→2`控制器 | 未完成 |
| 学习到的新Motif进入下一轮搜索 | 未完成 |
| 失败经验反馈给LLM | 未完成 |
| 通用候选级精确恢复 | 部分完成 |
| 多Seed论文级验证 | 未完成 |

## 三、必须优先解决的严重问题

### P0-1：通用后端静默忽略Cutoff语义

代码位置：

- `equivariant_nas/dsl/backends/e3nn_backend.py:378`
- `equivariant_nas/dsl/registry.py:462`
- `equivariant_nas/dsl/registry.py:558`

当前后端把`core.cutoff_envelope`列入支持操作，但实际执行为：

```python
elif op == "core.cutoff_envelope@1":
    output = resolved["x"][0]
```

因此：

- `cutoff`参数不影响数值结果；
- `order`参数不影响数值结果；
- 负数或无效cutoff也没有被静态拒绝；
- 两个语义不同的DSL程序可能执行成相同模型；
- 类型检查、等变测试和训练都可能通过，但程序语义没有被忠实执行。

这与早期简化e3nn后端产生巨大MAE的问题属于同一类风险：可执行不等于数值忠实。

验收标准：

1. 实现真实cutoff envelope数值公式。
2. 对`cutoff>0`、合法`order`进行静态检查。
3. 增加单元测试，证明改变cutoff会改变前向结果。
4. 增加边界测试，证明距离超过cutoff时权重按定义衰减或归零。
5. 未完成前禁止该Primitive进入正式NAS排名。

### P0-2：`exact_node_graph`标签过度声明

代码位置：

- `equivariant_nas/dsl/compiler.py:121`

当前所有非Legacy V1程序都会直接获得：

```text
mode=exact_node_graph
backend_family=e3nn_graph
backend_semantics_version=e3nn-graph-v1
```

该判断发生在具体后端构建和数值语义核验之前。结合Cutoff被实现成Identity的问题，`exact`这一名称会产生错误可信声明。

验收标准：

1. 将当前模式暂时改为`experimental_node_graph`或`partial_node_graph`。
2. `plan_lowering`必须读取具体后端的Support Report。
3. 每个Primitive必须同时具备静态类型实现、数值实现和语义一致性测试，才能进入`exact`集合。
4. Lowering Plan必须记录不支持、近似实现或未认证的节点。

### P0-3：当前搜索只有一个Region和一个Motif

代码位置：

- `equivariant_nas/dsl/regions.py:44`

当前只注册：

```text
Region：v1_readout
Allowed Op：motif.v1_multilevel_readout
```

因此LLM实际只能选择一个较早的Block作为辅助readout。它不能修改：

- Attention；
- Message Passing；
- Tensor Product；
- Radial Path；
- Norm和Gate；
- Representation；
- Macro结构；
- V2类SO(2)计算路径。

由于只有一个Region，Router调用目前不具有真实路由作用。

验收标准：

1. 注册至少`readout`、`attention`、`message`、`norm_gate`四类可信Region。
2. 每个Region至少包含Identity和两个可信Motif。
3. Router在测试中能够从多个Region中选择不同目标。
4. 编译器验证Region外节点、边、属性和输出契约保持不变。
5. 每个Region必须声明可执行Backend Capability。

### P0-4：最新“因子×修改半径”设计尚未落到代码

当前设计已经把两个维度区分为：

```text
横向因子：Representation、Operator、Action、Macro、Topology/Readout
纵向修改半径：R1、R2、R3、R4
```

但当前`RegionDefinition`和LLM协议中没有：

- `factor`；
- `mutation_radius`；
- `required_controls`；
- `factorial_pair`；
- R3联合变异的`P/A/B/AB`对照；
- R4新算子准入条件。

验收标准：

1. 新增Factor Registry和Mutation Radius枚举。
2. Router输出必须包含`factor`、`region_id`和`mutation_radius`。
3. R3变异必须自动生成父代、A、B、AB四类对照。
4. R4必须经过算子级、Block级等变测试和数值健康检查，才能进入短训练。

## 四、搜索工作流没有闭环的问题

### P1-1：语言进化是离线模块，新Motif没有进入下一轮搜索

代码位置：

- `scripts/run_dsl_evolution.py:225`
- `scripts/run_dsl_language_evolution.py:115`
- `equivariant_nas/dsl/evidence_store.py:248`

Motif发现、Held-out Replay和Admission已经实现，也能把新语言写入SQLite。

但每次搜索启动时仍然执行：

```python
primitives = core_registry()
motifs = reference_motif_registry()
```

搜索入口没有从Evidence Store加载上一轮发布的Language Version和Learned Motif。

当前流程实际是：

```text
搜索
→ 离线发现Motif
→ 写入SQLite
→ 结束
```

目标流程应为：

```text
搜索Cycle N
→ 发现并发布Motif
→ Cycle N+1加载新语言
→ LLM使用新Motif生成候选
→ 比较新语言是否提高合法率和搜索效率
```

验收标准：

1. `run_dsl_evolution.py`支持`--language-version`和`--evidence-store`。
2. 搜索启动时恢复完整Motif Registry并校验Hash。
3. Active Vocabulary中真实出现Learned Motif。
4. 测试证明下一Cycle能够生成包含新Motif的候选。
5. 无法恢复完整Motif定义时必须拒绝启动。

### P1-2：DSL搜索没有接入`8→4→2`多保真控制器

当前`run_quarter_multifidelity_cycles.py`仍然处理旧的Python候选：

```text
iteration_xxxx.py
```

DSL搜索输出为：

```text
iteration_xxxx_<architecture_id>.dsl.json
```

`run_dsl_evolution.py`只支持为所有候选设置同一个`max_steps`。8000-step父子实验使用的是专用Showcase Controller，不是通用多候选DSL搜索。

验收标准：

1. 新增DSL原生多保真控制器。
2. 第一阶段生成8个有效DSL候选并训练至8000 steps。
3. Top 4从原Checkpoint续训至80000 steps。
4. Top 2切换完整训练集并续训至250000 steps。
5. 所有晋级只使用Validation，Test全程锁定。
6. Fidelity、数据子集、Checkpoint和Architecture ID全部强绑定。

### P1-3：失败候选没有形成LLM可用的结构化记忆

当前LLM证据主要来自父代和少量Inspirations，失败候选通常只保存在`evolution.jsonl`中。

系统没有系统整理：

- 哪个Region经常生成非法Patch；
- 哪类Primitive经常导致数值爆炸；
- 哪类Lowering失败；
- 哪类机制多次无收益；
- 哪些Repair建议曾经成功。

验收标准：

1. 将失败原因标准化为Failure Evidence。
2. 按Factor、Region、Radius、Primitive和Diagnostic Code聚合。
3. Router和Critic Prompt必须接收相关成功与失败证据。
4. 记录失败证据是否降低后续同类错误率。

### P1-4：通用搜索没有候选级精确恢复

搜索数据库可以在迭代边界恢复，但如果进程在候选训练中崩溃：

- `current_candidate.json`可能仍存在；
- 训练Checkpoint可能已经保存；
- 重启后不会优先恢复同一候选；
- 系统可能重新调用LLM生成另一个候选；
- 部分训练目录会成为孤立证据。

验收标准：

1. 启动时检查`current_candidate.json`。
2. 若候选程序、协议Hash和Checkpoint完整，优先恢复原候选。
3. 恢复完成后再推进OpenEvolve迭代号。
4. 确定性失败达到上限后才允许放弃候选。

## 五、实验可信性和证据链问题

### P1-5：V2后端身份记录错误

代码位置：

- `equivariant_nas/dsl/backends/qm9_model.py:102`
- `equivariant_nas/dsl/backends/qm9_model.py:118`
- `equivariant_nas/dsl/backends/equiformer_v2_backend.py:243`

提供`equiformer_v2_root`时，运行时可能使用`EquiformerV2GraphBackend`，但模型元数据仍从静态Lowering Plan读取：

```text
backend_family=e3nn_graph
backend_semantics_version=e3nn-graph-v1
```

因此实际使用V2 SO(2)融合时，结果仍可能错误报告为普通e3nn后端。

此外，代码只要发现V2目录存在，就记录固定Commit字符串，没有验证真实Git Commit或文件Hash。

验收标准：

1. Backend实例返回真实Backend Family和Semantics Version。
2. Pipeline使用实际Graph Backend元数据，不使用静态占位值。
3. 核验V2源码Git Commit或关键文件SHA-256。
4. V1官方源码也必须记录Commit或文件Hash。
5. Backend源码身份进入Protocol ID和结果文件。

### P1-6：Architecture ID没有绑定实际Backend语义

`architecture_id`支持`backend_semantics_version`参数，但`Compiler.analyze`使用默认`backend-neutral-v1`。

同一AST可能：

- 在普通e3nn后端执行；
- 在V2融合后端执行；
- 产生不同数值语义；
- 仍共享同一个Architecture ID。

验收标准：

1. 区分Backend-neutral Program ID和Executable Architecture ID。
2. Executable ID必须包含Lowering Plan Hash、Backend Semantics和源码身份。
3. Evidence Store不能把不同Backend结果误合并为同一可执行架构。

### P1-7：外部Initial Metrics绕过协议与Test隔离核验

代码位置：

- `scripts/run_dsl_evolution.py:311`

通过`--initial-metrics`传入的结果会被直接加入OpenEvolve数据库，没有检查：

- `test_evaluated`；
- Architecture ID；
- Seed；
- Fidelity；
- 训练子集；
- Lowering模式；
- Protocol Hash；
- Checkpoint身份。

这可能让Test指标或其他实验的结果影响父代选择。

验收标准：

1. Initial Metrics必须包含签名完整的Protocol Manifest。
2. 强制验证`test_evaluated=false`。
3. 强制验证Architecture ID和Executable ID。
4. 强制验证Task、Seed、Fidelity和数据集身份。
5. 不匹配时拒绝加入数据库。

### P1-8：缓存可能复用旧代码产生的结果

代码位置：

- `equivariant_nas/dsl/pipeline.py:205`
- `equivariant_nas/dsl/pipeline.py:230`

当前DSL Protocol ID没有完整绑定：

- 当前Git Commit；
- Trainer代码Hash；
- Equiformer V1源码Hash；
- e3nn后端代码Hash；
- 数据路径和数据版本；
- `allow_data_transition`；
- `resume_model_only`；
- 完整训练超参数；
- Symmetry Warning阈值。

若已有有效`result.json`，Pipeline可能直接返回旧缓存。

验收标准：

1. 生成完整Runtime Manifest。
2. Protocol ID包含代码、数据、Backend和训练配置Hash。
3. 缓存命中前逐项比较Manifest。
4. 任一关键字段变化时创建新Run，不复用旧结果。

### P1-9：Showcase的LLM证据没有强绑定实际子代

代码位置：

- `scripts/run_dsl_showcase_pair.py:32`

Showcase Controller检查搜索目录中只有一个带Patch的记录，但没有重新编译`child_program`并比较Architecture ID。

理论上可能出现：

```text
Generation Evidence来自子代A
实际训练传入子代B
```

只要B同样属于`exact_hybrid`，当前控制器不会发现证据错配。

验收标准：

1. 重新编译`child_program`。
2. 比较Generation Record、Program文件、Evaluator结果和Checkpoint中的Architecture ID。
3. 四者必须完全一致。
4. Generation Evidence应记录Prompt Run ID和Program文件SHA-256。

## 六、等变性审计不足

### P1-10：对称性硬阈值过松

当前硬拒绝阈值约为：

```text
relative symmetry error > 0.25
```

即相对误差在`0.01～0.25`之间时，候选只产生Warning，仍然：

- `valid=true`；
- 只按Validation MAE排名；
- 可以成为OpenEvolve最佳候选。

对于等变架构自进化，25%的相对误差不能作为可信硬门槛。

### P1-11：数值审计采样量过小

当前审计主要使用：

- 一个Batch；
- 两个分子；
- 一次随机旋转；
- 一次固定平移；
- 一次节点逆序。

这属于Smoke Test，不是严格等变认证。

验收标准：

1. 使用预注册的多分子审计集。
2. 每个候选测试多次随机旋转、平移和置换。
3. 区分绝对误差、相对误差和接近零输出时的稳定尺度。
4. 对新算子同时执行算子级、Block级和整网级审计。
5. 正式搜索硬阈值应根据官方V1数值噪声分布校准，而不是直接设为0.25。

## 七、测试与工程问题

### P2-1：测试数量较多，但关键运行时测试在本地被跳过

使用正确命令：

```powershell
$env:PYTHONPATH='.'
pytest -q -ra tests
```

当前结果为：

```text
128 passed, 6 skipped
```

跳过的恰好包括：

- e3nn运行时；
- QM9模型运行；
- Dropout等变性；
- 非线性运行；
- V2运行；
- Trainer入口。

现有测试没有覆盖：

- Cutoff是否真实生效；
- Backend身份是否正确；
- 缓存是否随代码版本失效；
- Initial Metrics是否泄漏Test；
- Learned Motif是否进入下一轮搜索；
- DSL多保真晋级。

### P2-2：默认`pytest`命令无法可靠运行

直接执行`pytest`会出现：

- 项目包无法导入；
- 误收集`第三方源码/SPARK`中的测试；
- 第三方依赖缺失导致大量收集错误。

原因是`pyproject.toml`没有配置：

```text
pythonpath
testpaths
norecursedirs
```

验收标准：

1. 在Pytest配置中固定`testpaths=tests`。
2. 排除`第三方源码`、`reports`和`tmp`。
3. 配置项目根目录导入或以Editable模式安装。
4. 建立CPU静态测试和服务器运行时测试两套CI任务。

### P2-3：训练Python路径硬编码

DSL Pipeline直接使用：

```text
/home/20262202788/conda-envs/equiformer/bin/python
```

这会影响：

- 新服务器迁移；
- 本地测试；
- 其他用户复现；
- 环境版本升级。

验收标准：

1. 优先读取`EQUIFORMER_PYTHON`环境变量。
2. 将Python解释器路径写入Runtime Manifest。
3. 启动前验证Python、PyTorch、CUDA、e3nn和timm版本。

## 八、当前真实工作流

当前实际运行链路为：

```text
官方Equiformer V1父代
→ 获取唯一v1_readout Region
→ Router形式化选择该Region
→ Critic分析多层readout假设
→ Synthesizer选择一个辅助Block
→ 生成预定义multilevel-readout Motif
→ 编译器检查Region外Hash
→ exact_hybrid构建
→ 单一固定Step训练
→ Validation MAE写入OpenEvolve
```

该工作流已经完成并得到一个可信的单Seed积极结果，但它主要验证的是：

> 受控DSL局部编辑能否产生合法、保持等变约束并可训练的子代。

它没有验证：

> LLM能否在多个等变功能因子和多个结构区域中搜索，并发现V2类核心算子。

## 九、目标完整工作流

建议最终实现：

```text
冻结任务、数据、训练和预算协议
→ Factor Router选择科学探索方向
→ Radius Router选择R1～R4修改半径
→ Region Router选择合法结构区域或区域组合
→ Critic读取成功与失败证据
→ Synthesizer生成Typed Patch
→ Compiler检查类型、Region、Frozen Hash和Backend Capability
→ 算子级与Block级等变审计
→ 数值健康和资源门控
→ 8k低保真训练
→ Top 4续训至80k
→ Top 2续训至250k
→ 更新OpenEvolve种群与因子成功率
→ 从成功谱系发现新Motif
→ Held-out Replay和Admission
→ 发布新语言版本
→ 下一Cycle加载新语言继续搜索
```

## 十、建议修复顺序

### 第一阶段：恢复语义可信性

1. 修复Cutoff和所有近似Primitive。
2. 将`exact_node_graph`降级，建立逐Primitive语义认证。
3. 修复实际Backend身份和源码Hash记录。
4. 修复Architecture ID与Executable ID。
5. 修复缓存Manifest。

### 第二阶段：实现真正的多方向搜索

1. 建立Factor Registry。
2. 建立多Region Registry。
3. 建立R1～R4修改半径。
4. 实现R3的P/A/B/AB对照。
5. 让Router拥有真实可选空间。

### 第三阶段：统一搜索与训练调度

1. 将DSL候选接入`8→4→2`多保真控制器。
2. 实现候选级精确恢复。
3. 加强Test隔离和Initial Metrics核验。
4. 将失败证据反馈给Router和Critic。

### 第四阶段：完成语言自进化闭环

1. 从Evidence Store恢复新语言。
2. 下一Cycle真实使用Learned Motif。
3. 比较语言进化前后的合法率、创新率和搜索效率。
4. 防止Motif只对单一任务或单一谱系过拟合。

### 第五阶段：论文级验证

进行同预算对照：

```text
Raw Code
Factor Spec
Static DSL
Evolving DSL
```

至少测量：

- 解析成功率；
- 编译成功率；
- 等变审计通过率；
- 首次生成成功率；
- 平均Repair次数；
- Unique Rate；
- Mechanism Novel Rate；
- Useful Novel Rate；
- 找到优于V1候选所需的LLM调用数；
- 找到目标MAE所需GPU小时；
- 多Seed稳定性；
- 跨QM9目标或跨数据集泛化性。

## 十一、交给接手Agent的核心要求

接手Agent不应只增加更多类、Prompt或单元测试，而应优先回答以下问题：

1. 每个DSL Primitive是否真的执行了其声明的数值语义？
2. 每个正式候选是否使用可核验的Backend和源码版本？
3. Router是否真的有多个因子和Region可以选择？
4. DSL候选是否进入统一的多保真搜索？
5. 新语言是否真实影响下一轮候选生成？
6. 失败实验是否会改变后续搜索行为？
7. Test、缓存、Checkpoint和LLM证据是否与候选身份强绑定？
8. 最终结果是否证明了核心等变结构创新，而不仅是Readout级改动？

只有以上问题得到代码和实验层面的肯定回答，才能将系统称为完整的LLM驱动等变架构自进化工作流。
