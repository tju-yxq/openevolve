# 等变NAS功能因子本体与单因子到多因子路由设计

> 文档性质：文献证据版设计基线
>
> 当前结论：创新率Benchmark暂缓；优先完成因子划分、单因子选择和受控交互验证
>
> 适用对象：Equiformer V1/V2、QM9及后续二维/三维等变网络DSL
>
> 更新日期：2026年7月25日
> 原文证据：[等变NAS因子划分文献证据矩阵](../调研报告/等变NAS因子划分文献证据矩阵.md)

## 一、先说结论

用户提出的判断是正确的，但需要再精确一步。当前真正需要解决的不是一个问题，而是按依赖顺序排列的三个问题：

1. **因子发现与划分**：等变网络中哪些功能机制应当成为独立因子？每个因子拥有哪些计算节点、允许读取哪些边界输入、必须保持什么输出合同？
2. **单因子选择**：在父代架构和搜索历史已知时，本轮应该修改哪个因子？
3. **交互升级**：完成单因子主效应估计后，哪些因子之间存在足够证据，允许进行受控双因子或多因子修改？

其中第一项决定后两项是否成立。如果因子本身存在功能重叠，那么Router即使每轮只选择一个标签，实际修改仍会跨越多个机制，功能纠缠并没有被消除。

因此，当前方案不能继续把`REPRESENTATION/OPERATOR/ACTION/MACRO`直接当成最终因子，也不能把R1至R4修改半径当成并行搜索方向。正确结构应为：

```text
文献与架构计算图
→功能因子本体
→类型化区域与所有权
→强制单因子干预
→单因子主效应估计
→因子选择器
→交互证据检验
→受控双因子修改
→必要时才开放更高阶组合
```

## 二、为什么SPARK的因子不能直接移植

SPARK在CLRS程序架构中手工定义了`OPERATOR`和`ACTION`两个因子：前者负责算子选择与参数化，后者负责算子的调用、组合、消息连线、控制流和掩码。论文要求两个代码区域互斥，每轮ASR只选一个因子，未选区域保持一致。

但是SPARK论文也明确把“当前只使用两个手工定义区域，未来需要更细粒度或自动发现因子”列为局限。因此，SPARK证明的是**因子隔离编辑有价值**，而不是证明`OPERATOR/ACTION`是跨任务通用的最佳因子划分。

对等变网络而言，笼统的`OPERATOR`至少混合了：

- 几何基展开；
- 等变张量耦合；
- 标量注意力路由；
- 邻居聚合；
- 节点内状态变换。

笼统的`ACTION`又可能混合：

- 残差组合；
- 归一化；
- 等变非线性；
- 随机深度和Dropout；
- block级控制结构。

所以，照搬两个标签只会把功能纠缠重新包装成两个更大的编辑框。

## 三、论文共同给出的计算分解证据

这里不按研究者经验直接命名因子，而是比较不同等变架构论文如何反复拆解计算过程。

| 论文 | 原文中的主要计算分解 | 对因子划分的直接启示 |
|---|---|---|
| Tensor Field Networks | 几何滤波器$R(r)Y_l(\hat r)$、Clebsch–Gordan张量积、邻居卷积、自相互作用、等变非线性 | 几何编码、等变耦合、节点内更新不是同一机制 |
| e3nn | irreps作为数据类型、球谐作为方向表示、tensor product定义合法类型交互 | 表示模式首先是类型系统，不应当和普通局部算子混为一类 |
| SE(3)-Transformer | 不变注意力权重、等变value message、邻居聚合、自相互作用 | 路由权重和等变消息核可以独立变化 |
| EGNN | edge operation、coordinate operation、aggregation、node operation | 边消息、坐标更新、聚合、节点更新是不同角色；坐标更新应是可选能力 |
| DimeNet/GemNet | 径向/角向基、方向消息传递、聚合、原子更新、输出块 | 几何基和消息路径不能合并为一个宽泛Operator |
| PaiNN | 标量/向量表示、message block、atomwise update block、输出网络 | 消息形成与状态更新可以具有稳定边界 |
| NequIP | radial basis、radial MLP、spherical harmonics、tensor product、self-interaction、residual、gate | 当前`OperatorSpec`实际覆盖了多个可分机制 |
| MACE | message construction、product basis、update、readout | body/correlation order可以独立于message-passing depth变化 |
| SEGNN | steerable attributes、equivariant message、aggregation、node update | 几何属性、消息函数和节点更新应分别建模 |
| Equiformer V1 | atom/edge-degree embedding、equivariant graph attention、FFN、readout；attention内部又含message、attention weight和neighbor sum | Transformer block不是最小因子，至少还要沿消息、路由和状态变换拆分 |
| eSCN/Equiformer V2 | SO(3)到SO(2)卷积实现、SO(2)attention、S²/separable S² activation、改进normalization和rescaling | V1到V2的创新横跨多个相互独立的机制轴，不能用一个Operator标签解释 |
| Group Equivariant CNNs | 二维G-convolution、pointwise nonlinearity、G-pooling分别定义 | F3、F5和池化/读出职责在二维场景同样可分 |
| Steerable CNNs/General E(2)-Steerable CNNs | feature types、equivariant filter banks/kernel constraints、admissible nonlinearities分别定义 | G1类型系统与F3/F5边界并非三维分子网络特例 |
| SPARK | 手工功能区域、单因子token、冻结非选区域、可行性检查 | 因子必须对应互斥编辑支持集，而不只是自然语言名称 |

上述论文共同支持一个结论：等变网络应当先按**数据角色与计算职责**拆分，再按代码区域实现；不能反过来根据现有配置类的字段位置命名因子。逐项章节、PDF页码、拆分证据和证据等级见[等变NAS因子划分文献证据矩阵](../调研报告/等变NAS因子划分文献证据矩阵.md)。

## 四、因子发现的方法：如何决定“选什么因子”

### 4.1建立统一的类型化计算图

先把代表性架构转成同一种中间表示。节点表示基本操作，边表示张量或图结构依赖，边上必须带有：

- symmetry group，例如$SO(2)$、$SE(2)$、$SO(3)$、$O(3)$或$E(3)$；
- carrier，即node、edge、graph、coordinate或field；
- irrep degree和parity；
- multiplicity和shape；
- invariant或equivariant语义；
- 是否改变邻接关系、坐标或表示类型。

只有先统一到这种计算图，二维卷积型等变网络、三维分子网络和Equiformer V1/V2才能在同一因子框架下比较。

### 4.2根据功能职责生成候选分区

对每个节点回答一个问题：它主要改变的是什么？

```text
输入实体如何进入表示空间？
几何关系如何编码？
不同表示如何进行等变耦合？
信息沿哪些边、以什么权重传播和聚合？
聚合信息如何更新节点状态？
局部表示如何形成任务输出？
```

这些问题来自多篇论文反复出现的模块分解，不是从当前Python类名中反推出来的。

### 4.3用“拆分规则”消除因子内部纠缠

出现以下任一情况，应把候选因子继续拆开：

1. 因子内部节点回答两个不同的科学问题；
2. 文献中存在只替换其中一部分而保持另一部分不变的代表架构；
3. 一个子区域具有稳定的输入输出类型，可以独立编译；
4. 同一因子内不同修改的性能响应和失败模式长期无关；
5. LLM经常需要跨越该因子内部两个远距离子图才能完成一次修改。

例如SE(3)-Transformer明确把不变attention weight和等变value message分开，因此二者不应继续合并为一个“Attention Operator”因子。

### 4.4用“合并规则”避免过度碎片化

出现以下情况，应考虑合并两个候选区域：

1. 一个区域在保持边界类型不变时无法独立形成合法候选；
2. 两个区域几乎每次都必须共同迁移；
3. 拆分后某个区域只剩没有独立机制含义的参数开关；
4. 两个区域的所有权边界高度重叠，冻结补集无法实现；
5. 单独修改任一部分只会产生结构无效，而联合修改才构成一个原子语义操作。

例如tensor product path和与之绑定的合法输出irrep选择，若编译器不能在局部自动补全类型，就应作为一个原子耦合区域，而不能硬拆成两个因子。

### 4.5因子不是单个超参数

`num_heads=4→8`只是一个参数修改；“路由与聚合”才是功能因子。一个因子可以包含多种语义一致的原语和结构改写，但每次干预必须限定在该因子的所有权范围内。

## 五、建议采用的初始因子本体

以下划分是基于20篇二维/三维等变网络、等变NAS和LLM驱动NAS论文全文综合得到的**Factor Ontology v0.3**。F1至F6是功能因子族，用于组织科学问题；它们不是每轮Router直接选择的最小编辑单位。真正的单因子路由必须继续落到5.7节定义的叶子因子。该本体具有逐项原文证据，但仍需通过编译和干预测试验证，不能写成已经由既有论文证明的唯一正确本体。

### 5.1F1：输入提升与属性初始化

**科学问题**：离散实体属性和已有物理属性如何进入等变表示空间？

**拥有内容**：

- atom/node embedding；
- 原始非几何edge attribute embedding；
- 外部给定物理属性的类型化输入，例如velocity、force或spin；
- 初始标量、向量或更高阶特征的构造。

**只读依赖**：表示类型合同和任务输入定义。

**不拥有内容**：相对位置、径向基、球谐、消息传递、readout。

单独保留该因子的原因是，论文普遍把embedding与interaction block分开，而且属性初始化可以在不改变消息核的情况下独立变化。

### 5.2F2：几何关系编码

**科学问题**：坐标和邻接关系如何转换为后续等变计算可使用的几何描述？

**拥有内容**：

- relative position和distance；
- 邻域构图与cutoff envelope；
- radial basis；
- spherical/circular harmonics；
- 角、二面角和局部frame的构造；
- 由坐标和邻域推导的steerable edge/node attributes；
- edge-degree等纯几何统计嵌入。

**输出合同**：类型明确的标量几何特征、方向特征和边索引。

**边界限制**：改变球谐最高阶并导致全网irrep模式变化时，不再是普通F2修改，而升级为全局表示迁移。

### 5.3F3：等变交互与耦合核

**科学问题**：输入表示和几何表示通过何种合法耦合产生等变消息？

**拥有内容**：

- Clebsch–Gordan/tensor product paths；
- depth-wise tensor product；
- radial-conditioned coupling weights；
- SO(3)或SO(2)卷积实现；
- edge-frame变换；
- body/correlation order；
- 等变value message的构造。

**不拥有内容**：attention softmax、邻居reduction、残差更新和readout。

Equiformer V1的tensor product与Equiformer V2的SO(2)卷积替换主要属于该因子，而不是一个笼统的全模型版本标签。

### 5.4F4：路由、选择与聚合

**科学问题**：已生成的合法消息以什么权重、沿哪些边、用什么归约规则进入目标节点？

**拥有内容**：

- query/key兼容度；
- invariant attention logits；
- mask；
- segment softmax或其他归一化权重；
- head mixing；
- neighbor sum/mean；
- degree rescaling；
- 方向消息的路由选择。

**只读输入**：F3生成的等变消息，以及F2提供的边和几何特征。

**输出合同**：与目标状态类型一致的聚合消息。

SE(3)-Transformer把attention weight、value message和self-interaction明确分开，是F3、F4和F5应分离的直接证据。

### 5.5F5：节点状态变换与更新

**科学问题**：聚合消息如何与旧状态组合，并在保持等变的前提下形成下一层状态？

**拥有内容**：

- self-interaction或irrep-wise linear；
- equivariant FFN；
- gate、norm activation、S² activation和separable S² activation；
- equivariant normalization；
- residual add和局部block内更新顺序。

**不拥有内容**：跨block的宏观拓扑、消息产生和邻居聚合。

这里把归一化与非线性放入状态变换，是因为它们共同定义节点内表示如何更新；但实现时仍应设子区域标签，例如`F5.normalization`和`F5.activation`，以便发现内部仍存在的纠缠。如果单独修改这两个子区域能够稳定闭包，后续可把F5进一步拆分。

### 5.6F6：任务读出与输出构造

**科学问题**：节点或边上的表示如何转换为任务要求的不变或等变输出？

**拥有内容**：

- scalar selection；
- local prediction head；
- graph pooling；
- hierarchical或multi-level readout；
- 标量、向量、张量输出的最终类型构造。

**边界合同**：QM9极化率任务输出必须为graph-carried scalar；力预测等任务则需要vector输出合同。

当前DSL唯一认证区域`v1_readout`属于F6，因此现有端到端实验只验证了F6局部编辑，不能代表六因子搜索已经实现。

### 5.7真正用于单因子路由的叶子因子

如果Router只选择F2“几何关系编码”或F5“节点状态变换”，因子内部依旧太宽。因而需要两层本体：

```text
功能因子族：用于组织研究问题和跨架构泛化
→叶子因子：用于一次单因子干预和代码区域冻结
```

建议的叶子因子如下。

| 因子族 | 叶子因子 | 唯一拥有的主要机制 | 典型独立修改 |
|---|---|---|---|
| F1输入提升 | F1.1节点属性初始化 | atom/node embedding、初始node irreps | 改变元素嵌入或初始标量/向量通道的构造 |
| F1输入提升 | F1.2非几何边属性初始化 | bond type等原始edge attributes | 改变边属性嵌入，但不改相对坐标和几何基 |
| F2几何编码 | F2.1邻域与支持域 | neighbor graph、cutoff support | 改变邻域策略或cutoff调度 |
| F2几何编码 | F2.2径向编码 | distance、radial basis、envelope | Gaussian/Bessel基、基数量、包络结构 |
| F2几何编码 | F2.3角向、steerable属性与局部frame编码 | spherical/circular harmonics、geometry-derived attributes、angles、frame | 改变角向基或frame构造，但不改耦合核 |
| F2几何编码 | F2.4坐标或几何状态更新 | coordinate operation、equivariant displacement | EGNN类模型更新坐标；Equiformer中保持禁用 |
| F3等变耦合 | F3.1耦合路径代数 | tensor product path和合法输出irrep | 选择、稀疏化或重组CG路径 |
| F3等变耦合 | F3.2耦合权重参数化 | radial-conditioned weights、kernel MLP | 改变路径权重的生成方式 |
| F3等变耦合 | F3.3等变核实现族 | full SO(3)、G-steerable kernel、edge-frame、SO(2)convolution | 在边界合同不变时替换核实现族 |
| F3等变耦合 | F3.4交互阶数 | body/correlation order、two-hop interaction | 改变局部交互阶数，不等同于加深网络 |
| F4路由聚合 | F4.1消息打分 | query/key、invariant compatibility、logits | 改变注意力兼容函数 |
| F4路由聚合 | F4.2权重归一与动态选择 | softmax、normalizer、content-dependent mask | 改变路由权重的归一化和动态选择规则 |
| F4路由聚合 | F4.3邻居归约与尺度 | sum/mean、degree rescaling | 改变聚合和度数尺度校正 |
| F4路由聚合 | F4.4多头组织 | head split、head mixing | 改变并行路由结构，不改变value kernel |
| F5状态更新 | F5.1节点内线性与通道混合 | self-interaction、irrep linear、channel mixing | 在固定表示合同和固定非线性下改变线性混合 |
| F5状态更新 | F5.2等变非线性 | gate、norm activation、S² activation | 在类型合同内替换非线性机制 |
| F5状态更新 | F5.3归一化与尺度稳定 | equivariant norm、layer/instance norm | 改变等变归一化策略 |
| F5状态更新 | F5.4局部状态组合 | residual、局部skip、attention/FFN子层组合 | 改变同一block内部的新旧状态组合 |
| F6任务读出 | F6.1局部输出投影 | scalar selection、atomwise head | 改变节点级任务投影 |
| F6任务读出 | F6.2集合归约 | global sum/mean/attention pooling | 改变节点到图的归约 |
| F6任务读出 | F6.3多层融合与输出构造 | multi-level readout、final combiner | 融合不同block，但保持任务输出合同 |

**单因子修改的精确定义**是：每次只选择一个叶子因子，例如`F3.3`，而不是笼统地选择整个F3。功能因子族主要承担三项作用：组织DSL词汇、建立因子间依赖图、支持跨二维/三维架构迁移。

该层级也允许后续基于闭包测试调整粒度：如果`F5.4`仍然过宽，可以继续拆分；如果某两个叶子因子始终无法独立闭包，则可合并。调整必须由证据触发，而不是随意改名。

并非所有叶子因子都应在每个架构家族中启用。应由Capability Profile声明可用性：

- `F2.4`只对显式更新坐标或几何状态的EGNN类模型启用，Equiformer V1/V2固定坐标时禁用；
- `F3.4`主要对具有显式body/correlation order或多跳交互的模型启用；
- `F4.4`只对multi-head attention架构启用，不能当作所有等变网络的通用核心因子；
- `F6.3`只在计算图提供多个合法readout tap时启用。

这种“通用因子族+能力门控叶子”比强迫二维CNN、EGNN和Equiformer共享完全相同的可编辑字段更严谨。

### 5.8叶子因子的类型化边界合同

| 叶子因子 | 主要输入 | 区域输出合同 | 单因子修改时禁止触碰 |
|---|---|---|---|
| F1.1节点属性初始化 | 原始node/graph属性、G1类型声明 | 初始node-carried typed features | 邻域、几何基、message block和G1类型本身 |
| F1.2非几何边属性初始化 | 原始非几何edge属性、G1类型声明 | 初始edge-carried typed features | 相对坐标、径向/角向基和消息核 |
| F2.1邻域与支持域 | 坐标、batch、周期边界条件 | edge index和support/cutoff mask | edge message数值、耦合核、状态更新 |
| F2.2径向编码 | distance、cutoff support | edge-carried invariant radial features | 球谐、CG路径和radial-conditioned kernel结构 |
| F2.3角向、steerable属性与局部frame编码 | relative position、方向、邻域和G1阶数上限 | directional irreps、geometry-derived attributes、angles或edge frame | tensor product路径、attention和node update |
| F2.4坐标或几何状态更新 | 当前坐标及合法标量/向量位移信号 | 与输入群作用一致的新坐标 | node feature schema、读出和固定坐标模型的能力开关 |
| F3.1耦合路径代数 | typed features、F2角向特征、G1合法性规则 | 合同内的equivariant message | 路由权重、邻居归约和G1类型迁移 |
| F3.2耦合权重参数化 | radial features、固定path identifiers | 每条合法路径的不变标量权重 | radial basis定义、path集合和聚合规则 |
| F3.3等变核实现族 | 固定输入/输出irreps和几何frame | 与父代同边界类型的equivariant message | 改变外部类型、attention或readout |
| F3.4交互阶数 | 局部环境及固定表示合同 | 指定body/correlation order的message | block深度、全局感受野和任务头 |
| F4.1消息打分 | node/edge状态和只读几何上下文 | edge-carried invariant logits | value kernel、softmax/归一规则和aggregation |
| F4.2权重归一与动态选择 | invariant logits、合法动态mask | 归一化edge weights | F2.1静态邻域、message内容、neighbor reduction和node update |
| F4.3邻居归约与尺度 | weighted equivariant messages、edge index | target-node-carried aggregated message | 打分函数、耦合核和残差更新 |
| F4.4多头组织 | 多个同合同head输出 | 与父代相同类型的合并消息 | 单个head内部核、G1模式和非attention模型 |
| F5.1节点内线性与通道混合 | node state或aggregated message | 同一G1合同下的线性变换特征 | activation、normalization、residual topology和multiplicity迁移 |
| F5.2等变非线性 | typed node features及必要的不变gate | 相同irrep类型的非线性特征 | 线性层结构、normalization统计和残差连线 |
| F5.3归一化与尺度稳定 | typed node features | 相同irrep类型和shape的归一化特征 | activation函数、message aggregation和训练优化器 |
| F5.4局部状态组合 | old state和各子层固定输出 | 下一层node state，边界schema不变 | F5.1至F5.3内部实现、跨block宏观拓扑 |
| F6.1局部输出投影 | terminal或指定tap的typed features | node/edge级任务预测类型 | pooling、上游interaction和任务输出类型 |
| F6.2集合归约 | local predictions及segment ids | graph-level invariant/equivariant output | local head内部和上游表示 |
| F6.3多层融合与输出构造 | 多个声明过的readout taps | 与任务合同一致的最终输出 | 未声明block、上游冻结结构和任务类型 |

复合结构例如“完整Equiformer FFN”会同时包含F5.1线性、F5.2非线性、F5.3归一化和F5.4组合。它是一个motif，不是最小因子。单因子阶段只允许修改其中一个叶子区域；后续只有在交互证据成立时才允许联合改写整个motif。

两个容易混淆的尺度机制也必须按作用对象区分：F4.3只拥有由邻居数量或聚合规则产生的degree/message scaling；F5.3只拥有节点表示内部按channel、degree或norm计算的feature normalization。二者不得因为都含有“缩放”操作而合并。

## 六、哪些内容不应作为普通局部因子

### 6.1G1：对称群与表示模式

包括：

- group family；
- parity；
- $l_{\max}$；
- 各阶multiplicity；
- node/edge/field carrier布局；
- layerwise irrep schema。

它们决定大量模块的输入输出类型。改变它们会触发全网类型迁移，因此应由`Schema Migration`控制器处理，而不是让普通Router声称只改一个局部因子。

### 6.2G2：宏观拓扑与资源模式

包括：

- block数量；
- stage划分；
- 中间层spatial/group pooling和分辨率变化；
- 跨block skip topology；
- weight sharing；
- 全局感受野调度；
- 参数量、FLOPs和显存预算。

这类修改跨越多个局部区域，应作为`Macro Transformation`处理。

二维G-CNN中的中间层G-pooling若改变空间或群域分辨率，归入G2；只有最终把局部表示变成任务级不变量的pooling才归入F6.2。二者不能因为都叫pooling就共享同一叶子因子。

### 6.3训练策略不是架构因子

学习率、batch size、weight decay、普通Dropout率和训练步数属于训练协议。搜索阶段应保持冻结，避免把训练收益错误归因于架构。只有当随机深度本身改变推理时的结构语义时，才作为宏观结构机制单独研究。

## 七、如何证明一个因子划分是合格的

仅有论文依据仍不够，还要通过实现级测试。

| 检验 | 定义 | 目标 |
|---|---|---|
| NodeOwnershipOverlap | 同一可编辑节点被多个普通因子拥有的比例 | 必须为0；共享接口只能只读 |
| EditLeakRate | 实际diff落到未选因子或冻结脚手架的比例 | 趋近0 |
| FreezePassRate | 未选区域规范化AST/hash保持一致的候选比例 | 接近1 |
| BoundaryClosureRate | 修改只在选定区域内即可编译并满足类型合同的比例 | 足够高且稳定 |
| FactorCoverage | 可搜索原语被某一因子唯一拥有的比例 | 接近1 |
| EquivariancePassRate | 候选通过群变换数值测试的比例 | 合法候选必须为1 |
| CreditStability | 同类修改跨父代和随机种子的效果方向稳定性 | 用于判断因子是否仍过宽 |

如果某因子`BoundaryClosureRate`长期很低，应检查边界是否切得过细；如果`CreditStability`很低且内部修改类型差异很大，应继续拆分。

## 八、如何解决“怎样选择因子”

### 8.1不要一开始完全交给LLM自由路由

搜索初期没有足够历史证据，LLM容易反复选择看起来容易解释的因子，导致其他因子没有被公平探索。因此第一阶段应采用强制单因子覆盖：

```text
Capability Profile中启用的F1.1→F1.2→F2.1→…→F6.3
```

每次只允许修改一个已启用叶子因子，并为每个已启用叶子因子获得至少若干个合法候选。这里的目标不是立即找到最优架构，而是建立各叶子因子的：

- 合法生成率；
- 编译失败类型；
- 等变测试通过率；
- validation MAE主效应；
- 训练成本；
- 对不同父代的敏感性。

### 8.2有证据后再使用上下文因子选择器

当每个因子都有最小证据量后，Router针对父代$p$和因子$f$计算选择价值：

$$
U(f\mid p,H)=\widehat{\Delta}_{f}(p)+\beta\operatorname{Uncertainty}_{f}(p)-\lambda_1\operatorname{InvalidRisk}_{f}-\lambda_2\operatorname{Cost}_{f}.
$$

其中：

- $\widehat{\Delta}_{f}(p)$：该因子在相似父代上的预期validation收益；
- $\operatorname{Uncertainty}_{f}(p)$：尚未探索的不确定性，用于保留探索；
- $\operatorname{InvalidRisk}_{f}$：编译、接口或等变失败风险；
- $\operatorname{Cost}_{f}$：预计训练时间、显存和参数量成本；
- $H$：搜索历史和失败诊断。

LLM可以读取这些证据并提出因子选择及理由，但最终输出必须是一个受限token，例如`F3_COUPLING`。系统再用确定性规则检查该token和当前候选区域是否一致。

### 8.3因子选择和区域选择要分开

同一因子可能在多个block重复出现。建议流程为：

```text
Factor Router：选择修改哪种功能机制
→Region Router：选择该机制位于哪个block或哪些共享实例
→Directive Generator：根据历史生成本因子内的修改方向
→Scoped Synthesizer：只输出该区域的DSL patch
```

这样“选择F5状态更新”和“修改第几个block”不会混成同一个决策。

## 九、为什么必须先单因子，再多因子

### 9.1单因子阶段估计主效应

对父代$P$分别构造$P+A$和$P+B$，可以判断A、B各自是否有效，并积累失败模式。如果直接生成$P+A+B$，即使MAE改善也无法知道收益来自A、B还是二者交互。

### 9.2多因子不能靠“允许编辑两个区域”实现

直接开放两个区域仍会产生新的纠缠。受控双因子实验至少需要四个对应候选：

```text
P
P+A
P+B
P+A+B
```

对损失$L$定义交互项：

$$
I_{A,B}=L_{P+A+B}-L_{P+A}-L_{P+B}+L_P.
$$

$I_{A,B}<0$表示联合修改产生超出两个单因子主效应之和的协同；接近0表示近似可加；大于0表示负交互。

### 9.3双因子修改应串行编译、联合评估

推荐执行方式：

```text
父代P
→只修改因子A
→编译、类型检查、等变检查
→在P+A上只修改因子B
→再次编译、类型检查、等变检查
→得到P+A+B
→与P、P+A、P+B一起估计交互
```

这不是把A和B塞进一次自由Prompt，而是两个可审计的单因子patch组成一个联合实验。

### 9.4何时允许进入多因子阶段

至少满足以下条件之一：

1. 两个因子分别有稳定正主效应，并且接口依赖表明它们可能互补；
2. 单因子修改反复受另一个因子的固定边界限制；
3. 搜索历史中出现可复现的协同信号；
4. 文献明确表明某项创新由两个机制共同构成；
5. 单因子阶段已覆盖全部因子且搜索进入停滞。

多因子阶段初期只开放有显式依赖边的因子对，例如：

```text
F2几何编码→F3等变耦合
F3等变耦合→F4路由聚合
F4路由聚合→F5状态更新
F5状态更新→F6任务读出
```

不应直接开放任意$6\times6$组合。

## 十、Equiformer V1到V2应如何映射

V2不是一次不可分解的整体“新算子发现”，而是多个机制层面的创新组合。至少可以按下列方式回溯：

| V1/V2变化 | 主因子 | 只读依赖或可能交互 |
|---|---|---|
| tensor product计算替换为基于edge frame的SO(2)卷积 | F3.3等变核实现族 | 依赖F2.3方向/frame和G1表示模式 |
| SO(2)attention打分与权重生成 | F4.1消息打分、F4.2权重归一 | 读取F3生成的消息表示；两者应先分别验证 |
| S²/separable S² activation | F5.2等变非线性 | 依赖G1的球面表示合同 |
| normalization改进 | F5.3归一化与尺度稳定 | 不与非线性共用编辑区 |
| degree rescaling改进 | F4.3邻居归约与尺度 | 不归入F5归一化 |
| 更高$l_{\max}$和通道配置 | G1表示模式迁移 | 触发F2、F3和F5类型重编译 |

因此，本项目若希望从V1逐步发现类似V2的结构，合理路径是：先分别在F3.3、F4.1/F4.2、F5.2和F5.3等叶子因子中发现有效局部修改，再通过交互实验组合；而不是让LLM一次重写整个block。

## 十一、当前DSL与目标设计的差距

当前原语注册表已经包含：

- 几何原语：`relative_position`、`distance`、`radial_basis`、`spherical_harmonics`；
- 耦合原语：`tensor_product`、`to_edge_frame`、`from_edge_frame`、`so2_convolution`；
- 路由聚合原语：`invariant_compatibility`、`segment_softmax`、`segment_sum/mean`；
- 状态更新原语：`irrep_linear`、`residual_add`、`gate`、`equivariant_norm`、`s2_activation`、`separable_s2_activation`；
- 读出原语：`select_scalars`、`global_pool`。

但`regions.py`目前只认证`v1_readout`一个区域。因此当前完成的是：

```text
已有覆盖多个因子的原语词汇表
+已有F6读出区域的冻结补集验证
-尚未建立F1至F5的区域所有权和边界合同
-尚未实现基于证据的Factor Router
-尚未实现单因子主效应数据库和交互图
```

### 11.1当前DSL原语的初始唯一所有权

下表把现有`registry.py`中的全部原语映射到一个主叶子因子。它是编译器实现的初始规范，后续只有在闭包或重叠测试失败时才调整。

| DSL原语 | 主所有者 | 主要只读依赖或说明 |
|---|---|---|
| `core.identity` | F5.4局部状态组合 | 保持输入类型 |
| `core.irrep_linear` | F5.1节点内线性与通道混合 | 读取G1表示合同 |
| `core.irrep_concat` | F5.1节点内线性与通道混合 | 所有输入必须满足可拼接类型合同 |
| `core.residual_add` | F5.4局部状态组合 | 两侧表示类型必须相同 |
| `core.tensor_product` | F3.1耦合路径代数 | 读取F2.3角向特征和G1合法路径 |
| `core.scalar_activation` | F5.2等变非线性 | 只作用于标量 |
| `core.invariant_weight` | F4.2权重归一与动态选择 | 权重必须不变，value由F3提供 |
| `core.edge_lift` | F4.1消息打分 | node到edge的只读载体转换 |
| `core.segment_sum` | F4.3邻居归约与尺度 | 读取edge index |
| `core.global_pool` | F6.2集合归约 | node到graph载体转换 |
| `core.select_scalars` | F6.1局部输出投影 | 读取G1中的$l=0$通道 |
| `core.to_edge_frame` | F3.3等变核实现族 | frame由F2.3提供 |
| `core.from_edge_frame` | F3.3等变核实现族 | 输出恢复到全局表示合同 |
| `core.irrep_slice` | F5.1节点内线性与通道混合 | 不允许隐式修改G1模式 |
| `core.change_multiplicity` | G1表示模式迁移 | 不是普通局部叶子因子 |
| `core.relative_position` | F2.3角向与局部frame编码 | 读取坐标和edge index |
| `core.distance` | F2.2径向编码 | 读取relative position |
| `core.radial_basis` | F2.2径向编码 | 读取distance |
| `core.cutoff_envelope` | F2.2径向编码 | support变化由F2.1控制 |
| `core.spherical_harmonics` | F2.3角向与局部frame编码 | 最高阶受G1约束 |
| `core.norm_activation` | F5.2等变非线性 | 读取表示范数 |
| `core.gate` | F5.2等变非线性 | gate必须为不变量 |
| `core.equivariant_norm` | F5.3归一化与尺度稳定 | 不改变irrep类型 |
| `core.stochastic_depth` | G2宏观拓扑与资源模式 | 默认冻结，不进入首轮架构搜索 |
| `core.invariant_dropout` | 训练协议 | 默认冻结，不进入架构因子路由 |
| `core.segment_mean` | F4.3邻居归约与尺度 | 读取edge index和度数 |
| `core.segment_softmax` | F4.2权重归一与动态选择 | 输入必须为不变logits |
| `core.invariant_compatibility` | F4.1消息打分 | query/key必须产生不变量 |
| `core.so2_convolution` | F3.3等变核实现族 | 读取edge frame和G1表示合同 |
| `core.s2_activation` | F5.2等变非线性 | 读取球面表示合同 |
| `core.separable_s2_activation` | F5.2等变非线性 | 标量与高阶分支合同固定 |

这里暴露了两个实现缺口：F1输入提升、F2.1邻域支持域和F2.4坐标更新目前没有足够的显式DSL原语；F3.2耦合权重参数化、F3.4交互阶数、F4.4多头组织也没有独立原语。因此不能只靠现有registry声明本体已经覆盖完整。

## 十二、下一版实现顺序

### P0：冻结创新率Benchmark开发

创新率暂时只保留概念，不进入当前开发主线。

### P1：把Primitive Ownership Table编码进编译器

把11.1节的初始映射变成机器可检查的registry字段，并为每个DSL原语指定：

- 唯一主因子；
- 只读依赖；
- 输入输出类型；
- 是否允许局部改写；
- 是否会触发G1/G2全局迁移。

### P2：把Equiformer V1完整导入为类型化AST

不能只保留readout区域。需要覆盖embedding、geometry、coupling、routing、update和readout的完整计算图。

### P3：注册叶子因子区域并实现冻结补集证明

每个单因子patch必须满足：

$$
\operatorname{Edit}(P',P)\subseteq R_f,
$$

且所有非选区域规范化AST哈希不变。

### P4：实现Factorization Tests

先验证所有权重叠、边界闭包、等变性和冻结补集，不训练大规模候选。

### P5：运行均衡单因子搜索

每个叶子因子获得同量合法候选，建立主效应、失败率和成本记录。

### P6：实现证据条件化Factor Router

使用历史收益、不确定性、失败风险和成本选择一个因子；LLM负责语义推理，确定性控制器负责边界执行。

### P7：建立稀疏Interaction Graph

先测试相邻依赖因子对，使用$P/P+A/P+B/P+A+B$估计交互，再决定是否开放双因子进化。

## 十三、最终研究问题的精确定义

本项目当前要研究的不是“让LLM自由修改等变网络”，而是：

> 如何从等变神经网络的类型化计算图和跨架构文献共性中发现功能内聚、边界可闭包、补集可冻结的架构因子；如何利用搜索历史选择单一干预因子，并在获得可归因的主效应后，通过显式交互证据逐步开放受控多因子组合，从而降低LLM驱动等变NAS中的功能纠缠和无效候选成本？

这一问题比“设计几个修改策略”更准确，也比直接照搬SPARK的`OPERATOR/ACTION`更有研究价值。

## 十四、论文与本地全文

- [Tensor Field Networks](../调研报告/论文原文/06_等变因子划分依据/Tensor_Field_Networks.pdf)
- [SE(3)-Transformer](../调研报告/论文原文/06_等变因子划分依据/SE3_Transformer.pdf)
- [EGNN](../调研报告/论文原文/06_等变因子划分依据/EGNN.pdf)
- [DimeNet](../调研报告/论文原文/06_等变因子划分依据/DimeNet.pdf)
- [GemNet](../调研报告/论文原文/06_等变因子划分依据/GemNet.pdf)
- [PaiNN](../调研报告/论文原文/06_等变因子划分依据/PaiNN.pdf)
- [NequIP](../调研报告/论文原文/06_等变因子划分依据/NequIP.pdf)
- [Allegro](../调研报告/论文原文/06_等变因子划分依据/Allegro.pdf)
- [MACE](../调研报告/论文原文/06_等变因子划分依据/MACE.pdf)
- [SEGNN](../调研报告/论文原文/06_等变因子划分依据/SEGNN.pdf)
- [Equiformer V1](../调研报告/论文原文/01_等变网络与自动等变约束/2206.11990_Equiformer.pdf)
- [e3nn](../调研报告/论文原文/01_等变网络与自动等变约束/2207.09453_e3nn.pdf)
- [eSCN](../调研报告/论文原文/01_等变网络与自动等变约束/2302.03655_eSCN.pdf)
- [Equiformer V2](../调研报告/论文原文/01_等变网络与自动等变约束/2306.12059_EquiformerV2.pdf)
- [Group Equivariant NAS](../调研报告/论文原文/01_等变网络与自动等变约束/2104.04848_Group_Equivariant_NAS.pdf)
- [Equivariance-aware Architectural Optimization](../调研报告/论文原文/01_等变网络与自动等变约束/2210.05484_Equivariance_Aware_Architectural_Optimization.pdf)
- [SPARK](../调研报告/论文原文/04_LLM与进化搜索/2605.04057_SPARK.pdf)
- [Group Equivariant Convolutional Networks](../调研报告/论文原文/07_二维等变因子依据/1602.07576_Group_Equivariant_CNNs.pdf)
- [Steerable CNNs](../调研报告/论文原文/07_二维等变因子依据/1612.08498_Steerable_CNNs.pdf)
- [General E(2)-Equivariant Steerable CNNs](../调研报告/论文原文/07_二维等变因子依据/1911.08251_General_E2_Steerable_CNNs.pdf)

## 十五、一句话结论

现在最优先的不是设计更多路由策略，而是用文献共性、类型化计算图、唯一节点所有权和边界闭包测试确定可信因子；在此基础上先做单因子主效应搜索，再由交互证据决定哪些因子可以受控组合。
