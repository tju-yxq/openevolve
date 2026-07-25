# 架构语言驱动LLM生成新网络的相关论文核查

> 调研日期：2026-07-24  
> 研究问题：是否已经有人把神经网络架构表示成一种语言，再让大语言模型（Large Language Model，LLM）在该语言中生成新的架构？  
> 重点边界：区分自由Python代码、结构化配置、架构程序、领域专用语言（Domain-Specific Language，DSL）和带等变类型的DSL。  
> 原文归档：本轮使用的论文原文位于[本地论文库](论文原文/)，新增原文及哈希见[论文下载与阅读清单](论文原文/论文下载与阅读清单.md)。

## 1. 先说结论

结论不是简单的“有”或“没有”，而是分成四层。

1. **广义想法已经有人做过。**EvoPrompting、LLMatic、NNGPT和Delta-Based NAS都让LLM生成或修改描述神经网络的Python代码。因此，不能把“让LLM用一种编程语言生成新架构”本身写成首创。
2. **把架构作为程序或细粒度语言搜索也已经有人做过。**AutoML-Zero、Primer、Neural Architecture Search as Program Transformation Exploration、αNAS和Syno分别使用指令程序、TensorFlow程序、程序变换、属性引导合成或结构化算子合成产生预定义模块列表之外的新结构，但它们没有使用LLM完成等变架构生成。
3. **自动搜索等变约束也已经有人做过。**Autoequivariant Network Search、Equivariance-aware Architectural Optimization和逐层等变学习工作能够搜索群分解、群或子群约束，但没有构造面向三维等变算子的LLM生成DSL。
4. **在本次检索范围内，尚未发现完整覆盖以下组合的工作：**三维$SO(3)/O(3)$表示类型、等变算子DSL、LLM生成类型化架构程序、编译器与静态等变检查、三维分子或原子任务上的闭环NAS。

因此，当前项目可以主张的潜在研究空白不是“首次让LLM生成神经网络架构”，而是：

> 能否把三维等变网络架构表示为带有不可约表示、宇称、参考系和张量积路径类型的可执行DSL，并让LLM在这一受约束语言中合成结构候选，使候选在训练前即可完成解析、类型检查、编译和等变性验证？

上述“尚未发现”只对本报告记录的检索范围成立，不能写成未经限制的“世界上绝对没人做过”。

## 2. 什么才算“把架构变成一种语言”

### 2.1 五种容易混淆的表示

| 表示形式 | 示例 | 是否是专用架构语言 | 能否表达新结构 | 是否自动保证合法 |
|---|---|---:|---:|---:|
| 超参数向量 | 层数、宽度、head数量 | 否 | 很有限 | 通常只能保证取值范围 |
| 固定cell配置 | 操作编号、边连接表 | 弱结构语言 | 只能组合预定义操作 | 可通过schema部分保证 |
| 高层结构模板 | 模块类别、设计idea树 | 接近知识schema | 可以指导结构组合 | 通常不能静态证明 |
| 自由Python代码 | `nn.Module`及`forward` | 通用编程语言，不是架构DSL | 可以 | 主要依赖执行发现错误 |
| 类型化架构DSL | 原语、AST、类型、语义和编译器 | 是 | 可以组合出新motif | 可以在训练前拒绝非法候选 |

如果只把Equiformer配置写成YAML，再让LLM修改`num_layers`或`num_heads`，这不是本项目要研究的DSL，只是结构化超参数搜索。

### 2.2 一个完整架构DSL至少需要什么

建议将语言定义为：

$$
\mathcal{L}=(\Sigma,G,T,\operatorname{Sem},R,C)
$$

其中：

- $\Sigma$：可信架构原语，例如张量积、径向权重、门控、参考系旋转和邻居聚合；
- $G$：程序组合语法及抽象语法树（Abstract Syntax Tree，AST）；
- $T$：表示类型、形状、宇称、参考系和合法耦合路径；
- $\operatorname{Sem}$：每个程序的数学语义和运行语义；
- $R$：保持接口或满足证明义务的类型化重写规则；
- $C$：将DSL程序编译到PyTorch、e3nn或现有Equiformer实现的可信编译器。

LLM的输出不是一个“更优的数值”，而是一个新程序或类型化补丁：

```text
当前架构程序+语言规则+历史实验反馈
→LLM提出typed patch
→解析和类型检查
→规范化与去重
→编译
→数值等变性审计
→训练和验证
→结果进入下一轮搜索
```

## 3. 最直接的LLM架构生成工作

### 3.1 EvoPrompting：已经实现LLM生成架构程序

- 正式发表：NeurIPS 2023；arXiv首次提交于2023年。
- 原文证据：论文把代码语言模型作为进化搜索的mutation和crossover operator，根据父代程序生成候选架构，再训练并用验证集评价。
- 表示形式：完整架构代码，属于自由程序生成。
- 关键区别：没有面向等变网络的专用类型系统；合法性主要依靠代码执行和训练暴露。

这篇论文足以否定“以前没有人让LLM基于语言生成新架构”这一宽泛主张。

### 3.2 LLMatic：LLM代码变异与质量多样性搜索

- 正式发表：GECCO 2024，DOI：[10.1145/3638529.3654017](https://doi.org/10.1145/3638529.3654017)。
- 原文证据：LLMatic使用LLM对定义神经网络的代码产生有意义变体，并以质量多样性（Quality-Diversity，QD）archive保持不同候选。
- 表示形式：神经网络代码，而不是具有形式语义的架构DSL。
- 关键区别：没有irrep类型、张量积耦合规则、编译证明和三维等变任务。

### 3.3 NNGPT：让LLM在闭环中学习生成PyTorch架构

- 发表状态：2026年预印本，[arXiv:2601.02997](https://arxiv.org/abs/2601.02997)。
- 原文证据：代码LLM在22轮循环中生成完整PyTorch卷积网络；候选通过解析、实例化和dummy forward后，使用低保真准确率与MinHash-Jaccard新颖度筛选，再转成prompt-code对进行LoRA微调。
- 论文报告：CIFAR-10有效生成率稳定在50.6%，最高74.5%；这说明自由代码空间仍存在大量无效候选。
- 关键区别：没有DSL、静态类型系统或等变约束；所谓valid主要是Python可解析、模型可实例化和forward可执行。

NNGPT是当前项目必须加入的直接基线，因为它真的让LLM通过实验反馈逐步内化架构生成先验。

### 3.4 Delta-Based NAS：让LLM生成架构代码补丁

- 发表状态：2026年预印本，[arXiv:2605.04903](https://arxiv.org/abs/2605.04903)。
- 原文证据：LLM不再从头输出完整模型，而是输出修改现有架构的unified diff；论文报告输出从200行以上缩短到约30至50行，并提高有效生成率。
- 表示形式：软件工程代码diff，不是架构AST上的类型化事务。
- 关键区别：补丁可以保留父代代码骨架，但仍无法在训练前证明等变性，也不能区分语义等价改写与真正结构创新。

它说明“只生成局部patch”也不是首创。当前项目必须把创新落在**领域语义、类型保证和可验证结构重写**，而不是patch形式本身。

### 3.5 SPARK与Structuring Open-Ended NAS：结构化搜索，但不是形式DSL

SPARK使用Operator和Action功能因子、作用域选择和局部代码编辑降低functional entanglement。Structuring Open-Ended NAS先定义高层架构属性模板，再让LLM从论文中填充设计知识，并通过多种mutation探索源代码空间。

二者都比无约束代码生成更结构化，但本次全文核查没有发现它们提供以下完整机制：

- 架构程序的形式语法和AST类型规则；
- $SO(3)/O(3)$不可约表示类型；
- 张量积路径的静态合法性推导；
- DSL到等变运行模块的可信编译；
- 整体等变性的构造性证明。

因此，不能把SPARK简单称为DSL，也不能声称我们的贡献只是“像SPARK一样拆分因子”。

## 4. 已有的程序化NAS与DSL思想

### 4.1 AutoML-Zero：低级指令语言可以发现完整学习算法

- 正式发表：ICML 2020，[PMLR正式页面](https://proceedings.mlr.press/v119/real20a.html)。
- 原文证据：学习算法被表示为由`Setup`、`Predict`和`Learn`构成的程序，指令对小型内存执行基础数学操作。
- 价值：证明细粒度原语能组合出非预制结构。
- 局限：搜索空间极稀疏；论文指出即使简单任务的有效算法也可能低至$10^{-12}$量级，因此依赖极高候选吞吐量。

它直接支持本项目使用强类型剪除不合法组合，否则三维等变原语空间会更快爆炸。

### 4.2 Primer：在TensorFlow程序primitive层搜索新Transformer结构

- 正式发表：NeurIPS 2021，[arXiv:2109.08668](https://arxiv.org/abs/2109.08668)。
- 原文证据：搜索空间由定义Transformer TensorFlow程序的低级primitive构成，进化搜索发现了squared ReLU和Q/K/V投影后的depthwise convolution。
- 价值：证明程序级搜索能够发现预定义block列表中没有的新motif。
- 局限：不是LLM驱动，也没有等变类型。

Primer说明“从V1产生V2级变化”不能只靠枚举完整算子名称，必须允许多primitive组合和多节点重写。

### 4.3 程序变换式NAS：架构修改可以是组合重写

- 正式发表：ASPLOS 2021，DOI：[10.1145/3445814.3446753](https://doi.org/10.1145/3445814.3446753)。
- 原文证据：论文将NAS操作表达成程序变换，将编译变换和架构变换放入统一框架，并由简单变换组合出新的tensor convolution。
- 价值：为typed rewrite和事务式多节点编辑提供直接先例。
- 局限：目标主要是推理性能，没有LLM和群等变语义。

### 4.4 αNAS：按程序属性搜索并合成满足属性的架构

- 正式发表：OOPSLA 2022，DOI：[10.1145/3563329](https://doi.org/10.1145/3563329)。
- 原文证据：αNAS不直接在具体网络图中盲目变异，而是在shape、depth和mixing等抽象程序属性空间中搜索，再通过程序合成得到满足属性的具体架构。
- 价值：是“先检查属性，再花训练预算”的最近形式化NAS先例。
- 局限：没有LLM、三维irrep和等变构造。

### 4.5 Syno：结构化合成预定义列表之外的新算子

- 正式发表：ASPLOS 2025，DOI：[10.1145/3676642.3736118](https://doi.org/10.1145/3676642.3736118)。
- 原文证据：Syno使用结构化程序合成产生神经算子实现，而不局限于从完整算子列表中选择。
- 价值：证明“新算子发现”需要较低层的索引、循环或组合原语以及语义等价和代价验证。
- 局限：不是LLM生成的等变架构DSL。

## 5. 已有等变架构搜索做到了什么

### 5.1 Autoequivariant Network Search

- 发表状态：2021年预印本，[arXiv:2104.04848](https://arxiv.org/abs/2104.04848)。
- 方法：利用群分解结果构造较小群的等变网络，并用深度Q学习在缩减空间中搜索等变性与网络规模之间的平衡。
- 局限：主要是二维群等变网络；不是LLM，也不是三维张量积程序DSL。

### 5.2 Equivariance-aware Architectural Optimization

- 正式发表：ICLR 2023，[OpenReview](https://openreview.net/forum?id=a6rCdfABJXg)。
- 方法：提出equivariance relaxation morphism和混合群等变层，并以进化NAS和可微NAS选择群或子群约束。
- 局限：搜索“采用哪种等变约束”，不是搜索完整的$SO(3)$算子计算图。

### 5.3 对当前研究空白的含义

已有论文证明“等变性可以成为搜索变量”，因此不能声称首次自动搜索等变网络。当前差异在于：

- 搜索对象从群或子群选择升级为带类型的三维等变程序；
- 搜索动作从固定模块替换升级为类型安全的路径、motif和宏结构重写；
- 生成器从强化学习或连续权重升级为LLM程序合成；
- 验证从训练后性能检查升级为训练前静态证明义务加数值证伪。

## 6. 逐项覆盖矩阵

| 工作 | 架构是程序或结构语言 | 使用LLM生成 | 静态类型或属性检查 | 等变搜索 | 三维$SO(3)$等变DSL |
|---|---:|---:|---:|---:|---:|
| EvoPrompting | 是，自由代码 | 是 | 否 | 否 | 否 |
| LLMatic | 是，自由代码 | 是 | 否 | 否 | 否 |
| NNGPT | 是，完整PyTorch | 是 | 否，主要靠执行 | 否 | 否 |
| Delta-Based NAS | 是，代码diff | 是 | 否，主要靠执行 | 否 | 否 |
| SPARK | 是，受限代码区域 | 是 | 接口和shape检查 | 否 | 否 |
| Structuring Open-Ended NAS | 高层知识模板加源代码 | 是 | 非形式化模板 | 否 | 否 |
| AutoML-Zero | 是，低级指令程序 | 否 | 基础操作约束 | 否 | 否 |
| Primer | 是，TensorFlow程序 | 否 | 程序构造约束 | 否 | 否 |
| 程序变换式NAS | 是，程序变换 | 否 | 变换合法性与容量代理 | 否 | 否 |
| αNAS | 是，属性加程序合成 | 否 | 是，抽象属性 | 否 | 否 |
| Syno | 是，算子合成语言 | 否 | 是，结构和语义约束 | 否 | 否 |
| Autoequivariant Network Search | 固定群分解空间 | 否 | 群论构造 | 是 | 否 |
| Equivariance-aware Optimization | 固定群或子群空间 | 否 | 群约束 | 是 | 否 |
| 当前拟议系统 | 类型化架构DSL | 是 | 是 | 是 | 是 |

矩阵显示，单独看每一列都不是空白，潜在新颖性来自最后一行的**交叉组合及其可验证实现**。

## 7. “从V1产生V2”应该如何严谨定义

“DSL能够手工编码V1和V2”只证明语言表达力，不证明LLM具备架构发现能力。

需要区分三个实验层级：

1. **表达性测试**：研究者分别用DSL写出V1和V2，验证编译结果与原实现一致。
2. **受指导重写测试**：告诉LLM目标机制或提供V2论文知识，检查它能否生成合法的V1到V2类型化事务。这证明语言和生成接口可用，但不属于独立发现。
3. **盲发现测试**：生成器只看到V1、基础原语、类型规则和实验反馈，不看到V2实现、V2论文motif或训练语料中的V2代码；检查是否独立发现V2风格结构或其他更优结构。

如果要在论文中使用“发现”或“rediscovery”，至少需要第三层，并进行训练语料泄漏审计。否则只能说“DSL能够表达并生成V2风格重写”。

## 8. 当前项目可以和不可以主张什么

### 8.1 可以争取验证的主张

1. 提出首个面向三维等变网络架构合成的typed DSL，显式表示irrep、parity、frame和tensor product path。
2. 提出LLM驱动的typed architecture patch生成机制，使大部分数学非法候选在训练前被拒绝或修复。
3. 相较自由Python生成、代码diff和固定配置搜索，在相同LLM与评价预算下提高合法率、等变通过率、结构新颖度和单位GPU成本的有效改进率。
4. 在不开放任意Python逃生口的情况下表达V1、V2及至少一个两者之外的合法混合结构。

这些仍是待实验验证的研究假设，不能在实现和实验前写成已证实结论。

### 8.2 不可以主张的内容

- “首次使用LLM生成神经网络架构”；
- “首次把神经网络架构表示成程序”；
- “首次使用局部代码patch进行LLM-NAS”；
- “首次自动搜索等变网络”；
- “OpenEvolve或SPARK绝对无法产生V2”；
- “只要DSL能写出V2，就证明LLM发现了V2”。

## 9. 实验上必须比较哪些基线

为了证明收益来自DSL而不是额外prompt知识或更小搜索空间，建议至少比较：

1. **自由代码生成**：EvoPrompting、LLMatic或OpenEvolve风格；
2. **代码diff生成**：Delta-Based NAS风格；
3. **因子约束代码编辑**：SPARK风格；
4. **结构知识模板**：Structuring Open-Ended NAS风格；
5. **非LLM的同一DSL搜索**：随机、进化或属性引导搜索；
6. **本项目typed DSL+LLM**。

不同表示方式无法使用逐字相同的prompt，因为输出契约不同。公平性应保持以下信息和预算一致：

- 相同父代、任务描述和可用文献知识；
- 相同历史候选与性能反馈；
- 相同LLM或等价能力模型；
- 相同LLM调用数、token预算和候选训练预算；
- 相同训练、验证、晋级和最终test隔离协议；
- 额外报告编译拒绝、执行失败和等变失败所节省的GPU成本。

核心指标不应只有最终MAE，还应包括：语法合法率、类型通过率、等变通过率、唯一AST比例、结构级修改比例、有效改进率、达到目标MAE的GPU小时和完整训练后的稳定性。

## 10. 对DSL实现路线的直接修订

本轮论文核查给出五个具体要求：

1. **不能只生成完整代码。**NNGPT已经覆盖该方向，而且有效率仍受执行错误限制。
2. **不能只生成普通diff。**Delta-Based NAS已经覆盖代码diff；本项目应生成AST级typed transaction。
3. **不能只做因子路由。**SPARK已经覆盖因子条件编辑；本项目必须增加形式类型、编译语义和等变证明义务。
4. **不能只建高层idea模板。**Structuring Open-Ended NAS已经覆盖论文知识结构化；本项目需要可执行语言和可信检查器。
5. **必须允许多节点结构重写。**Primer、程序变换式NAS和Syno说明新motif通常由多个primitive协同形成，单字段编辑无法承载V1到V2级变化。

最小可运行版本应先实现：

```text
GeoTensor类型
→基础等变primitive
→AST与解析器
→irrep/frame类型检查
→V1 motif
→可信编译器
→旋转、平移和置换测试
→V2 motif及V1到V2事务
→LLM typed patch接口
```

## 11. 检索范围、证据质量与局限

### 11.1 检索路径

- 时间主体：2020年至2026年7月，补充必要奠基工作；
- 关键词族：`LLM neural architecture search`、`code-level NAS`、`program synthesis NAS`、`architecture DSL`、`grammar-guided NAS`、`equivariant NAS`、`open-ended NAS`；
- 数据源：arXiv API、DBLP、Crossref、PMLR、ACM DOI、OpenReview、正式论文PDF和官方代码链接；
- 引文扩展：从EvoPrompting、LLMatic、αNAS、Equivariance-aware Optimization和2026年LLM-NAS论文的相关工作继续追踪；
- 原文阅读：对本报告主要结论依赖的论文核读摘要、方法、表示形式、生成输出、合法性检查和实验部分。

### 11.2 本轮新增全文

- From Memorization to Creativity: LLM as a Designer of Novel Neural Architectures，16页；
- Delta-Based Neural Architecture Search: LLM Fine-Tuning via Code Diffs，19页。

两篇均已下载到本地并完成全文机制核查。它们目前是预印本，不能当作已同行评审论文。

### 11.3 证据边界

- “正式发表”由正式proceedings、DOI、DBLP或OpenReview交叉核验；
- “方法做了什么”以PDF原文为主要证据；
- OpenAlex本轮返回429，未将其自动摘要或引用量用于结论；
- 搜索没有发现完整相同系统，只能说明当前检索范围内存在空白，不能证明绝无遗漏；
- 2026年工作仍快速出现，投稿前必须重新刷新检索。

## 12. 最终判断

用户提出的想法可以被精确地拆成三个已有部分和一个尚待验证的交叉创新：

```text
已有：架构作为程序或细粒度语言
+已有：LLM生成或修改架构代码
+已有：等变约束和等变架构搜索
→潜在创新：LLM在可编译、可静态验证的三维等变架构DSL中进行程序合成
```

所以，对外最严谨的说法是：

> 已有工作分别探索了LLM代码级NAS、程序合成NAS和等变架构搜索，但它们尚未形成一个面向三维等变网络的类型化架构语言。我们的目标是将群表示约束转化为LLM可操作的语法、类型和重写规则，使LLM生成的不是任意Python代码，而是可在训练前完成合法性与等变性验证的架构程序。

## 参考文献

1. Chen A, Dohan D M, So D R. EvoPrompting: Language Models for Code-Level Neural Architecture Search. NeurIPS, 2023. [arXiv](https://arxiv.org/abs/2302.14838)
2. Nasir M U, Earle S, Cleghorn C W, et al. LLMatic: Neural Architecture Search via Large Language Models and Quality Diversity Optimization. GECCO, 2024. [DOI](https://doi.org/10.1145/3638529.3654017)
3. Khalid W, Ignatov D, Timofte R. From Memorization to Creativity: LLM as a Designer of Novel Neural Architectures. arXiv preprint, 2026. [arXiv](https://arxiv.org/abs/2601.02997)
4. Adhikari S P, Timofte R, Ignatov D. Delta-Based Neural Architecture Search: LLM Fine-Tuning via Code Diffs. arXiv preprint, 2026. [arXiv](https://arxiv.org/abs/2605.04903)
5. Liu Z, Liu Y, Wang J, et al. Structured Progressive Knowledge Activation for LLM-Driven Neural Architecture Search. arXiv preprint, 2026. [arXiv](https://arxiv.org/abs/2605.04057)
6. Sakuma Y, Yoshimura M, Gröpl M, et al. Structuring Open-Ended NAS: Semi-Automated Design Knowledge Structuring with LLMs for Efficient Neural Architecture Search. arXiv preprint, 2026. [arXiv](https://arxiv.org/abs/2605.19247)
7. Real E, Liang C, So D R, et al. AutoML-Zero: Evolving Machine Learning Algorithms From Scratch. ICML, 2020. [PMLR](https://proceedings.mlr.press/v119/real20a.html)
8. So D R, Mańke W, Liu H, et al. Primer: Searching for Efficient Transformers for Language Modeling. NeurIPS, 2021. [arXiv](https://arxiv.org/abs/2109.08668)
9. Turner J, Crowley E J, O'Boyle M F P. Neural Architecture Search as Program Transformation Exploration. ASPLOS, 2021. [DOI](https://doi.org/10.1145/3445814.3446753)
10. Jin C, Phothilimthana P M, Roy S. Neural Architecture Search using Property Guided Synthesis. OOPSLA, 2022. [DOI](https://doi.org/10.1145/3563329)
11. Zhuo Y, Su Z, Zhao C, Gao M. Syno: Structured Synthesis for Neural Operators. ASPLOS, 2025. [DOI](https://doi.org/10.1145/3676642.3736118)
12. Basu S, Magesh A, Yadav H, Varshney L R. Autoequivariant Network Search via Group Decomposition. arXiv preprint, 2021. [arXiv](https://arxiv.org/abs/2104.04848)
13. Maile K, Wilson D G, Forré P. Equivariance-aware Architectural Optimization of Neural Networks. ICLR, 2023. [OpenReview](https://openreview.net/forum?id=a6rCdfABJXg)
14. Pelleriti N, Nelaturu S H, Zhou Z, et al. What Do Evolutionary Coding Agents Evolve? arXiv preprint, 2026. [arXiv](https://arxiv.org/abs/2605.20086)
