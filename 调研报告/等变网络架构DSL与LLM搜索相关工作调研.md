# 等变网络架构 DSL 与 LLM 搜索相关工作调研

## 1. 调研结论

当前研究设想可以表述为：构建一种带有不可约表示（irrep）类型、等变组合规则、执行语义和合法重写机制的等变神经网络架构领域专用语言（DSL），使用大语言模型作为启发式程序合成器，根据历史训练结果提出新算子、子图或重写规则，再由可信检查器保证候选的类型、形状和等变合法性，最终使用 QM9 极化率验证集 MAE 判断性能。

这是一种“形式化约束与经验优化结合”的神经架构搜索，而不是纯粹的形式化证明：

- DSL 和类型系统负责定义哪些程序合法；
- 等变检查器负责验证数学结构和实现；
- LLM 负责生成具有潜力的候选，而不是证明候选最优；
- QM9 训练与验证负责判断经验性能；
- 最终只能报告搜索中发现的最佳候选，不能声称全局最优。

在本次检索覆盖的近年 CCF A 类会议及直接相关预印本中，没有发现一篇工作同时实现以下四项：

1. 三维 SO(3)/O(3) irrep 类型系统；
2. 可组合、可编译的等变算子 DSL；
3. LLM 生成并迭代改进类型安全的架构重写；
4. 在 QM9 等三维分子任务上进行多保真架构搜索。

现有工作分别覆盖了程序合成 NAS、低级 primitive 搜索、LLM 代码级 NAS、等变性架构优化、合法神经网络图生成和算子规则归纳。这些方向的交叉仍存在清晰的研究空白。

## 2. 对研究设想的严格定义

### 2.1 DSL 的组成

可以将等变架构 DSL 形式化为：

$$
\mathcal{L}=(\Sigma,G,T,\operatorname{Sem},R)
$$

其中：

- $\Sigma$：基础变量和 primitive 集合；
- $G$：架构程序的组合语法；
- $T$：irrep 类型和合法性判断规则；
- $\operatorname{Sem}(p)$：DSL 程序 $p$ 的数学与 PyTorch 执行语义；
- $R$：保持类型或满足证明义务的架构重写规则。

候选架构至少需要满足：

$$
\Gamma\vdash p:\tau
$$

以及相应的等变关系：

$$
f_p(g\cdot x)=\rho(g)f_p(x).
$$

### 2.2 不应交给 LLM 修改的可信规则

以下规则属于可信内核，应由群表示理论和人工验证的 primitive 固定：

- Clebsch–Gordan 三角耦合规则；
- 宇称组合规则；
- 各 primitive 的输入与输出 irrep；
- 残差相加的类型一致性；
- 标量性质预测头的不变性；
- 坐标平移、旋转和原子置换的变换规律。

LLM 不能因为某次训练结果更好就修改这些数学规则。

### 2.3 可以由 LLM 搜索的内容

LLM 可以在可信规则下提出：

- 新的合法张量积路径组合；
- 不同阶特征的通道分配和交互方式；
- Gate、S² activation、归一化和残差的组合；
- Rotate-to-edge-frame、SO(2) mixing、Rotate-back 等 primitive 形成的新子图；
- 基于历史实验结果的派生设计原则；
- 带前置条件、后置类型和资源约束的子图重写模板。

更准确的目标是让 LLM 归纳和搜索“派生设计规则”，而不是修改等变数学公理。

### 2.4 与形式化方法的关系

该体系接近反例引导归纳合成（CEGIS）：

```text
LLM提出候选程序
→ DSL解析
→ 类型与等变检查
→ 得到静态反例或实现错误
→ 训练并获得性能反馈
→ 根据反例和性能继续生成候选
```

但 QM9 验证 MAE 是经验指标，不能被形式证明。因此更准确的定位是：

> 受形式化方法启发的、性能反馈驱动的神经符号等变架构搜索。

## 3. 近年 CCF A 类直接相关工作

### 3.1 Neural Architecture Search using Property Guided Synthesis

- 作者：Charles Jin、Phitchaya Mangpo Phothilimthana、Sudip Roy
- 会议：OOPSLA 2022，CCF A
- DOI：<https://doi.org/10.1145/3563329>
- arXiv：<https://arxiv.org/abs/2205.03960>
- 代码：Google Research `abstract_nas`

这是与当前设想最接近的形式化 NAS 工作。论文把神经网络计算图视为程序，将具体架构抽象为程序属性，并在抽象属性空间内进行变异，再通过程序合成生成满足目标属性的具体子图。

其主要属性包括：

- shape property：输入输出形状；
- depth property：非线性路径深度；
- mixing property：输入元素和输出元素之间的信息混合关系。

工作流为：

```text
选择父架构子图
→ 静态推导程序属性
→ 修改抽象属性
→ 合成满足新属性的子图
→ 替换原子图
→ 训练和评价
```

与本项目的对应关系：

| αNAS | 本项目 |
|---|---|
| shape/depth/mixing 属性 | irrep、阶数、张量积路径、等变误差 |
| 普通 DNN 计算图 | 三维等变计算图 |
| property-guided synthesis | equivariance-guided synthesis |
| 随机属性变异 | LLM 引导的属性与子图重写 |
| 静态 shape 推导 | irrep 类型与群作用检查 |

主要差距是 αNAS 没有 LLM、三维 irreps、等变算子语义和分子任务。

### 3.2 AutoML-Zero: Evolving Machine Learning Algorithms From Scratch

- 会议：ICML 2020，CCF A
- 正式页面：<https://proceedings.mlr.press/v119/real20a.html>

AutoML-Zero 使用由基础数学操作构成的低层指令语言。程序由 `Setup`、`Predict` 和 `Learn` 三个函数构成，每个函数是一系列包含操作和内存地址的指令。进化算法通过插入、删除和替换指令搜索完整学习算法。

其重要启示是：当搜索语言足够基础时，搜索对象可以从超参数和预制模块扩大到完整计算路径，甚至发现梯度归一化、权重平均和乘性交互等结构。

主要差距是它没有等变类型系统，搜索空间极度稀疏，需要非常高的候选吞吐量。

### 3.3 Primer: Searching for Efficient Transformers for Language Modeling

- 会议：NeurIPS 2021，CCF A
- arXiv：<https://arxiv.org/abs/2109.08668>

Primer 将 Transformer 表示为由 TensorFlow primitive 和多个 subprogram 构成的程序 DNA。Self-attention 不是不可修改的基本算子，而是由低级 primitive 组合形成的复合程序。

搜索最终发现了包括 squared ReLU 和 attention 后 depthwise convolution 在内的结构。这证明 primitive 级程序搜索能够产生超出预制操作列表的新结构。

其不足是缺乏强类型和形式约束，大量随机程序无效或无法稳定训练。本项目的 irrep 类型系统可以降低这种无效搜索比例。

### 3.4 AutoBERT-Zero: Evolving BERT Backbone from Scratch

- 会议：AAAI 2022，CCF A
- DOI：<https://doi.org/10.1609/aaai.v36i10.21311>
- arXiv：<https://arxiv.org/abs/2107.07445>

AutoBERT-Zero 将注意力结构表示为由基础数学操作构成的 DAG，并使用 Operation-Priority Evolution 根据历史候选表现调整不同操作的选择优先级。

它说明可以从 primitive 组合中搜索新注意力结构，而不是只在预定义 attention 变体之间选择。但其搜索引导是统计式 UCB，不是 LLM 推理，也没有形式语义和等变约束。

### 3.5 EvoPrompting: Language Models for Code-Level Neural Architecture Search

- 会议：NeurIPS 2023，CCF A
- DOI：<https://doi.org/10.48550/arXiv.2302.14838>
- arXiv：<https://arxiv.org/abs/2302.14838>

EvoPrompting 使用语言模型作为进化搜索中的 mutation 和 crossover operator，直接生成完整神经网络代码。

它证明 LLM 能够根据优秀父代程序和实验反馈生成新的架构，但代码空间缺少正式类型系统，主要依靠执行和训练暴露问题。

本项目可以将自由代码生成替换为：

```text
LLM生成Typed DSL Patch
→ 静态类型检查
→ 等变检查
→ 编译为PyTorch
```

### 3.6 Design Principle Transfer in Neural Architecture Search via Large Language Models

- 会议：AAAI 2025，CCF A
- DOI：<https://doi.org/10.1609/aaai.v39i21.34463>

该工作让 LLM 从已有架构、组件和性能结果中归纳自然语言 design principles，再利用原则排除不理想的架构区域，并根据新实验逐步更新原则。

它与“从实验中推导更优规则”最接近，但其规则是自然语言启发式，不是可执行或可形式检查的架构重写。

本项目可进一步把自然语言原则转化为：

```text
rewrite rule
+ 前置类型条件
+ 输出类型
+ 等变证明义务
+ 资源约束
```

### 3.7 Equivariance-aware Architectural Optimization of Neural Networks

- 会议：ICLR 2023，当前 CCF 目录为 A 类
- arXiv：<https://arxiv.org/abs/2210.05484>

该工作提出 equivariance relaxation morphism 和 `[G]`-mixed equivariant layer，并分别使用进化和可微 NAS 搜索每层应保留的群或子群等变约束。

它证明等变性本身可以成为架构搜索变量，而不必在所有层中完全固定。

主要差距是：

- 主要面向离散群和图像；
- 搜索群或子群，而不是完整算子程序；
- 没有 SO(3) irrep DSL；
- 没有 LLM 和分子任务。

### 3.8 NNSmith: Generating Diverse and Valid Test Cases for Deep Learning Compilers

- 会议：ASPLOS 2023，CCF A
- DOI：<https://doi.org/10.1145/3575693.3575707>
- arXiv：<https://arxiv.org/abs/2207.13066>

NNSmith 不是 NAS，而是深度学习编译器测试工具。它为算子定义输入类型、输出类型、属性约束和 shape transfer function，使用 symbolic tensor、symbolic integer 和 SMT solver 增量构造能够通过类型检查的计算图。

它可以作为等变 DSL 检查器的重要工程蓝本：

```text
普通类型：dtype + rank + shape
扩展类型：dtype + shape + irrep + parity + multiplicity
```

### 3.9 NeuRI: Diversifying DNN Generation via Inductive Rule Inference

- 会议：ESEC/FSE 2023，CCF A
- DOI：<https://doi.org/10.1145/3611643.3616337>
- arXiv：<https://arxiv.org/abs/2302.02261>

NeuRI 从算子的输入、属性和输出记录中，利用归纳程序合成自动发现输入合法性约束和输出 shape 传播规则。其规则语言由简单算术 grammar 构成。

它证明算子规则可以部分自动归纳。但群表示耦合等数学规则不能仅依靠有限运行样本推断，应作为可信规则固定；自动归纳更适合扩展派生规则、shape约束和资源模型。

## 4. 直接相关但未核验为 CCF A/B 正式会议的工作

### Autoequivariant Network Search via Group Decomposition

- arXiv：<https://arxiv.org/abs/2104.04848>

该工作利用群分解和强化学习搜索适当的群等变结构，说明自动搜索对称性具有直接先例。但本次未找到可核验的 CCF A/B 正式会议版本，因此不将其计入正式会议论文列表。

### SPARK

- 标题：Structured Progressive Knowledge Activation for LLM-Driven Neural Architecture Search
- arXiv：<https://arxiv.org/abs/2605.04057>
- 代码：<https://github.com/AIM-ResearchLab/SPARK>

SPARK 使用 ASR、RC 和 SAR 在 `OPERATOR`、`ACTION` 两个人工代码区域中进行局部 LLM 修改，并实施语法、接口、shape 和 masking 检查。它提供搜索控制策略，但不是正式 DSL。

其适合放置在等变 DSL 上层：

```text
ASR选择Typed IR修改范围
→ RC生成局部改进目标
→ SAR生成DSL Patch
→ DSL检查器和编译器保证合法
```

## 5. 现有研究覆盖矩阵

| 研究方向 | 代表工作 | 已解决 | 尚未解决 |
|---|---|---|---|
| 属性引导程序合成NAS | αNAS | 抽象属性、静态推导、子图合成 | LLM、3D等变 |
| 基础操作程序搜索 | AutoML-Zero、Primer、AutoBERT-Zero | primitive级新结构发现 | 强类型、等变证明 |
| LLM代码级NAS | EvoPrompting、SPARK | LLM变异与程序生成 | 正式DSL、合法性保证 |
| LLM归纳设计原则 | LAPT | 从历史架构提炼语言原则 | 原则不可执行、不可证明 |
| 等变性架构搜索 | EquiNAS | 搜索群/子群约束 | 3D等变算子程序搜索 |
| 合法神经网络图生成 | NNSmith | 类型与SMT约束 | 不优化任务性能 |
| 自动算子规则归纳 | NeuRI | 自动发现shape和输入规则 | 不处理等变语义和NAS |

## 6. 建议的研究定位

建议将研究问题写为：

> 能否将三维等变神经网络表示为一种带 irrep 类型和等变证明义务的领域专用语言，并利用 LLM 在该语言中执行实验反馈驱动的程序合成，从而自动发现比人工模板搜索更优的等变算子和网络结构？

建议的三个方法贡献为：

1. **Equivariant Architecture DSL**：定义 primitive、类型、语法、语义、规范化和编译后端；
2. **Proof-carrying LLM Architecture Evolution**：LLM 输出 DSL patch、修改理由、前置条件、输出类型和证明义务；
3. **Counterexample- and Performance-guided Search**：联合利用静态错误、等变反例、资源错误、训练曲线和 QM9 验证 MAE 引导搜索。

## 7. 与当前四因子搜索的关系

当前 `MACRO / REPRESENTATION / OPERATOR / ACTION` 四因子可以保留，但应从简单配置字段升级为 DSL 的顶层非终结符或作用域：

```text
Architecture
├── MacroProgram
├── RepresentationProgram
├── OperatorProgram
└── ActionProgram
```

SPARK 的 ASR 可以选择其中一个作用域，LLM 随后在该作用域内部生成一个或多个类型安全的子图修改。候选不应被限制为每次只改变一个数值字段；“每次只选择一个因子”与“因子内部只能改变一个节点”是两回事。

## 8. 不能声称全局最优的原因

要证明全局最优，需要有限搜索空间、完整枚举、统一充分训练、随机性控制和精确目标比较。当前实验不满足这些条件。

论文中应使用：

- best-found architecture；
- empirically superior architecture；
- Pareto-optimal candidate within the evaluated archive；
- search-efficient discovery。

不应使用 `provably optimal architecture` 或 `global optimum`。

## 9. 本次检索范围与证据边界

- 时间主体：2020年至2026年7月；
- 会议主体：ICML、NeurIPS、AAAI、ICLR、OOPSLA、ASPLOS、ESEC/FSE 等 CCF A 会议；
- 补充来源：arXiv、论文正式 DOI、PMLR、AAAI Proceedings、官方代码仓库；
- 核心论文均读取了摘要、方法、实验或结论部分，不以搜索摘要作为唯一依据；
- 未发现完整覆盖“3D等变DSL + LLM类型安全重写 + QM9 NAS”的工作，但该结论仅限于本次检索范围，不表示绝无遗漏。
