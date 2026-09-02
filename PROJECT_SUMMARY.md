# Qwen2.5-1.5B 专家模型合并项目总结

## 一句话介绍

针对 Qwen2.5-1.5B `base`、`math`、`coder` 模型配置不完全一致的问题，本项目设计并验证了 MLP 子空间的局部模型合并方案，在尽量保留数学推理能力的同时补强代码生成能力。

## 项目背景

三个 checkpoint 虽然属于同一模型系列，但 `max_position_embeddings`、`rope_theta`、`sliding_window` 等配置存在差异。直接进行全参数合并可能破坏注意力和位置编码相关行为，因此项目将实验范围限制在 MLP 层，并通过结构检查脚本确认模型是否具备合并条件。

## 技术方案

- **合并方向**：以 `math` 作为基座，向 `coder` 注入 MLP 子空间能力。
- **合并范围**：比较全 MLP、排除层和指定层范围，重点搜索 9-18 层。
- **合并方法**：Task Arithmetic、TIES、层选择式 SLERP。
- **实验参数**：扫描 lambda、TIES density 和 SLERP `t`，统一生成 mergekit YAML 并保存实验配置。
- **评测任务**：GSM8K 衡量数学推理，HumanEval 衡量代码生成，MMLU 作为通用能力参考。
- **自动化入口**：`merge/` 负责合并，`eval/` 负责评测、结构检查、参数比较和归一化结果统计。

## 代表结果

最佳代表配置：`math + coder / MLP layers 9-18 / TIES / lambda=0.05 / density=0.85`

| 模型/配置 | GSM8K | MMLU | HumanEval | core_score |
|---|---:|---:|---:|---:|
| `base` | 0.6300 | 0.6095 | 0.3720 | 2.0000 |
| `math` | 0.7407 | 0.4375 | 0.3110 | 2.0118 |
| `coder` | 0.5792 | 0.5374 | 0.4024 | 2.0013 |
| TIES 最佳代表 | 0.7293 | 0.4310 | 0.3902 | 2.2068 |

相对 `math`，代表模型的 GSM8K 从 `0.7407` 变为 `0.7293`，HumanEval 从 `0.3110` 提升到 `0.3902`。这说明局部 MLP 合并可以在数学能力基本保持的同时改善代码能力，但 MMLU 下降表明通用能力仍存在代价。

## 关键结论

1. 全参数合并风险较高，MLP-only 是更适合本实验模型差异的保守方案。
2. `math -> coder` 比 `coder -> math` 更适合作为能力迁移方向。
3. MLP 9-18 层对代码能力注入最敏感。
4. TIES 在当前实验中的代表结果最高，但 SLERP 接近，说明合并方向和层选择比具体算法更重要。
5. HumanEval 样本量较小，`0.3780`、`0.3902` 等小幅差异需要谨慎解释。

## 工程亮点

- 将合并方法、层范围、lambda、density 和分层权重参数化，支持单次运行与扫描实验。
- 通过统一评测入口固定 few-shot、解码参数和随机种子，输出 JSON、CSV 与日志。
- 保留 mergekit YAML、实验报告、CSV 结果和可视化图，便于审阅实验过程。
- 提供模型结构、配置、权重 key/shape 和 tokenizer 资产检查工具，降低不兼容合并风险。

## 简历描述

### 中文

设计并实现 Qwen2.5-1.5B 数学/代码专家模型合并实验框架，针对 checkpoint 配置差异将合并范围限制在 MLP 子空间，系统比较 Task Arithmetic、TIES 与层选择式 SLERP。通过层消融和参数扫描定位 MLP 9-18 层为关键迁移区域；最佳 TIES 配置在保持 GSM8K 0.7293 的同时，将 HumanEval 从 math 模型的 0.3110 提升至 0.3902，并配套实现统一评测、结果汇总和模型兼容性检查工具。

### English

Built a reproducible Qwen2.5-1.5B expert-model merging pipeline for math and code capability transfer. Restricted merging to the MLP subspace to account for checkpoint configuration differences, and benchmarked Task Arithmetic, TIES, and layer-selective SLERP. Layer ablations identified MLP layers 9-18 as the most effective transfer region; the best TIES configuration preserved GSM8K at 0.7293 while improving HumanEval from 0.3110 for the math checkpoint to 0.3902, supported by unified evaluation and mergeability-check tooling.

## 面试讲解主线

- **问题**：为什么不能直接全参数合并？三个 checkpoint 的上下文和位置编码配置不同。
- **方法**：为什么选择 MLP？先隔离风险较低、可控的参数子空间，再比较多种合并算法。
- **发现**：为什么是 9-18 层？层消融显示去掉该范围后 HumanEval 和核心分数下降最明显。
- **权衡**：为什么不宣称全面超过 base？最佳模型提升了核心数学/代码组合目标，但 MMLU 有明显下降。
- **反思**：HumanEval 样本量有限，density 0.75-0.85 的差异很小，需要重复测评后再做更强结论。
