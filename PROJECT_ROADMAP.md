# Qwen Merge 项目升级路线

本文档用于记录将当前 Qwen2.5-1.5B 模型合并实验仓库升级为成熟简历项目的目标、优先级与完成状态。

状态说明：

- `[x]`：已经基本完成。
- `[ ]`：尚未完成。
- `[~]`：部分完成，仍需整理或工程化。

## 总目标

将当前项目从“课程/探索型实验仓库”升级为“可以写入简历、可复现、可展示、可信赖”的模型合并项目。

最终希望达到：

- 打开 GitHub 后，读者能在 2 分钟内理解项目价值。
- 跟随文档后，读者能在本地复现最佳合并配置。
- 仓库能体现模型合并、实验设计、工程实现、结果分析与自动化检查能力。

## P0：项目包装

- [x] 重写 `README.md`，改成更清晰的简历项目风格。
- [x] 明确项目背景：Qwen `base/math/coder` 架构不完全一致，直接全层合并风险较高。
- [x] 明确核心方法：只在 MLP 子空间中比较 Task Arithmetic、TIES 与 SLERP。
- [x] 给出核心结论：`math + coder/mlp` 是更有效方向，MLP 9-18 层是关键范围。
- [x] 保留最终实验报告：`outputs/merge_experiment_report.md`。
- [x] 增加结果可视化图：`outputs/figures/`。
- [x] 在 `README.md` 中加入更醒目的最佳结果表和关键图。
- [x] 在 `README.md` 中加入项目亮点摘要，方便简历和面试阅读。

## P1：可复现性

- [x] 提供环境说明文档：`ENVIRONMENT_SETUP.md`。
- [x] 说明需要本地准备模型权重，仓库不上传 `models/` 和 `merge_outputs/`。
- [x] 在报告中给出 Task Arithmetic、TIES、SLERP 与 baseline 测评命令。
- [x] 增加 `requirements-merge.txt` 或 `environment_merge.yml`。
- [x] 增加 `requirements-eval.txt` 或 `environment_eval.yml`。
- [x] 增加 `configs/paths.example.yml`，统一说明本地模型路径与输出路径。
- [x] 增加一键合并脚本：`scripts/run_best_merge.sh`。
- [x] 增加一键测评脚本：`scripts/run_best_eval.sh`。
- [x] 增加完整复现脚本：`scripts/reproduce_best.sh`。

## P2：代码工程化

- [x] 保留统一合并入口：`merge/single_expert_merge.py`。
- [x] 保留 SLERP 实验入口：`merge/slerp_experiment.py`。
- [x] 保留统一测评入口：`eval/evaluate_models.py`。
- [x] 保留归一化分数脚本：`eval/normalized_score.py`。
- [x] 当前脚本已经可用，并已补充复现、绘图和质量检查入口。
- [x] 抽象通用路径管理逻辑。
- [x] 抽象层选择解析逻辑。
- [x] 抽象 mergekit YAML 生成逻辑。
- [x] 为主要脚本补充更完整的 `--help` 参数说明。
- [x] 为关键函数增加 docstring。
- [x] 增加更友好的错误提示，例如模型路径不存在、CSV 指标缺失、任务名错误等。

## P3：结果可信度

- [x] 保留 GSM8K、MMLU、HumanEval 的 CSV 测评结果。
- [x] 报告中定义 `core_score` 与 `score_3task`。
- [x] 报告中加入基线对比、λ 扫描、分层消融、density 扫描、SLERP 扫描与方法对比。
- [x] 当前报告结果数字已和 `eval_results/` 校验一致。
- [x] 结果可视化已经完成，但绘图过程尚未脚本化保存。
- [x] 增加 `scripts/plot_results.py`，从 CSV 自动生成 `outputs/figures/`。
- [x] 增加 `scripts/check_results.py`，自动检查报告表格数字是否与 CSV 一致。
- [x] 增加 `scripts/check_project.py`，检查图片引用、CSV 字段和配置文件。
- [x] 明确说明 HumanEval 单次测评可能存在波动。
- [~] 已增加 `scripts/repeat_stability.py`；仍需在具备模型权重和评测环境后执行重复测评并记录结果。

## P4：自动化与质量检查

- [x] `.gitignore` 已忽略模型权重、合并输出、日志、缓存等不应提交内容。
- [x] 增加 `.gitattributes`，统一文本文件换行符。
- [x] 增加 GitHub Actions 工作流。
- [x] 自动检查 Python 语法。
- [x] 自动检查 YAML 是否可解析。
- [x] 自动检查 CSV 指标字段是否存在。
- [x] 自动检查 Markdown 图片引用是否有效。
- [x] 增加 `make check` 或等价命令，统一运行本地检查。

## P5：展示与简历材料

- [x] 已有完整实验报告。
- [x] 已有可视化结果图。
- [x] 增加 `PROJECT_SUMMARY.md`，用一页说明项目背景、方法、结果与亮点。
- [x] 增加简历描述版本，包括中文与英文。
- [x] 增加面试讲解提纲：问题、方法、难点、结果、反思。
- [x] 增加 `scripts/infer_compare.py`，本地对比 `base/math/coder/merged` 输出。
- [~] 已增加 PDF 导出脚本；仍需在安装 Pandoc/LaTeX 后生成 `outputs/merge_experiment_report.pdf`。

## 推荐执行顺序

1. 完善 `README.md`，加入项目亮点、最佳结果表和核心图。
2. 增加环境文件与一键复现实验脚本。
3. 增加绘图脚本和结果校验脚本。
4. 增加 GitHub Actions 与 `.gitattributes`。
5. 增加 `PROJECT_SUMMARY.md` 和简历/面试材料。

## 当前阶段判断

当前仓库已经具备：

- 完整实验过程。
- 可读实验报告。
- 关键结果 CSV。
- 多种合并方法对比。
- 可视化结果。

当前最主要短板是：

- 合并脚本的公共路径、层解析和 YAML 生成逻辑仍可进一步抽取。
- 需要在真实模型环境中完成最佳 TIES/SLERP 的重复测评。
- PDF 文件需要 Pandoc/LaTeX 环境才能生成。

因此，基础升级任务已完成；剩余重点是抽取合并公共模块，以及在时间允许时进行 HumanEval 重复测评和增加本地推理对比工具。
