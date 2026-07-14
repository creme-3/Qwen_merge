# Qwen2.5-1.5B MLP 模型合并实验

本仓库用于复现 Qwen2.5-1.5B `math` 与 `coder` 专家模型的 MLP 子空间合并实验。实验目标不是让合并模型在所有任务上全面超过 `base`，而是验证能否在数学推理能力和代码生成能力之间取得更好的折中。

## 1. 实验结论

当前实验的主要结论如下：

- **合并方向**：以 `math` 作为 `base_model`，向 `coder` 注入 MLP 子空间能力。
- **关键层范围**：MLP `9-18` 层最关键。
- **最佳代表配置**：`ties_math_plus_coder_mlp_layers_9_18_lam_0p05_dens_0p85`。
- **核心目标**：重点关注 GSM8K 与 HumanEval；MMLU 作为通用能力稳定性参考。
- **主指标口径**：
  - GSM8K：`exact_match, flexible-extract`
  - MMLU：`acc, none`
  - HumanEval：`pass@1, create_test`

完整实验分析见：

```text
outputs/merge_experiment_report.md
```

精简结果表见：

```text
outputs/merge_results_summary.csv
```

## 2. 环境配置

环境配置说明见：

```text
ENVIRONMENT_SETUP.md
```

建议使用两个 Conda 环境：

- `merge`：运行 `mergekit` 与模型合并脚本。
- `model_merge`：运行 `lm-eval` 与模型测评脚本。

## 3. 仓库结构

当前提交版仓库只保留复现实验所需的代码、配置、CSV 结果和报告。

```text
.
├── README.md
├── ENVIRONMENT_SETUP.md
├── eval/
│   ├── evaluate_models.py
│   ├── normalized_score.py
│   ├── check_mergeability.py
│   ├── compare_model_params.py
│   └── compare_base_pairs.py
├── eval_results/
│   └── *.csv
├── merge/
│   ├── single_expert_merge.py
│   ├── slerp_experiment.py
│   └── experiments/
│       └── *.yml
└── outputs/
    ├── merge_experiment_report.md
    └── merge_results_summary.csv
```

说明：

- `models/` 不随仓库提交，复现实验时需在本地准备原始模型。
- `merge_outputs/` 不随仓库提交，合并模型可由脚本重新生成。
- `eval_results/` 仅保留报告引用的 CSV 结果，`.json` 和 `.log` 已清理。

## 4. 文件职责

| 路径 | 作用 |
|---|---|
| `merge/single_expert_merge.py` | Task Arithmetic 与 TIES 的主合并脚本，支持层选择、排除层、分层权重和 density 扫描。 |
| `merge/slerp_experiment.py` | 层选择式 SLERP 合并脚本。 |
| `merge/experiments/` | 与报告结果对应的 mergekit YAML 配置。 |
| `eval/evaluate_models.py` | 统一测评脚本，调用 `lm-eval` 运行 GSM8K、MMLU、HumanEval。 |
| `eval/normalized_score.py` | 根据 `base` 分数计算 `core_score` 和 `score_3task`。 |
| `eval/check_mergeability.py` | 检查模型结构、配置和参数形状是否适合合并。 |
| `eval/compare_model_params.py` | 比较两个模型之间的参数差异。 |
| `eval/compare_base_pairs.py` | 比较 `base`、`math`、`coder` 之间的参数差异。 |
| `eval_results/` | 保存最终报告引用的 CSV 测评结果。 |
| `outputs/merge_experiment_report.md` | 最终实验报告。 |
| `outputs/merge_results_summary.csv` | 精简结果汇总表。 |

## 5. 复现主实验

运行前进入项目根目录：

```bash
cd ~/Qwen_merge
```

### 5.1 TIES 主结果

合并模型：

```bash
conda activate merge
python merge/single_expert_merge.py \
  --method ties \
  --base-name math \
  --expert coder \
  --preset mlp \
  --layers 9-18 \
  --lambda 0.05 \
  --density 0.85
```

测评模型：

```bash
conda activate model_merge
python eval/evaluate_models.py \
  --scan \
  --models-dir merge_outputs \
  --models ties_math_plus_coder_mlp_layers_9_18_lam_0p05_dens_0p85 \
  --tasks gsm8k mmlu humaneval \
  --output eval_results/ties_math_plus_coder_mlp_layers_9_18_lam_0p05_dens_0p85-all.json 
```

### 5.2 Task Arithmetic 代表结果

合并模型：

```bash
conda activate merge
python merge/single_expert_merge.py \
  --method task_arithmetic \
  --base-name math \
  --expert coder \
  --preset mlp \
  --exclude-layers 0-8 \
  --lambda 0.05
```

测评模型：

```bash
conda activate model_merge
python eval/evaluate_models.py \
  --scan \
  --models-dir merge_outputs \
  --models ta_math_plus_coder_mlp_exclude_layers_0_8_lam_0p05 \
  --tasks gsm8k mmlu humaneval \
  --output eval_results/ta_math_plus_coder_mlp_exclude_layers_0_8_lam_0p05-all.json 
```

### 5.3 SLERP 代表结果

合并模型：

```bash
conda activate merge
python merge/slerp_experiment.py \
  --start math \
  --end coder \
  --layers 9-18 \
  --t 0.125
```

测评模型：

```bash
conda activate model_merge
python eval/evaluate_models.py \
  --scan \
  --models-dir merge_outputs \
  --models slerp_math_to_coder_mlp_layers_9_18_t_0p125 \
  --tasks gsm8k mmlu humaneval \
  --output eval_results/slerp_math_to_coder_mlp_layers_9_18_t_0p125-all.json 
```

## 6. 结果汇总

测评完成后可运行：

```bash
conda activate model_merge
python eval/normalized_score.py --results-dir eval_results
```

归一化分数定义：

```text
core_score = GSM8K / base_GSM8K + HumanEval / base_HumanEval
score_3task = core_score + MMLU / base_MMLU
```

其中 `base` 的 `core_score = 2.0000`，`score_3task = 3.0000`。

## 7. 不随仓库提交的内容

以下内容已清理或由 `.gitignore` 忽略：

- `models/`：原始模型权重，需本地准备。
- `merge_outputs/`：合并模型输出，可由脚本重新生成。
- `eval_results/*.json`、`eval_results/*.log`：测评原始 JSON 与日志，可重新生成。
- `__pycache__/`、`.tmp_recipes/`、`.vscode/`：缓存或本地临时文件。

