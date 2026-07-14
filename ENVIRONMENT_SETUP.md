# Qwen Merge 环境配置说明

本项目建议使用两个独立 Conda 环境：

- `merge`：负责模型合并，主要运行 `mergekit-yaml` 与 `merge/` 下脚本。
- `model_merge`：负责模型测评，主要运行 `lm-eval` 与 `eval/` 下脚本。

将两个环境拆开可以减少依赖冲突，也方便在显存紧张时分别排查合并和测评问题。

## 1. 基础目录约定

默认项目位于：

```bash
cd /mnt/d/Qwen_merge
```

模型目录约定：

```text
models/
  Qwen2.5-1.5B-base/
  Qwen2.5-1.5B-math/
  Qwen2.5-1.5B-coder/
```

合并输出目录：

```text
merge_outputs/
```

测评结果目录：

```text
eval_results/
```

## 2. 合并环境：`merge`

创建环境：

```bash
conda create -n merge python=3.10 -y
conda activate merge
```

安装 PyTorch。若本机 CUDA 可用，安装与驱动匹配的 CUDA 版本；例如：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

安装合并相关依赖：

```bash
pip install mergekit transformers accelerate safetensors sentencepiece protobuf pyyaml
```

验证：

```bash
which mergekit-yaml
python - <<'PY'
import torch, transformers
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("transformers:", transformers.__version__)
PY
```

常用合并命令示例：

```bash
python merge/single_expert_merge.py \
  --method ties \
  --base-name math \
  --expert coder \
  --preset mlp \
  --layers 9-18 \
  --lambda 0.05 \
  --densities 0.75 0.8 0.85
```

## 3. 测评环境：`model_merge`

创建环境：

```bash
conda create -n model_merge python=3.10 -y
conda activate model_merge
```

安装 PyTorch：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

安装测评相关依赖：

```bash
pip install transformers accelerate datasets evaluate sentencepiece protobuf safetensors
pip install lm-eval[api]
```

HumanEval 需要执行生成代码，评测脚本会默认设置：

```bash
export HF_ALLOW_CODE_EVAL=1
```

如果数据集已经缓存到本地，建议使用离线模式减少网络波动：

```bash
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

常用测评命令示例：

```bash
python eval/evaluate_models.py \
  --scan \
  --models-dir merge_outputs \
  --models ties_math_plus_coder_mlp_layers_9_18_lam_0p05_dens_0p85 \
  --tasks gsm8k mmlu humaneval \
  --output eval_results/ties_math_plus_coder_mlp_layers_9_18_lam_0p05_dens_0p85-all.json \
  --offline
```

如果只想快速检查流程，可以加 `--limit`：

```bash
python eval/evaluate_models.py \
  --scan \
  --models-dir merge_outputs \
  --models ties_math_plus_coder_mlp_layers_9_18_lam_0p05_dens_0p85 \
  --tasks gsm8k \
  --limit 10 \
  --output eval_results/smoke_test.json
```

## 4. 结果汇总

归一化分数可通过：

```bash
python eval/normalized_score.py --results-dir eval_results
```

本项目报告中的主指标口径为：

```text
GSM8K     = exact_match, flexible-extract
MMLU      = acc, none
HumanEval = pass@1, create_test
```

其中：

```text
core_score = GSM8K / base_GSM8K + HumanEval / base_HumanEval
score_3task = core_score + MMLU / base_MMLU
```

## 5. 常见问题

### 5.1 Hugging Face 网络报错

如果看到类似 `Network is unreachable`，但日志显示使用 cached dataset，一般不会影响已有缓存的完整评测。建议：

```bash
python eval/evaluate_models.py ... --offline
```

### 5.2 CUDA 显存不足

合并和测评均建议单进程运行。运行前检查：

```bash
nvidia-smi
```

如果显存残留异常，优先关闭其他 Python 进程；必要时重启 WSL 或重启机器。

### 5.3 tokenizer 警告

如果偶发 tokenizer 网络或 regex 警告，先确认模型目录中的 tokenizer 文件完整，并尽量使用稳定网络或离线缓存。该警告通常与 Hugging Face 访问状态有关，不一定代表模型权重损坏。

