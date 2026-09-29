# Router 评测

本目录收录 Router M0 三分类数据集、Q4/Q8/F16 量化质量与 CPU 耗时评测脚本，以及 Jev 三分类评测脚本。这里不收录历史运行结果。请从项目根目录执行命令，输出写到本地临时目录。

| 评测 | 数据集 | 入口 | 主要指标 |
|---|---|---|---|
| M0 60 题 / 量化质量矩阵 | `datasets/router_m0_test_v1.jsonl`（60 题） | `python -m eval.router.router_m0.matrix` | 准确率、macro F1、各类召回率、协议通过率、逐题总耗时 |
| Q4/Q8/F16 CPU 耗时矩阵 | `datasets/router_perf_v1.jsonl`（3 个合成输入） | `python -m eval.router.run_router_perf` | 端到端耗时、首 token 耗时、prefill/decode、吞吐、峰值 RSS |
| Jev 三分类 | 同一套 M0 60 题 | `python -m eval.router.run_jev` | 准确率、macro F1、各类召回率、逐题总耗时、错误数 |

M0 数据集的类别、语言及上下文分布见 [数据集说明](datasets/ROUTER_DATASET_CARD.md)。其 SHA-256 为 `4eebf40bad255264efd014e12c1a28297803d97770d2afa716985a16d50fb53b`。不要把这套测试题用于训练或调参；公开后，它也不再是私有留出集。

## 环境

先安装项目根目录的 `requirements.txt`，再安装本目录的 `requirements.txt`（CPU 性能采样使用 `psutil`）。本地模型评测还需要 `llama-server` 可执行文件。仓库发布版仅带 Q4_K_M 模型；完整矩阵需要自行把同一 Router V3 的 Q8_0 和 F16 GGUF 放在 `models.example.json` 指定的位置。所有模型依次运行，不同时占用内存。

```powershell
python -m pip install -r requirements.txt -r eval/router/requirements.txt
python -m eval.router.router_m0.dataset_validation --dataset eval/router/datasets/router_m0_test_v1.jsonl --phase phase02 --sha eval/router/datasets/router_m0_test_v1.sha256
python -m eval.router.run_jev --validate-only
```

## 本地模型质量

在 PowerShell 中设置 `$server` 为本机 `llama-server.exe` 的路径。只测发布版 Q4 时加 `--model-id`；测完整矩阵时去掉该参数，并先放好 Q8_0、F16 模型。

```powershell
$server = 'C:\path\to\llama-server.exe'
python -m eval.router.router_m0.matrix --dataset eval/router/datasets/router_m0_test_v1.jsonl --sha eval/router/datasets/router_m0_test_v1.sha256 --registry eval/router/models.example.json --llama-server $server --output-root .tmp/router-eval/quality --model-id qwen3-1.7b-router-sft-v3-q4-k-m
```

质量脚本启动本地服务，对 60 题逐一预测，写出 `predictions.jsonl`、`scorecard.json` 和 `lineage.json`。`scorecard.json` 包含质量指标与单题总耗时；它不测流式首 token 耗时。

## 本地模型耗时

下面是 Q4 的示例。将 `--model` 和 `--name` 换成 Q8_0、F16 对应模型后分别运行，即可得到量化耗时矩阵。每个输入先预热，再测 5 次；默认场景为 `warm_prefix_hit`，与原 CPU 矩阵口径一致。不要把 3 个合成输入上的意图匹配率当作 60 题准确率。

```powershell
python -m eval.router.run_router_perf baseline --name q4-k-m --llama-server $server --model models/router/qwen3-1.7b-router-sft-v3/Qwen3-1.7B-Router-SFT-V3-Q4_K_M.gguf --dataset eval/router/datasets/router_perf_v1.jsonl --output-root .tmp/router-eval/performance --threads 4 --threads-batch 4 --cache-type-k f16 --cache-type-v f16 --scenario warm_prefix_hit --repeats 5
```

该脚本生成 `measurements.jsonl`、`scorecard.json`、`manifest.json` 等本地文件。首 token 耗时从请求发出计至首次非空流式内容；prefill/decode 使用服务端 token timing。

## Jev

在项目的 `backend/.env` 填写 `JEV_API_KEY`；使用网关时，再按对应服务填写 `JEV_BASE_URL` 和 `JEV_MODEL`。这些变量与项目运行时的 Jev Router 共用。密钥不要写进本目录或提交到 Git。

```powershell
python -m eval.router.run_jev --output .tmp/router-eval/jev-m0.json
```

脚本只向 Jev 发送当前问题和有界会话上下文，不发送标准答案或数据集元数据；逐题记录预测、错误类型和端到端耗时。`--validate-only` 只校验数据集，不调用 API。
