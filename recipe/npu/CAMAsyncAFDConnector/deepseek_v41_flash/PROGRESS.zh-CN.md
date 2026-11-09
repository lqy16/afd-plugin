# DSV4.1 Flash A5 AFD 适配进度

更新时间：2026-11-09；设备联调状态截至 2026-11-06。

目前服务器上的 Attention 和 FFN 均已越过模型构造和权重加载，进入异步
connector 初始化，在检查四个 `afd_async_*` 接口时停止。A5 构建、扩展加载
和 Python 模型导入已经完成；CAM 通信域、真实推理和精度测试尚未完成。

## 1. 版本与目录

| 组件 | 版本 / 提交 |
| --- | --- |
| AFD-plugin 基线 | `v0.32.0rc1`，`652e249872bc968fc645c68ba6683182e0f69368` |
| vLLM | `0.32.0`，`5df63a7203cdb9328d824a27b682e77c7b9928be` |
| vLLM-Ascend | `f9c8e0bdb263d7a8c0270536f1ea00fa7373a84e` |
| CANN | `9.2.1`，`/usr/local/Ascend/cann-9.2.1` |
| Python / torch / torch-npu | `3.12.16` / `2.10.0+cpu` / `2.10.0.post7` |
| 设备 | 单机 8 卡 Ascend 950DT / A5 |

本地开发仓库：`/root/projects/codex-workspace/vllm-a5-dsv41/afd-plugin`。
服务器容器内仓库：`/home/l00961550/afd-plugin`。
分支：`codex/dsv41-flash-a5-afd`。
Git 身份：`liuqingyuan <liuqingyuan19@huawei.com>`。
适配代码及联调修复维护在上述开发分支。

容器内的 vLLM / vLLM-Ascend 入口分别为 `/vllm-workspace/vllm` 和
`/vllm-workspace/vllm-ascend`，沿用现有 editable 安装和目录软链接。

模型路径：

```text
/mnt/xds/sfs/DeepSeek-V4.1-Flash/DeepSeek-V4.1-Flash
```

目标模型为 `DeepseekV41ForCausalLM`：40 层、384 路由专家、H=5120，
SWA=128，含 Engram，使用 FP8 线性层和 FP4 专家权重。

## 2. 已完成的代码

| 部分 | 实现 |
| --- | --- |
| 模型注册 | 新增 NPU 的 `AFDDeepseekV41ForCausalLM`，由 AFD worker 选择 |
| Attention 模型 | 继承原生 V4.1 decoder/model forward，保留 HC、`pre_mix`、共享 KV、indexer 和 Engram |
| A/F 边界 | Attention 执行 gate、top-k、shared experts；路由专家调用复用已有 V4 A3 CAM proxy |
| FFN 模型 | 构造本侧 routed experts，按照任务携带的 `layer_idx` 选择计算层 |
| 权重加载 | 一次遍历完成角色筛选，保留原名称和 tensor，交给原生 loader 做映射、量化和 EP 分片 |
| bounded replay | 增加模型专用 ReplayState，按本 Attention DP 决定压缩，复用原生 metadata/context/scatter |
| Engram | 保留原生流程，多 Attention DP 使用 CPU offload 和共享主机表 |

主要代码位于 `afd_plugin/model_executor/models/npu/`：

- `deepseek_v41.py`：模型、角色构造及远端 MoE 调用。
- `deepseek_v41_weights.py`：checkpoint 权重归属与筛选。
- `deepseek_v41_state.py`：本 DP 的 replay 状态及原生 Engram 状态接入。

公共 runner、connector 实现和算子代码没有修改。本轮仍使用已有 A3
`afd_async_dispatch_send`、`afd_async_dispatch_recv`、`afd_async_combine_send`、
`afd_async_combine_recv` 接口，尚未接入 CAM 仓库的 async-megamoe 三算子。

### 联调修复已同步本地

| 问题 | 修复 |
| --- | --- |
| FFN 的层工厂不接受 `prefix=` | 改为 `lambda prefix: ...`，测试替身也使用真实的关键字调用 |
| Attention 缺少 `gate.bias_vl` 参数 | 按原生 `vision_n_layers > 0` 条件创建 FP32、无梯度参数并正常加载 |
| FFN 尝试加载未构造的 `aligner` | 两侧过滤顶层 aligner、vision、image、MTP 权重，兼容有无 `model.` 前缀；保留层内 `gate.bias_vl` |
| `enable_dsa_cp` 不被容器配置类接受 | 从 A5 启动命令删除，保留默认 PCP1/DCP1 |

### 验证进展

服务器结果来自本轮用户提供的日志：

| 阶段 | 结果 |
| --- | --- |
| `SOC_VERSION=950` 编译和安装 | 成功 |
| `afd_plugin._C_ascend` 加载 | 成功，路径在开发仓库内 |
| A2E/E2A 接口注册检查 | 通过，未执行内核数值测试 |
| V4.1 模型和 ModelState 导入 | 成功 |
| Attention 模型构造、权重加载 | 已越过，进入 `connector.init_afd_connector()` |
| FFN 模型构造、权重加载 | 已越过，进入 `start_ffn_server_loop()` 的 connector 初始化 |
| 四个异步接口 | 未注册，两侧均在此退出 |
| CAM 通信、请求执行、数值精度、性能 | 尚未验证 |

本地使用 CPU PyTorch 2.10 做接口和状态测试；最新 56 项定向检查通过，
包含角色构造、原始权重迭代、可选 gate 参数、层号传递和 replay 状态重置。
这些测试不执行 NPU 算子。

2026-11-09 复查完整本地 `tests/unit`：**985 passed、155 skipped、
60 subtests passed**。Ruff 与三个新增生产模块的 mypy 检查通过；mypy 使用
`--follow-imports=skip`，未校验本机缺少的外部 vLLM/vLLM-Ascend 类型。

## 3. 在新容器中安装

使用已应用完整适配代码及上述修复的仓库，沿用容器现有 CANN 环境。
保留现有 vLLM、vLLM-Ascend 和 torch 等依赖：

```bash
cd /home/l00961550/afd-plugin

export SOC_VERSION=950
export AFD_BUILD_ASCEND_OPS=1
export AFD_SKIP_ACLNN_BUILD=0
export AFD_BUILD_JOBS=8
export MAX_JOBS=8
export ASCEND_CANN_PACKAGE_PATH="$ASCEND_HOME_PATH"

python3 -m pip install --no-deps --no-build-isolation -v -e .
```

该流程构建插件算子包和 Python 扩展，插件 vendor 安装在仓库的
`afd_plugin/_cann_ops_custom/vendors/afd-plugin/`，不替换原 CANN 安装包。
950 配置当前只选择 A2E/E2A，主动排除四个异步接口。

安装显示 `0.32.0rc2.dev0+g652e24987.d20261106` 是 setuptools-scm 根据
基线提交和未提交修改生成的开发版本号，不表示切换到了另一个 rc2 提交。

安装后的 Python 修改只需重启对应服务；修改 C++ 或 AscendC 才需要重编译。

验证扩展与模型导入：

```bash
VLLM_PLUGINS=ascend,ascend_model,ascend_model_loader,ascend_service_profiling,ascend_kv_connector,afd VLLM_USE_V2_MODEL_RUNNER=1 python3 - <<'PY'
import torch
import torch_npu
from vllm.plugins import load_general_plugins
load_general_plugins()

from afd_plugin.compat.npu import ensure_afd_ascend_ops_loaded
ensure_afd_ascend_ops_loaded()

import afd_plugin._C_ascend as extension
from afd_plugin.model_executor.models.npu.deepseek_v41 import AFDDeepseekV41ForCausalLM
from afd_plugin.model_executor.models.npu.deepseek_v41_state import AFDDeepseekV41ModelState
print("Extension:", extension.__file__)
print("Model/state imports: PASS")

for name in (
    "afd_async_dispatch_send", "afd_async_dispatch_recv",
    "afd_async_combine_send", "afd_async_combine_recv",
):
    print(name, "registered=", hasattr(torch.ops.afd_ascend, name))
PY
```

当前预期四个异步接口均为 `False`。

## 4. 单机 4A＋4F 启动

先停止之前占用这 8 卡的同步服务。两个终端分别前台启动 FFN 和 Attention，
日志直接看终端，不使用后台管理脚本。

| 项目 | Attention | FFN |
| --- | --- | --- |
| 设备 | 0、1、2、3 | 4、5、6、7 |
| 角色并行 | DP4、TP1 | EP4、TP1 |
| API 端口 | 7100 | 7101 |
| DP RPC 端口 | 17100 | 17101 |
| CAM rendezvous | 两侧均为 `127.0.0.1:12499` | 同左 |

FFN 启动参数中的 `data-parallel-size=4` 沿用 AFD 的进程组织方式，配合
expert parallel 构成四个专家分片，每卡每层 96 个专家。

首轮调试采用最大序列 4096、chunk budget 512、每个 Attention DP 最多
一个请求。eager 开启，DBO/MTP/CP/PP 关闭，prefix caching/EMS 关闭，
bounded replay 保留。`--no-async-scheduling` 关闭 vLLM 调度器的异步调度，
AFD 的异步执行仍由 `async_dp=true` 开启。

以下指令复现本轮启动流程；当前仍会在异步接口检查处停止。

两个终端都先执行：

```bash
export MODEL_PATH=/mnt/xds/sfs/DeepSeek-V4.1-Flash/DeepSeek-V4.1-Flash
export VLLM_PLUGINS=ascend,ascend_model,ascend_model_loader,ascend_service_profiling,ascend_kv_connector,afd
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0
export AFD_ASYNC_CAM_LAYERED_GMM=0
export BATCH_SIZE_FACTOR=0.0625
export HCCL_BUFFSIZE=512
export OMP_NUM_THREADS=4
export ASCEND_GLOBAL_LOG_LEVEL=3
```

终端 1，FFN：

```bash
ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 \
vllm serve "$MODEL_PATH" \
    --served-model-name dsv41-afd \
    --host 127.0.0.1 --port 7101 --api-server-count 1 \
    --data-parallel-address 127.0.0.1 \
    --data-parallel-rpc-port 17101 \
    --data-parallel-size 4 --data-parallel-size-local 4 \
    --tensor-parallel-size 1 --pipeline-parallel-size 1 \
    --enable-expert-parallel \
    --all2all-backend flashinfer_all2allv \
    --tokenizer-mode deepseek_v41 \
    --quantization deepseek_v4_fp8 --dtype bfloat16 \
    --enforce-eager --no-async-scheduling \
    --no-enable-dbo --no-enable-eplb \
    --max-model-len 4096 \
    --max-num-batched-tokens 512 --max-num-seqs 1 \
    --enable-chunked-prefill --long-prefill-token-threshold 0 \
    --no-enable-prefix-caching --swa-bounded-replay \
    --engram-config '{"cpu_offload":true,"dp_shared_memory":true}' \
    --limit-mm-per-prompt '{"image":0}' \
    --gpu-memory-utilization 0.85 --block-size 128 --seed 1024 \
    --additional-config '{
      "enable_cpu_binding":false,
      "enable_fused_mc2":0,
      "multistream_dsv4_dsa_overlap":false,
      "afd":{
        "role":"ffn",
        "connector":"CAMAsyncAFDConnector",
        "async_dp":true,
        "host":"127.0.0.1",
        "port":12499,
        "num_attention_ranks":4,
        "num_ffn_ranks":4,
        "compute_gate_on_attention":true,
        "connector_extra_config":{
          "dynamicQuant":1,
          "attn_ranks_per_dp":1,
          "hccl_buffer_size":512
        }
      }
    }'
```

终端 2，Attention：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
vllm serve "$MODEL_PATH" \
    --served-model-name dsv41-afd \
    --host 127.0.0.1 --port 7100 --api-server-count 1 \
    --data-parallel-address 127.0.0.1 \
    --data-parallel-rpc-port 17100 \
    --data-parallel-size 4 --data-parallel-size-local 4 \
    --tensor-parallel-size 1 --pipeline-parallel-size 1 \
    --enable-expert-parallel \
    --all2all-backend allgather_reducescatter \
    --tokenizer-mode deepseek_v41 \
    --quantization deepseek_v4_fp8 --dtype bfloat16 \
    --enforce-eager --no-async-scheduling \
    --no-enable-dbo --no-enable-eplb \
    --max-model-len 4096 \
    --max-num-batched-tokens 512 --max-num-seqs 1 \
    --enable-chunked-prefill --long-prefill-token-threshold 0 \
    --no-enable-prefix-caching --swa-bounded-replay \
    --engram-config '{"cpu_offload":true,"dp_shared_memory":true}' \
    --limit-mm-per-prompt '{"image":0}' \
    --gpu-memory-utilization 0.85 --block-size 128 --seed 1024 \
    --additional-config '{
      "enable_cpu_binding":false,
      "enable_fused_mc2":0,
      "multistream_dsv4_dsa_overlap":false,
      "afd":{
        "role":"attention",
        "connector":"CAMAsyncAFDConnector",
        "async_dp":true,
        "host":"127.0.0.1",
        "port":12499,
        "num_attention_ranks":4,
        "num_ffn_ranks":4,
        "compute_gate_on_attention":true,
        "connector_extra_config":{
          "dynamicQuant":1,
          "attn_ranks_per_dp":1,
          "hccl_buffer_size":512
        }
      }
    }'
```

停止时在两个终端分别按 `Ctrl+C`。

## 5. 当前停止位置与后续工作

两侧都在 `ensure_cam_async_ops_available()` 检查失败：

```text
RuntimeError: Async CAM requires all four torch.ops.afd_ascend.afd_async_* operators.
```

`csrc/npu/ascend_kernels/operator_registry.json` 的 950 项只有 A2E/E2A，
`csrc/npu/pybind/CMakeLists.txt` 同时排除了异步接口绑定。
报错里的重新编译建议是通用提示；相同配置重编译仍不会产生这四个接口。

后续顺序：

1. 接入 A5 异步通信算子及 connector，完成通信域初始化。
2. 接入本权重的 MXFP4 FFN 计算；现有 helper 仅有 NONE/W8A8/W4A8 分支。
3. 用短请求对照原生同步基线，检查 logits/token、路由缩放和 shared expert 合并。
4. 检查多 DP 异步请求、chunked prefill 和 bounded replay，再扩大序列长度并采集性能。
5. 后续再接入 CAM 通信算子仓库提供的 async-megamoe 三算子。

算子接通、Attention 服务就绪后，使用以下请求做 prefill-only 冒烟测试：

```bash
curl -sS --max-time 10 http://127.0.0.1:7100/v1/models

curl -sS --max-time 300 http://127.0.0.1:7100/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{
      "model":"dsv41-afd",
      "prompt":"The capital of France is",
      "max_tokens":1,
      "temperature":0,
      "ignore_eos":true
    }'
```

Prefill-only 通过最多生成一个 token 实现，不配置独立 Decode 服务或 KV 传输。
