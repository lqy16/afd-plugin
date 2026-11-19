# DeepSeek-V4.1-Flash A5 framework development

This branch runs DeepSeek V4.1 Flash on A5 with `CAMAsyncMegaAFDConnector`.
Build the split operators with `AFD_BUILD_ASCEND_OPS=1 SOC_VERSION=950 python3 -m pip install --no-deps --no-build-isolation .`.
Use the model, tokenizer and TP1/CP1 options below. Set `afd.async=true`,
`compute_gate_on_attention=true`, `enable_fused_mc2=0`, and
`connector_extra_config={"attn_ranks_per_dp":1}` on both roles.
The FFN operator selects the ready DP and layer on device.
See the [Mega runtime configuration](../../../../docs/npu/MEGA_EAGER.md).

## Historical ordinary CAM prototype

The remaining notes document the earlier ordinary CAM prototype. Its installation
commands and connector settings apply only to that historical checkout.

This adapter connects native DSV4.1 Attention to the existing DSV4 A3 Async CAM
FFN path. It retains the A3 operators for the first framework implementation.
The A5 Async CAM operator registrations and this checkpoint's MXFP4 FFN
computation are the next device integration tasks.

See the [current progress and complete launch commands](PROGRESS.zh-CN.md)
for the A5 container installation, 4A/4F topology and server debugging results.

## Source versions

| Component | Commit |
| --- | --- |
| AFD base, `v0.32.0rc1` | `652e249872bc968fc645c68ba6683182e0f69368` |
| vLLM, `v0.32.0` | `5df63a7203cdb9328d824a27b682e77c7b9928be` |
| Container vLLM-Ascend | `f9c8e0bdb263d7a8c0270536f1ea00fa7373a84e` |

Use the existing A5 container's vLLM, vLLM-Ascend, PyTorch, torch-npu and CANN
packages. Install the Python adapter from this checkout without replacing them:

```bash
cd /home/l00961550/afd-plugin
AFD_BUILD_ASCEND_OPS=0 python3 -m pip install \
    --no-deps --no-build-isolation -e .
```

This command installs Python code only. It does not install or build the A3
CAM operators. The repository's existing NPU installation procedure owns that
separate native extension/package.

## Execution path

```text
Attention DP, TP1                         FFN EP, TP1
native Attention / delayed HC
native router + shared experts
    Async CAM dispatch, layer_id ------> native routed expert weights[layer_id]
    native shared MLP overlaps           existing FFN GMM helper
    Async CAM combine <----------------- expert output
native HC / next layer / sampling
```

`DeepseekV41ForCausalLM` becomes `AFDDeepseekV41ForCausalLM` only in AFD
workers. Ordinary workers continue using the native model. Attention owns KV,
Engram and request/sampling state; the FFN daemon uses the connector's layer ID
to choose its local expert layer. Checkpoint names and tensors pass through one
role filter, then the native V4.1 loader handles mapping, quantization and EP
sharding.

Native V4.1 decoder/model `forward` methods are inherited. This preserves
`pre_mix`, shared KV/indexer state and the bounded replay layer group. Replay
admission is local to each asynchronous Attention DP; its row gathering,
attention metadata and result scattering use the native implementation.

## Configuration for the first device debugging pass

Start from the existing DSV4 Async CAM recipe and change the model path,
tokenizer and quantization to this checkpoint. Use these settings on both roles:

```bash
export MODEL_PATH=/mnt/xds/sfs/DeepSeek-V4.1-Flash/DeepSeek-V4.1-Flash
export VLLM_PLUGINS=ascend,ascend_model,ascend_model_loader,ascend_service_profiling,ascend_kv_connector,afd
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0
export AFD_ASYNC_CAM_LAYERED_GMM=0
```

```text
--tokenizer-mode deepseek_v41
--quantization deepseek_v4_fp8 --dtype bfloat16
--tensor-parallel-size 1 --pipeline-parallel-size 1
--enable-expert-parallel --enforce-eager
--no-enable-dbo --no-enable-eplb --no-async-scheduling
--max-model-len 4096 --max-num-batched-tokens 512 --max-num-seqs 1
--enable-chunked-prefill --no-enable-prefix-caching
--swa-bounded-replay
--engram-config '{"cpu_offload":true,"dp_shared_memory":true}'
--limit-mm-per-prompt '{"image":0}'
```

Set the existing connector's `afd.role` to `attention` or `ffn`, with
`connector="CAMAsyncAFDConnector"`, `async_dp=true`,
`compute_gate_on_attention=true`, and
`connector_extra_config={"dynamicQuant":1,"attn_ranks_per_dp":1}`. Declare
matching Attention/FFN rank counts and use separate devices and service/RPC
ports, as in the DSV4 recipe. Keep the existing transport's buffer sizing.

The role topology determines Attention DP and FFN EP independently. A native
DP8/EP8 run using all eight devices does not leave devices for a separate FFN
group. Choose an A/F split that fits this checkpoint before device execution.
Prefill-only requests use `max_tokens=1`; no Decode service or KV transfer is
needed. This does not use the native `enable_fused_mc2=2` MegaMoE path.

## Remaining device work

1. The retained extension exposes `afd_async_dispatch_send`,
   `afd_async_dispatch_recv`, `afd_async_combine_send` and
   `afd_async_combine_recv` on A3. Its Ascend 950 build does not register
   these Async APIs. Keep this visible when first starting A5 workers.
2. The reused FFN helper handles unquantized, W8A8 and W4A8 experts. The
   provided checkpoint uses FP8 activations/linear weights and FP4 expert
   weights; its quantized FFN branch still needs A5 integration.
3. This change does not use `async-megamoe` or its three split operators.
   Graph execution, DBO and speculative decoding are outside this first path.

## CPU checks

```bash
python3 -m pytest \
    tests/unit/model_executor/models/npu/test_deepseek_v41.py \
    tests/unit/model_executor/models/npu/test_deepseek_v41_weights.py \
    tests/unit/model_executor/models/npu/test_deepseek_v41_state.py \
    tests/unit/model_executor/models/test_model_utils.py
```

These checks cover role-specific construction/loading, task layer association,
the reused CAM proxy handoff and replay state reset. They do not execute NPU
operators. Native/AFD numerical comparison will follow device integration.
