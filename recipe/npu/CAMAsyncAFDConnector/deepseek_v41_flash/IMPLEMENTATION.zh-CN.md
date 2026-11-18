# DSV4.1 Flash A5 AFD 框架适配记录

截至 2026-11-06，服务器上的 Attention 和 FFN 已越过模型构造、权重加载，
两侧均在异步算子接口检查处停止。完整安装、直接启动指令和当前进度见
[PROGRESS.zh-CN.md](PROGRESS.zh-CN.md)。

## 代码版本

| 组件 | 基线提交 |
| --- | --- |
| AFD-plugin `v0.32.0rc1` | `652e249872bc968fc645c68ba6683182e0f69368` |
| 容器 vLLM `v0.32.0` | `5df63a7203cdb9328d824a27b682e77c7b9928be` |
| 容器 vLLM-Ascend | `f9c8e0bdb263d7a8c0270536f1ea00fa7373a84e` |

开发分支：`codex/dsv41-flash-a5-afd`。Git 作者配置为
`liuqingyuan <liuqingyuan19@huawei.com>`。适配代码及后续联调修复维护在该分支。

目标权重为 `DeepseekV41ForCausalLM`，40 层、384 路由专家、H=5120、
SWA=128、FP8 线性层与 FP4 专家，含 Engram。使用已有 A3 DSV4 Async CAM
接口完成框架连接，尚未接入 async-megamoe。

## 原生 V4 与 V4.1 的关键差异

| 项目 | 对适配的影响 | 实现 |
| --- | --- | --- |
| V4.1 延迟 HC，通过 `pre_mix` 跨层传递状态 | 直接套用 V4 decoder forward 会改变计算顺序 | 继承原生 V4.1 decoder 和 model 的 forward |
| V4.1 Attention 的 KV 来源层、indexer 和共享缓冲 | 逐层新建 V4 Attention 无法保留共享关系 | Attention 侧调用原生 V4.1 Model 构造函数，注入角色层工厂 |
| V4.1 MoE 新增 `already_sequence_parallel` 参数 | 原 V4 remote proxy 不接受这个参数 | 增加匹配 V4.1 调用的 proxy 入口，首版使用 TP1 |
| 原生 bounded replay 使用 DP 协调，异步 DP 下会直接跳过压缩 | 原样复用会使 AFD 请求失去 bounded replay | 模型专用 ReplayState 按本 DP 决定保留行，复用原生压缩、metadata、context 与 scatter |
| Engram 查询可能使用跨 DP 通信 | 各 Attention DP 进度独立时无法依赖查询阶段的全局同步 | 多 DP 使用原生 `dp_shared_memory` 主机表模式；EDP1 可使用本地表 |
| FP8/MXFP4 权重与 A3 W8A8/W4A8 FFN 路径不同 | 框架可连接，但专家计算分支仍需后续调试 | 保留原 A3 FFN helper，记录量化接入点 |

## 新增代码

### 模型适配：`deepseek_v41.py`

文件位于 `afd_plugin/model_executor/models/npu/`。

- `AFDDeepseekV41DecoderLayer`：Attention 侧构造原生 Attention、HC、norm、
  Engram 和本地 shared experts，将路由专家替换为 CAM proxy。FFN 侧只构造
  原生 routed MoE，不先构造完整 Attention 模型再删除。
- `AFDDeepseekV41AttentionGateRemoteMoE`：沿用 V4 的 router、dispatch、
  shared expert 重叠计算、combine 流程，接受 V4.1 的 MoE 参数。
- `AFDDeepseekV41Model`：Attention 侧保留原生 shared KV、Engram 和 replay
  初始化；FFN 侧构造专家层集合，按任务携带的 `layer_idx` 选择计算层。
- `AFDDeepseekV41ForCausalLM`：保留原生 loader、采样及模型入口，提供已有
  Async FFN runner 要求的模型接口。

任务路径为 Attention gate → CAM dispatch（携带层号）→ FFN 按层取权重
→ 原有 GMM helper → CAM combine → Attention 的 HC 与下一层。

### 权重筛选：`deepseek_v41_weights.py`

遍历 checkpoint 一次，保留原始名称和 tensor 对象，随后交给原生 loader：

| 权重 | Attention | FFN |
| --- | --- | --- |
| Attention、HC、层 norm、Engram | 加载 | 不构造、不加载 |
| routed experts | 不构造、不加载 | 加载并按 EP 分片 |
| shared experts | 加载并本地计算 | 不构造、不加载 |
| gate | 加载并计算路由 | 保留原生 MoE 的 gate 参数，供 loader/量化初始化使用 |
| embedding、最终 norm、head | 保留原生模型生命周期 | 沿用已有 V4 适配的公共权重策略 |
| 顶层 aligner、vision、image、MTP | 本轮文本请求不加载 | 不加载 |

`ffn`/`mlp` 等名称转换、packed projection、专家映射、量化 scale 和 EP
分片全部由容器中的原生 V4.1 loader 处理。

### Replay 状态：`deepseek_v41_state.py`

`AFDDeepseekV41ReplayState.prepare()` 删除本模型 replay 准入阶段的跨 DP
协调，按本 DP 的请求长度选择需要保留的行。完整 prompt logprobs 请求继续
保留全部输出。每步清空上一批的 replay 行与 context，避免长短请求切换时
复用旧状态。profile/capture 阶段沿用原生提前返回逻辑。

`AFDDeepseekV41ModelState` 通过原生 `get_model_state_cls()` 扩展点接入，
保留 Engram 的请求历史、输入准备及执行结束处理。

## 其他修改

- 注册 `AFDDeepseekV41ForCausalLM` 的延迟加载路径；仅 AFD NPU worker
  改写自身的模型配置，普通 vLLM worker 继续使用原生模型。
- 将 `deepseek_v41` 纳入已有 DSV4 CAM 配置检查。
- 增加开发配置说明、模块文档和 CPU 测试。
- 公共 runner、connector、C++/AscendC 算子和 vLLM/vLLM-Ascend 源码均未修改。

## 检查结果

本机使用隔离的 CPU PyTorch 2.10 环境进行检查：

- 当前模型、权重筛选、replay 与配置选择的定向检查为 56 项通过。
  测试使用真实 CPU tensor 和上游接口替身，不执行 NPU 计算。
- 联调修复后的完整 `tests/unit` 为 985 passed、155 skipped、
  60 subtests passed；运行时相关项目在 CPU 环境中跳过。
- Ruff 检查与格式检查通过。
- 三个新增生产模块的 mypy 检查通过，使用 `--follow-imports=skip`；
  外部 vLLM/vLLM-Ascend 类型未在本机验证。
- Python editable 安装通过，设置 `AFD_BUILD_ASCEND_OPS=0`，未构建算子。

服务器已完成 `SOC_VERSION=950` 的编译安装和扩展加载。原生/AFD 数值比较
及性能测试尚未开始，异步通信仍停在四个接口缺失的检查处。

## 联调修复

| 问题 | 本地代码修复 |
| --- | --- |
| FFN 层工厂不接受 `prefix=` | 将 `lambda layer_prefix` 改为 `lambda prefix`；测试替身也按原生接口使用关键字调用 |
| Attention 加载 `gate.bias_vl` 时 KeyError | 按原生 `vision_n_layers > 0` 条件创建 FP32、无梯度的参数，不跳过 checkpoint 权重 |
| FFN 加载 `aligner` 时 KeyError | 在两侧的角色筛选中排除 aligner、vision、image、MTP 的顶层路径，兼容有无 `model.` 前缀 |
| 启动参数 `enable_dsa_cp` 不被接受 | 从 A5 启动配置移除；当前原生配置类没有此字段 |

## 服务器后续调试顺序

1. 处理现有 A3 Async CAM 在 A5 上的算子注册/API 接入。当前 950 registry
   只有 `a2e/e2a`；四个 `afd_async_*` 算子仍是 A3 路径。
2. 接入此权重的 MXFP4 FFN 计算。现有 helper 的分支是 NONE/W8A8/W4A8，
   首次进入量化 FFN 时会暴露这个差异。
3. 完成短请求的原生/AFD 精度对比，再测多 DP 异步、chunked prefill 与
   bounded replay。之后再替换为 async-megamoe。

安装和配置项见同目录 `README.md`。Prefill-only 继续以 `max_tokens=1`
测试，不需要独立 Decode 服务或 KV 传输。
