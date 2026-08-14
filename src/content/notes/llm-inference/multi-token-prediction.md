---
title: "论文解读：多 Token 预测如何让大模型更强、更快"
summary: "完整拆解 Meta MTP 论文的方法、训练目标和实验，并结合 vLLM 的 MTP Speculator、DeepSeek MTP 层、RejectionSampler、KV Cache 与调度实现。"
published: 2026-08-14
category:
  - 论文阅读
  - 论文解读
paperId: "meta-multi-token-prediction"
topics:
  - MTP
  - Multi-Token Prediction
  - 解码加速
  - Speculative Decoding
  - Meta
---

## 论文信息

- 论文：[Better & Faster Large Language Models via Multi-token Prediction](https://arxiv.org/abs/2404.19737)
- 作者：Fabian Gloeckle、Badr Youbi Idrissi、Baptiste Rozière、David Lopez-Paz、Gabriel Synnaeve
- 机构：Meta FAIR、École des Ponts ParisTech、Université Paris-Saclay
- 首次发布：2024-04-30
- 会议：ICML 2024

这篇论文是理解 **Multi-Token Prediction（MTP）** 最合适的起点。它讨论的并不只是一个推理加速技巧，而是一个同时影响预训练目标、模型能力和解码方式的完整方案。

## 一句话结论

传统语言模型在每个位置只预测下一个 Token；MTP 让共享主干同时预测后续多个 Token，使训练信号关注更长程的结构，并让额外预测头可以直接用于自推测解码。

论文的核心发现是：

- MTP 的收益会随着模型规模增大而增强；
- 对代码生成、摘要和算法推理的帮助最明显；
- 它不保证所有自然语言基准都提升；
- 额外预测头可在不引入独立 Draft Model 的情况下实现自推测解码；
- 4-Token 模型在论文实验中获得约 `2.7×–3.0×` 的生成加速。

## 1. 为什么 Next-Token Prediction 不够

标准语言模型使用 Next-Token Prediction（NTP）：

```text
给定 x₁ … xₜ，只预测 xₜ₊₁
```

它简单、稳定，并且可以利用任意无标注文本。但论文认为它存在两个问题。

### 过度关注局部模式

在 Teacher Forcing 训练中，模型每一步都会看到真实历史。只预测下一个 Token 时，大量训练信号来自标点、固定搭配、缩进和局部语法等容易判断的内容。

真正决定后续文本走向的“关键选择点”反而较少。例如在代码中，选择一个错误变量、分支或算法后，后续很多 Token 都会受到影响，但 NTP 只直接惩罚当前一步。

### 训练和生成存在分布差异

训练时，模型始终基于正确历史预测；生成时，模型必须依赖自己之前的输出。一旦早期生成错误，错误会继续累积。

MTP 通过要求模型从当前位置同时考虑更远的未来，迫使隐状态编码更多长期信息。

## 2. MTP 的训练目标

对于标准 NTP，损失可以写成：

```text
L₁ = -Σₜ log P(xₜ₊₁ | x₁…xₜ)
```

MTP 将目标扩展为后续 `n` 个 Token：

```text
Lₙ = -Σₜ Σᵢ₌₁ⁿ log P(xₜ₊ᵢ | zₜ)
```

其中 `zₜ` 是共享 Transformer 主干根据当前上下文得到的隐状态。

模型结构由三部分组成：

1. 一个共享 Transformer 主干；
2. `n` 个独立预测头，每个头负责一个未来位置；
3. 一个共享的 Unembedding 矩阵，将隐状态映射到词表 Logits。

需要注意：论文中的多个预测头是**并行且相互独立**的。第 `i` 个头直接从共享表示预测第 `i` 个未来 Token，不依赖前一个预测头的输出。

这与 DeepSeek-V3 的顺序式 MTP 不同，后文会单独比较。

## 3. 如何控制训练显存

MTP 最直接的问题是 Logits 显存。

假设词表大小为 `V`，同时物化 `n` 个预测头的 Logits 和梯度，会产生近似 `O(nV)` 的显存开销。对于数万甚至十几万词表，这会迅速成为瓶颈。

论文采用顺序执行预测头的方法：

1. 主干只前向一次，得到共享表示；
2. 依次执行每个预测头的前向和反向；
3. 将梯度累积到共享主干；
4. 当前头完成后立即释放其 Logits 和梯度；
5. 再执行下一个预测头。

这样峰值显存从：

```text
O(nV + d)
```

降低为：

```text
O(V + d)
```

其中 `d` 是隐藏维度。

论文正文将其描述为理论上几乎没有额外训练时间和显存开销。不过附录给出了更谨慎的结果：受 FSDP 通信与计算没有充分重叠影响，大模型训练仍有约 `2%–9%` 的实测时间开销，小模型上的开销更高。作者认为这是实现问题，而不是方法本身的必要成本。

## 4. 实验结果说明了什么

### 收益随模型规模增大

作者从头训练了约 300M 到 13B 参数的代码模型。小模型使用 MTP 有时不如 NTP，但随着模型规模增加，MTP 的优势逐渐明显。

13B 模型相对同等规模 NTP 基线：

- HumanEval 解决的问题约增加 12%；
- MBPP 解决的问题约增加 17%；
- 代码任务平均提升约 15%。

这说明 MTP 需要足够的模型容量来同时学习短期预测和长期结构。

### 4-Token 通常是较好的平衡点

在 7B、32K 词表、200B 代码 Token 的实验中，作者比较了 `n=1、2、4、6、8`。

`n=4` 在 HumanEval 和 MBPP 上整体表现最好，但 APPS 的最佳值更接近 `n=6`。论文因此没有给出固定最优值，而是认为 `n` 与数据分布、词表和任务相关。

### 对代码比普通文本更有效

代码具有更强的长程约束：

- 括号和缩进必须闭合；
- 变量需要前后一致；
- 函数签名决定后续实现；
- 算法选择影响整段程序。

因此从当前状态预测多个未来 Token，更容易迫使模型学习程序结构。

### 自然语言结果并非全面提升

论文的自然语言实验更复杂：

- 2-Token 模型在多个选择题基准上与 NTP 大致持平；
- 4-Token 模型在部分选择题基准上有所下降；
- 摘要任务上，2-Token 和 4-Token 模型均有提升；
- GSM8K 的结果会随着训练数据量和 `n` 改变，结论不稳定。

所以不能简单认为“预测更多 Token 一定让模型更强”。收益主要出现在生成、代码和需要长程规划的任务中。

### 后期微调不能替代预训练

附录尝试在已训练好的 Llama 2 上加入 MTP 继续训练，但没有得到显著收益。作者认为，突然改变训练目标会破坏原有初始化。

这意味着 MTP 更适合作为预训练阶段就存在的目标，而不是模型完成后再临时添加的插件。

## 5. 为什么 MTP 可能有效

论文给出两个解释，但都仍属于假设而非严格证明。

### 关键选择点获得更高权重

如果某个 Token 会决定后续多个 Token，那么从更早位置预测这些未来 Token 时，与该选择相关的错误会被重复计入损失。

论文推导认为，在简化场景中：

- 关键选择点获得约 `n(n+1)/2` 的隐式权重；
- 普通局部 Token 的权重约为 `n`。

因此 MTP 会相对强调那些真正影响后续结构的决策。

### 强化相邻 Token 之间的互信息

对于未来两个 Token `X` 和 `Y`，NTP 只优化 `H(X)`；2-Token Prediction 同时优化 `H(X) + H(Y)`。

经过信息论分解后，`X` 与 `Y` 的互信息项在 MTP 中获得更高权重。直观上，模型更关注“当前选择如何影响后续内容”。

### 更容易形成 Induction Head

在合成实验中，小模型使用 2-Token Prediction 更容易学会 Induction：

```text
前面出现 A → B
后面再次出现 A
模型倾向预测 B
```

当模型足够大或数据本身足够高质量时，这项优势会减弱，因为 NTP 也能学会相同模式。

## 6. MTP 如何加速推理

训练完成后有两种使用方式。

### 只保留主预测头

直接丢弃额外预测头，模型退化为普通自回归模型。此时：

- 推理接口不变；
- 不会因为 MTP 增加推理开销；
- 仍可能保留 MTP 预训练带来的能力提升。

### 使用额外预测头做自推测解码

额外预测头可以一次提出多个候选 Token，然后由主模型验证。它与普通 Speculative Decoding 的区别是：

- 不需要加载独立 Draft Model；
- Draft Head 与主模型共享绝大多数计算和权重；
- 候选质量已在预训练阶段得到优化。

论文在 7B、4-Token 模型上的贪心解码实验得到：

- 代码生成约 `3.0×` 加速；
- 普通文本约 `2.7×` 加速；
- 代码中三个候选平均接受约 2.5 个；
- 加速在不同 Batch Size 下相对稳定。

对于 8-Byte Prediction 模型，论文报告约 `6.4×` 加速。但 Byte-Level 模型的序列更长，不能直接将这个数字与常规 Token 模型横向比较。

## 7. 与 Medusa、EAGLE 和 DeepSeek-V3 的区别

### MTP 论文

主要目标是改善预训练：

- 多个独立 Transformer 预测头；
- 所有头从同一共享表示并行预测；
- 同时研究能力提升和自推测解码。

### Medusa

主要目标是推理加速：

- 在已有模型上增加多个解码头；
- 通过树状候选和 Tree Attention 一次验证多个分支；
- 更偏向 Post-Training 和部署阶段。

### EAGLE

主要目标也是推理加速：

- 不直接在 Token 空间做简单预测；
- 使用特征级预测构建高质量 Draft；
- 重点优化候选接受率和验证效率。

### DeepSeek-V3 MTP

DeepSeek-V3 受到本文启发，但采用顺序式 MTP：

- 每个 MTP 模块接收上一深度表示和对应真实 Token；
- 不同预测深度保持完整因果链；
- 输出头与主模型共享；
- 首要目标是增加训练信号密度和模型能力；
- 推理时可以删除 MTP 模块，也可以复用于推测解码。

因此，“MTP”不是唯一固定结构，而是一类让模型在训练时显式面向多个未来 Token 的方法。

## 8. 工程实现时最值得关注的点

### 公平比较需要固定参数或计算预算

论文增加预测头时，会从共享主干中移除相同数量的层，以保持总参数量近似一致。否则更好的结果可能只是来自额外参数。

### 不要同时物化所有词表 Logits

如果直接计算 `[batch, sequence, n, vocabulary]`，显存会迅速爆炸。顺序执行预测头、梯度累积和及时释放 Logits 是关键实现细节。

### 同时评估能力与加速

建议至少分开评估：

- 训练吞吐和峰值显存；
- 代码、摘要、推理等生成任务；
- 标准语言模型损失与选择题基准；
- 候选接受率；
- 每次 Forward 接受的平均 Token 数；
- 不同 Batch Size 下的端到端吞吐和延迟。

### 推理收益依赖候选接受率

理论上使用 `k` 个预测头最多接近 `k×` 加速，但实际还要扣除：

- 候选验证计算；
- Tree Attention 或 Block Verification 开销；
- 低接受率导致的无效草稿；
- 额外 KV Cache 和调度成本；
- Kernel Launch 与显存带宽瓶颈。

## 9. vLLM 中的 MTP 实现

下面以 2026 年 8 月的 vLLM V1 文档与主干源码为参考。vLLM 的 Speculative Decoding 模块仍在快速演进，类名和目录可能随版本变化，但 Draft、Verify、Accept/Reject 和 KV Cache 管理这条主链路是稳定的。

### 9.1 先区分论文 MTP 与 vLLM 的 `method="mtp"`

Meta 论文提出的是一种**训练架构**：

- 共享 Transformer 主干；
- 多个相互独立的未来 Token 预测头；
- 预测头在预训练时共同参与损失；
- 推理时可丢弃，也可用于 Self-Speculative Decoding。

vLLM 不负责训练这些预测头。它做的是：

1. 识别模型 Checkpoint 中已经存在的原生 MTP 层；
2. 将 MTP 层加载为 Speculator；
3. 使用 MTP 层产生 Draft Token；
4. 用目标模型批量验证；
5. 将接受的 Token 提交给请求状态。

因此，并不是任意 Hugging Face CausalLM 都能通过添加 `"method": "mtp"` 获得 MTP。模型架构和 Checkpoint 必须包含 vLLM 已支持的 MTP 权重。

当前 vLLM 文档中的原生 MTP 路线包括 DeepSeek 系列、MiMo，以及使用专用 Assistant Checkpoint 的 Gemma 4 等。若模型没有原生 MTP，应改用 EAGLE、外部 Draft Model、N-Gram 或其他 Speculator。

### 9.2 启动配置

当前推荐的统一方法名是 `mtp`：

```bash
vllm serve <model> \
  --tensor-parallel-size 1 \
  --speculative-config '{
    "method": "mtp",
    "num_speculative_tokens": 1
  }'
```

Python API 对应写法：

```python
from vllm import LLM

llm = LLM(
    model="<native-mtp-model>",
    tensor_parallel_size=1,
    speculative_config={
        "method": "mtp",
        "num_speculative_tokens": 1,
    },
)
```

配置解析位于 [`vllm/config/speculative.py`](https://github.com/vllm-project/vllm/blob/main/vllm/config/speculative.py)。

它会完成几件重要工作：

- 将旧版本的 `deepseek_mtp` 等方法名归一为 `mtp`；
- 如果没有显式指定 Draft Model，MTP 会引用目标模型自身的 Checkpoint；
- Draft 部分的量化配置会与目标模型对齐；
- 校验模型架构是否支持 MTP；
- 设置 `num_speculative_tokens`、Draft Sampling 和 CUDA Graph 等运行参数。

`num_speculative_tokens=1` 表示每轮额外提出一个 Draft Token。它看起来不像“同时预测很多 Token”，但如果该 Token 被接受，目标模型还能发出一个 Bonus Token，因此一次迭代仍可能提交两个 Token。

建议先从 `1` 开始，而不是直接设置很大的值。更深的 Draft 会增加循环、KV Cache、验证长度、CUDA Graph Shape 和低接受率带来的浪费。

### 9.3 vLLM 的初始化链路

在较新的模块化 GPU Worker 实现中，初始化链路可以简化为：

```text
SpeculativeConfig(method="mtp")
        │
        ▼
init_speculator(...)
        │
        ▼
MTPSpeculator
        │
        ├── 继承 AutoRegressiveSpeculator
        └── 加载目标模型内的 MTP 层
```

入口可参考：

- [`vllm/v1/worker/gpu/spec_decode/__init__.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/spec_decode/__init__.py)
- [`vllm/v1/worker/gpu/spec_decode/mtp/speculator.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/spec_decode/mtp/speculator.py)

在其他 V1 版本中，同一逻辑可能通过 `SpecDecodeBaseProposer` 或 `EagleProposer` 体系组织。不要只根据某个版本的类名理解架构，应关注三个稳定接口：

- 输入目标模型 Hidden State；
- 生成 `K` 个 Draft Token；
- 将 Draft Token 和可选 Draft Probability 交给 Verifier。

### 9.4 DeepSeek MTP 层如何映射到代码

DeepSeek 的 MTP 与 Meta 论文的并行独立头不同。它使用顺序 MTP 模块，并保持不同预测深度之间的因果链。

vLLM 中的核心实现位于：

[`vllm/model_executor/models/deepseek_mtp.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/deepseek_mtp.py)

`DeepSeekMultiTokenPredictorLayer` 的主要组成包括：

- `enorm`：对当前 Token Embedding 做 RMSNorm；
- `hnorm`：对上一层 Hidden State 做 RMSNorm；
- `eh_proj`：将两者拼接后从 `2H` 投影回 `H`；
- `mtp_block`：一层 DeepSeek Decoder Layer；
- `shared_head`：与主模型共享或兼容的归一化和词表输出头。

可以将其前向过程近似理解为：

```python
embedding = RMSNorm(current_token_embedding)
hidden = RMSNorm(previous_hidden_state)

mtp_input = concat(embedding, hidden)
mtp_input = linear_2h_to_h(mtp_input)

draft_hidden = deepseek_decoder_layer(mtp_input)
draft_logits = shared_lm_head(draft_hidden)
```

源码中还会：

- 将位置 0 的无效 MTP Embedding 置零；
- 管理 Residual 与 Final Norm 前后的 Hidden State；
- 对 MoE、Tensor Parallel、Sequence Parallel 做适配；
- 检查量化 Checkpoint 是否真的包含 MTP 层；
- 为 DeepSeek-V3.2 的稀疏 MLA 路径维护 Top-K Index Buffer。

如果量化模型只导出了主模型权重，没有导出 MTP 层，vLLM 会提示 MTP 权重缺失。这类 Checkpoint 即使目标模型本身能正常推理，也不能直接启用 MTP。

### 9.5 Draft Token 如何产生

核心 Proposer 路径可参考：

[`vllm/v1/spec_decode/llm_base_proposer.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/spec_decode/llm_base_proposer.py)

它接收的关键输入包括：

- `target_token_ids`：目标模型本轮处理的 Token；
- `target_positions`：位置索引；
- `target_hidden_states`：目标模型输出的 Hidden State；
- `next_token_ids`：主模型刚采样出的 Token；
- `common_attn_metadata`：Paged KV Cache 和 Attention 元数据；
- `sampling_metadata`：Temperature、随机数生成器等采样信息；
- `slot_mappings`：Draft KV Cache 写入位置。

第一步 Draft 的逻辑是：

```text
目标模型 Hidden State
      +
刚生成的目标 Token Embedding
      │
      ▼
MTP Decoder Layer
      │
      ▼
Draft Logits → Draft Token
```

当 `num_speculative_tokens == 1` 时，vLLM 在第一次 MTP Forward 后直接返回 `[batch, 1]` 的 Draft Token。

当 `K > 1` 时，vLLM 会进入自回归 Draft 循环：

1. 将上一步 Draft Token 作为下一步输入；
2. 更新 Position、Sequence Length 和 Slot Mapping；
3. 为 Draft 层重新构造 Attention Metadata；
4. 执行下一层或下一轮 MTP Forward；
5. 采样新的 Draft Token；
6. 最后将结果堆叠为 `[batch, K]`。

这意味着 vLLM 中的深层 MTP Draft 通常不是“一次 Kernel 同时吐出 K 个 Token”，而是用轻量 MTP 层完成 K 步 Draft，再让昂贵的目标模型一次验证整段候选。

### 9.6 Target Verification

得到 Draft Token 后，vLLM 将每个请求的候选追加到本轮验证输入中。目标模型对 Draft 区间执行一次 Forward，产生每个位置的 Target Logits。

概念流程如下：

```text
请求已有上下文
      │
      ├── MTP 提出 d₁, d₂, ... dₖ
      │
      ▼
目标模型一次验证这些位置
      │
      ▼
Target Logits p₁, p₂, ... pₖ
      │
      ▼
RejectionSampler
      │
      ├── 接受最长合法前缀
      ├── 在首个拒绝处恢复采样
      └── 全部接受时追加 Bonus Token
```

Verifier 不是简单比较整段字符串是否完全一致。它按位置处理，一旦出现首个拒绝，后续 Draft 就失去有效因果前提，必须全部丢弃。

### 9.7 Greedy 与随机采样的接受规则

vLLM V1 的验证逻辑位于：

[`vllm/v1/sample/rejection_sampler.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/rejection_sampler.py)

#### Greedy

对于贪心解码：

```text
若 draft_token == argmax(target_logits)
    接受
否则
    拒绝该位置及其后所有 Draft
    使用目标模型 Token
```

#### Random Sampling

设 Draft 分布为 `q(x)`，目标模型分布为 `p(x)`，Draft Token 的接受概率为：

```text
min(1, p(x) / q(x))
```

如果拒绝，则从修正分布中恢复采样：

```text
normalize(max(p(x) - q(x), 0))
```

这样才能在理论上保持目标模型原始分布不变。

当前 vLLM 实现会将不同请求的 Draft Token 展平，通过：

- `num_draft_tokens`
- `cu_num_draft_tokens`
- `max_spec_len`

描述 Ragged Batch，再由 Triton Kernel 并行完成最长前缀接受和恢复采样。

输出 Buffer 的形状近似为：

```text
[batch_size, max_spec_len + 1]
```

未使用位置由 `PLACEHOLDER_TOKEN_ID` 填充。

### 9.8 Bonus Token 与 KV Cache

如果所有 Draft Token 都被接受，目标模型已经计算出了 Draft 末尾位置之后的分布。此时可以再采样一个完全来自目标模型的 Bonus Token。

因此一轮最多输出：

```text
K 个接受的 Draft + 1 个 Bonus Token
```

但 Bonus Token 是在目标 Forward 结束后采样的，本轮尚未为它生成 KV Cache。下一轮必须将它重新送入模型以补齐 KV。

这也是 Speculative Decoding 工程实现中容易出错的地方：

- Scheduler 中的逻辑长度已经增加；
- KV Cache 中却还没有 Bonus Token；
- 下一轮 Input Preparation 必须识别并回填；
- 抢占、异步调度或请求重新入队时不能遗留 `-1` Placeholder。

vLLM 历史上多次修复过 Bonus Token 与 KV Cache 同步问题。阅读源码时要区分“已经返回给用户的 Token”和“已经写入 KV Cache 的 Token”。

### 9.9 Paged KV Cache 与 Draft 状态

MTP 层本身也可能包含 Attention，因此它需要自己的 Attention Metadata 和 Slot Mapping。

vLLM 会为 Draft 侧：

- 构造 Per-Group 和 Per-Layer Attention Metadata；
- 根据 Block Table 找到物理 KV Block；
- 更新 Draft Position 和 Sequence Length；
- 在多步 Draft 时推进 Slot Mapping；
- 在请求发生拒绝后调整有效长度；
- 让被拒绝的尾部 KV 在逻辑上失效并可被后续覆盖。

对于不同模型族，KV 共享方式不同。例如 Gemma 4 Assistant 路径会显式让 Assistant 层与目标模型共享 KV Cache；DeepSeek MTP 则按照其 MTP Decoder Layer 和模型配置建立对应 Draft Attention 状态。

不能笼统认为所有 MTP 都“完全复用目标 KV”或“拥有一份完整独立 KV”，具体取决于模型适配器。

### 9.10 CUDA Graph 与 Batch Shape

启用 MTP 后，目标模型一次 Decode 的 Query Length 不再固定为 1，而接近：

```text
num_speculative_tokens + 1
```

这会影响：

- CUDA Graph Capture Size；
- Padded Batch 与 Ragged Batch 策略；
- Attention Backend 对 Query Length 的支持；
- 每轮可调度 Token 数；
- `max_num_batched_tokens` 的实际消耗；
- GPU 显存 Profiling。

较新的 vLLM Runner 会根据 `K+1` 配置 Decode Graph，但模型和 Attention Backend 仍可能有限制。

例如部分 DeepSeek-V3.2 Sparse MLA Kernel 目前只稳定支持较小的 MTP 深度。设置 `num_speculative_tokens > 1` 前，应检查使用的 vLLM 版本、Attention Backend、量化 Checkpoint 和硬件。

### 9.11 为什么高并发下不一定更快

将普通 Decode 每轮耗时记作 `T_target(1)`，MTP 一轮近似包含：

```text
T_mtp_draft(K) + T_target_verify(K + 1) + T_accept
```

一轮平均提交 Token 数记作：

```text
E[L] = 1 + 平均接受的 Draft Token 数
```

只有当：

```text
E[L] / (T_mtp_draft + T_target_verify + T_accept)
>
1 / T_target(1)
```

MTP 才真正有收益。

低并发时，目标 Decode 常受内存带宽限制，多验证几个位置可能不会同比增加耗时，因此 MTP 很容易降低 TPOT。

高并发时：

- 普通 Continuous Batching 已能充分利用 GPU；
- Verification 增加的 Token 会挤占 Batch Token Budget；
- MTP Draft 产生额外 Kernel 和同步；
- 每个请求接受长度不同，Batch 更不规则；
- 更深的 K 会降低后续位置接受率。

所以 MTP 通常首先优化单请求或低并发延迟，而不是保证所有高吞吐场景都提升。

### 9.12 应该监控哪些指标

仅观察 Tokens/s 不足以判断 MTP 是否有效。至少需要对比：

- TTFT；
- TPOT / ITL；
- 端到端延迟；
- Request Throughput；
- Output Token Throughput；
- Draft Acceptance Rate；
- Mean Acceptance Length；
- 每个 Draft 位置的接受率；
- Draft、Verify、Sample 各阶段耗时；
- KV Cache 使用率；
- GPU 利用率与显存带宽；
- 满足 SLO 的 Goodput。

vLLM 提供的核心指标包括：

```text
vllm:spec_decode_num_drafts
vllm:spec_decode_num_draft_tokens
vllm:spec_decode_num_accepted_tokens
vllm:spec_decode_num_accepted_tokens_per_pos
```

可以计算：

```text
acceptance_rate =
  accepted_tokens / draft_tokens

mean_acceptance_length =
  1 + accepted_tokens / num_drafts
```

第二个指标比单纯接受率更接近每轮实际能前进多少 Token。

### 9.13 建议的 vLLM A/B 测试

先运行无 MTP 基线：

```bash
vllm serve <model> \
  --tensor-parallel-size <tp> \
  --max-model-len <len>
```

再只增加 MTP：

```bash
vllm serve <model> \
  --tensor-parallel-size <tp> \
  --max-model-len <len> \
  --speculative-config '{
    "method": "mtp",
    "num_speculative_tokens": 1
  }'
```

测试时固定：

- 相同模型和量化方式；
- 相同 TP、PP、EP；
- 相同 Prompt/Output Length 分布；
- 相同 Sampling 参数；
- 相同 `max_num_seqs` 和 `max_num_batched_tokens`；
- 相同 Prefix Cache、Chunked Prefill 设置；
- 充分 Warmup；
- 分别测试并发 `1、4、16、64` 等负载点。

然后再尝试增加 `num_speculative_tokens`。不要一开始同时修改 MTP 深度、Batch Size、CUDA Graph 和调度参数，否则无法定位收益来源。

### 9.14 常见问题

#### Checkpoint 缺少 MTP 权重

一些量化或裁剪后的 Checkpoint 只包含主模型，没有 MTP 层。vLLM 会在权重加载阶段报告缺失。

#### 旧版本仍显示 `draft_model`

如果本应使用 MTP 的模型被识别成通用 Draft Model，应升级到包含对应模型适配的 vLLM 版本，不要强行绕过架构检查。

#### `K` 越大反而越慢

常见原因包括：

- 后续位置接受率快速下降；
- Draft 循环和 Kernel Launch 增多；
- Verification Token 数变大；
- CUDA Graph Shape 不匹配；
- Attention Backend 不支持深 Draft；
- 高并发下 GPU 已经饱和。

#### MTP 打开后显存不足

除了模型权重，还需要考虑：

- MTP 层权重；
- Draft Attention 的 KV Cache；
- 更长 Verification Query；
- 额外 CUDA Graph；
- Draft/Target Logits 和采样 Buffer。

#### 输出与基线不一致

贪心模式下应检查 Draft Verification 和 Bonus Token；随机采样模式下还要确认 Draft Probability、随机数状态、Top-K/Top-P 和 Rejection Sampling 路径是否受当前版本支持。

### 9.15 理论概念与 vLLM 对照

- 论文中的共享主干，对应 vLLM 中的目标模型主体；
- 论文中的额外预测头，对应模型 Checkpoint 中的原生 MTP 层；
- 论文中的候选未来 Token，对应 Proposer 返回的 Draft Token；
- 论文中的自推测解码，对应 MTP Speculator + Target Verification；
- 接受的最长前缀，由 `RejectionSampler` 产生；
- 全部接受后的额外输出，对应 Bonus Token；
- 候选序列状态，由 Scheduler、Attention Metadata 和 Paged KV Cache 共同维护。

这套映射比记忆类名更重要，因为 vLLM 的源码目录会变化，而这几类职责不会消失。

### 9.16 vLLM 相关源码

- [MTP 使用文档](https://docs.vllm.ai/en/latest/features/speculative_decoding/mtp/)
- [Speculative Decoding 总览](https://docs.vllm.ai/en/latest/features/speculative_decoding/)
- [SpeculativeConfig](https://github.com/vllm-project/vllm/blob/main/vllm/config/speculative.py)
- [MTP Speculator](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/spec_decode/mtp/speculator.py)
- [Base Proposer](https://github.com/vllm-project/vllm/blob/main/vllm/v1/spec_decode/llm_base_proposer.py)
- [DeepSeek MTP Model](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/deepseek_mtp.py)
- [Rejection Sampler](https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/rejection_sampler.py)
- [GPU Model Runner](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/model_runner.py)

## 10. 论文的局限

1. 最主要的正向结果来自代码数据，普通自然语言收益不稳定；
2. 实验最大模型为 13B，不能直接外推到所有前沿模型；
3. 论文宣称接近零训练开销，但附录中的现有实现仍有额外成本；
4. 最优预测长度依赖数据、任务和 Tokenizer；
5. 预训练收益与推理加速收益容易被混为一谈，实际上两者可以独立存在；
6. 自推测解码数字来自特定模型、硬件和贪心生成设置，不能直接视为线上服务收益；
7. 在已有模型上后加 MTP 的效果较弱，需要从训练目标设计阶段考虑。

## 11. 我的理解

这篇论文最重要的价值不是“多预测几个 Token”，而是重新审视了 Next-Token Prediction 的训练信号。

NTP 把每个位置视为同等重要的局部分类问题；MTP 则通过未来多个位置的联合监督，让那些会影响后续结构的选择获得更强训练信号。

它同时连接了两个方向：

- **训练侧**：更密集的监督、更好的样本效率和长程表示；
- **推理侧**：利用额外预测头做不依赖独立 Draft Model 的自推测解码。

但这两个收益不应混为同一个结论。即使部署时删除所有额外预测头，MTP 仍可能改善模型能力；反过来，推理加速是否成立，还取决于接受率、验证实现和实际服务负载。

## 延伸阅读

- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)：顺序式 MTP 与推测解码应用
- [Medusa](https://arxiv.org/abs/2401.10774)：多解码头与 Tree Attention
- [EAGLE](https://arxiv.org/abs/2401.15077)：特征级 Draft 模型
- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
