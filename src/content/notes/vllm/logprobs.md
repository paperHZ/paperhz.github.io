---
title: "vLLM logprobs 说明"
summary: "从 Transformer 前向、logits 与 log_softmax 数据流解释 vLLM 的 logprobs、prompt_logprobs 和 rank，并复盘 PD 分离下未初始化张量导致的 OverflowError。"
published: 2026-08-20
wide: true
category:
  - 大模型推理
  - vLLM 框架
  - vLLM logprobs 说明
topics:
  - vLLM
  - logprobs
  - prompt_logprobs
  - PD 分离
  - 故障分析
---

> 本文所有返回样例、数值、rank 均在一个 PD 分离部署的 decode 实例上实跑抓取
> （vLLM 0.24.x，一个 vocab 120832 的中文 MoE 模型），非手写示意。

---

## 一、logprobs 是什么

logprob = **log probability**，模型对某个 token 的对数概率，范围 `(-∞, 0]`：

| logprob | 概率 | 含义 |
|---|---|---|
| `0.0` | 100% | 完全确定 |
| `-0.69` | ~50% | 比较确定 |
| `-2.3` | ~10% | 一般 |
| `-5.5` | ~0.4% | 模型觉得意外 |

换算 `概率 = exp(logprob)`。用对数是因为概率连乘容易下溢，取对数后变累加。

### 两个参数，作用完全不同

| 参数 | 作用对象 | 典型用途 |
|---|---|---|
| `logprobs` | **输出**的每个 token | 生成置信度、备选词 |
| `prompt_logprobs` | **输入**的每个 token | 给已有文本打分（困惑度、选项打分、AI 检测） |

两者可同时用。`prompt_logprobs` 需服务端带 `--enable-prompt-tokens-details`。

---

## 二、logprobs 从哪来：Transformer 的哪一步产生它

用 **Qwen3-8B** 举例（标准 dense 结构，比 MLA + MoE 的模型更容易看清数据流）。
参数取自其 `config.json`：

```text
hidden_size=4096   num_hidden_layers=36   num_attention_heads=32
num_key_value_heads=8 (GQA)   head_dim=128   intermediate_size=12288
vocab_size=151936   tie_word_embeddings=false
```

### 2.1 一次前向的完整数据流

假设输入 3 个 token（`今天` / `天气` / `很好`），`seq_len=3`：

```text
输入 token ids                                        shape
   [6685, 15346, 13778]                               (3,)
        │
        ▼  ① embed_tokens：查表，把 id 变向量
   hidden_states                                      (3, 4096)
        │
        ├──────────── ② 36 层 DecoderLayer 循环 ────────────┐
        │   每层内部（vllm/model_executor/models/qwen3.py:171）│
        │                                                    │
        │     input_layernorm (RMSNorm)                      │
        │            ▼                                       │
        │     self_attn  ← Q:32头 / KV:8头 (GQA)，读写 KV cache│
        │            ▼    + residual                         │
        │     post_attention_layernorm                       │
        │            ▼                                       │
        │     mlp: gate/up (4096→12288) → SiLU → down       │
        │            ▼    + residual                         │
        │   ────────────── 重复 36 次 ──────────────          │
        └────────────────────────────────────────────────────┘
        │
        ▼  ③ 最终 norm
   hidden_states                                      (3, 4096)
        │            ↑ 到这里，每个位置都有一个 4096 维向量
        │              「位置 i 的向量」= 模型读完前 i+1 个 token 后的理解
        │
        ▼  ④ lm_head：线性投影到词表维度   ← logits 在这一步诞生
   logits                                             (3, 151936)
        │            qwen3.py:338  logits = self.logits_processor(self.lm_head, hidden_states)
        │
        ▼  ⑤ log_softmax：归一化成对数概率  ← logprobs 在这一步诞生
   logprobs                                           (3, 151936)
                     sampler.py:316  return logits.log_softmax(dim=-1, dtype=torch.float32)
```

**关键点：第 ④⑤ 步对「每个位置」都算了一遍。**
`logits` 的形状是 `(seq_len, vocab_size)` —— 不是只有最后一个位置。
这是 Transformer 的固有行为（训练时要靠它一次算出所有位置的 loss），不是为
logprobs 特意加的计算。

### 2.2 同一份 logprobs，两个参数各取一部分

```text
   logprobs  (3, 151936)   ← 每一行都是「读完前 i+1 个 token 后，对下一个词的预测」
   ┌─────────────────────────────────────────────────────────┐
   │ 行[0]  读完「今天」        → 预测下一个词的 151936 维分布  │ ← prompt_logprobs 取
   │ 行[1]  读完「今天天气」    → 预测下一个词的分布            │ ← prompt_logprobs 取
   │ 行[2]  读完「今天天气很好」→ 预测下一个词的分布            │ ← 用于真正生成 + logprobs 取
   └─────────────────────────────────────────────────────────┘
            行[0..n-2] 平时被直接丢弃 ──┐
                                        └→ prompt_logprobs 就是把它们留下来返回
            行[n-1] 用于采样下一个 token ─→ logprobs 返回这一步的候选
```

| | 取 logprobs 的哪些行 | 对齐关系 |
|---|---|---|
| `prompt_logprobs` | prefill 阶段的**每一行** | 行[i] 的预测 vs 真实的 token[i+1] |
| `logprobs` | decode 阶段每步的那**一行** | 该行的预测 vs 实际采样出的 token |

所以 `prompt_logprobs` 数组第 0 项恒为 `null` —— 位置[0] 是 `今天` 本身，
它前面没有任何行来预测它。

### 2.3 从 logprobs 到返回给你的 JSON

`sampler.py:319 gather_logprobs()` 做三件事（每一行 151936 个数不可能全返回）：

```python
topk_logprobs, topk_indices = torch.topk(logprobs, num_logprobs, dim=-1)   # 模型的 top-N 猜测
token_logprobs = logprobs.gather(-1, token_ids)                            # 真实 token 的那一个
token_ranks    = batched_count_greater_than(logprobs, token_logprobs)      # 真实 token 排第几
indices  = torch.cat((token_ids, topk_indices), dim=1)                     # 真实的拼在 topk 前面
logprobs = torch.cat((token_logprobs, topk_logprobs), dim=1)
```

这解释了三个现象：

1. **为什么 `prompt_logprobs=N` 返回 N+1 项** —— `cat` 把「真实 token」拼在
   「top-N 猜测」前面。所以 `=0` 意味着「只要真实 token，不要额外候选」。
2. **为什么会出现你 prompt 里没有的词** —— 那些是 `topk` 出来的模型猜测。
3. **`rank` 从哪来** —— `batched_count_greater_than`，见第四节。

### 2.4 计算成本

`prompt_logprobs` 的额外开销**不在前向**（`logits` 本来就要算），而在：

- `log_softmax` 要对 `(seq_len, 151936)` 全量做 —— prompt 越长越贵
- `topk` + `gather` + rank 统计
- GPU→CPU 传输那份 `(seq_len, N+1)` 张量

这也是崩溃 bug 的土壤：为了承接这份数据，vllm 会**按整个 prompt 长度预分配一块
CPU 张量**（`empty_cpu`），再随 chunked prefill 分片填充 —— 分配与填充分离，
就有了「分配了但没填」的窗口。详见第六节。

---

## 三、`prompt_logprobs` 为什么会返回"我没写的词"

这是最容易困惑的地方。**模型不是在猜你的 prompt，你的 prompt 一个字都没改。**

实测证明（同一 prompt，加不加 `prompt_logprobs`）：

```text
不加: text='，我们决定'
加了: text='，我们决定'      ← 输出完全相同
prompt_logprobs 字段: 不加=None / 加了=3 项
```

### 原理

Transformer 处理 prompt 时，会**在每个位置都顺带算出"下一个词的概率分布"** ——
这是自回归模型的固有机制（训练时要靠它一次算出所有位置的 loss）。所以哪怕只是读一段文本：

```text
读到「今天」        → 内部算出：下一个词分布（，12% / 给大家 2% / 天气 0.4% ...）  ← 平时丢弃
读到「今天天气」    → 内部算出：下一个词分布（晴朗 12% / 真好 11% / 很好 6% ...）  ← 平时丢弃
读到「今天天气很好」→ 内部算出：下一个词分布                                      ← 用于真正生成
```

前两个分布平时被**直接丢掉**（prompt 已给定，不需要模型决定）。
`prompt_logprobs` 唯一做的事就是**把这些本来要丢的中间结果读出来返回**。

所以准确说法是：

> 模型顺手记录了「如果让我来写，我会写什么」，然后告诉你「你实际写的那个词，在我的排序里是第几名」。

正因为模型的猜测和你的真实文本**不一样**，这个对比才有价值。如果模型的猜测总是和
你的文本完全一致（rank 全是 1），这个功能就毫无信息量了。

---

## 四、`rank` 是什么

源码定义（`vllm/v1/sample/ops/logprobs.py:27`）：

```python
return (x >= values).sum(-1)
```

**rank = 全词表里 logprob ≥ 该 token 的元素个数**。因为 `>=` 含自己，所以**最小值是 1**（不是 0）。
无并列时就等于概率排名。

实测验证（10 元素小词表）：`最大→rank=1`、`第2→2`、`第4→4`、`最小→10`。
并列会累加：三个 `-1.0` 并列时查 `-1.0` 返回 `rank=3`（词表大时并列极少，可忽略）。

### 完整实例：`prompt = "今天天气很好"`

**第一步：先看 tokenize 结果**（不是按字切的，这是理解 rank 的前提）：

```text
[0] id=6685   '今天'
[1] id=15346  '天气'
[2] id=13778  '很好'
```

所以 `prompt_logprobs` 数组长度 = 3，每项对应一个 token 位置。

**位置[0]** — 真实 token `'今天'`，前文为空

```text
→ null      首个 token 没有前文，算不出条件概率
```

**位置[1]** — 真实 token `'天气'`，前文只有 `'今天'`

```text
rank=1     '，'      logprob=-2.102     ← 模型的猜测（top-1），你的 prompt 里没有
rank=2     '给大家'   logprob=-3.793     ← 模型的猜测（top-2），你的 prompt 里没有
rank=31    '天气'    logprob=-5.491     ← 你 prompt 里真实的 token
```

**为什么 `天气` 排 31？** 模型只看到「今天」，它认为更可能接「，」「给大家」「早上」……
「今天天气」当然通顺，但没有「今天，」那么高频。所以真实 token 在 120832 个候选里
排第 31 位（前 0.026%，其实很正常）。

**位置[2]** — 真实 token `'很好'`，前文变成 `'今天天气'`

```text
rank=1     '晴朗'    logprob=-2.101
rank=2     '真好'    logprob=-2.233
rank=5     '很好'    logprob=-2.797     ← 真实 token，从 31 名跃升到 5 名
```

前文多了「天气」两个字，模型立刻知道该接天气类形容词。**上下文越多，模型预测越准。**

### 怎么读这份数据

| 你看到的 | 是什么 |
|---|---|
| `rank=1`、`rank=2` 那几项 | **模型的猜测**（top-N 候选），你 prompt 里通常没有 |
| `rank` 数值偏大的那一项 | **你 prompt 里真实的 token**，rank 表示"模型觉得它有多不可能" |

**rank 越大 = 模型越意外。** 这正是困惑度打分的原理。

### rank 的实际用途

1. **判断"模型是否意外"比 logprob 更直观** —— `-5.49` 是绝对值看不出相对位置，
   `rank=31` 立刻告诉你"前面还有 30 个更可能的"。
2. **AI 生成文本检测** —— 机器生成的文本 rank 普遍很低（模型总选自己最爱的词），
   人写的更散。这是 DetectGPT 类方法的基础。
3. **定位异常位置** —— 某位置 rank 突然飙到几万，往往是错别字、tokenize 边界问题或不通顺处。

---

## 五、请求会返回什么（实跑数据）

### 5.1 `logprobs: 2` —— 输出侧

```json
{"prompt":"今天天气很好","max_tokens":3,"temperature":0.0,"logprobs":2}
```

返回 `choices[0].logprobs`：

```json
{
  "text_offset":    [0, 1, 3],
  "tokens":         ["，", "我们", "决定"],
  "token_logprobs": [-0.684, -2.969, -1.115],
  "top_logprobs": [
    {"，": -0.684, ",": -1.610},
    {"我们": -2.969, "阳光": -2.983},
    {"决定": -1.115, "一起去": -2.464}
  ]
}
```

- `tokens` / `token_logprobs` —— 实际生成的 token 及其 logprob
- `top_logprobs[i]` —— 第 i 步的候选表（含实际采样的 + 另 `logprobs` 个）
- 第 2 步 `我们(-2.969)` vs `阳光(-2.983)` 几乎持平 → 模型在这里很犹豫
- `text_offset` —— 每个 token 在输出文本中的字符起始位置

### 5.2 `prompt_logprobs` 的取值含义

| 取值 | 每项返回 | 说明 |
|---|---|---|
| `null`（默认） | 字段为 `None` | 不启用 |
| `0` | **1 项** | 只要真实 token，不要额外候选 |
| `2` | **3 项** | 真实 token + top-2 候选 |
| `N` | **N+1 项** | 真实 token 必返回，无论它排第几 |

源码依据 `sampler.py:360`：`torch.cat((token_ids, topk_indices))` —— 真实 token 拼在 topk 前面。

**注意 `prompt_logprobs: 0` 不是"不要"**，而是"只要真实 token"。
现场崩溃的那个请求用的正是 `0`，很像是客户端默认值透传。

字段：key 是 token_id，`logprob` 是对数概率，`rank` 见第四节，`decoded_token` 是解码后的字符串。

### 5.3 典型用途

**困惑度 / 文本流畅度**

```text
PPL = exp(-mean(所有 prompt token 的 logprob))
```

越低越"像模型认知里的正常文本"。可用于机器生成检测、文本质量筛选。

**选项打分（无需生成）** —— 把每个选项分别拼进 prompt，比较各自 logprob 总和取最高。
比让模型输出 A/B/C/D 更稳定，常用于评测集。

**输出置信度** —— `logprobs` 里 top-1 与 top-2 差距很小时说明模型在猜，
可据此触发人工复核或改走更大模型。

---

## 六、PD 分离下的崩溃（本次修复）

**现象**：PD 分离的 decode 实例启动约 81 秒即崩溃退出，就绪探针在较长窗口后才
判定重启，一天内循环 7 次、累计 73 次 `OverflowError`。报错栈：

```text
OverflowError: out of range integral type conversion attempted
  at vllm/tokenizers/detokenizer_utils.py  convert_ids_list_to_tokens
  <- vllm/v1/engine/logprobs.py            _update_prompt_logprobs
  <- vllm/v1/engine/async_llm.py           output_handler   ← 进程内唯一的输出协程
```

关键在最后一层：`output_handler` 是整个进程**唯一**的输出处理协程，而它的
`try` 包在 `while True` 外面。任何一个请求在这里抛异常 → 协程退出 →
`propagate_error` 把异常投给所有在飞请求 → APIServer 退出、EngineCore 收
SIGTERM → **一个坏请求打挂整台机器**。

### 6.1 流程图：正常 vs 崩溃

```text
┌────────────────────── 单机（非 PD）：正常 ──────────────────────┐
│  请求(prompt_logprobs=0)                                        │
│        ▼                                                        │
│  _get_prompt_logprobs_dict()                                    │
│        ├─ empty_cpu(1399, 1)  ← torch.empty，未初始化脏内存       │
│        │                                                        │
│        ├─ 本地 prefill 分片进行：                                 │
│        │     num_computed_tokens = 0 → 512 → 1024 → 1400        │
│        │     num_logits > 0，每片都执行 copy_()                  │
│        │     ████████████████████████  张量被完整填满 ✅          │
│        │                                                        │
│        └─ 交付 → decode → 返回合法 logprobs                       │
└─────────────────────────────────────────────────────────────────┘

┌───────────── PD 分离 D 侧（kv_role=kv_consumer）：崩溃 ─────────────┐
│  P 侧完成 prefill，通过 FlexibleConnector 把 KV 推给 D              │
│        ▼                                                           │
│  D 侧 num_computed_tokens 被外部 KV 直接顶到 1400（prompt 末尾）      │
│        ▼                                                           │
│  _get_prompt_logprobs_dict()                                       │
│        ├─ empty_cpu(1399, 1)  ← 同样 torch.empty                   │
│        │                                                           │
│        ├─ start_idx     = 1400                                     │
│        │  start_tok     = 1401                                     │
│        │  num_remaining = 1400 - 1401 = -1        ← 负数！           │
│        │  num_tokens(1) > -1  → else 分支                           │
│        │       num_logits = -1                                     │
│        │       prompt_logprobs_dict[req] = tensors  ← ❶ 交付脏张量   │
│        │                                                           │
│        ├─ if num_logits <= 0: continue            ← ❷ 跳过所有 copy_ │
│        │  ░░░░░░░░░░░░░░░░░░░░░░░░  张量一个字节都没写 ❌            │
│        ▼                                                           │
│  output 侧: logprobs.py:147 _update_prompt_logprobs                │
│        │  token_ids.flatten().tolist()                             │
│        │  → 实测 1399 个值中 246 个越界、1153 个落在 [0,vocab) 内     │
│        ▼                                                           │
│  detokenizer_utils.py:100 convert_ids_list_to_tokens               │
│        │  tokenizer.decode([-1094407996])                          │
│        ▼                                                           │
│  ❌ OverflowError: out of range integral type conversion attempted  │
│        ▼                                                           │
│  异常抛在 AsyncLLM.output_handler —— 进程内唯一的输出协程            │
│  try 包在 while 外 → 协程退出 → propagate_error 毒化所有在飞请求     │
│        ▼                                                           │
│  APIServer Shutting down → EngineCore SIGTERM → 整个实例挂          │
│  存活仅 81 秒；探针要 60 分钟才判定重启 → 全天循环 7 次              │
└────────────────────────────────────────────────────────────────────┘
```

### 6.2 四个必要条件（缺一不可）

1. PD 分离，且被测实例是 D 侧（`kv_role=kv_consumer`）
2. 请求带 `prompt_logprobs`（非 None），服务端有 `--enable-prompt-tokens-details`
3. 请求经 P 侧转发（带 `kv_transfer_params.remote_prefill`），使 `num_computed_tokens`
   被外部 KV 顶到 prompt 末尾
4. `torch.empty` 拿到的内存含越界值 —— 概率性，实测命中率 **17.6% ~ 27.7%**，
   长 prompt / 高并发下几乎必然

> 单机部署无法复现：prefill 一定在本地执行，`copy_` 会把张量填满。

### 6.3 修复：只改一处（交付点后移）

`vllm/v1/worker/gpu_model_runner.py`，**+10 -1 行**：

```python
             else:
                 num_logits = num_remaining_tokens
                 completed_prefill_reqs.append(req_id)
-                prompt_logprobs_dict[req_id] = logprobs_tensors   # 删除：过早交付

             if num_logits <= 0:
                 # 本步没有任何 copy_ 发生，不能交付未初始化的张量
                 continue

+            # 本步会填 [start_idx, start_idx+num_logits)；只在末片真正写入后才发布
+            if req_id in completed_prefill_reqs:
+                prompt_logprobs_dict[req_id] = logprobs_tensors
```

`num_logits > 0` 的正常路径（本地 prefill、chunked prefill 末片）**行为完全不变**。

### 6.4 刻意不做的三件事

评审原则：**宁可让问题暴露，也不做防御性掩盖**（上线前有各种场景压测）。
以下三处曾经加过，评审后全部撤回：

| 撤回项 | 为什么是掩盖而非修复 |
|---|---|
| `empty_cpu` 改 `torch.zeros` | 脏内存变成合法 id `0` → decode 出一串 `'!'`，把可见崩溃变成静默错误结果；且每请求多一次全长 zeros 开销。上游用 `torch.empty` 是有意的，语义是"马上会被 `copy_` 填满" |
| detokenizer 加词表范围校验 | 越界 id 本身就是上游 bug 的信号，静默降级空串会掩盖它。**更关键的是它并不彻底**：实测一块脏张量的 1399 个值里，246 个越界被降级，另外 **1153 个落在 `[0, vocab)` 内，照样解成看着合法、实则编造的 token** —— 业务拿到格式完整、数值合理、全是假数据的 logprobs，比直接抛异常难查得多 |
| `output_handler` per-request 异常隔离 | "单请求毒化全局协程"是独立的架构问题，与本 bug 无关，应单独评估；其中的逐条重试还依赖 `process_outputs` 幂等，未经验证 |

**三种方案下业务实际收到什么**（实测数据）：

| 方案 | D 侧业务拿到的 `prompt_logprobs` |
|---|---|
| 无修复 | 崩溃，整台机器挂 + 60 分钟重启循环 |
| 只加范围校验 | 1399 项：246 空串 + **1153 个从随机内存解出的假 token** ← 最危险 |
| 零初始化 + 校验 | 1399 项全是 id `0` → 1399 个 `'!'` ← 静默错误 |
| **只改交付点（最终方案）** | **不交付** —— 业务明确知道没拿到，无假数据 |

关键：`num_logits <= 0` 时 `copy_` 执行 **0 次**，张量里**没有一格是真实 logprob**。
所以不存在"本该有 1153 个正确值被丢掉"—— 修复不是少给，是**不再谎报**。
