---
title: "MegaMoE：如何实现 warp 级通算融合"
summary: "把 MoE 一层的 dispatch、FC1、SwiGLU、FC2、combine 压进同一个 CUDA kernel，用 symmetric memory 把通信降到 warp 级。"
published: 2026-09-11
category:
  - 大模型推理
  - 推理优化
  - 算子、编译器与硬件
  - GEMM、Attention 与 MoE 算子
topics:
  - 大模型推理
  - MoE
  - CUDA
  - 通算融合
  - 性能优化
wide: true
zhihu: "https://zhuanlan.zhihu.com/p/2067944896175068638"
---
MegaMoE 是把 MoE 一整层的 5 个阶段(dispatch、FC1、SwiGLU、FC2、combine)全部压进一个 CUDA kernel 里跑完，实现 warp 级的通算融合。

## 1. 想解决什么问题

先说清楚 MoE 一层长什么样。假设 8 个 rank 组成 EP 并行, 每个 rank 上 32 个 expert, 每个 token 选 top-8。一次 forward 里,MoE 一层要做的事情按顺序是:

1. **Dispatch**:每个 rank 把自己 token 的 top-k 路由结果送到对应 expert 所在 rank
2. **L1 GEMM**:每个 expert 对收到的 token 做 `[M, hidden] × [2·I, hidden]^T`,输出 `[M, 2·I]`
3. **SwiGLU + 量化**:激活函数 + 乘 top-k 权重,输出 `[M, I]` 并量化成 FP8
4. **L2 GEMM**:再做 `[M, I] × [hidden, I]^T`,输出 `[M, hidden]`
5. **Combine**:每个 expert 把结果送回 token 原来的 rank,做 top-k reduce

```python
recv_x  = dispatch(x, topk_idx, topk_weights)      # EP all-to-all,把 token 送到 expert 所在 rank
l1_y    = grouped_gemm(recv_x, l1_weights)         # FC1: [M, hidden] × [E, 2·I, hidden] → [M, 2·I]
l1_y    = swiglu(l1_y) * topk_weights              # 激活 + 路由权重(→ FP8 或 BF16)
l2_y    = grouped_gemm(l1_y, l2_weights)           # FC2: [M, I] × [E, hidden, I] → [M, hidden]
y       = combine(l2_y)                            # EP all-to-all 反向 + top-k reduce
```

传统做法是把这 5 步跑成 5 个独立 CUDA kernel, 通信用 NCCL 或者 NVSHMEM, 中间数据全走 HBM 中转。看起来挺自然, 但有几个绕不开的问题, 如下图所示：

```text
stream A (comm):   [dispatch] . . . . . . . . [combine]
stream B (compute):          [L1 GEMM][SwiGLU][L2 GEMM]
```

![传统 5 kernel 方案下通信与计算的时间线](/images/megamoe/baseline-pipeline.jpg)

首先，**通信和计算重叠的问题**，即使我们实现通算重叠，但 stream 级的 overlap 太粗。

比如说，两条 stream 只能在”整个 kernel 级别”重叠,做不到”细粒度重叠”，dispatch 内部有”写 metadata → wait → pull payload”两个子阶段, GEMM 只能等 dispatch 整体完成才启动。其实，第一批 token 到达时,GEMM 完全可以开跑。

其次，**NCCL kernel 抢 SM 资源**，NCCL kernel 会占用一部分的 SM （8-16）, 这也会导致 GEMM 计算拿到的实际 SM 数会有折扣。

然后，**中间激活反复往返 HBM**，就是多个 kernel 的实现方式，天然会导致 kernel 间的数据传递是通过 HBM 的，白白浪费了多次的拷贝耗时。

最后，**SM idle 导致的耗时浪费和 kernel launch 的 overhead**。GEMM 最后一批 tile 只用几个 SM,其他 SM 已经空转等 kernel 退出。5 个 kernel 加起来这个”尾巴”叠了 5 次。

MegaMoE 就是奔着这 4 个问题去的, 方案是**把 5 步塞进同一个 CUDA kernel**。**通信不再是独立 kernel, 而是 kernel 内某些 warp 直接对远端 GPU 做 load/store**。

## 2. 融合成一个 kernel 到底难在哪

想把通信塞进 kernel, 第一个问题就是 CUDA kernel 根本无法访问或识别 “其他 rank”。

kernel 里只有指针,`st.global` 只能写”当前进程虚拟地址空间里的地址”。要让 warp 里的一条 store 指令能写到别的 GPU 上, 得先让”远端 GPU 的内存”以指针形式出现在当前进程的地址空间里。

我们先来详细分析下这件事：

首先，GPU kernel 里根本没有 rank 概念。它能做的操作只有:

```cpp
*ptr = value;        // 写某个地址
value = *ptr;        // 读某个地址
```

ptr 是当前进程 GPU 上的一个 64-bit 虚拟地址。kernel 不认识”rank 3 的 GPU”,“进程 5 的分配”这种东西 —— 这些是 host 侧的概念。

所以要让 kernel 内做跨 rank 访问, 唯一的路径就是让”远端 rank 的 GPU 内存”以一个普通指针的形式, 出现在当前进程的地址空间里。一旦拿到这样的指针,kernel 就能像访问本地内存一样访问它。这里的关键就是，这个指针必须在 kernel 启动之前就存在。因为kernel 内部没有能力”临时申请”一个能访问远端的指针。

那么，现在的问题是：host 侧怎么让 rankA 拿到 “能访问 rank B 的 GPU 内存”的指针？

CUDA 上有三种主流机制:

(1) CUDA IPC 底层物理机制

Rank B 分配一块 GPU 内存, 把它的 handle 序列化传给 rank A,rank A 通过 cudaIpcOpenMemHandle 把它 mmap 到自己的地址空间。之后 rank A 就拿到一个”指向 rank B 内存”的指针。

这其实就是 symmetric memory 底层用的机制(单节点内)。 跨节点用 NVSHMEM 或 GPUDirect RDMA 原理类似。

(2) NCCL / NVSHMEM 集合通信库封装

这些库封装了跨 GPU 通信, 但它们内部要么用 CUDA IPC(节点内) 要么用 RDMA(节点外), 本质是一样的。区别在于 NCCL 把这些机制藏在 collective API 后面 —— 你调用 ncclAllToAll(…), 它会启动自己的 kernel, 做完了退出。

不过其中关键的是，NCCL kernel 里做的事情你的 kernel 做不了, 因为 NCCL 的指针映射是 NCCL 内部准备好的, 不暴露给你的 kernel。但问题是是 kernel 就会占用 SM 等资源。

(3) torch.distributed._symmetric_memory

这就是 PyTorch 对 CUDA IPC 的一层薄包装, 目的就是把指针表还给用户，让自定义 Kernel 拥有调度主导权:

> “对称内存（Symmetric Memory）”的强约束——所有 Rank 分配了完全相同大小的显存块，并约定以各自的 local_base 为锚点，通过统一的虚拟地址跨度（Offset）建立映射关系。

```python
buffer = symm_mem.empty(num_bytes, dtype=torch.int8, device='cuda')
handle = symm_mem.rendezvous(buffer, group=group)
# handle.buffer_ptrs[rank_idx] 是 rank_idx 的 buffer 在当前进程地址空间的指针
```

rendezvous 做的事情就是:每个 rank 分配本地 buffer → 通过 CUDA IPC 交换 handle → 每个 rank 打开所有其他 rank 的 handle → 得到一张 buffer_ptrs[num_ranks] 表。

之后 kernel 里就能:

```text
// sym_buffer.map(local_ptr, dst_rank) 的实现, 基本就是:
//   offsets[dst_rank] + local_ptr
// 其中 offsets[i] = buffer_ptrs[i] - buffer_ptrs[my_rank]
layout/sym_buffer.cuh:24-31
```

这就是”symmetric memory 让融合可行”的物理机制。

所以综上所述，要想将通信融合入 kernel 同时还要更细粒度控制资源，这就是 symmetric memory 的用武之地。

PyTorch 2.5+ 的 `torch.distributed._symmetric_memory` 干的事情就是:每个 rank 分配一块**大小和布局完全一样**的本地 buffer,通过 CUDA IPC 交换基地址,让每个 rank 都拿到一张 `buffer_ptrs[num_ranks]` 表。之后 kernel 里就能用一条加法算出远端地址:

```text
remote_addr = local_addr + (buffer_ptrs[dst_rank] - buffer_ptrs[my_rank])
```

这个公式依赖两个前提:

1. **所有 rank 的 buffer 布局完全一致**, 同一字段在所有 rank 上 offset 相同, 否则单靠”基地址差”算不出正确远端地址
2. **offset 表在 kernel 启动前就准备好**, 不能在 kernel 里做 IPC 握手

第一条就是”symmetric”这个词的物理来源。第二条决定了 buffer 必须**一次性按 worst case 预分配**, 因为 kernel 启动后没机会重新协商布局。

当然很明显，这条约束的代价是显存浪费（对于多卡 rank 数据不均的情况）。

以默认配置(8 rank × 8192 tokens × top-8)为例, 理论上每个 local expert 最多可能收到所有 rank 的所有 token × 8 top-k 路由,pool 得预留 66048 × 8 = 52 万行。而路由均匀时每个 expert 平均只收 2000 行,32 个 expert 加起来 6.4 万行,**利用率约 12%**。仅 `l1_acts` 就占 3.6 GiB。加上 combine buffer(每个 top-k 槽位一份完整输出,`[num_topk, num_max_tokens, hidden] × BF16`)接近 1 GB,整块 symmetric buffer 能到 10 GB 量级。

这个浪费换来的是”kernel 内可以随时对远端做 load/store,粒度到 warp 级”。

对于 EP256~2048 规模、每层都是 MoE 的训练场景,HBM 宽裕、NVLink 是瓶颈,这个 tradeoff 划算。反之,小 EP 推理场景 HBM 紧张, 不一定合适。

目前，MegaMoE 已经把大部分 pool 改成了 ring buffer, 只有 combine buffer 和少量 metadata 表还是 worst case 布局, 浪费问题比早期版本好很多。

## 3. 性能收益到底从哪来

在讲实现细节之前, 先算一下融合到底能带来多少收益, 就是我们先来定义一下问题。

用源码里的默认例子代入：

```text
num_tokens/rank        = 8192
hidden = 7168, inter   = 3072
num_experts = 384, topk = 6, ranks = 8

每 rank 的 expert 输入行数 num_recv_tokens ≈ 49152

# 计算量(测试脚本里的公式)
routed_flops = 2 * num_recv_tokens * hidden * inter * 3 ≈ 6.5 TFLOPs / rank

# HBM 传输(权重 + 激活,FP8/FP4)
HBM_bytes    ≈ 1-2 GB / rank

# NVLink 传输(dispatch pull + combine write-back)
NVL_bytes    = num_recv_tokens * hidden * 3 ≈ 1.05 GB / rank
```

在 GB200 上(FP8 ~2500 TFLOPS, HBM ~4 TB/s, NVLink ~1.8 TB/s bi-dir):

| 阶段 | 理论下界 |
| --- | --- |
| 计算 | (6.5 TFLOPs / 2500 TFLOPS 峰值) = 2.6 ms |
| HBM | (1.5 GB / 4 TB/s) = 0.4 ms |
| NVLink | (1.05 GB / 900 GB/s 单向) = 1.2 ms |
| 完全串行(baseline 上限) | 4.2 ms |
| 完全 overlap(下界) | max = 2.6 ms |

理论 speedup 上限 ≈ 4.2 / 2.6 ≈ **1.6x**。 所以说使用 MegaMoE 理论上的收益最高 60% 。

但这里要注意一个重要前提:上面这个 1.6x 是**相对”完全串行”的 baseline**算的, 也就是把 dispatch、GEMM、combine 严格按顺序跑一遍的最坏情况。

真实生产系统里几乎没人这么做, 大家或多或少都上了 kernel 级的通算掩盖(双 stream + event 同步、CUDA Graph、或者 DeepEP 那种独立 comm kernel 配合 GEMM stream), 这些方案能吃掉相邻 kernel 之间大约 40-60% 的通信开销。

所以 MegaMoE 相对不同 baseline 的实际加速比大约是:

| baseline | 相对 speedup |
| --- | --- |
| 纯串行 5 kernel | ~1.6x |
| 双 stream 通算掩盖(DeepEP legacy) | 1.2 - 1.35x |
| CUDA Graph + 双 stream 深度调优 | 1.1 - 1.2x |

也就是说, 如果你已经是双 stream 掩盖过的方案, MegaMoE 的相对收益会缩到 20-35%。真正只属于 MegaMoE 独有的收益是”当前层内部 dispatch 和 GEMM 的 block 级重叠”、”L1→L2 消除中间 HBM 往返”、”SM 资源不被通信 kernel 抢占”这三条,加起来大概就是这 20-35%。

当然如果还能通过 PDL 提前启动后继 kernel，进一步压缩 launch latency 和 kernel 边界空泡，那么 MegaMoE 的相对增益还会继续缩小。

具体分解到 5 个收益源:

**第一, warp 级通信-计算重叠**。传统方案是 stream 级重叠,粒度是”整个 dispatch 完成”。MegaMoE 把粒度缩到”一个 192 行的 pool block 完成”:dispatch warp 拉完一个 block 就更新 `l1_full_count`,GEMM warp 一看到 counter 到位立即开跑。**从”整层等整层”到”块等块”**。

**第二, 中间 GMEM 往返消除**。L1 epilogue 直接把 SwiGLU 后的 FP8 结果写进 `l2_acts`,这块 GMEM 就是 L2 GEMM 的输入。BF16 中间激活和 FP8 中间激活的两次 GMEM 往返完全消失。

**第三, 持续 SM 占用(3-8%)**。传统方案每个 kernel 尾部都有 SM idle;MegaMoE 是 persistent kernel,一次 launch 占满所有 SM 跑到最后,尾巴 idle 只有一次。

**第四, kernel launch 消除**。5 次 launch → 1 次,省几十 μs。训练场景可以忽略,推理小 batch 才明显。

**第五, SM 资源竞争消除(2-5%)**。NCCL kernel 不再抢 SM,dispatch warp 就 4 个 warp,和 GEMM warp 在同一 SM 上并存。

所以，总的来说，MegaMoE 的收益核心是”把 stream 级 overlap 降到 warp 级 overlap”。symmetric memory 只是让这件事在物理上成为可能。

## 4. MegaMoE 的整体架构

![MegaMoE 的整体架构](/images/megamoe/architecture.jpg)

### 4.1 5 种不同 warp 和 buffer

kernel 里同时驻留 5 种不同职责的 warp, 每种 warp 做完自己的活就等着下一个任务:

| warp 类型 | 数量(每 CTA) | 干什么 |
| --- | --- | --- |
| Dispatch | 4 | 路由统计、写远端 metadata、从远端 pull payload |
| Token loader | 1 | 把 activation 从 GMEM 搬到 shared memory |
| Weight loader | 1 | 把 weight 从 GMEM 搬到 shared memory |
| MMA issue | 1 | 发 UMMA 指令 |
| Epilogue | 8 | SwiGLU、量化、远端 push combine、本地 reduce |

再加上 1 个 warp 跑 scheduler, 一个 CTA 一共 16 个 warp, 512 个线程。整个 grid 起 148 个 CTA(SM100 上一个 CTA 一个 SM),每 2 个 CTA 组成一个 cluster。

数据面用到的 buffer 分五类:

- **Workspace**:控制面 counter 和 metadata 表(几个 MB)
- **Input buffers**:用户 tensor 的 view(x、x_sf、topk_idx、topk_weights)
- **Routed L1/L2 acts**:dispatch 到 L1 输入、L1 输出到 L2 输入的中间 activation,用 ring buffer 节省显存
- **Shared L1/L2 acts**:shared expert 的中间 activation
- **Combine buffer**:combine 阶段跨 rank 写回的 partial result

这些 buffer 拼成一整块 symmetric buffer,布局在 kernel 启动前静态规划好,启动后所有 rank 看到完全一致的字段 offset。

### 4.2 5 级流水线阶段映射

| 流水线级数 | 逻辑阶段名 | 对应 Warp 类型 | 核心动作（数据变换） |
| --- | --- | --- | --- |
| Stage 1 | 路由预取与元数据准备 | Dispatch | 查路由表、更新统计 counter；从远端（其他 rank）pull payload，将待计算的输入张量指针/元数据摆放到 Workspace 中。 |
| Stage 2 | 数据搬运（显存→共享内存） | Token loader + Weight loader | 根据 Stage 1 给出的地址，将 Activation 和 Weight 从 GMEM 批量搬运到 Shared Memory（双缓冲机制准备就绪）。 |
| Stage 3 | 矩阵乘计算发射 | MMA issue | 在 Shared Memory 数据 ready 后，发射 UMMA（异步矩阵乘）指令，启动 Tensor Core 计算。 |
| Stage 4 | 激活后处理 | Epilogue（前半部分） | 对 MMA 结果执行 SwiGLU 非线性激活，并进行量化（Quantization），将 FP32/FP16 结果转为低比特。 |
| Stage 5 | 归约与结果写出 | Epilogue（后半部分） + Dispatch | 在 Rank 内做 本地 Reduce；跨 Rank 时由 Dispatch warp 将量化后的 partial result 远端 Push Combine（写回远端显存），并清理/更新远端 metadata。 |

**第一，三级同步精确卡位**：

- **跨 Rank Barrier 第 1 次（Pull 之前）**：卡在 **Stage 1 开始前**，确保所有 Rank 的路由元数据全局可见。
- **跨 Rank Barrier 第 2 次（Combine reduce 之前）**：卡在 **Stage 5 开始前**，确保所有 Rank 的 MMA 计算和量化全部完成，才能安全做跨 GPU 原子加。
- **跨 Rank Barrier 第 3 次（Workspace 清理后）**：卡在 **Stage 5 结束后**，释放当前迭代占用的远端 buffer。

**第二，静态规划的 Symmetric Buffer**：所有 Rank 看到完全一致的字段 offset，意味着 **Stage 1 的 Dispatch warp 可以无锁地直接通过指针偏移访问 Stage 5 需要写的远端地址**，流水线各级之间不存在地址解析延迟，完全硬编码对齐。

**第三，Ring Buffer 的存在**：Routed L1/L2 acts 使用 ring buffer，说明 **Stage 2（Load）与 Stage 3（MMA）之间存在生产者-消费者重叠**。当 MMA warp 正在计算第 N 批数据时，Loader warp 已经在搬运第 N+1 批数据到 Ring Buffer 的另一块区域——这正是流水线持续充满（Fully Pipelined）的典型标志。

## 5. 详细流程: 从 dispatch 到 combine

下面按 kernel 内的时间顺序过一遍。

### 5.1 Dispatch 第一阶段: 写远端 metadata

![Dispatch 第一阶段：写远端 metadata](/images/megamoe/dispatch-metadata.jpg)

Dispatch 第一阶段写远端的 metadata，相当于把 All-to-All 通信协议里“握手 + 路由寻址”的部分拆出来，用手动原子操作（Atomic）硬编码在了 CUDA 内核里，目的是为了隐藏通信延迟并优化大块连续内存写入。

这一步具体干的事情是：

1.先发极小的元数据（仅存“哪个 Top-K 槽位要去哪个专家”）。

2.利用原子操作在分布式环境下无锁地划分好每个 SM 的写入区间。

3.等所有 SM 的元数据都落地且全局可见后，下一步才会根据这些元数据，真正去搬运 token 的隐藏层向量。

下面我们来看看其是如何实现的：

在开始前，我们先来了解下 TMEM：

> SM100 引入的 Tensor Memory(TMEM)是一块独立于 shared memory 和寄存器的新存储,专门给 UMMA(tensor core)当 accumulator 用。它的组织是：

1. 32 个 datapath(列),每个 datapath 对应 warp 里一个 lane。这个 32 是硬件常量, 不能改，这迫使数据必须提前排布好。
2. 每列 512 行,每行 32 bit(所以每列 2 KB,一个 warp 一次 TMEM 分配总共 64 KB)。
3. 一个 lane 只能访问自己那一列。lane 5 想读 lane 6 的数据?做不到。硬件上物理上就没这个通路。 这一条非常反直觉,值得停一下。Shared memory 是任何 lane 都能访问任何地址的; TMEM 不是,它是”每 lane 一列”的分片存储。

**第一遍遍历：不传输仅计数**

首先，每 4 个 warp 负责一个 SM（流多处理器），全网共 592 个这样的工作组，并行处理输入 token。

每个 warp 每次处理 5 个 token。每个 token 会路由到 Top-6 个专家（MoE 特性），所以一次处理 5×6=30 条路由记录，正好塞进 32 个线程（lane）里。

第一遍遍历数每个 SM 内每个 expert 收到多少条路由, 写在 shared memory 里做 SM 内局部统计。相当于仅统计当前 SM 可以看见的局部结果。

**第二部遍历前：原子计数器**

第二遍遍历前, 先用一次 64-bit 原子加把局部统计聚合到 GMEM 全局 count, 顺便**用返回的旧值当作”当前 SM 在这个 expert 上的起始 slot 偏移”**。这一步一次原子加干三件事: 全局聚合、拿 SM 起始 offset、高 32 位累加 1 记录 SM 到达数。 相当于 每个 SM 抢着去摸一把计数器，摸到的号码就是它往货架上放货的起始位置，绝对不会冲突。

**第二部遍历：写数据**

这次遍历，每条路由项要算出目标 rank（即该专家落在哪个 GPU 上）。

先在当前 SM 的 Shared Memory 里原子加 1，拿到槽位编号（slot），加上刚才抢到的全局起始偏移，就得到全局唯一的写入地址。

此时写入不是写 token_id，而是写 token_id × topk + topk_slot。原因：接收方（目标 rank）在后续做 Combine（加权合并）时，必须知道这条路由是来自发送方的第几个 Top-K 候选项，否则无法把计算结果填回原始 token 的对应位置。

写完 metadata 之后过一次 grid_sync + nvlink_barrier, 确保所有 rank 的 metadata 都全局可见, 才能进入下一阶段。

### 5.2 Dispatch 第二阶段: 从远端 pull payload

![Dispatch 第二阶段：从远端 pull payload](/images/megamoe/dispatch-pull.jpg)

现在目标 rank 已经知道每个 local expert 收到多少条路由、分别从哪些源 rank 来。

dispatch warp 从”我是发送方”的视角切换成”我是接收方”, 按目标 rank 的视角把 payload(token 数据、SF、topk_weight)从远端拉回来填进本地 pool。

这块为什么要设计为从远端拉取而不是 push: 主要原因是 NVLink push 时，必须等待接收方的显存控制器完成物理写入并返回 ACK，这个事务才算完成。而 Pull 模式让“数据传输完成”和“数据落盘”变成了两个独立事件，中间隔着接收方的 L2 缓存，可以流水起来。

这一阶段最烧脑的两件事:

**第一件事, 反解 (pool_slot → src_rank, token_id_in_rank)**。

pool 里每一行属于哪个 src_rank 的哪条路由, 不是简单按 rank 顺序拼接的,而是**按交错 round-robin 排列**: 每一轮里各 rank 各出一条,某个 rank 耗尽就退出下一轮。这样安排的目的是让 pull 阶段的远端 TMA load 分散到不同 src_rank,把 NVLink 带宽摊开使用,不是集中打一个链路。

反解算法用”iterative min-peeling”:

```text
Round 1: active={0,1,2}, min=2, length=2 → [r0 r1 r2 r0 r1 r2]
Round 2: r1 耗尽,active={0,2}, min=2, length=2 → [r0 r2 r0 r2]
Round 3: r2 耗尽,active={0}, length=1 → [r0]
```

> 为什么不用“按 Rank 顺序拼接”？（比如 [r0 r0 r0, r1 r1, r2]） 如果按顺序拼，接收方在 Pull 数据时，会先把 Rank 0 的数据全拉完，再拉 Rank 1，再拉 Rank 2

给定一个 pool_slot, 一层一层剥掉不含它的完整 round,直到落在某一 round 里,再算出它在这一 round 里对应哪个 rank、这个 rank 的第几条。整个过程用 warp reduce 并行完成(`__reduce_add_sync`、`__reduce_min_sync`),不用 loop。

**第二件事, TMA overlap**。

让计算和通信完全重叠。

- MA Load（远端读）：从其他 Rank 的 HBM（显存）读 3.5KB 数据，延迟 300-500 纳秒（这在 GPU 里已经是非常慢的延迟了）。
- TMA Store（本地写）：把读回来的数据写进本地的 Pool（显存）。
- 如果不做 Overlap：发 Load → 傻等 500ns → 收到数据 → 发 Store（又等几十 ns）。这 500ns 里，Warp 的算术单元全在空转，浪费大量算力。

整体的“流水线”流程为：

1. 发起 Chunk 0 的 Load（不等它回来）。
2. 马上发起 Chunk 1 的 Load。
3. 回来处理 Chunk 0 的 Store。
4. 处理 Chunk 1 的 Store。

故意推迟最后一个 Chunk 的 Store：

因为 SF（缩放因子，224 字节）和 Weight（权重，4 字节）数据量极小，它们不走 TMA，而是由 Warp 内 32 个 Lane 每人搬一个 uint32 直接写入（顺便做 4×32 转置，方便后续 GEMM 直接读取）。 这些小事本来可以穿插在大数据搬运的空隙里做。如果立即把最后一块大数据的 Store 做完，Warp 可能会进入空闲状态，而此时 SF 还没搬完。所以故意让最后一个 TMA Store 晚点执行，利用这“垃圾时间”让 Lane 们把 SF 和 Weight 搬完，最后再统一 Store 收尾。这叫“用访存延迟掩盖控制流开销”。

### 5.3 GEMM 主体: 三级 K 维展开

![GEMM 主体的三级 K 维展开](/images/megamoe/gemm-pipeline.jpg)

现在 pool 里有数据了, GEMM warp 开始接手。三种 warp 各司其职，目标是实现双 CTA 协作 + 三层循环分块 + 双缓冲 TMEM”的 GEMM 流水线:

- **Token loader**: 从显存（Pool）把输入矩阵 A 和缩放因子 SFA 搬进 Shared Memory，必须等 l1_full_count 信号
- **Weight loader**: 从显存把权重矩阵 B 和缩放因子 SFB 搬进 Shared Memory，不需要等
- **MMA issue**: 把 Shared Memory 里的 SF 搬到 TMEM，然后发 UMMA 指令让 Tensor Core 做矩阵乘，必须等两个 loader 都搬完

三种 warp 通过 mbarrier 组成生产者消费者流水线。`full_barriers[stage_idx]` 的 init 值是 4(2 CTA × 2 loader),表示需要 4 个生产者都 arrive 才让 MMA issue 开始; `empty_barriers[stage_idx]` init 值是 1,表示 MMA issue 一个 arrive 就还给 loader 复用。

2-CTA cluster 沿 M 维切半:leader CTA 负责前半行,non-leader 负责后半行。A tile 走 multicast(TMA 硬件层面广播给两 CTA),B tile 每 CTA 各自加载不同的 n_block。

K 维展开分三层:

- `num_k_blocks`:整个 tile 的 K 分成几块,每块对应一次 TMA copy(pipeline stage 消费一次)
- `BLOCK_K / UMMA_BLOCK_K`:每次 UMMA_BLOCK_K=128 的 K 段对应一次 UTCCP SF 搬运
- `UMMA_BLOCK_K / UMMA_K`:每条 UMMA 指令覆盖 UMMA_K=32 的 K 单位，Tensor Core 指令粒度

三层循环的粒度从粗到细, 恰好匹配 TMA copy 粒度、UTCCP 搬运粒度、UMMA 指令粒度这三种硬件单位。

MMA issue warp 只在 leader CTA 上跑,因为硬件规定 2-CTA UMMA 只让 leader 发指令。scheduler mainloop 也只在 leader CTA 上跑,通过 cluster barrier 广播 task 给整个 cluster。

TMEM accumulator 也做双缓冲： (`kNumEpilogueStages = 2`) 表示 TMEM（Tensor Memory）有两个 Stage 做乒乓缓冲。 MMA 在 stage 0 算的时候 epilogue 可以从 stage 1 读回上一个 tile 的结果, 反之亦然。

### 5.4 L1 Epilogue: SwiGLU + FP8 量化 + 写 l2_acts

```text
[TMEM Accumulator]
    │ (UMMA 计算结果：gate/up 交错排列)
    ▼
SM100_TMEM_LOAD (读到寄存器)
    │ (每个 Lane 拿到 gate0, gate1, up0, up1)
    ▼
寄存器内 SwiGLU (无需 warp shuffle)
    │ (out0 = silu(gate0)*up0*weight, out1 = ...)
    ▼
FP8 量化
    ├── 8 个 Lane reduce 出 amax
    ├── 邻近 2 个 Warp 交换 amax
    └── 算出 scale，cast 到 FP8
    ▼
stmatrix.trans 写入 Shared Memory (Swizzle 布局)
    │ (无 Bank Conflict)
    ▼
TMA Store → GMEM (`l2_acts`)
    │ (同一个物理地址，被两个 TMA Descriptor 共享)
    ▼
更新 `l2_full_count` → 唤醒 L2 Loader
更新 `l1_empty_count` → 释放 Ring Slot
```

**第一步，从 TMEM Load 到寄存器。**

这条指令一次 Load 让 Warp 里 32 个 Lane 各拿到 4 个 uint32，覆盖 8 行 token × 一段 N 通道。

> 为什么是 8 行？ 因为 UMMA 指令的 M 维输出是 8（这是 Tensor Core 的硬件约束），一个 Warp 一次算 8 个 token 的一小段通道。

Load 完成后：每个 Lane 的 4 个寄存器里装的是(gate0, gate1, up0, up1)——正好是同一个 token 的相邻通道的 Gate 和 Up 值。

**这里有个关键细节**:因为 UMMA 用了 swap A/B,TMEM 里连续 4 个寄存器对应 (gate, gate, up, up) 而不是普通顺序。这个约束追根溯源,来自 Python 侧对权重做的 `_interleave_weights(gran=8)`,gate 和 up 沿 N 维按 8 通道交错。硬件 fragment 分布配上 8 通道交错刚好让 gate/up 落到同一 lane 的 (0,1) 和 (2,3) 位置,方便就地做 SwiGLU 而不用做 warp shuffle。

**第二步，寄存器内做 SwiGLU。**

SwiGLU 公式 `SiLU(gate) * up * topk_weight` 在寄存器里算完之后, 还要做 amax 用于 FP8 量化。

不需要 warp shuffle（跨线程通信），因为 Gate 和 Up 已经在同一个 Lane 的相邻寄存器里了。这是_interleave_weights(gran=8)带来的布局红利。

```cpp
float scale = 240.0f / amax;  // 把最大值缩放到 FP8 的最大表示范围
uint8_t fp8_val = fp32_to_fp8(activation * scale);  // 转成 E4M3 格式
```

拿到最终 amax 算出 UE8M0 scale, 把 activation 缩放到 E4M3 表示范围,cast 成 FP8,用 `stmatrix.trans` 一条指令写进 shared memory 的 swizzle 布局, 避免 Bank Conflict。

**第三步，TMA Store：零拷贝跨阶段通信。**

最后用 TMA store 把 shared memory 里的 FP8 数据搬到 GMEM 的 `l2_acts`。**这块 GMEM 就是 L2 GEMM 的输入 buffer**,两个不同的 TMA descriptor(`tensor_map_l1_output` 写视角,`tensor_map_l2_acts` 读视角)描述同一块内存。中间那 GB 级别的 HBM 往返就这样消失了。

写完之后更新两个 counter:`l2_full_count` 通知 L2 loader “这一块可以开算”,`l1_empty_count` 释放 ring slot 让 dispatch pull 可以覆盖。

### 5.5 L2 Epilogue:远端 push 到 combine buffer

L2 GEMM 结束后没有 SwiGLU 也没有量化, 直接把 BF16 accumulator 送回 token 原来所在的 rank。

每一行 pool token 都带着一个 12 字节的 `token_src_metadata`,记录它来自哪个 (src_rank, src_token_id, src_topk_slot)。这份 metadata 是 dispatch pull 阶段写下的,存在 workspace 里贯穿整个 kernel。

L2 epilogue 从 TMEM 读回 accumulator,cast 成 BF16,用 stmatrix 写进 shared memory 的 swizzle 布局。然后**每 lane 读一个 float4 = 16 字节,用一条 `st.global` 通过 `sym_buffer.map` 写到远端**:

```text
combine_token_buffer[dst_topk_idx][dst_token_id][hidden_chunk] on dst_rank
```

一个 warp 里 16 lane 一次 write 覆盖完整 BLOCK_N=128 个 BF16 channel。**通信就是一条普通 store,没有 fence,没有 NCCL 调用,和写本地 GMEM 从代码形态上无差别**。这是”kernel 内融合”最直接的体现。

combine buffer 按 top-k 槽位分开存储,形状 `[num_topk, num_max_tokens, hidden]`。每个 expert 只写自己对应的那个 top-k 槽位,不同 expert 写不同槽位,天然无冲突。这就避免了跨 rank 原子加(否则 hidden 维几千次原子加,NVLink 承受不住)。代价是每个 rank 得预留 `num_topk × num_max_tokens × hidden × 2` 的 BF16 buffer,接近 1 GB。

Shared expert 走 combine buffer 的第 (topk+1) 个槽位。

写完之后所有 rank 过一次 `nvlink_barrier`,保证所有 rank 都完成 push,才能进入下一步。

### 5.6 Combine reduce + Dispatch cleanup(并行)

Kernel 的最后一步, 两组 warp 各干各的活并行推进, 分别对应两件事：

- Combine reduce
- Dispatch cleanup

**Combine reduce**(epilogue warps 做): 每个 warp 负责一个原始 token。先读 `topk_idx` 判断这个 token 命中了哪几个槽位, 得到一个 warp mask; 然后按 hidden chunk 切分(hidden=7168 装不下寄存器,切成 2 个 chunk 处理); 每个 chunk 内用 `__ffs(mask)-1` 迭代弹出待处理槽位,用双缓冲 TMA load 把 combine buffer 里的 partial 加载到 shared memory, 累加到寄存器里的 float2(用 fp32 累加避免 BF16 精度损失),最后 cast 成 BF16 用 TMA store 写到最终输出 `y`。

**Dispatch cleanup**(dispatch warps 做): 把 workspace 里各种 counter 清零, 为下次 kernel 调用做准备。SM 0 负责全局 counter(`expert_send_count`、task counters、shared L2 counter), 其他 SM 分摊每个 expert 的 per-rank counter 和 ring counter。**清理粒度精确到”这次 kernel 实际用过的位置”**,不清整个 workspace。

cleanup 期间顺便更新 `cumulative_local_expert_recv_stats`(用户提供的可选 tensor,跨多次 kernel 累积每个 expert 的接收 token 数,用于监控 expert 均衡度)。这是零成本 side effect,反正 cleanup 要读 recv_count,顺手加上去。

Cleanup 和 combine 并行进行,访问的数据完全不冲突。整个 kernel 尾部时间打平到 `max(cleanup, combine)`,不是相加。

Cleanup 结束后 dispatch warps 过最后一次 `nvlink_barrier`, 等所有 rank 都清完 workspace 再退出。Epilogue warps 完成 combine 后直接退出, 不用等这个 barrier, 因为 combine 只写用户输出, 不影响下次 kernel。

两组 warps 独立完成 shutdown。

## 6. 三个值得单独讲的设计

### 6.1 布局规划和物理分配解耦

MegaMoE 里 buffer 布局是一个”关系描述”, 不是”内存分配”。

```cpp
const auto mega_buffer = layout::MegaMoEBuffer(
    nullptr, hidden, intermediate_hidden,
    num_ranks, num_experts, num_max_tokens_per_rank,
    num_topk, num_ring_tokens, num_sf_ring_tokens, with_sf,
    num_shared_experts
);
```

Kernel 启动前 host 端调用 `MegaMoEBuffer` 构造函数, 注意第一个参数是 nullptr。构造函数会串接调用各段 buffer 的 Buffer(layout, ranks, tokens, base=前段.get_end_ptr()),此时所有 base 都从 nullptr 出发累加,get_end_ptr() 返回的其实是累积字节数当作指针来看。

```cpp
CUTLASS_HOST_DEVICE int64_t get_num_bytes() const {
    return static_cast<uint8_t*>(combine_token_buffer.get_end_ptr())
           - static_cast<uint8_t*>(workspace.base);
}
```

拿到总字节数之后,Python 层用 `symm_mem.empty(num_bytes)` 分配真实内存,再调 `slice_input_buffers(buffer)` 把 offset 关系绑定到真实地址上。这个 lambda 只是把 `torch::from_blob(buffer.data_ptr() + offset, shape, dtype)` 套一遍。

这种解耦的好处是 **同一份布局关系可以复用到不同的 raw buffer**:每次 kernel 调用不用重新算布局,也可以跨多次 forward 共用同一块 symmetric buffer(实际上就是这么做的)。

### 6.2 双 signal 双 sign 的 self-resetting barrier

`grid_sync` 和 `nvlink_barrier` 都用了一个巧妙的技巧: barrier 不需要在下一轮之前显式清零。

`grid_sync` 用单个 uint32 counter,每次 barrier 完成后翻一次最高位:非 SM 0 的 CTA 各加 1,SM 0 的 CTA 加 `0x80000000 - (kNumSMs - 1)`,总增量固定 = `0x80000000`。等待时观察 `(new_value ^ old_value) & 0x80000000` 是否翻转,翻转就说明本轮 barrier 到齐。counter 永远单调递增,不用重置。

`nvlink_barrier` 用两个 int signal + 一个状态 counter,状态 4 位一轮:signal[0]+1、signal[1]+1、signal[0]-1、signal[1]-1。

加法轮结束 signal 从 0 到 kNumRanks,减法轮结束 signal 从 kNumRanks 回到 0。两个 signal 轮流用,永远在 0 和 kNumRanks 之间震荡,不用清零。

这个思路在 CPU 侧 barrier 实现(比如 pthread barrier、Java Phaser)里也常见,叫 “double buffering barrier”。放到 GPU 上省去了 reset kernel 的开销,是很划算的优化。

### 6.3 warp 分工到 kernel 最后一刻

传统 CUDA kernel 里所有 warp 做同一件事, warp specialization 是 SM90 之后才流行起来的模式。MegaMoE 是我见过 warp 分工最细的公开 kernel:5 类 warp × persistent kernel × task queue 的组合,让每一个 warp 全程都有活干。

最典型的例子是 kernel 尾部:combine reduce 和 dispatch cleanup 完全并行,两组 warps 各自跑各自的收尾逻辑,共享 shared memory 但访问的 GMEM 完全不冲突。整个 kernel 尾部时间打平到 `max(combine, cleanup)`。

同样的思路贯穿全 kernel:dispatch pull 期间 GEMM 已经在跑,L1 epilogue push 到 `l2_acts` 期间 L2 loader 已经在 poll `l2_full_count`,scheduler 和 GEMM warp 用双 stage mbarrier 队列解耦。**没有一个 warp 空转**。

## 7. 什么场景值得走这条路

MegaMoE 不是万能方案,它的性能收益强烈依赖于场景。

**适合的场景**:

- 大规模 EP 训练(EP256 以上)。这时候 NVLink 是瓶颈,warp 级 overlap 收益最大
- HBM 宽裕的机器(GB200 有 192 GB,能容忍 10 GB 的 symmetric buffer 一次性预留)
- 每一层都是 MoE(比如 DeepSeek-V3)。融合收益按层数累加
- 追求 bit-exact 精度的场景。MegaMoE 相对 baseline 是 bit-exact 等价(单独 GEMM + SwiGLU 也能拿到同样结果),不是近似融合

**不适合的场景**:

- 小 EP 推理(EP8 以下)。通信量小,融合空间不大
- HBM 紧张的 fine-tune 场景。10 GB 预留吃不消
- 非 MoE 模型或者稀疏 MoE(topk 很小、路由极度不均)
- Hopper 或更早架构。MegaMoE 大量依赖 SM100 独有指令(TMEM、UMMA、UTCCP、stmatrix.trans),没法迁移

一句话:MegaMoE 是拿”worst-case HBM 预留 + 高度硬件绑定”换”kernel 内通算融合”。这个 tradeoff 只有在 HBM 宽裕、通信是瓶颈、SM100+、模型全 MoE 时才划算。放到别的场景,大概率还是老老实实用 DeepEP + 独立 GEMM 更合适。

## 8. 总结

读完这份代码最大的感受是, MegaMoE 里每一个设计决定都不是”发明”, 而是”约束推导”。

想在 kernel 内做跨 rank 融合,推出必须用 symmetric memory;symmetric memory 要 O(1) 远端寻址, 推出所有 rank 布局必须对称; 布局对称, 推出必须 worst-case 预分配; worst-case 预分配,推出显存浪费不可避免, 只能用 ring buffer 压一部分数据面。

这些推导每一步都没太多的选择余地。看到最后你会发现, MegaMoE 长成现在这个样子, 基本是被三个物理约束：

1. 通信是 kernel 内的普通指令
2. 远端寻址延迟和本地一致
3. buffer 提前建立。

这也是建议每个做 GPU 系统的人都去读一遍这份代码的原因:

它示范了”当所有可优化的地方都做了极致优化”之后, 一个 kernel 能长成什么样。

SwiGLU 融合、gate/up 8 通道交错、UTCCP 4×32 转置、amax 跨非连续 lane group reduce、iterative min-peeling 反解、双 signal barrier、warp 分工到 kernel 最后一刻, 每一个细节单独拿出来都值得写一篇长文，有非常多的细节，多看看总能发现些新东西。
