---
title: "大模型推理优化全景图"
summary: "从性能指标出发，梳理模型压缩、KV Cache、解码、调度、分布式并行、算子和生产部署之间的关系。"
published: 2026-08-14
category:
  - 大模型推理
  - 推理优化
  - 总览
topics:
  - 大模型推理
  - 性能优化
  - 知识地图
---

大模型推理优化不是单一技巧，而是模型、算法、运行时、分布式系统和硬件共同作用的结果。不同技术解决的瓶颈并不相同，开始优化前应先判断问题发生在 **Prefill** 还是 **Decode** 阶段。

Prefill 一次处理输入上下文，通常更受计算能力影响；Decode 自回归地产生 Token，需要反复读取模型权重和访问 KV Cache，通常更受显存带宽与内存容量影响。

## 推理基础与性能评测

性能指标需要分开观察：

- **TTFT**：请求到达后生成首个 Token 的时间，主要包含排队和 Prefill；
- **TPOT / ITL**：首个 Token 之后，相邻输出 Token 的平均间隔；
- **端到端延迟**：整个请求从进入系统到生成结束的时间；
- **Throughput / QPS**：单位时间内处理的 Token 或请求数量；
- **Goodput**：满足 TTFT、TPOT 等服务目标的有效吞吐量；
- **P50 / P95 / P99**：用于观察普通请求和尾部请求的延迟差异。

只有带着真实的输入长度、输出长度、并发量和 SLO 做基准测试，优化结果才具有参考价值。

## 模型压缩

模型压缩主要降低权重体积、显存占用和内存访问量。

- **量化**：FP8、FP4、INT8、INT4，以及 GPTQ、AWQ、SmoothQuant；
- **量化对象**：权重、激活和 KV Cache 可以采用不同精度；
- **剪枝与稀疏化**：减少参数或有效计算量，但需要硬件和算子支持；
- **知识蒸馏**：用较小模型学习大模型能力，换取更低部署成本。

低比特不必然带来同比例加速。实际收益取决于目标硬件是否存在对应的高效计算内核，以及精度损失是否可以接受。

## Attention 与 KV Cache

KV Cache 避免 Decode 时重复计算历史 Token，但会随上下文长度、层数和并发量持续增长。

- **MQA、GQA、MLA**：减少需要保存的 Key 和 Value；
- **FlashAttention、FlashInfer**：减少 Attention 的显存读写和中间张量；
- **PagedAttention**：按块管理 KV Cache，降低预分配和内存碎片；
- **Prefix Cache、RadixAttention**：复用相同系统提示、会话或文档前缀；
- **量化与压缩**：以更低精度保存 KV；
- **淘汰与 Offload**：在 GPU、CPU 或外部存储之间管理缓存；
- **滑动窗口与稀疏 Attention**：控制长上下文需要保留和访问的范围。

## 解码加速

普通自回归解码每次前向传播只确认一个 Token，串行依赖限制了生成速度。

- **Speculative Decoding**：小模型先提出多个候选 Token，大模型一次验证；
- **EAGLE、MTP、Medusa、DFlash**：使用特征预测或多 Token 预测构造候选；
- **N-Gram / Suffix Matching**：从已有文本模式中直接提出候选；
- **并行解码**：尝试同时预测或验证多个位置；
- **结构化解码**：使用 FSM 等方式约束 JSON、语法或工具调用输出。

推测解码更适合低批量、候选接受率较高的工作负载；草稿生成成本过高时，额外计算可能抵消收益。

## 批处理与请求调度

服务端需要同时处理长度不同、到达时间不同的请求。

- **Continuous / In-flight Batching**：在每次迭代后加入新请求、移除已完成请求；
- **Chunked Prefill**：切分长输入，避免一个 Prefill 长时间阻塞其他请求；
- **优先级与抢占**：根据 SLO、请求类型和剩余工作量安排执行顺序；
- **Cache-aware Scheduling**：优先安排能够命中 Prefix Cache 的请求；
- **Multi-LoRA Batching**：在共享基础模型上批量处理不同 LoRA 适配器；
- **准入控制**：在过载时限制请求，保护尾部延迟。

调度优化本质上是在单请求延迟、系统吞吐量和公平性之间做取舍。

## 分布式推理

当单张 GPU 无法容纳模型或无法达到目标性能时，需要组合多种并行策略。

- **Tensor Parallel**：把每层矩阵计算切分到多张 GPU；
- **Pipeline Parallel**：把不同模型层放到不同设备；
- **Data Parallel**：复制模型并行处理不同请求；
- **Expert Parallel**：分布 MoE 模型中的专家；
- **Context / Sequence Parallel**：切分超长序列和 KV Cache；
- **混合并行**：根据节点内、节点间带宽组合多种策略。

并行度越高，通信和同步开销通常越大。GPU 拓扑、NVLink、PCIe 和 InfiniBand 会直接影响最终性能。

## Prefill-Decode 分离

Prefill 偏计算密集，Decode 偏显存带宽密集。PD Disaggregation 将两者部署在不同实例或硬件池中：

- 分别调整 Prefill 和 Decode 的并行策略；
- 独立扩缩容两类 Worker；
- 避免长 Prefill 干扰正在生成的请求；
- 通过高速网络或专用服务传输 KV Cache；
- 根据输入输出长度动态调整两侧资源比例；
- 为不同阶段选择不同类型或不同精度的加速器。

这种方案增加了 KV Cache 生命周期、跨节点传输、失败恢复和资源配比的复杂度，更适合有明确规模和 SLO 的集群。

## 算子、编译器与硬件

这一层负责让相同计算更贴近硬件峰值：

- 融合 GEMM、Attention、归一化、采样和 MoE 算子；
- 使用 CUDA Graph 减少 CPU 调度和 Kernel Launch 开销；
- 使用 Triton、CUTLASS 或专用 CUDA Kernel；
- 通过 `torch.compile`、TensorRT 等执行图优化；
- 调整混合精度、Tensor Layout 和内存访问模式；
- 针对 GPU、NPU、CPU 的计算和带宽特征选择实现。

## 推理引擎与生产部署

具体引擎是上述技术的组合，而不是独立的优化类别：

- **vLLM**：PagedAttention、Continuous Batching、Prefix Cache、Chunked Prefill；
- **SGLang**：RadixAttention、Cache-aware Scheduling、结构化执行；
- **TensorRT-LLM**：NVIDIA 硬件上的量化、融合算子、并行和推测解码；
- **llama.cpp**：面向本地、CPU 和消费级设备的低比特推理；
- **TGI**：Hugging Face 生态中的生产推理服务。

生产环境还需要持续监控队列深度、GPU 利用率、KV Cache 使用率、吞吐量、错误率和各延迟分位数，并结合真实流量做容量规划。

## 建议的优化顺序

1. 固定工作负载和质量基线，拆分 TTFT、TPOT 与排队延迟；
2. 选择适配模型和硬件的推理引擎；
3. 优先启用成熟的批处理、Paged KV Cache 和高效 Attention；
4. 在显存或带宽受限时评估量化；
5. 根据请求特征评估 Prefix Cache、Chunked Prefill 和推测解码；
6. 单机达到瓶颈后，再设计多 GPU 并行或 Prefill-Decode 分离；
7. 最后通过算子、编译器和硬件专项调优逼近性能上限。

## 参考资料

- [Efficient LLM Inference: A Survey](https://www.cs.uoregon.edu/Reports/AREA-202606-Nguyen.pdf)
- [vLLM Documentation](https://docs.vllm.ai/en/latest/)
- [TensorRT-LLM Overview](https://nvidia.github.io/TensorRT-LLM/latest/overview.html)
- [SGLang](https://github.com/sgl-project/sglang)
- [Hugging Face Transformers Optimization Overview](https://huggingface.co/docs/transformers/v5.7.0/optimization_overview)
