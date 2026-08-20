export interface TaxonomyItem {
	name: string;
	children?: readonly TaxonomyItem[];
}

export const knowledgeTaxonomy: readonly TaxonomyItem[] = [
	{
		name: '大模型推理',
		children: [
			{
				name: '模型理论',
				children: [
					{
						name: '数学与计算基础',
						children: [
							{ name: '向量、矩阵与张量' },
							{ name: '矩阵乘法与线性变换' },
							{ name: '梯度与链式法则' },
							{ name: '概率分布与最大似然' },
							{ name: '数值精度与稳定性' },
							{ name: '计算与空间复杂度' },
						],
					},
					{
						name: '基础概念与算子',
						children: [
							{ name: 'LayerNorm、RMSNorm 与 BatchNorm' },
							{ name: 'ReLU、GELU 与 SiLU' },
							{ name: 'GLU、GeGLU 与 SwiGLU' },
							{ name: 'Softmax 与 LogSumExp' },
							{ name: 'Cross Entropy 与 KL Divergence' },
							{ name: 'Embedding 与 Linear' },
							{ name: '残差连接与 Dropout' },
						],
					},
					{
						name: 'Token 与输入表示',
						children: [
							{ name: 'Token、词表与 Tokenizer' },
							{ name: 'BPE、WordPiece 与 SentencePiece' },
							{ name: 'Special Token 与 Chat Template' },
							{ name: 'Token Embedding' },
							{ name: 'RoPE、ALiBi 与位置编码' },
							{ name: '上下文窗口与位置外推' },
						],
					},
					{
						name: 'Transformer 结构',
						children: [
							{ name: 'Encoder、Decoder 与 Decoder-Only' },
							{ name: 'Transformer Block' },
							{ name: 'Causal Mask 与双向注意力' },
							{ name: 'FFN、Gated FFN 与 SwiGLU' },
							{ name: 'Pre-Norm 与 Post-Norm' },
						],
					},
					{
						name: '注意力机制',
						children: [
							{ name: 'Attention 基础' },
							{ name: 'Q、K、V 与缩放点积注意力' },
							{ name: 'MHA、MQA、GQA 与 MLA' },
							{ name: '稀疏、线性与滑动窗口注意力' },
						],
					},
					{
						name: 'LLM 架构',
						children: [
							{ name: 'Dense 与 MoE' },
							{ name: '专家路由与负载均衡' },
							{ name: '参数共享与权重绑定' },
							{ name: 'Llama、Qwen 与 DeepSeek 结构对比' },
						],
					},
					{
						name: '训练与对齐原理',
						children: [
							{ name: '预训练与 Scaling Law' },
							{ name: 'Multi-Token Prediction (MTP)' },
							{ name: 'SFT 与指令微调' },
							{ name: 'RLHF、DPO 与 GRPO' },
							{ name: '奖励模型与过程监督' },
						],
					},
					{
						name: '生成原理',
						children: [
							{ name: 'Forward、Logits 与 LM Head' },
							{ name: '自回归生成' },
							{ name: 'Greedy、Sampling 与 Beam Search' },
							{ name: 'Temperature、Top-K 与 Top-P' },
							{ name: '重复惩罚与停止条件' },
						],
					},
				],
			},
			{
				name: '推理优化',
				children: [
					{ name: '总览' },
					{
						name: '性能分析与评测',
						children: [
							{ name: 'Prefill 与 Decode' },
							{ name: 'TTFT、TPOT 与 ITL' },
							{ name: '吞吐量、QPS 与 Goodput' },
							{ name: 'P50、P95、P99 与 SLO' },
							{ name: '显存、计算与带宽瓶颈' },
							{ name: 'Roofline 与 Profiling' },
							{ name: 'Benchmark 与容量规划' },
						],
					},
					{
						name: '模型表示与压缩',
						children: [
							{ name: 'FP16、BF16、FP8 与 FP4' },
							{ name: 'INT8、INT4 与 Weight-Only' },
							{ name: 'PTQ 与 QAT' },
							{ name: 'GPTQ、AWQ 与 SmoothQuant' },
							{ name: '剪枝与结构化稀疏' },
							{ name: '知识蒸馏与小模型' },
						],
					},
					{
						name: 'Attention 执行与 KV Cache',
						children: [
							{ name: 'FlashAttention 与 FlashInfer' },
							{ name: 'KV Cache 原理与显存估算' },
							{ name: 'PagedAttention' },
							{ name: 'Prefix Cache 与 RadixAttention' },
							{ name: 'KV Cache 量化与压缩' },
							{ name: 'Offload、淘汰与迁移' },
						],
					},
					{
						name: '解码与生成加速',
						children: [
							{ name: 'Speculative Decoding' },
							{ name: 'Draft Model 与并行验证' },
							{ name: 'Medusa 与 EAGLE' },
							{ name: 'MTP 自推测解码' },
							{ name: 'N-Gram 与 Self-Speculative' },
							{ name: 'Lookahead 与并行解码' },
							{ name: '接受率与收益边界' },
						],
					},
					{
						name: 'Runtime 与请求调度',
						children: [
							{ name: 'Continuous 与 In-Flight Batching' },
							{ name: 'Chunked Prefill' },
							{ name: '优先级、抢占与准入控制' },
							{ name: 'Cache-Aware Scheduling' },
							{ name: 'Multi-LoRA Batching' },
							{ name: '延迟、吞吐与公平性' },
						],
					},
					{
						name: '并行与分布式执行',
						children: [
							{ name: 'DP、TP 与 PP' },
							{ name: 'EP 与 MoE 推理' },
							{ name: 'Context 与 Sequence Parallel' },
							{ name: '混合并行策略' },
							{ name: 'AllReduce、AllGather 与 AllToAll' },
							{ name: 'NVLink、RDMA 与 InfiniBand' },
						],
					},
					{
						name: 'Prefill-Decode 分离',
						children: [
							{ name: 'PD Disaggregation' },
							{ name: 'KV Cache 跨节点传输' },
							{ name: 'Prefill 与 Decode 独立扩缩容' },
							{ name: '分布式 KV Cache 服务' },
							{ name: 'NIXL 与传输后端' },
							{ name: '路由、调度与故障恢复' },
						],
					},
					{
						name: '算子、编译器与硬件',
						children: [
							{ name: 'GPU、SM、Warp 与 Tensor Core' },
							{ name: 'HBM、Cache 与内存带宽' },
							{ name: 'GEMM、Attention 与 MoE 算子' },
							{ name: '算子融合与 CUDA Graph' },
							{ name: 'Triton 与 CUTLASS' },
							{ name: 'torch.compile 与 TensorRT' },
						],
					},
					{
						name: '推理引擎与生产服务',
						children: [
							{ name: 'Transformers 与 llama.cpp' },
							{ name: 'vLLM 与 SGLang' },
							{ name: 'TensorRT-LLM、TGI 与 Triton' },
							{ name: 'API、流式输出与并发控制' },
							{ name: '扩缩容、多租户与限流' },
							{ name: '监控、Tracing 与容量规划' },
							{ name: 'Benchmark、排障与回归测试' },
						],
					},
				],
			},
			{
				name: 'vLLM 框架',
				children: [{ name: 'vLLM logprobs 说明' }],
			},
		],
	},
];
