export const siteConfig = {
	title: 'paperHZ',
	description: '把学习过程写成一张可以反复抵达的地图。',
	author: 'paperHZ',
	github: 'https://github.com/paperHZ',
	navigation: [
		{ label: '时间流', href: '/' },
		{ label: '论文', href: '/papers/' },
		{ label: '主题', href: '/topics/' },
	],
} as const;

export interface TaxonomyItem {
	name: string;
	children?: readonly TaxonomyItem[];
}

// 预先声明的学习路线。即使某个细项还没有文章，也会保留在左侧知识树中。
export const knowledgeTaxonomy: readonly TaxonomyItem[] = [
	{
		name: '大模型推理',
		children: [
			{ name: '总览' },
			{
				name: '推理基础与性能评测',
				children: [
					{ name: 'Prefill 与 Decode' },
					{ name: 'TTFT、TPOT 与 ITL' },
					{ name: 'Throughput、QPS 与 Goodput' },
					{ name: '延迟分位数与 SLO' },
					{ name: '计算、显存与通信瓶颈' },
					{ name: 'Benchmark 与容量规划' },
				],
			},
			{
				name: '模型压缩',
				children: [
					{ name: 'FP8、FP4、INT8 与 INT4' },
					{ name: 'GPTQ、AWQ 与 SmoothQuant' },
					{ name: '权重、激活与混合量化' },
					{ name: '剪枝与结构化稀疏' },
					{ name: '知识蒸馏' },
				],
			},
			{
				name: 'Attention 与 KV Cache',
				children: [
					{ name: 'Attention 基础' },
					{ name: 'MHA、MQA、GQA 与 MLA' },
					{ name: 'FlashAttention 与 FlashInfer' },
					{ name: 'PagedAttention' },
					{ name: 'Prefix Cache 与 RadixAttention' },
					{ name: 'KV Cache 量化与压缩' },
					{ name: 'KV Cache 淘汰与 Offload' },
					{ name: '长上下文与滑动窗口' },
				],
			},
			{
				name: '解码加速',
				children: [
					{ name: 'Sampling 与 Beam Search' },
					{ name: 'Speculative Decoding' },
					{ name: 'Draft Model 与并行验证' },
					{ name: 'EAGLE、MTP 与 Medusa' },
					{ name: 'N-Gram 与无模型推测' },
					{ name: '并行与多 Token 解码' },
					{ name: '结构化与约束解码' },
				],
			},
			{
				name: '批处理与请求调度',
				children: [
					{ name: 'Static 与 Dynamic Batching' },
					{ name: 'Continuous 与 In-flight Batching' },
					{ name: 'Chunked Prefill' },
					{ name: '优先级、抢占与准入控制' },
					{ name: 'Cache-aware Scheduling' },
					{ name: 'Multi-LoRA Batching' },
					{ name: '延迟与吞吐量权衡' },
				],
			},
			{
				name: '分布式推理',
				children: [
					{ name: 'Tensor Parallel' },
					{ name: 'Pipeline Parallel' },
					{ name: 'Data Parallel' },
					{ name: 'Expert Parallel' },
					{ name: 'Context 与 Sequence Parallel' },
					{ name: 'MoE 专家放置与负载均衡' },
					{ name: 'AllReduce、NVLink 与 InfiniBand' },
				],
			},
			{
				name: 'Prefill-Decode 分离',
				children: [
					{ name: 'PD Disaggregation' },
					{ name: 'KV Cache 跨节点传输' },
					{ name: 'Prefill 与 Decode 独立扩缩容' },
					{ name: '分布式 KV Cache 服务' },
					{ name: '请求路由与负载均衡' },
					{ name: '异构 GPU 推理' },
				],
			},
			{
				name: '算子、编译器与硬件',
				children: [
					{ name: 'GEMM、Attention 与 MoE 算子' },
					{ name: '算子融合与 CUDA Graph' },
					{ name: 'Triton 与 CUTLASS' },
					{ name: 'torch.compile' },
					{ name: 'TensorRT 图优化' },
					{ name: '混合精度与内存布局' },
					{ name: 'GPU、NPU 与 CPU 推理' },
				],
			},
			{
				name: '推理引擎与生产部署',
				children: [
					{ name: 'vLLM' },
					{ name: 'SGLang' },
					{ name: 'TensorRT-LLM' },
					{ name: 'llama.cpp 与 TGI' },
					{ name: '服务监控与容量规划' },
					{ name: '多租户、限流与故障恢复' },
					{ name: '生产 Benchmark 与参数调优' },
				],
			},
		],
	},
];

export function withBase(path = '/') {
	const base = import.meta.env.BASE_URL;
	const cleanPath = path.replace(/^\/+/, '');

	return cleanPath ? `${base}${cleanPath}` : base;
}
