import type { CollectionEntry } from 'astro:content';

export type Paper = CollectionEntry<'papers'>;

export const paperStatus = {
	unread: '待读',
	reading: '阅读中',
	done: '已完成',
} as const;

const organizationOrder = [
	'DeepSeek',
	'Qwen',
	'Google',
	'OpenAI',
	'Kimi',
	'智谱',
	'Anthropic',
	'Meta',
	'Mistral',
];

export const inferencePaperTopics = [
	'投机解码',
	'Attention 与 KV Cache',
	'解码与生成加速',
	'量化与模型压缩',
	'推理服务与系统',
] as const;

type InferencePaperTopic = (typeof inferencePaperTopics)[number];

const inferenceTopicOverrides: Record<string, readonly InferencePaperTopic[]> = {
	'openai-generating-long-sequences-with-sparse-transformers': [
		'Attention 与 KV Cache',
	],
	'google-multi-query-attention': ['Attention 与 KV Cache'],
	'google-speculative-decoding': ['解码与生成加速'],
	'mistral-mistral-7b': ['Attention 与 KV Cache'],
	'mistral-ministral-3': ['量化与模型压缩'],
	'deepseek-v2': ['Attention 与 KV Cache'],
	'deepseek-native-sparse-attention': ['Attention 与 KV Cache'],
	'qwen2-5-1m': ['Attention 与 KV Cache'],
	'kimi-mooncake': ['Attention 与 KV Cache', '推理服务与系统'],
	'kimi-moba': ['Attention 与 KV Cache'],
	'kimi-linear': ['Attention 与 KV Cache'],
};

const speculativeDecodingPaperIds = new Set([
	'google-speculative-decoding',
	'meta-multi-token-prediction',
	'inference-blockwise-parallel-decoding',
	'inference-speculative-sampling',
	'inference-draft-and-verify',
	'inference-medusa',
	'inference-eagle',
	'inference-lookahead-decoding',
	'inference-eagle-2',
	'inference-eagle-3',
	'inference-specinfer',
	'inference-rest',
	'inference-sequoia',
	'inference-layerskip',
	'inference-magicdec',
	'inference-dflash',
	'inference-dspark',
]);

const paperAreas = [
	{
		name: '基础模型与架构',
		keywords: [
			'基础模型',
			'基座模型',
			'模型架构',
			'Transformer',
			'Attention',
			'MoE',
			'稀疏模型',
			'语言模型',
		],
	},
	{
		name: '预训练与 Scaling',
		keywords: [
			'预训练',
			'持续预训练',
			'Scaling',
			'缩放',
			'计算最优',
			'数据配比',
			'训练稳定性',
			'优化器',
		],
	},
	{
		name: '后训练与对齐',
		keywords: [
			'后训练',
			'指令微调',
			'RLHF',
			'RLAIF',
			'强化学习',
			'偏好',
			'奖励模型',
			'安全对齐',
			'GRPO',
			'PPO',
			'RLVR',
			'GSPO',
		],
	},
	{
		name: '推理与 Reasoning',
		keywords: [
			'推理与 Reasoning',
			'推理模型',
			'数学推理',
			'形式化推理',
			'定理证明',
			'思维链',
			'Chain-of-Thought',
			'测试时计算',
			'过程监督',
			'验证器',
		],
	},
	{
		name: '推理优化',
		keywords: [],
	},
	{
		name: '投机解码',
		keywords: [],
	},
	{
		name: 'Attention 与 KV Cache',
		keywords: [],
	},
	{
		name: '解码与生成加速',
		keywords: [],
	},
	{
		name: '量化与模型压缩',
		keywords: [],
	},
	{
		name: '推理服务与系统',
		keywords: [],
	},
	{
		name: 'Agent 与工具使用',
		keywords: [
			'智能体',
			'Agent',
			'工具使用',
			'工具调用',
			'网页浏览',
			'GUI',
			'机器人控制',
			'环境交互',
			'软件工程',
		],
	},
	{
		name: 'RAG、记忆与长上下文',
		keywords: [
			'检索',
			'外部记忆',
			'条件记忆',
			'长上下文',
			'百万上下文',
			'128K 上下文',
			'长文本',
			'文档问答',
		],
	},
	{
		name: '多模态',
		keywords: [
			'多模态',
			'视觉',
			'图像',
			'视频',
			'音频',
			'语音',
			'OCR',
			'CLIP',
			'文生图',
		],
	},
	{
		name: '代码模型',
		keywords: ['代码', 'Code LLM', '编程', '编码智能体', '软件工程'],
	},
	{
		name: '评测、安全与可解释性',
		keywords: [
			'安全',
			'评测',
			'红队',
			'可解释',
			'稀疏自编码器',
			'欺骗',
			'越狱',
			'审计',
			'错位',
			'内部威胁',
			'可检查性',
		],
	},
] as const;

export function getPaperAreas(paper: Paper) {
	const searchable = [
		paper.data.title,
		paper.data.titleZh ?? '',
		paper.data.summary,
		paper.data.series,
		...paper.data.topics,
	]
		.join(' ')
		.toLocaleLowerCase('zh-CN');

	const matches = paperAreas
		.filter((area) =>
			area.keywords.some((keyword) =>
				searchable.includes(keyword.toLocaleLowerCase('zh-CN')),
			),
		)
		.map((area) => area.name);

	const explicitInferenceTopics: InferencePaperTopic[] = [
		...(paper.data.inferenceTopics.length > 0
			? paper.data.inferenceTopics
			: (inferenceTopicOverrides[paper.id] ?? [])),
	];
	if (
		speculativeDecodingPaperIds.has(paper.id) &&
		!explicitInferenceTopics.includes('投机解码')
	) {
		explicitInferenceTopics.push('投机解码');
	}
	if (explicitInferenceTopics.length > 0) {
		matches.push('推理优化', ...explicitInferenceTopics);
		matches.sort(
			(a, b) =>
				paperAreas.findIndex((area) => area.name === a) -
				paperAreas.findIndex((area) => area.name === b),
		);
	}

	return matches.length > 0 ? matches : ['基础模型与架构'];
}

export function sortPapers(papers: Paper[]) {
	return [...papers].sort(
		(a, b) => b.data.paperDate.getTime() - a.data.paperDate.getTime(),
	);
}

export function getPaperOrganizations(papers: Paper[]) {
	const counts = new Map<string, number>();

	for (const paper of papers) {
		for (const organization of paper.data.organizations) {
			if (!organizationOrder.includes(organization)) continue;
			counts.set(organization, (counts.get(organization) ?? 0) + 1);
		}
	}

	return [...counts.entries()]
		.map(([name, count]) => ({ name, count }))
		.sort((a, b) => {
			const aIndex = organizationOrder.indexOf(a.name);
			const bIndex = organizationOrder.indexOf(b.name);
			return (
				(aIndex < 0 ? Number.MAX_SAFE_INTEGER : aIndex) -
					(bIndex < 0 ? Number.MAX_SAFE_INTEGER : bIndex) ||
				a.name.localeCompare(b.name)
			);
		});
}

export function getPaperTopics(papers: Paper[]) {
	const counts = new Map<string, number>();

	for (const paper of papers) {
		for (const topic of getPaperAreas(paper)) {
			counts.set(topic, (counts.get(topic) ?? 0) + 1);
		}
	}

	return [...counts.entries()]
		.map(([name, count]) => ({ name, count }))
		.sort(
			(a, b) =>
				paperAreas.findIndex((area) => area.name === a.name) -
				paperAreas.findIndex((area) => area.name === b.name),
		);
}
