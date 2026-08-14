import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';
import { z } from 'astro/zod';
import anthropicPapers from './data/papers/anthropic.json';
import deepseekPapers from './data/papers/deepseek.json';
import googlePapers from './data/papers/google.json';
import inferencePapers from './data/papers/inference.json';
import kimiPapers from './data/papers/kimi.json';
import metaPapers from './data/papers/meta.json';
import mistralPapers from './data/papers/mistral.json';
import openaiPapers from './data/papers/openai.json';
import qwenPapers from './data/papers/qwen.json';
import zhipuPapers from './data/papers/zhipu.json';

const notes = defineCollection({
	loader: glob({ pattern: '**/*.{md,mdx}', base: './src/content/notes' }),
	schema: z.object({
		title: z.string(),
		summary: z.string(),
		published: z.coerce.date(),
		updated: z.coerce.date().optional(),
		category: z.array(z.string()).min(1),
		topics: z.array(z.string()).default([]),
		draft: z.boolean().default(false),
		zhihu: z.url().optional(),
		paperId: z.string().optional(),
	}),
});

const papers = defineCollection({
	loader: async () => [
		...deepseekPapers,
		...qwenPapers,
		...googlePapers,
		...openaiPapers,
		...kimiPapers,
		...zhipuPapers,
		...anthropicPapers,
		...metaPapers,
		...mistralPapers,
		...inferencePapers,
	],
	schema: z.object({
		id: z.string(),
		title: z.string(),
		titleZh: z.string().optional(),
		summary: z.string(),
		paperDate: z.coerce.date(),
		readDate: z.coerce.date().optional(),
		organizations: z.array(z.string()).min(1),
		series: z.string(),
		topics: z.array(z.string()).min(1),
		inferenceTopics: z
			.array(
				z.enum([
					'Attention 与 KV Cache',
					'解码与生成加速',
					'量化与模型压缩',
					'推理服务与系统',
				]),
			)
			.default([]),
		status: z.enum(['unread', 'reading', 'done']).default('unread'),
		arxivUrl: z.url(),
		noteId: z.string().optional(),
		draft: z.boolean().default(false),
	}),
});

export const collections = { notes, papers };
