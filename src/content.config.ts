import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';
import { z } from 'astro/zod';

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
	}),
});

export const collections = { notes };
