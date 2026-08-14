import type { CollectionEntry } from 'astro:content';
import type { TaxonomyItem } from '../site.config';

export type Note = CollectionEntry<'notes'>;

export interface CategoryNode {
	name: string;
	path: string[];
	children: CategoryNode[];
	notes: Note[];
	order: number;
}

export function sortNotes(notes: Note[]) {
	return [...notes].sort(
		(a, b) => b.data.published.getTime() - a.data.published.getTime(),
	);
}

export function formatDate(date: Date) {
	return new Intl.DateTimeFormat('zh-CN', {
		year: 'numeric',
		month: '2-digit',
		day: '2-digit',
	}).format(date);
}

export function buildCategoryTree(
	notes: Note[],
	taxonomy: readonly TaxonomyItem[] = [],
): CategoryNode[] {
	const seedTaxonomy = (
		items: readonly TaxonomyItem[],
		parentPath: string[] = [],
	): CategoryNode[] =>
		items.map((item, index) => {
			const path = [...parentPath, item.name];
			return {
				name: item.name,
				path,
				children: seedTaxonomy(item.children ?? [], path),
				notes: [],
				order: index,
			};
		});

	const roots = seedTaxonomy(taxonomy);

	for (const note of notes) {
		let level = roots;
		const currentPath: string[] = [];

		for (const [index, segment] of note.data.category.entries()) {
			currentPath.push(segment);
			let node = level.find((item) => item.name === segment);

			if (!node) {
				node = {
					name: segment,
					path: [...currentPath],
					children: [],
					notes: [],
					order: Number.MAX_SAFE_INTEGER,
				};
				level.push(node);
			}

			level = node.children;
			if (index === note.data.category.length - 1) node.notes.push(note);
		}
	}

	const sortLevel = (nodes: CategoryNode[]) => {
		nodes.sort(
			(a, b) => a.order - b.order || a.name.localeCompare(b.name, 'zh-CN'),
		);
		for (const node of nodes) {
			node.notes = sortNotes(node.notes);
			sortLevel(node.children);
		}
	};

	sortLevel(roots);
	return roots;
}

export function getTopics(notes: Note[]) {
	const counts = new Map<string, number>();

	for (const note of notes) {
		for (const topic of note.data.topics) {
			counts.set(topic, (counts.get(topic) ?? 0) + 1);
		}
	}

	return [...counts.entries()]
		.map(([name, count]) => ({ name, count }))
		.sort((a, b) => b.count - a.count || a.name.localeCompare(b.name, 'zh-CN'));
}

export function getReadingMinutes(note: Note) {
	const characterCount = note.body?.replace(/\s/g, '').length ?? 0;
	return Math.max(1, Math.ceil(characterCount / 500));
}
