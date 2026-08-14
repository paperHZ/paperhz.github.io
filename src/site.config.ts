import { knowledgeTaxonomy } from './knowledge-taxonomy';

export type { TaxonomyItem } from './knowledge-taxonomy';
export { knowledgeTaxonomy };

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

export function withBase(path = '/') {
	const base = import.meta.env.BASE_URL;
	const cleanPath = path.replace(/^\/+/, '');

	return cleanPath ? `${base}${cleanPath}` : base;
}
