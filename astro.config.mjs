// @ts-check
import { defineConfig } from 'astro/config';

const repository = process.env.GITHUB_REPOSITORY?.split('/')[1];
const owner = process.env.GITHUB_REPOSITORY_OWNER;
const isUserSite =
	repository?.toLowerCase() === `${owner}.github.io`.toLowerCase();

// GitHub Actions 会自动推断项目站点的子路径；本地开发始终使用根路径。
const base =
	process.env.BASE_PATH ??
	(process.env.GITHUB_ACTIONS && repository && !isUserSite ? `/${repository}` : '/');
const site = process.env.SITE_URL ?? (owner ? `https://${owner}.github.io` : 'http://localhost:4321');

export default defineConfig({
	site,
	base,
	trailingSlash: 'always',
	markdown: {
		shikiConfig: {
			theme: 'github-light',
			wrap: true,
		},
	},
});
