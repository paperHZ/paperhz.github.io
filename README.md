# paperHZ

一个面向长期学习记录的静态知识站：左侧是分类树，首页是从近到远的时间流，主题负责跨分类连接文章。内容全部使用 Markdown，构建结果可以直接部署到 GitHub Pages。

## 本地运行

需要 Node.js 22.12 或更高版本。

```bash
npm install
npm run dev
```

浏览器打开 `http://localhost:4321`。发布前可执行：

```bash
npm run build
npm run preview
```

## 先修改个人信息

编辑 `src/site.config.ts`：

- `title`：站点名称；
- `author`：作者名；
- `github`：右上角 GitHub 链接；
- `description`：站点简介。

## 写一篇文章

在 `src/content/notes/` 的任意位置新增 `.md` 文件：

```yaml
---
title: "文章标题"
summary: "一句话摘要"
published: 2026-08-14
updated: 2026-08-15 # 可选
category:
  - 大模型推理
  - 推理优化
  - Attention 执行与 KV Cache
  - PagedAttention
topics:
  - vLLM
  - Transformer
draft: false
zhihu: "https://zhuanlan.zhihu.com/p/..." # 可选
---

从这里开始写正文。
```

- `category` 是左侧目录中的唯一层级路径，可以继续增加层级；
- `topics` 可以有多个，会生成主题聚合页；
- `draft: true` 的文章不会参与构建；
- 有 `zhihu` 时，文章页自动显示知乎跳转按钮。

## 添加论文

在 `src/data/papers/` 对应机构的 JSON 文件中增加一条记录。仓库只保存轻量索引，不保存论文 PDF 或原文：

```json
{
  "id": "deepseek-r1",
  "title": "论文英文标题",
  "titleZh": "可选的中文标题",
  "summary": "论文解决的问题",
  "paperDate": "2025-01-22",
  "organizations": ["DeepSeek"],
  "series": "推理模型路线",
  "topics": ["推理与 Reasoning", "强化学习"],
  "status": "unread",
  "arxivUrl": "https://arxiv.org/abs/..."
}
```

每篇论文只保存一次元数据，网站会根据 `organizations` 和 `topics` 自动生成机构路线与主题索引，论文标题直接跳转 arXiv。

当前索引包含九家主要大模型机构及推理系统研究社区的 189 篇核心论文，并设有独立的投机解码与推理优化专题。

## 发布到 GitHub Pages

1. 在 GitHub 创建仓库。个人主页建议命名为 `你的用户名.github.io`；
2. 将代码推送到 `main` 分支；
3. 打开仓库的 **Settings → Pages**，将 Source 设为 **GitHub Actions**；
4. 等待 `Deploy to GitHub Pages` 工作流完成。

部署配置会自动区分个人主页仓库和普通项目仓库，并处理项目站点的子路径。若使用自定义域名，再按 Astro 的 GitHub Pages 文档设置 `SITE_URL` 和 `public/CNAME`。

## 目录

```text
src/
├── content/notes/       # Markdown 文章
├── data/papers/         # 按机构拆分的论文 JSON 索引
├── components/          # 目录和导航
├── layouts/             # 页面外壳
├── pages/               # 时间流、论文、主题和详情页
├── styles/global.css    # 全站视觉样式
└── site.config.ts       # 个人站点配置
```
