---
title: "欢迎来到 paperHZ"
summary: "这是一篇示例文章，用来说明分类、主题、时间流和知乎链接如何协同工作。"
published: 2026-08-14
category:
  - 站点
  - 使用说明
topics:
  - 写作
  - 知识管理
zhihu: "https://www.zhihu.com/"
---

这是一个为长期积累设计的个人学习站点。它有两种互补的阅读方式：

- **左侧目录**回答“这篇内容属于哪里”；
- **首页时间流**回答“最近学了什么”。

## 新增一篇文章

在 `src/content/notes/` 下新建 Markdown 文件。文件夹可以按照自己的习惯组织，真正展示在目录中的层级由文章头部的 `category` 决定。

```yaml
---
title: "文章标题"
summary: "用一句话说明这篇文章解决什么问题。"
published: 2026-08-14
category:
  - 大模型推理
  - 推理优化
  - Attention 执行与 KV Cache
  - PagedAttention
topics:
  - vLLM
  - Transformer
---
```

一篇文章只能有一条分类路径，但可以拥有多个主题。这样目录保持稳定，横向关联也不会丢失。

## 同步到知乎

文章在知乎发布后，只需要增加一个字段：

```yaml
zhihu: "https://zhuanlan.zhihu.com/p/你的文章编号"
```

文章标题下方会自动出现“在知乎阅读”按钮。删除该字段，按钮就不会显示。

> 建议先在这里保留 Markdown 原稿，再把排版后的版本同步到其他平台。仓库始终是内容的唯一来源。
