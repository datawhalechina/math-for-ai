# worklog

## 2026-08-29 PR #25 审查 + 修复(fix/lagrangre-refs 分支)

### 任务背景

审查 open PR [#25](https://github.com/datawhalechina/math-for-ai/pull/25)(ksk2023:fix2 → main,用 KaTeX 公式替换 ch12_3 失效的 simpletex 图床外链),结论为接受合并;用户随后要求:修复审查发现的问题、提交到 fix/ 分支、起 docsify server 测试对应章节的图片/公式加载、修复全部干净文件中的 Lagrangre 拼写。

### 审查结论(2026-08-29 上午)

- PR #25 动机成立:原图床外链 `img.simpletex.net/pdf/BzBdZhAh/...png` 实测 HTTP 502。
- 公式正确性:下载 MML 原书 PDF(mml-book.github.io,17.5 MB)提取第 383 页式 (12.34) 逐项比对,PR 公式与原书完全等价(⟨w,xₙ⟩ 与 w^⊤xₙ 记号差异为等价改写,且与 12.35 中 xₙ^⊤ 风格一致)。
- 记号与仓库惯例一致(\boldsymbol、\:、\tag);`git merge-tree` 与 GitHub API 均确认可干净合并。

### 本分支改动(基于 origin/main c1936b0,合并 PR #25 后)

提交清单:cb6e526(merge PR #25)→ af287a0(ch12_3 修复)→ 00bc55f(拼写)→ ba6fbc2(sidebar 死链)→ 本提交(worklog)。

1. **Merge PR #25**(保留贡献者署名):`fed29235` 合入,PR 合入 main 后会被 GitHub 自动标记为 Merged。
2. **ch12_3.md 修复**:
   - (12.34) aligned 补 `&`(对齐 `=`)与 `\\`(PR #25 缺失,渲染会挤成单行——Copilot 审阅意见确认有效);
   - (12.35) 原把三条偏导合在一个 aligned 块只挂单 tag,正文引用的 (12.36)/(12.37) 悬空 → 拆为三个独立公式块分别挂 tag(实测 KaTeX 不支持 aligned 内多 tag,报 "Multiple \tag");
   - 对偶函数两条公式补 `\tag{12.39}`/`\tag{12.40}`(正文已引用,此前悬空);
   - (12.51a)/(12.51b) 双 tag 为 KaTeX 不支持 → 按原书两行编号拆为两个独立 `$$` 块(经原书 PDF p.388 核实确为 12.51a/12.51b);
   - 本文件 13 处 "Lagrangre" 拼写修正。
3. **Lagrangre 全仓拼写修正(干净文件范围)**:ch12.md(1)、ch12_2.md(1)、ch10_2.md(2)、ch7_习题.md(3)、ch12_3.md(13)。规则:乘数/乘子 → **Lagrange**,函数 → **Lagrangian**,对偶 → **Lagrange**(对应原书 Lagrange multipliers / the Lagrangian / Lagrange duality)。
4. **_sidebar.md 死链**:移除 `ch8/ch8_习题.md`、`ch10/ch10_习题.md` 两个链接。根因:521191d "delete unneeded files" 删除了这两个文件但漏更 sidebar。

### 测试与验证(全部通过)

- **KaTeX 离线审计**:仓库自带 tools/katex_check.js(站点同款 KaTeX),覆盖 5 个受影响文件全部 **415 个公式块(display+inline),ALL PASS**。注:katex_audit.py 硬编码了 Linux node 路径,本机(Windows node)跑不了,改用同款 checker + 等价提取脚本验证。
- **docsify serve 实测**(npx docsify-cli,端口 63159):5 个页面(/#/ch12/ch12、ch12_2、ch12_3、ch10/ch10_2、ch7/ch7_习题)全部 200;真实浏览器(chromium via puppeteer-core,ms-playwright 已装 chromium-1228)实测:
  - 图片 17 张全部加载成功(broken=0)。关键结论:`attachments/X.png` 写法在 docsify hash 路由下解析为 `/attachments/X.png`(文档 base 为根),0572878 的路径改动**不是回归**;历史模式 `../attachments/` 在根 base 下解析结果相同。
  - KaTeX 渲染计数 18+131+91+149+26=415,**katexErrors=0**,与离线审计数字完全吻合。
  - 修复前用 history 路由(/ch12/ch12_2)curl 模拟得到的 /chN/attachments/ 404 是误报场景,不对应 docsify 实际 hash 路由行为。

### 问题记录

- [遗留] ch7_2.md(7 行)、ch7_3.md(8 行)仍有 "Lagrangre" 拼写问题:两文件在本地工作区有未提交修改(与 origin/main 新提交 29 文件交集),为避免污染 WIP 未动;待 WIP 提交时一并修正。
- [遗留] `/favicon.ico` 404:全站无 favicon,index.html 也未配置,轻微。
- [遗留] 每页 console 报 `Docsify plugin error: Cannot read properties of undefined (reading 'unshift')`:index.html 插件栈的既有问题,与内容无关,建议后续排查插件配置(疑似 search/emoji 插件加载顺序)。
- [遗留] ch12_3.md "它可以通过 http://fouryears.eu/... 访问。" 一句为翻译残句(b 的计算式缺失,原书脚注语境),需补译,超出本任务范围未动。
- [备注] 本地主工作区 main 落后 origin/main 2 提交且 45 个文件有未提交修改(29 个与远端新提交重叠),本轮全部工作在独立 worktree(math-for-ai-prfix)完成,主工作区未动。主工作区同步需先处理 WIP,建议:确认 WIP 内容 → 提交或 stash → fast-forward。

### 反思

1. 图片路径验证差点误判:先用 curl 模拟 history 路由得出"全站图片 404"的结论,再意识到 docsify 是 hash 路由、浏览器 base 永远是根——**SPA 的资源解析必须用真实浏览器验证**,静态 curl 只能测"文件是否存在",不能测"页面里怎么解析"。Playwright 的 chromium 二进制 + puppeteer-core 组合是零下载成本的可靠方案。
2. 多重证据链再次起效:公式改动对照原书 PDF(12.51a/b 编号)、KaTeX 实测(Multiple \tag 限制)、浏览器渲染计数与离线审计数字互相印证(415=415),避免了"看起来对"。
3. WIP 与远端提交重叠时,独立 worktree 是零风险操作面:任何主工作区状态都不受影响,push 也不依赖本地 main 的新旧。

