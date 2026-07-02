# Plug-and-Play Module Extractor

一个用于从论文 PDF 和/或配套源码中判断、提取、复现深度学习即插即用模块的 Codex skill。

## 功能

- PDF-only：读取论文，判断是否存在可独立复用的模块；如果存在，生成 paper-derived reconstruction brief，辅助复现模块代码；如果不存在，明确终止。
- PDF + code：结合论文和源码定位模块实现，优先按目标类/函数名抽取，并自动包含同文件顶层依赖。
- 验证链路：支持语法检查、导入检查、smoke test、接口比对、论文候选对齐、批量样例回归。

## 目录

```text
plug-and-play-module-extractor/
  SKILL.md
  agents/openai.yaml
  references/
  scripts/
```

`样例/` 是本地测试资料目录，不会提交到仓库。

## 安装

把 skill 目录复制到 Codex skills 目录，例如：

```powershell
Copy-Item -Recurse .\plug-and-play-module-extractor $env:USERPROFILE\.codex\skills\
```

之后在 Codex 中使用：

```text
Use $plug-and-play-module-extractor to analyze this paper PDF and extract the plug-and-play module.
```

## 常用命令

提取 PDF 文本：

```powershell
python .\plug-and-play-module-extractor\scripts\extract_pdf_text.py paper.pdf --output paper.txt
```

分析论文是否包含即插即用模块：

```powershell
python .\plug-and-play-module-extractor\scripts\analyze_paper_modules.py paper.txt
```

生成 PDF-only 复现简报：

```powershell
python .\plug-and-play-module-extractor\scripts\build_reconstruction_brief.py paper.txt --output brief.md
```

扫描源码候选模块：

```powershell
python .\plug-and-play-module-extractor\scripts\find_module_candidates.py path\to\code
```

按名称抽取源码模块：

```powershell
python .\plug-and-play-module-extractor\scripts\extract_python_module.py source.py --name ModuleName --include-main --output extracted.py
```

验证生成模块：

```powershell
python .\plug-and-play-module-extractor\scripts\smoke_test_module.py extracted.py --python python
```

## 回归测试

完整回归套件会覆盖脚本编译、skill 校验、样例评估和模拟 generated batch 评估：

```powershell
python .\plug-and-play-module-extractor\scripts\run_regression_suite.py .\样例 --python python --validator "$env:USERPROFILE\.codex\skills\.system\skill-creator\scripts\quick_validate.py" --report-json regression-report.json
```

如果测试环境中的 `python` 没有 PyTorch，可以传入一个已安装 CPU 版 torch 的解释器。

## 注意

- 本仓库只提交 skill 本体和脚本，不提交论文/代码案例。
- PDF-only 复现必须标明 `paper-derived`，不能声称来自官方源码。
- PDF + code 抽取优先保留源码实现，只移除训练、数据、日志、checkpoint 等非模块逻辑。
