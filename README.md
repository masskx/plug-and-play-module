# Plug-and-Play Module Extractor

一个从论文和/或源码中判断、提取、复现深度学习**即插即用模块**，并导出为可直接运行代码的 Codex / Claude skill。

如果审查之后确实没有可用模块，skill 只会输出结论（“经审查，未发现可用的即插即用模块”）和依据，不会凑数、不会生成代码。

## 支持的四种输入

| 情形 | `source_mode` | 代码依据 |
|---|---|---|
| 有论文 + 有代码 | `pdf-plus-code` | `source-code-derived`，论文用于定位和核对 |
| 有论文、无代码 | `pdf-only` | `paper-derived`，依据公式/算法/结构图复现，不冒充官方实现 |
| 无论文、只有代码 | `code-only` | `source-code-derived`，由代码检查确定模块边界 |
| 只有论文标题 / arXiv 号 | `title-only` | 先检索并核实论文（及官方代码），再按上面三种处理；无法核实则停止并索要 PDF，绝不凭标题臆造 |

详细流程见 `plug-and-play-module-extractor/references/input_modes.md`。

## 工作流

```text
triage_inputs ─▶ (resolve_paper_by_title) ─▶ extract_pdf_text
      │
      ▼
review_gate  = analyze_paper_modules + find_module_candidates + match_paper_to_code
      │               └─ 再按 references/module_detection.md 人工核对每个候选
      ├─ no ─▶ 输出“经审查，未发现可用的即插即用模块”及依据，结束
      └─ yes / uncertain
             ▼
   extract_python_module（源码）  /  build_reconstruction_brief（论文）
             ▼
   check_module_contract --run  +  compare_module_interface --fidelity
```

“可用”的判定标准（接口明确、细节完整、可脱离流程、可复用、非平凡、可运行）见 `references/module_detection.md`。关键词（module / layer / ablation 等）本身不构成证据，脚本会单独统计公式、张量形状、算法/结构图引用、即插即用声明、模块消融等结构性证据，以及论文贡献类型（数据集 / 综述 / 理论 / 系统 / 训练策略会直接判为 `no`）。

## 目录

```text
plug-and-play-module-extractor/
  SKILL.md
  agents/openai.yaml
  references/   # input_modes / module_detection / paper_reconstruction / output_contract
  scripts/
tests/          # 自包含回归测试与合成样例（无需私有论文）
```

`样例/` 是本地私有测试资料目录，不会提交到仓库。

## 安装

把 skill 目录复制到 skills 目录，例如：

```powershell
Copy-Item -Recurse .\plug-and-play-module-extractor $env:USERPROFILE\.codex\skills\
```

```bash
cp -r plug-and-play-module-extractor ~/.claude/skills/
```

之后使用：

```text
Use $plug-and-play-module-extractor to review this paper PDF and export the plug-and-play module.
```

## 常用命令

判断输入类型并给出下一步：

```bash
python scripts/triage_inputs.py paper.pdf path/to/code
python scripts/triage_inputs.py "Paper Title"
```

只有标题时检索论文（需要访问 arXiv；被屏蔽时会返回 `network_unavailable`，改用联网搜索工具）：

```bash
python scripts/resolve_paper_by_title.py --title "Paper Title" --download-dir ./paper --github
```

提取论文文本并做审查：

```bash
python scripts/extract_pdf_text.py paper.pdf --output paper.txt
python scripts/review_gate.py --paper-text paper.txt --code-root path/to/code --json
python scripts/review_gate.py --code-root path/to/code            # 只有代码
python scripts/review_gate.py --paper-text paper.txt --lang zh    # 只有论文，中文结论
```

扫描代码候选（按 plug_score 分级，并列出风险：本地导入、注册器、config 依赖、CUDA 算子等）：

```bash
python scripts/find_module_candidates.py path/to/code --min-tier possible
python scripts/match_paper_to_code.py paper.txt path/to/code
```

按名称抽取源码模块（自动带上同文件依赖，并内联同仓库其他文件中的辅助函数 / 常量）：

```bash
python scripts/extract_python_module.py path/to/code/pkg/blocks.py --name ModuleName \
  --include-main --root path/to/code --prune-imports --output extracted.py
```

生成论文复现简报（含模块卡片与假设清单）：

```bash
python scripts/build_reconstruction_brief.py paper.txt --output brief.md
```

验证导出的模块（独立性 / 无未定义名称 / 无硬编码 CUDA / 无联网下载 / 运行烟雾测试）：

```bash
python scripts/check_module_contract.py extracted.py --run --python python
python scripts/compare_module_interface.py path/to/code/blocks.py extracted.py --fidelity
```

## 回归测试

自包含测试（不依赖私有样例；未安装 PyTorch 时相关用例自动跳过）：

```bash
python -m unittest discover -s tests -v
```

完整回归（可选传入私有 `样例/` 目录）：

```bash
python plug-and-play-module-extractor/scripts/run_regression_suite.py [样例] --python python --report-json regression-report.json
```

## 注意

- 本仓库只提交 skill 本体、脚本和合成测试样例，不提交真实论文/代码案例。
- paper-derived 复现必须标明 `paper-derived` 并附假设清单，不能声称来自官方源码。
- 源码抽取优先保留原实现，只移除训练、数据、日志、checkpoint、配置系统、注册器等非模块逻辑。
- 下载的 PDF / 仓库一律视为不可信数据，只读取不执行。
