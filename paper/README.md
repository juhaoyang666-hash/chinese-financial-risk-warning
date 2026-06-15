# LaTeX 论文编译说明

主论文源码：`paper/main.tex`

本文主要使用 LaTeX 写作，`paper/main.pdf` 为正式排版和阅读基准；`paper/main.docx` 是按课程提交要求从 LaTeX 源码生成的 Word 版本。

当前服务器已安装 `tectonic`，推荐使用下面命令编译：

```bash
cd /data1/yangjuhao/NLP/paper
XDG_CACHE_HOME=/data1/yangjuhao/NLP/.tectonic-cache tectonic -X compile main.tex
```

或使用 Makefile：

```bash
cd /data1/yangjuhao/NLP/paper
make
```

如果本机已安装完整 TeX Live，也可以使用 XeLaTeX 编译中文论文：

```bash
cd /data1/yangjuhao/NLP/paper
xelatex -interaction=nonstopmode main.tex
xelatex -interaction=nonstopmode main.tex
```

输出：

- `paper/main.pdf`
- `paper/main.docx`（使用 `python scripts/build_word_doc.py` 从 LaTeX 源码生成）

当前论文源码使用 `ctexart` 文档类，适合中文学术论文。若换机器后提示 `tectonic: command not found` 或 `xelatex: command not found`，需要先安装 Tectonic 或 TeX Live 的中文支持组件，例如 `texlive-xetex`、`texlive-lang-chinese`、`latexmk`。

生成 Word 提交版：

```bash
cd /data1/yangjuhao/NLP
python scripts/build_word_doc.py
```
