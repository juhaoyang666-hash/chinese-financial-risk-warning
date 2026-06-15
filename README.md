# 中文金融舆情风险智能预警系统

学号：10225001426  
姓名：杨巨豪  
课程大作业题目：基于 Qwen3-8B QLoRA 的中文金融舆情风险智能预警系统

## 说明

本项目面向中文金融风控场景，研究金融新闻、投诉文本和社交媒体文本中的负面主体风险识别。核心任务是：判断文本是否包含针对某个金融主体的负面风险信息，并抽取风险指向的主体。项目在 FINNSP 数据集上比较传统机器学习、中文编码器、金融领域编码器和 Qwen3-8B QLoRA 指令微调模型，并进一步扩展为一个包含风险事件、主体画像、审核队列、API、审核台和漂移监控原型的风控预警系统。

正式论文主要使用 LaTeX 撰写，以 PDF 版为正式排版和阅读基准；Word 版按课程提交要求由 LaTeX 源码生成，便于老师直接打开查看。提交压缩包根目录下会直接放置：

- `10225001426_杨巨豪_基于Qwen3-8B_QLoRA的中文金融舆情风险智能预警系统.pdf`
- `10225001426_杨巨豪_基于Qwen3-8B_QLoRA的中文金融舆情风险智能预警系统.docx`

`paper/` 目录中同时保留 LaTeX 源码、PDF 和 Word 原始生成文件，便于检查。

## 项目结构

```text
.
├── README.md                         # 项目说明与轻量复现指南
├── 作业要求.txt                       # 课程大作业要求
├── requirements.txt                  # Python 依赖
├── paper/
│   ├── main.tex                      # LaTeX 论文源码
│   ├── main.pdf                      # 论文 PDF
│   ├── main.docx                     # 论文 Word 版
│   └── README.md                     # 论文编译说明
├── scripts/                          # 数据处理、训练、评估、制图、打包脚本
├── src/risk_nlp/                     # 核心 schema、指标、风险事件、数据库与评分逻辑
├── app/                              # FastAPI 服务与 Streamlit 审核台
├── tests/                            # 单元测试
├── data/processed/
│   ├── dataset_stats.json            # 数据统计
│   └── finnsp_sample_redacted.jsonl  # 少量脱敏样例
├── outputs/
│   ├── results_table.csv/md          # 主实验结果表
│   ├── figures/                      # 论文图表
│   ├── analysis/                     # 错误分析与级联模拟
│   ├── external/                     # FLARE 外部泛化结果
│   ├── robustness/                   # 反事实压力测试
│   ├── monitoring/                   # 漂移监控快照
│   └── tabular_risk/                 # OpenML 信用违约风控实验
└── dist/
    └── 10225001426_杨巨豪_大作业.zip  # 最终提交包
```

提交包不包含模型权重、全量训练数据、SQLite 数据库或预测明细文件。

## 主要结果

FINNSP 验证集主任务结果：

| 模型 | Accuracy | F1 | Macro-F1 | Entity-F1 |
|---|---:|---:|---:|---:|
| TF-IDF + LR | 0.9300 | 0.9331 | 0.9299 | 0.0000 |
| TF-IDF + SVM | 0.9500 | 0.9522 | 0.9499 | 0.0000 |
| Chinese RoBERTa | 0.9540 | 0.9557 | 0.9539 | 0.0000 |
| Chinese MacBERT | 0.9580 | 0.9597 | 0.9579 | 0.0000 |
| FinBERT2-large | 0.9620 | 0.9638 | 0.9619 | 0.0000 |
| Qwen3-8B zero-shot | 0.9060 | 0.9150 | 0.9049 | 0.5526 |
| Qwen3-8B few-shot | 0.9340 | 0.9392 | 0.9335 | 0.6273 |
| Qwen3-8B QLoRA | 0.9720 | 0.9732 | 0.9719 | 0.8331 |

其他结果摘要：

- 主模型 invalid JSON rate 为 `0.0000`。
- 实体在文率为 `0.9864`，幻觉实体率为 `0.0136`。
- FinBERT2 初筛 + Qwen3-8B QLoRA 复核的离线级联模拟中，LLM 调用率为 `0.5320`，F1 为 `0.9710`，Entity-F1 为 `0.8386`。
- FLARE-zh-NSP 外部二分类泛化中，Qwen3-8B QLoRA 取得 Accuracy `0.9700`、F1 `0.9715`、Macro-F1 `0.9699`。该外部集缺少实体级 gold，因此不报告外部 Entity-F1。
- OpenML/UCI 信用卡违约 tabular 风控实验中，LightGBM 取得 AUC `0.7753`、KS `0.4283`、PSI `0.0024`。

## 轻量复现

以下命令用于老师快速检查代码可运行性和结果文件生成逻辑。完整 Qwen3-8B QLoRA 训练需要本地模型和多卡 GPU，不建议在批改时重新训练。

### 1. 安装依赖

```bash
conda activate /data1/yangjuhao/envs/llm
pip install -r requirements.txt
```

### 2. 运行单元测试

```bash
python -m unittest discover -s tests
```

### 3. 编译论文 PDF

```bash
cd paper
make
cd ..
```

### 4. 生成 Word 版论文

```bash
python scripts/build_word_doc.py
```

输出文件：

- `paper/main.pdf`
- `paper/main.docx`

### 5. 重新汇总结果和图表

```bash
python scripts/collect_results.py
python scripts/make_figures.py --predictions outputs/qwen3-8b-qlora-main-eval/predictions.jsonl
python scripts/analyze_predictions.py
python scripts/simulate_cascade.py
python scripts/monitor_drift.py
```

### 6. 重新打包提交文件

```bash
python scripts/build_submission.py \
  --student_id 10225001426 \
  --name 杨巨豪 \
  --kind 大作业
```

输出：

- `dist/10225001426_杨巨豪_大作业.zip`

## 关键脚本说明

| 脚本 | 用途 |
|---|---|
| `scripts/prepare_data.py` | 下载并处理 FINNSP 数据，将原始标签统一为 JSON schema |
| `scripts/train_encoder.py` | 训练 TF-IDF、SVM、RoBERTa、MacBERT、FinBERT2 等基线 |
| `scripts/train_llm_qlora.py` | Qwen3-8B QLoRA 指令微调 |
| `scripts/evaluate_llm.py` | LLM zero-shot、few-shot、QLoRA 评估 |
| `scripts/analyze_predictions.py` | 可靠性与错误分析 |
| `scripts/simulate_cascade.py` | FinBERT2 到 Qwen3 的级联风控模拟 |
| `scripts/build_risk_events.py` | 构造风险事件弱标签 |
| `scripts/build_entity_profiles.py` | 生成主体风险画像 |
| `scripts/monitor_drift.py` | 生成离线漂移监控快照 |
| `scripts/train_tabular_risk.py` | OpenML 信用违约 tabular 风控实验 |
| `scripts/build_word_doc.py` | 从 LaTeX 源码生成 Word 论文 |
| `scripts/build_submission.py` | 生成最终提交压缩包 |

## 系统演示

项目包含 FastAPI 服务和 Streamlit 审核台。若需要本地演示，可先确认 Qwen3-8B 基座模型和 LoRA adapter 路径存在，再运行：

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
streamlit run app/dashboard.py --server.port 8501
```

API 路由包括：

- `POST /score`
- `POST /batch_score`
- `GET /entity/{name}`
- `GET /review_queue`
- `GET /metrics`

论文中使用的审核台截图位于 `outputs/demo_screenshots/streamlit_dashboard_home.png`。

## 注意事项

- 提交包只包含代码、论文、脱敏样例、指标、图表和说明文件。
- 不提交模型权重、adapter 权重、checkpoint、全量数据、SQLite 数据库和完整预测明细。
- 风险类型、严重程度和风险评分属于风控扩展弱标签，不是人工标注多分类任务。
- 漂移监控是基于 FINNSP train/eval 的离线原型，不等同于生产环境时间序列监控。
