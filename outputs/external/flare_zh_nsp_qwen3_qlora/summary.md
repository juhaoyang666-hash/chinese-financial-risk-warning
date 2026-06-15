# FLARE-zh-NSP Qwen3-8B QLoRA 外部泛化评估

该实验使用 FINNSP 上微调得到的 Qwen3-8B QLoRA 主模型，对 FLARE-zh-NSP test 集进行真实模型推理。FLARE-zh-NSP 在当前处理流程中只提供二分类有/无标签，因此本结果只报告二分类外部泛化指标；模型生成的实体保留在预测文件中用于质检，但不计算 Entity-F1。

- 数据集：ChanceFocus/flare-zh-nsp，test，500 条，正例 259 条，负例 241 条。
- 模型：Qwen3-8B QLoRA 主模型，adapter `outputs/qwen3-8b-qlora-main/adapter`。
- 设置：关闭 thinking，max_new_tokens=128，zero-shot 外部推理。
- fallback_demo_records=0。

| 方法 | Accuracy | Precision | Recall | F1 | Macro-F1 | invalid JSON rate |
|---|---:|---:|---:|---:|---:|---:|
| 关键词弱监督 baseline | 0.8760 | 0.8774 | 0.8842 | 0.8808 | 0.8758 | - |
| Qwen3-8B QLoRA real inference | 0.9700 | 0.9552 | 0.9884 | 0.9715 | 0.9699 | 0.0000 |

混淆矩阵：TP=256，FP=12，TN=229，FN=3。相对关键词弱监督 baseline，Qwen3-8B QLoRA 的 F1 提升 0.0908，Accuracy 提升 0.0940。

平均单条推理耗时为 1.2245 秒。模型在 268 条样本中输出了非空实体，但由于外部集缺少实体 gold，该项不作为实体级评估结论。
