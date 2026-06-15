# 风控文本漂移监控

| 指标 | baseline | current | delta |
|---|---:|---:|---:|
| 样本数 | 4000 | 500 | - |
| 正例率 | 0.5447 | 0.5060 | -0.0387 |
| 平均文本长度 | 224.6302 | 242.2900 | 17.6598 |
| 平均实体数 | 0.9407 | 0.8420 | -0.0988 |
| 幻觉实体率 | N/A | 0.0136 | - |
| invalid JSON rate | N/A | 0.0000 | - |

- baseline 来源：FINNSP train gold weak events。
- current 来源：FINNSP eval gold weak events + Qwen3 reliability metrics。
- 是否建议重训：false。
- 该报告是离线 train/eval 快照监控原型，不等同于生产环境时间序列漂移监控。

```json
{
  "baseline": {
    "source": "FINNSP train gold weak events",
    "num_records": 4000,
    "positive_rate": 0.54475,
    "avg_text_length": 224.63025,
    "avg_entity_count": 0.94075,
    "hallucination_rate": null,
    "invalid_json_rate": null,
    "risk_level_distribution": {
      "high": 2801,
      "medium": 962
    }
  },
  "current": {
    "source": "FINNSP eval gold weak events + Qwen3 reliability metrics",
    "num_records": 500,
    "positive_rate": 0.506,
    "avg_text_length": 242.29,
    "avg_entity_count": 0.842,
    "hallucination_rate": 0.013605442176870748,
    "invalid_json_rate": 0.0,
    "risk_level_distribution": {
      "high": 321,
      "medium": 100
    }
  },
  "positive_rate_delta": -0.03874999999999995,
  "avg_text_length_delta": 17.659750000000003,
  "avg_entity_count_delta": -0.09875,
  "retrain_recommendation": false,
  "monitoring_scope": "This is an offline train/eval snapshot for the demo risk system, not a production time-series drift monitor."
}
```
