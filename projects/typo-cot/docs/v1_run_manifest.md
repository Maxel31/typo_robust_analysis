# v1 実行アーカイブ

**実行日時**: 2026-05-20 13:28 - 17:30 (JST)
**ログ**: `figures_20260520_230228/pipeline_run.log`

## モデル × ベンチマーク

| モデル | gsm8k | mmlu | mmlu_pro | arc | commonsense_qa |
|---|---|---|---|---|---|
| Llama-3.2-1B-Instruct | ✓ | ✓ | ✓ | ✓ | ✓ |
| Llama-3.2-3B-Instruct | ✓ | ✓ | ✓ | ✓ | ✓ |
| gemma-3-1b-it | ✓ | ✓ | ✓ | ✓ | ✓ |
| gemma-3-4b-it | ✓ | ✓ | ✓ | ✓ | ✓ |
| Mistral-7B-Instruct-v0.3 | ✓ | ✓ | ✓ | ✓ | ✓ |

合計 5 モデル × 5 ベンチマーク = 25 ペア × 6 摂動条件 (k=1/2/4/8 importance + k=4 random + k=4 bottom_k)

## 摂動条件

- LXT (importance) × k ∈ {1, 2, 4, 8}
- Random × k=4
- Bottom-k (Anti-LXT) × k=4

## 注意

- **`analysis_20260520_230228/`**: 旧 analyzer による出力（回答スパン未検出フィルタ未適用）
- **`figures_20260520_230228/`**: 旧 analyzer 由来の Figure/Table（Table 5 は gsm8k/mmlu/mmlu_pro のみ）

## 後続版（v2）

`outputs/baseline/`, `outputs/perturbed/`, `outputs/analysis/`, `outputs/figures/` 配下に
新しい結果を生成する。v2 では:
- 拡張モデル（Gemma-3 12B/27B, Qwen2.5 0.5B/1.5B/3B/7B/32B）を追加
- analyzer 側の回答スパン未検出フィルタ適用済み
- `outputs/figures/exclusion_summary.{csv,tex}` 追加
- Table 5 に arc/commonsense_qa を含める
