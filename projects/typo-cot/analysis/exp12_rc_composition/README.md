# 実験12: R_C 組成分析 (H12, M2 測定)

各設定 (31 = 既存25 + MATH6) の clean 側 R_C top-10 を 4 カテゴリに分類し、
設定レベルの組成シェアを Δρ(top10)・削除RD と回帰/相関する。

## R_C ランキング (再構築ローダー)
`rc_word_ranking_from_cot_pt` (exp/02 commit fef3958)。`_cot.pt` の CoT 領域語を
**符号付きスコア降順**で並べ、上位 10 エントリを採る。
**Mistral は必須の再構築ローダー**: word_scores が空白マーカー欠落で 1 語に結合
(degenerate) するため、`token_scores` を `full_text = build_prompt + generated_text` に
**貪欲整列**して語ランキングを再構築 (`word_scores_degenerate` で自動判定・切替)。
本実験の 31 設定中、Mistral 6 + Llama/Gemma/Qwen の一部 (計 22) で再構築が発火。

## 4 カテゴリ操作的定義 (rc_classifier.py, POS 代理 + テンプレートマッチ)
top-10 の各エントリを空白分割し構成語ごとに判定、優先順位
**conclusion > numeric > content > function** で 1 エントリ 1 カテゴリに割当
(10 スロット上でシェア和 = 1.0。演算記号のみのエントリ "=", "*" は分類対象外)。
- **conclusion (結論句定型)**: 答え宣言語彙 {answer, correct, final, therefore,
  thus, hence, conclusion, boxed, option, choice} / MC 選択肢ラベル (`(C).`,`D)` 等) /
  改行直後の答え行リード (`...\nThe answer is` の The/So/Therefore)。
- **numeric (数値)**: 英字を含まない数字 (通貨・桁区切り・小数・% 許容)。
- **content (内容語)**: 機能語ストップリスト外の英字語 (名詞/動詞/形容詞/副詞の代理)。
- **function (機能語)**: ストップリスト該当語・句読点のみ。

## 結果 (rc_composition_by_setting.csv, correlations.json)

### 組成の実像
R_C top-10 は **内容語 (reasoning 語) が支配** (share_content 0.25–0.76)。
結論句定型は少数派だが家系で明瞭に差:

| 家系 | conclusion シェア(平均) | Δρ(top10) 平均 | Δρ>0 割合 |
|---|---|---|---|
| Gemma | 0.169 | +0.252 | 92% |
| Llama | 0.136 | +0.383 | 83% |
| Mistral | **0.012** | **−0.088** | 17% |

### 事前登録判定 (verdict)
| 予測 | 閾値 | 結果 | 判定 |
|---|---|---|---|
| Gemma/Llama MC 結論句 | >0.5 | 平均 0.130, 0/16 | **不成立** |
| Mistral 結論句 | <0.3 | 平均 0.012, 6/6 | 成立 (自明) |
| GSM8K/MATH 数値+内容 | >0.7 | 平均 0.693, 5/11 | 部分 |
| \|r(結論句, Δρ)\| 全31 | ≥0.7 | 0.184 | **不成立** |

### 機構は成立方向 (閾値は誤較正)
- **MC 設定に限定**すると `r(結論句シェア, Δρ(top10)) = +0.705` (n=20, p=0.0005) で
  **|r|≥0.7 を満たす**。全31 で薄まるのは GSM8K/MATH (答え定型が数値=numeric に流れ、
  結論句軸が意味を持たない) が混入するため。
- 削除RD は事前登録の「内容語+数値」とは無相関 (r=−0.001) だが、**結論句シェアと
  r=+0.516 (p=0.008)** で連動。すなわち削除感受性も答え定型帰属と結びつく。

**→ H12 は「強形(結論句が top-10 の過半を占める)」としては REFUTED。**
一方、**機構的主張「fixed 後の相関変化 Δρ は帰属が答え定型へ張り付く度合いと連動する」は
MC 領域 (r=0.71) と家系対比 (Gemma/Llama=正 Δρ・高結論句 / Mistral=負 Δρ・ほぼ0結論句) で
方向的に支持**。閾値 (>0.5) は「R_C top-10 が答え定型で満たされる」という前提が過大で、
実際は答え定型は少数派だが Δρ を弁別する軸として機能する、と修正すべき。

## 生成物
- `rc_composition_by_setting.csv` … 31 設定 × 4 カテゴリシェア + Δρ + 削除RD
- `correlations.json` … 回帰/相関 (all31 / MC / family_contrast) + 判定
- `rc_top10_examples.json` … 各設定の代表 top-10 と分類 (監査用)
- `rc_classifier.py` … 4 カテゴリ分類器  / `run_exp12.py` … パイプライン

## 再現
```
python run_exp12.py   # torch で _cot.pt を読み分類・集計 (GPU 不要, CUDA_VISIBLE_DEVICES="")
```
補足: 設定あたり最大 500 サンプルでシェア推定 (`MAX_SAMPLES`)。Δρ は exp/04
delta_rho_table.json、削除RD は exp/02 `<setting>_core/summary.json` の
k=4 (top_rc_unrestricted − stratum_matched_random) をソース直接参照。
