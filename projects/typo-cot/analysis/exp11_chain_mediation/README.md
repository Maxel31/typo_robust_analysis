# 実験11: 連鎖媒介分析 (H11) — 修復 → 分岐(KL_sum) → flip

統一仮説 ERDC 連鎖の **S1/G(内部修復) → S2(分岐) → 読み出し(flip)** リンクを、
既存のサンプル別データを結合して接続する。

## データソース (読み取り専用)
- **repair(サンプル×摂動語)**: `exp-09-inner-repair/.../results/exp9/word_rows_*.jsonl`
  各行 = 1 摂動語 (`repair_score`, `r_q`, `zipf_freq`, `split_increment`, `sample_id`)。
- **KL_sum(サンプル)**: `exp-01-03-transplant/.../exp01_03/<setting>/divergence/<sid>.json` の `kl_sum`。
- **flip(サンプル)**: 同 `<setting>/outcomes.json`。TE flip = `answers["B"] != answers["A"]`
  (B=(typo,typo), A=(clean,clean))。included = `not exclude and a_correct`。

## サンプル特徴 (word_rows をサンプルで集約)
- `repair_min` = 4 摂動語の**最小**修復スコア (弱リンク仮説, 主予測子)
- `repair_mean` = 平均 (感度分析)
- 統制: `zipf_mean`, `split_mean`(分割増分), `rq_mean` (摂動語平均)

## 設定内 2 段回帰 (予測子は設定内 z 化)
1. 第1段 (OLS): `KL_sum ~ repair_min + Zipf + split + R_Q`  → `stage1_repair_coef`
2. 第2段a (logit): `flip ~ repair_min + 統制`  → 総効果 a
3. 第2段b (logit): `flip ~ repair_min + KL_sum + 統制`  → 直接効果 a′
4. **媒介率** = (a − a′) / a  (KL_sum 投入による repair 係数の減衰率)

横断: `BinomialBayesMixedGLM` で `flip ~ repair_z (+ kl_sum_z) + (1|setting)`、
および設定固定効果ロジット (ロバストネス) で pooled 媒介率。

## 事前登録判定と結果 (mediation_pooled.json)

**判定基準**: 第1段が負に有意が過半 かつ repair 直接効果が KL_sum 統制で ≥50% 減衰。

| グループ | 設定数 | 第1段 負×有意 | 媒介率 中央値 | pooled GLMM 媒介率 |
|---|---|---|---|---|
| **core5 (主分析)** | 50 | 35 (**70%**) | **0.523** | **0.577** |
| core5 × MC | 30 | 28 (**93%**) | 0.635 | — |
| all (Qwen 含む) | 53 | 36 (68%) | 0.524 | 0.581 |
| 感度: repair_mean (core5) | — | — | — | 0.638 |

**→ H11 は SUPPORTED**。修復(弱リンク)の flip への効果の約 **58%** が分岐(KL_sum)を
経由する。KL_sum の flip 係数は +0.505 (強正: 分岐大 → flip 大)。MC タスクで特に強い
(第1段 93% が負有意)。→ **S1/G → S2 接続を支持**。

### 反証・境界条件
- **MATH シャード (11 設定)**: 第1段 負有意 0/11、媒介率中央値 **−0.40**。
  MATH では repair↑ が KL_sum を下げず、媒介が成立しない。すなわち MATH では
  「修復は分岐を介さず読み出し段に直接効く/別経路」と記録すべき反例。
- MATH 除外の core5 (MC+GSM8K) が主結論を担う。

## Qwen の扱い (Track C dedup 改修)
exp-01-03 の Qwen `outcomes.json` は multi_trigger 過剰除外バグ (改修前) のため
included が崩壊 (arc で 4)。Track C 改修 (commit 4052b2c,
`build_cell_inputs(dedup_same_answer_triggers=True)`) の per-sample dedup-on `exclude`
を `dump_qwen_dedup_exclude.py` で再生成し (`qwen_dedup_exclude.json`)、Qwen 設定の
included 判定を上書き。exclude 数は Track C 集計と一致 (arc: 500)。flip/a_correct は
dedup で不変。Qwen は検証扱いで主判定 (core5) には非算入。

## 生成物
- `exp11_sample_table.parquet` … サンプル×特徴 (85,802 行, 60 設定)
- `mediation_by_setting.csv` … 設定別 2 段係数・媒介率
- `mediation_pooled.json` … pooled/GLMM + 判定
- `dump_qwen_dedup_exclude.py` / `qwen_dedup_exclude.json` … Qwen dedup-on 上書き

## 再現
```
python dump_qwen_dedup_exclude.py   # Qwen dedup-on exclude 再生成 (CPU)
python run_exp11.py                 # 結合・回帰・判定
```
