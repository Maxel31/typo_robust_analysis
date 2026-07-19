# 実験16 骨格: 統一 GLMM 用 features_partial

後続の統一 GLMM
```
flip ~ repair_min + KL_sum + fixed_Jaccard + ROUGE
       + 設定レベルモデレーター + (1|item) + (1|setting)
```
に投入できる **サンプル×特徴** 中間データフレーム (partial)。

## features_partial.parquet
- **85,802 行 / 60 設定** (6 model × (arc,csqa,gsm8k,math,mmlu,mmlu_pro) × {importance,random})
- 全特徴が揃う included サンプル (repair+KL+ROUGE): **34,190 行**

### 列 (由来)
| 列 | 由来 | 意味 |
|---|---|---|
| `setting,model,benchmark,condition,family,task_type,is_core` | exp11/12 | キー・層 |
| `sample_id,included,flip,cot_changed` | exp11 (exp01_03) | item キー・アウトカム |
| `repair_min,repair_mean` | exp11 (exp9 word_rows) | S1/G 内部修復 (弱リンク/平均) |
| `kl_sum` | exp11 (exp01_03 divergence) | S2 分岐量 |
| `zipf_mean,split_mean,rq_mean,n_words` | exp11 | 摂動語統制 |
| `rouge_l_f1` | Step0 `cot_rouge_l_f1` | **ROUGE** (clean vs typo CoT) |
| `cot_jaccard_top10` | Step0 | **Jaccard** (R_C top10 の clean vs typo 重なり) |
| `share_conclusion/numeric/content/function`, `share_content_plus_numeric` | exp12 | 設定モデレーター (組成) |
| `delta_rho_top10,deletion_rd_k4` | exp12 (exp04/exp02) | 設定モデレーター |

### 非 null カバレッジ
`repair_min` 1.00 / `kl_sum` 0.69 / `rouge_l_f1` 0.68 / `cot_jaccard_top10` 0.68 /
`share_conclusion` 0.78 / `delta_rho_top10` 0.78 / `deletion_rd_k4` 0.74。
(kl_sum は included かつ divergence dump のあるサンプルのみ。設定モデレーターは
Δρ/削除RD が存在する model×benchmark のみ。)

## 未追加の特徴 (後続実験で結合)
- **Gini**: 実験13 完了後に追加 (R_C 集中度)。
- **noCoT_flip**: 実験14 完了後に追加 (CoT 無し条件の flip)。
- **fixed_Jaccard (fixed-target 版)**: 本 partial では Step0 の default Jaccard
  (`cot_jaccard_top10`, typo vs clean) を暫定投入。exp/04 fixed-target の
  `j_fixed` (per-sample) を厳密投入する場合は
  `exp-04-fixed-target/.../prod{,_math}/analysis/<bench>/<model>/k4_fixed_target/full_results.json`
  から結合すること。

## 再現
```
python ../exp11_chain_mediation/run_exp11.py   # exp11_sample_table.parquet を先に生成
python ../exp12_rc_composition/run_exp12.py     # rc_composition_by_setting.csv を先に生成
python build_features.py                        # 結合 → features_partial.parquet
```
Qwen 行は exp11 の dedup-on `included` を継承済み。
