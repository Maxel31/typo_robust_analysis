# フォローアップ実行計画 (ユーザー考察 2026-07-19)

本書はユーザー考察(2026-07-19)で提示された残宿題7件を、担当トラック別に整理した
実行計画である。各項目に目的・データソース・完了基準を記す。Track A(本エージェント)
の3件は本セッションで完了し、確定表を docs / analysis に統合済み(§Track A 参照)。

## トラック分担

| 宿題 | 内容 | トラック | 状態 |
|---|---|---|---|
| 1 | 実験6 ρ(J_method\|R) 保持表の完結 + H6 判定 | **Track A** | 完了 |
| 2 | 実験7 R_Q 偏在の可視化と統合 (H7-4 判定) | **Track A** | 完了 |
| 5 | 実験3 相補性の実証補強の統合 (前半/後半分布) | **Track A** | 完了 |
| 3 | (Track B 担当) | Track B | 本書の対象外 |
| 4 | (Track B 担当) | Track B | 本書の対象外 |
| 6 | (Track C 担当) | Track C | 本書の対象外 |
| 7 | Tsuji+2025 関連記述の全面書き直し(実験8の早期層局在確定を反映) | 執筆トラック | 本書の対象外 |

注: 宿題3・4・6・7 の詳細定義はユーザー考察本文および各トラックの担当エージェントに
帰属する。本書は Track A(宿題1・2・5)の目的・データソース・完了基準・確定結果を
正典として記録する。

---

## Track A: 宿題1 — 実験6 ρ(J_method|R) 保持表の完結(最優先)

**目的**: 「内的ランキング安定性 J@10 と出力頑健性の相関(実験4の ρ(J|R))が
帰属手法 (AttnLRP=R_C / G×I / IG / rollout) と帰属フリーの LOO を替えても
保持されるか」= H6 (Attribution Method Convergence) を判定する。

**データソース**:
- 帰属ファミリー: exp-06-attribution worktree `results/attribution_family/*/results.json`
  (36シャード、n=300/seed=42、IG は m=16)。集計済み JSON は同
  `results/attribution_family/aggregate_attribution_family.json`(コミット 1ea65e3)。
- LOO: 同 worktree `results/loo/{setting}_{clean,lxt4}_{occ,type}/results.json`
  (16ディレクトリ = M3×B2×{clean,lxt4}×occ の 12 + Gemma×B2×{clean,lxt4}×type の 4)。
- R (出力頑健性) と flip: アーカイブ
  `outputs/analysis/{bench}/{model}/k4_importance/full_results.json` の
  `sample_results[].{answer_changed(=flip), cot_metrics.rouge_l.f1(=R),
  cot_metrics.jaccard.top10(=R_C の J@10)}`。

**手続き(検証で判明した重要補正)**:
- exp-06 の既存集計(analyze_attribution_family.py / analyze_loo_rankings.py)は
  `ρ(J_method@10|R)` を **Spearman(J_method@10, ROUGE-L)** として算出していた
  (dev_notes 表2、0.55〜0.85、18/18有意)。これは実験4/Step0 の ρ(J|R) =
  「flip を目的変数、ROUGE-L を統制した J@10 の**偏相関**(残差+Pearson)」とは
  別統計であることを確認した(記法の衝突)。
- そこで全手法(R_C / G×I / IG / rollout / LOO)について、per-sample J@10 を
  archive の flip・ROUGE-L と sample_id 結合し、Step0 `reproduce.py::_partial_corr`
  と同一手続きで **partial_corr(J@10, flip | ROUGE-L)** を再算出。Holm(m=30)を適用。
- 既存の Spearman(J, ROUGE-L) は二次統計として併記(dev_notes 表2を完全再現し検証)。

**完了基準**: 手法×設定の ρ保持表(偏相関 + Holm)+ H6 判定の明言。
出力 `analysis/exp6_rho_preservation/`(build_rho_preservation.py / preservation_table.csv
/ summary.json / README.md)+ exp-06 dev_notes 追記。

**確定結果(§実験6 all_results_by_setting.md, README.md 参照)**:
- **R_C 参照**: 偏相関は全6設定で負・Holm有意(n300 で −0.24〜−0.62)。実験4を再現。
- **LOO(帰属フリー)**: 全6設定で負・Holm有意(−0.18〜−0.44)。R3 の leave-one-out
  要求への最終回答 = 相関構造は AttnLRP 固有ではない。**H6 の中核証拠**。
- **勾配系 (IG / G×I)**: 自由記述 GSM8K では負・有意(IG 3/3, G×I 2/3)だが、
  多肢選択 MMLU では減衰し非有意。
- **rollout**: 偏相関は全6設定でほぼ0(Holm有意 0/6)。Spearman(J,ROUGE) は最大
  (0.75〜0.85)だが flip に対する追加説明力は無い。
- **判定: H6 = 条件付き支持**(符号は 21/24 で保持、偏相関の完全な符号+有意保持は
  LOO 6/6・勾配系は GSM8K のみ、rollout は棄却)。詳細は hypothesis_registry.md H6。

---

## Track A: 宿題2 — 実験7 R_Q 偏在の可視化と統合

**目的**: 校正器の残余ギャップ(ボトルネック)が、タイポが帰属重み R_Q を強く
担う語に集中するか = H7-4(Corrector Bottleneck の R_Q 偏在サブ仮説)を判定する。

**データソース**: `analysis/exp7_tables/`(2026-07-18 作成)
- `rq_mannwhitney.csv`(100行 = 75設定×校正器 + 25プール、Holm m=25)
- `restoration_rates.csv` / `flip_rates.csv`
- R_Q = 校正後生成 results.json `perturbed_tokens[].importance_score`(語単位は
  貪欲・語順保存マッチの最大値)。

**手続き**: 既存 CSV の数値を再確認(検証)し、プール25設定の Holm有意数・AUC
分布・失敗語 vs 復元語の R_Q 中央値差を集計、md へ統合。

**完了基準**: R_Q 偏在表 + H7-4 判定文。

**確定結果(検証済み)**: プール25設定中 Holm有意 **17/25**、AUC = P(R_Q_failed >
R_Q_restored) 平均 **0.539**(>0.5 が **23/25**)、median(R_Q failed − restored)
平均 **+0.050**。→ **H7-4 = 支持(方向一貫、効果は小)**。復元失敗語は復元語より
R_Q が高い側に偏るが、AUC≈0.54 と効果量は小さく、「高 R_Q への集中」は
決定的でなく確率的偏り。ボトルネックの主因は復元率そのもの(neural 0.886 >
llm 0.734 > spellfix 0.663)であり、byte-identical 復元で flip=0% が確定的証拠。

---

## Track A: 宿題5 — 実験3 相補性の実証補強の統合

**目的**: KL ピーク(摂動伝播の起点、CoT 前半)と R_C 上位(答え決定点、CoT 後半)
の空間的相補性(H3 修正版 Complementarity)を、設定別の前半/後半分布表として
実証補強し統合する。

**データソース**: `analysis/exp3_kl_rc_spatial/`(Phase A-2 作成、全60設定)
- `quadrant_table.csv`(60設定 × {kl_front_frac, kl_back_frac, rc_front_frac,
  rc_back_frac, overlap_coefficient, wasserstein_distance})
- `complementarity_stats.json` / `summary.json`

**手続き**: 既存数値を再確認(検証)し、考察の要求形式(設定別の前半KL率/後半R_C率)
に整えて md へ統合。

**完了基準**: 空間分布表(前半/後半)+ Complementarity 判定の再確認。

**確定結果(検証済み)**: 60設定。KL 大域平均位置 **0.359** vs R_C **0.541**
(Mann-Whitney p≈0)。設定別平均 前半KL率 = importance **0.749** / random 0.606、
後半R_C率 = **0.620**(両条件)。相補パターン(KL前半優位 かつ R_C後半優位)= **48/60
設定**(importance 24/30, random 24/30)。Wasserstein importance **0.285** / random
0.168(構造化算術ほど分離大)。→ **H3(Complementarity)= 支持(再確認)**。

---

## md 統合(全宿題共通)

- `docs/all_results_by_setting.md`: 実験6節に ρ保持表、実験7節に R_Q偏在表、
  実験3節に空間分布(前半/後半)表を追記(完了)。
- `docs/experiment_details.md`: 上記3件の手続き・判定を追記(完了)。
- `docs/hypothesis_registry.md`: H6 判定を Pending → 条件付き支持に更新、
  H7-4 / H3 の Track A 再確認を反映(完了)。

## 規約遵守

- GPU 不使用(全て CPU 集計)。
- 数値はソース(results.json / CSV / archive full_results.json)から直接転記。
- 他 worktree は読み取りのみ(exp-06 の dev_notes 追記のみ例外的に許可)。
- コミットは本体チェックアウトで実施、push なし。
