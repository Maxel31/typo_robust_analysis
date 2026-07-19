# 全設定別 実験結果一覧 (2026-07-19 時点)

結果ファイルからの直接抽出。ソース: 各 worktree の results/ 配下。
実験1+3/2/6(LOO)/10 の表は `analysis/all_results_extract/extract_tables.py`
(読み取り専用) で再抽出可能 (出力: 同ディレクトリ `extracted_tables.md`)。

> **2026-07-19 更新**: Phase B(ERDC 拡張)の新規完了分 **実験11(連鎖媒介)・実験12(R_C組成)・
> 実験13(読み出し集中度)・実験14(no-CoT ショートカット)・実験17(行動修復)**、
> **敵対的レビュー A(A1 Mistral監査 / A2 restore反証)・B(B1/B4/B5/B6)・C(実験6 ρ保持 修正版)**、
> **防御実験 D1(重要語優先校正オラクル, データ構築完了)** を追記(本ファイル末尾の「Phase B / ERDC 拡張」節)。
> 数値は各 `analysis/` 配下の JSON/CSV から直接転記。既存の確定判定・表は不変。

## 目次

- [Step 0: master table](#step-0-master-table-規模と再現検証)
- [実験4: fixed-target Δρ](#実験4-fixed-target-δρ-25設定-ktop10) / [MATH拡張](#実験4-拡張-math-500-δρ-6モデル-ktop10-b10000)
- [実験5: matched統制 McNemar](#実験5-matched統制-mcnemar-25設定)
- [実験1: CoT移植 4セル分解 (50設定)](#実験1-cot移植-4セル分解-50設定-mmlu-は-p0p1-統合)
- [実験3: KL divergence プロファイル (50設定)](#実験3-kl-divergence-プロファイル-50設定)
- [実験2: コア対比 (25設定)](#実験2-コア対比-25設定-無制限top-r_c-vs-層内マッチ統制-delete)
- [実験7: 校正器3段 精度](#実験7-校正器3段-精度-25設定3校正器) / [within-run 検証](#実験7-補遺-within-run-byte-identical-検証-75設定--25設定3校正器-2026-07-19-完走)
- [実験9: repair係数 (50設定)](#実験9-設定別-repair係数-50設定-clean正解条件付き)
- [実験8: activation patching 3モデル統合](#実験8-activation-patching-3モデル統合-m3b22条件--12条件-s2-kl-recovery)
- [実験10: スコープ拡張 (R1/Qwen/MATH/自然typo)](#実験10-スコープ拡張)
- [実験6: 帰属手法比較 + LOO](#実験6-帰属手法ファミリー比較-m3b22条件3手法--36シャード-2026-07-1819)
- [GLMM 最終推定](#glmm-最終推定-実験1--実験5-pooled-2026-07-18)
- [実験1+3 / 実験9 拡張グリッド (進行中)](#実験13--実験9-拡張グリッド-進行中-検証バッチ分-2026-07-18)
- [Phase B / ERDC 拡張: 実験11-17 + 敵対的レビュー + 防御D1 (2026-07-19)](#phase-b--erdc-拡張-実験11-17--敵対的レビュー--防御d1-2026-07-19)

## データソース対応表

WT = `.claude/worktrees`。results/ は各 worktree の `projects/typo-cot/` 配下。

| 実験 | worktree | 結果パス |
|---|---|---|
| Step 0 | exp-step0 | `docs/dev_notes_step0.md` (wave2 節)、master table manifest |
| 実験1+3 | exp-01-03-transplant | `results/exp01_03/*/summary.json` (mmlu は `__p0/__p1` 統合) |
| 実験2 | exp-02-target-deletion | `results/prod/exp2/*_core/summary.json` |
| 実験4 (25設定) | exp-04-fixed-target | `results/prod/delta_rho/delta_rho_table.json` |
| 実験4 (MATH) | exp-04-fixed-target | `results/prod_math/delta_rho/delta_rho_table.json` |
| 実験5 | exp-05-matched-control | `results/prod/exp5/mcnemar_summary.csv` |
| 実験6 (帰属比較) | exp-06-attribution | `results/attribution_family/*/summary.json` |
| 実験6 (LOO) | exp-06-attribution | `results/loo/*/summary.json` |
| 実験7 (精度) | exp-07-correctors | `results/prod/exp7/`、`analysis/exp7_tables/` |
| 実験7 (within-run) | exp-07-correctors | `docs/dev_notes_07_correctors.md`、`results/prod/exp7/within_run/` |
| 実験8 | exp-08-patching | `docs/dev_notes_08_patching.md`、`results/prod/exp8/*/{lxt4,rnd4}/*.json` |
| 実験9 | exp-09-inner-repair | `results/prod/exp9/`、`results/exp9/summary_*.json` (拡張) |
| 実験10 | exp-10-scope | `outputs/{baseline,perturbed}/*/summary.json`、`analysis/exp10_natural_typo/` |
| GLMM | 本体 | `analysis/glmm_final/` |
| Holm | 本体 | `analysis/holm_correction/` |
| 採択基準 | 本体 | `analysis/adoption_criteria/check.csv` |

> **採択基準フラグ (2026-07-18, 宿題6)**: 採択基準 = clean精度 ≧ チャンス+10pt
> (チャンス: MMLU 25% / MMLU-Pro 10% / ARC 25% / CSQA 20% / GSM8K 0%)。
> 機械的確認 (`analysis/adoption_criteria/check.csv`) の結果、未達は
> **gemma-3-1b-it_mmlu_pro (clean 15.4% < 20%)** と
> **Llama-3.2-1B-Instruct_mmlu_pro (clean 19.0% < 20%)** の2設定のみ。
> 他ベンチのチャンス水準でも追加の未達なし。該当2設定の行には
> 「※採択基準未達(淡色・付録送り)」を付す。

## Step 0: master table 規模と再現検証

ソース: exp-step0 worktree `docs/dev_notes_step0.md` (wave2 取込 2026-07-18)。

- **wave1 (v1)**: 150 parquet (25設定×6条件)、238,855 行 (うち clean 39,810行)。
  summary.json 150/150 セル一致、table5 90/90、偏相関 25/25 (atol 1e-9)、
  span 除外 25/25 一致 (union 除外率 全体 13.19%、最大 38.59% = Mistral×GSM8K)。
- **wave2 (2026-07-18)**: 67 セル追加で **217 parquet / 342,653 行** に拡張。
  内訳: v1 150セル 238,855行 (フル再ビルドで 150/150 byte 一致) /
  Anti-LXT-4 (k4_bottom_k) 25セル 39,805行 / MATH-500 再生成 (M5×3条件)
  15セル 7,500行 / Qwen2.5-7B (B5×3 + math×3) 18セル 33,936行 /
  R1蒸留 (gsm8k/math/mmlu×3条件, `<think>` 形式) 9セル 22,557行。
- 検証: `--verify` で 217 エントリ・移行元 520 ファイルの sha256 と行数 OK。
  スモーク accuracy 175/175 (anti_lxt4 25セル含む)、偏相関 25/25、span 除外 25/25。
  R1 の strict span 失敗は clean 11.3〜16.2% / 摂動 12.6〜28.0% (v1 モデルより高め)。

## 実験4: fixed-target Δρ (25設定, k=top10)

| 設定 | n | ρ_default | ρ_fixed | Δρ | Δρ CI95 | p |
|---|---|---|---|---|---|---|
| gemma-3-1b-it_gsm8k | 962 | -0.495 | -0.476 | +0.019 | [-0.017, +0.056] | 0.31 |
| gemma-3-1b-it_mmlu | 2487 | -0.465 | 0.029 | +0.494 | [+0.467, +0.522] | 0.0002 |
| gemma-3-1b-it_mmlu_pro ※採択基準未達(淡色・付録送り) | 937 | -0.385 | 0.034 | +0.420 | [+0.381, +0.458] | 0.0002 |
| gemma-3-1b-it_arc | 1162 | -0.287 | -0.065 | +0.222 | [+0.180, +0.263] | 0.0002 |
| gemma-3-1b-it_commonsense_qa | 1219 | -0.523 | -0.071 | +0.452 | [+0.407, +0.498] | 0.0002 |
| gemma-3-4b-it_gsm8k | 1141 | -0.509 | -0.539 | -0.030 | [-0.064, +0.004] | 0.087 |
| gemma-3-4b-it_mmlu | 2552 | -0.493 | -0.174 | +0.319 | [+0.286, +0.352] | 0.0002 |
| gemma-3-4b-it_mmlu_pro | 970 | -0.572 | -0.149 | +0.423 | [+0.375, +0.471] | 0.0002 |
| gemma-3-4b-it_arc | 1164 | -0.460 | -0.175 | +0.285 | [+0.231, +0.335] | 0.0002 |
| gemma-3-4b-it_commonsense_qa | 1219 | -0.689 | -0.333 | +0.356 | [+0.314, +0.398] | 0.0002 |
| Llama-3.2-1B-Instruct_gsm8k | 1033 | -0.566 | -0.730 | -0.164 | [-0.194, -0.133] | 0.0002 |
| Llama-3.2-1B-Instruct_mmlu | 2388 | -0.631 | -0.057 | +0.574 | [+0.541, +0.608] | 0.0002 |
| Llama-3.2-1B-Instruct_mmlu_pro ※採択基準未達(淡色・付録送り) | 1146 | -0.707 | -0.043 | +0.664 | [+0.617, +0.713] | 0.0002 |
| Llama-3.2-1B-Instruct_arc | 1159 | -0.704 | -0.037 | +0.668 | [+0.619, +0.717] | 0.0002 |
| Llama-3.2-1B-Instruct_commonsense_qa | 1201 | -0.685 | -0.124 | +0.561 | [+0.512, +0.610] | 0.0002 |
| Llama-3.2-3B-Instruct_gsm8k | 1022 | -0.529 | -0.544 | -0.015 | [-0.048, +0.020] | 0.39 |
| Llama-3.2-3B-Instruct_mmlu | 2448 | -0.618 | -0.110 | +0.508 | [+0.472, +0.542] | 0.0002 |
| Llama-3.2-3B-Instruct_mmlu_pro | 998 | -0.681 | -0.135 | +0.546 | [+0.501, +0.592] | 0.0002 |
| Llama-3.2-3B-Instruct_arc | 1143 | -0.670 | -0.090 | +0.580 | [+0.525, +0.636] | 0.0002 |
| Llama-3.2-3B-Instruct_commonsense_qa | 1151 | -0.663 | -0.166 | +0.497 | [+0.445, +0.553] | 0.0002 |
| Mistral-7B-Instruct-v0.3_gsm8k | 810 | -0.338 | -0.499 | -0.162 | [-0.201, -0.127] | 0.0002 |
| Mistral-7B-Instruct-v0.3_mmlu | 2591 | -0.294 | -0.366 | -0.072 | [-0.101, -0.043] | 0.0002 |
| Mistral-7B-Instruct-v0.3_mmlu_pro | 1150 | -0.339 | -0.333 | +0.006 | [-0.034, +0.045] | 0.78 |
| Mistral-7B-Instruct-v0.3_arc | 1145 | -0.308 | -0.389 | -0.082 | [-0.123, -0.040] | 0.0002 |
| Mistral-7B-Instruct-v0.3_commonsense_qa | 1205 | -0.380 | -0.455 | -0.076 | [-0.114, -0.037] | 0.0004 |

> **Holm 確定値 (2026-07-18, m=25)**: ρ_fixed (k=top10) の raw 有意 21/25 →
> **Holm 有意 19/25** (`analysis/holm_correction/exp4_rho_fixed_holm.csv`)。
> Holm 非有意の6設定 = Llama-1B {arc, mmlu_pro}、Gemma-1B {arc, commonsense_qa,
> mmlu, mmlu_pro} (いずれも ρ_fixed ≈ 0 まで減衰した多肢選択設定)。

### 実験4 拡張: MATH-500 Δρ (6モデル, k=top10, B=10,000)

ソース: exp-04-fixed-target worktree `results/prod_math/delta_rho/delta_rho_table.json` (2026-07-18)。

| 設定 | n | ρ_default | ρ_fixed | Δρ | Δρ CI95 | Δρ p | ρ_fixed Holm p |
|---|---|---|---|---|---|---|---|
| gemma-3-1b-it_math | 162 | -0.254 | -0.236 | +0.018 | [-0.067, +0.112] | 0.69 | 0.0079 |
| gemma-3-4b-it_math | 216 | -0.296 | -0.254 | +0.042 | [-0.005, +0.093] | 0.084 | 0.00065 |
| Llama-3.2-1B-Instruct_math | 288 | -0.373 | -0.296 | +0.077 | [+0.016, +0.142] | 0.015 | 1.7e-06 |
| Llama-3.2-3B-Instruct_math | 313 | -0.251 | -0.154 | +0.097 | [+0.064, +0.132] | 0.0002 | 0.013 |
| Mistral-7B-Instruct-v0.3_math | 295 | -0.162 | -0.306 | -0.144 | [-0.213, -0.077] | 0.0002 | 5.4e-07 |
| Qwen2.5-7B-Instruct_math | 274 | -0.120 | -0.157 | -0.036 | [-0.090, +0.010] | 0.15 | 0.013 |

判定: **自由記述 (MATH) で「内的軸頑健」が6モデル全てで再現**。ρ_fixed は全6設定で
Holm 有意に残存 (Holm p ≤ 0.013)、|Δρ| ≤ 0.144 で、多肢選択で見られた大幅減衰
(+0.42〜+0.67) は生じない。Δρ の Holm 有意は Llama-3B (+0.097) と Mistral (−0.144)
のみで、Mistral は GSM8K と同じく fixed で相関がむしろ強化される方向を MATH でも再現
(Qwen も同符号傾向、n.s.)。

## 実験5: matched統制 McNemar (25設定)

| 設定 | n | acc_clean | acc_LXT4 | acc_MatchedRnd4 | リスク差(全) | 条件付きリスク差 | 条件付きp | 有意 |
|---|---|---|---|---|---|---|---|---|
| gemma-3-1b-it_gsm8k | 1318 | 0.404 | 0.330 | 0.325 | -0.0053 | +0.0413 | 0.092 | False |
| gemma-3-1b-it_mmlu | 2850 | 0.412 | 0.378 | 0.381 | +0.0032 | +0.0655 | 0.0002 | True |
| gemma-3-1b-it_mmlu_pro ※採択基準未達(淡色・付録送り) | 1400 | 0.154 | 0.152 | 0.155 | +0.0029 | +0.0556 | 0.21 | False |
| gemma-3-1b-it_arc | 1172 | 0.473 | 0.415 | 0.438 | +0.0222 | +0.0523 | 0.029 | True |
| gemma-3-1b-it_commonsense_qa | 1221 | 0.464 | 0.358 | 0.399 | +0.0410 | +0.0758 | 0.0074 | True |
| gemma-3-4b-it_gsm8k | 1319 | 0.835 | 0.782 | 0.809 | +0.0265 | +0.0345 | 0.0021 | True |
| gemma-3-4b-it_mmlu | 2850 | 0.632 | 0.586 | 0.610 | +0.0239 | +0.0549 | 8.8e-07 | True |
| gemma-3-4b-it_mmlu_pro | 1400 | 0.373 | 0.355 | 0.367 | +0.0121 | +0.0747 | 0.00056 | True |
| gemma-3-4b-it_arc | 1172 | 0.816 | 0.764 | 0.778 | +0.0137 | +0.0293 | 0.018 | True |
| gemma-3-4b-it_commonsense_qa | 1221 | 0.727 | 0.650 | 0.676 | +0.0262 | +0.0519 | 0.002 | True |
| Llama-3.2-1B-Instruct_gsm8k | 1319 | 0.361 | 0.335 | 0.337 | +0.0015 | +0.0231 | 0.41 | False |
| Llama-3.2-1B-Instruct_mmlu | 2850 | 0.451 | 0.386 | 0.429 | +0.0428 | +0.1379 | 1.2e-16 | True |
| Llama-3.2-1B-Instruct_mmlu_pro ※採択基準未達(淡色・付録送り) | 1400 | 0.190 | 0.178 | 0.196 | +0.0186 | +0.1165 | 0.001 | True |
| Llama-3.2-1B-Instruct_arc | 1172 | 0.527 | 0.446 | 0.485 | +0.0384 | +0.1100 | 7.2e-06 | True |
| Llama-3.2-1B-Instruct_commonsense_qa | 1221 | 0.583 | 0.489 | 0.512 | +0.0229 | +0.0899 | 0.00019 | True |
| Llama-3.2-3B-Instruct_gsm8k | 1319 | 0.705 | 0.640 | 0.653 | +0.0136 | +0.0376 | 0.02 | True |
| Llama-3.2-3B-Instruct_mmlu | 2850 | 0.627 | 0.564 | 0.590 | +0.0256 | +0.0700 | 7.9e-09 | True |
| Llama-3.2-3B-Instruct_mmlu_pro | 1400 | 0.368 | 0.321 | 0.361 | +0.0400 | +0.1437 | 3.3e-09 | True |
| Llama-3.2-3B-Instruct_arc | 1172 | 0.768 | 0.697 | 0.718 | +0.0205 | +0.0656 | 2.4e-05 | True |
| Llama-3.2-3B-Instruct_commonsense_qa | 1221 | 0.731 | 0.618 | 0.678 | +0.0598 | +0.0896 | 3.1e-06 | True |
| Mistral-7B-Instruct-v0.3_gsm8k | 1319 | 0.433 | 0.400 | 0.413 | +0.0136 | +0.0595 | 0.0042 | True |
| Mistral-7B-Instruct-v0.3_mmlu | 2850 | 0.644 | 0.581 | 0.611 | +0.0302 | +0.0550 | 3.4e-07 | True |
| Mistral-7B-Instruct-v0.3_mmlu_pro | 1400 | 0.349 | 0.324 | 0.334 | +0.0100 | +0.0675 | 0.0061 | True |
| Mistral-7B-Instruct-v0.3_arc | 1172 | 0.781 | 0.724 | 0.761 | +0.0367 | +0.0689 | 3.7e-07 | True |
| Mistral-7B-Instruct-v0.3_commonsense_qa | 1221 | 0.740 | 0.656 | 0.680 | +0.0238 | +0.0432 | 0.0097 | True |

> **Holm 確定値 (2026-07-18, m=25)**: 条件付き McNemar の raw 有意 22/25 →
> **Holm 有意 16/25** (`analysis/holm_correction/exp5_holm.csv`)。Holm で
> 落ちる6設定 = Llama-3B_gsm8k、Mistral {commonsense_qa, mmlu_pro}、
> Gemma-1B {arc, commonsense_qa}、Gemma-4B_arc。

## 実験1: CoT移植 4セル分解 (50設定, mmlu は p0+p1 統合)

ソース: exp-01-03-transplant worktree `results/exp01_03/*/summary.json`。mmlu は
`__p0/__p1` 両シャードの flip_count を合算して率を再計算 (restore は
n_te_flipped 加重、TE照合は n_total 加重)。n_incl = 非除外 ∧ A セル正解。
**restore が主推定量** = TE flip のうち clean CoT 強制 (Cセル, do(CoT:=clean))
で元の答えに復帰した率 (反事実的に一意)。TE/DE/IE = 4 セル (B/C/D) の flip **リスク**
であって効果の加法分解ではない (GLMM 交互作用 $\approx-5.9$ = サブ加法, DE+IE≠TE)。
**IE/TE は記述的比率**であり「媒介割合」ではない (非加法下で proportion-mediated は
未定義)。**within-run ノイズ = 0**: 4 セルは同一ラン・同一バッチ (greedy) 生成のため
セル間比較に再現性ノイズは乗らない (実験7 within-run: byte-identical→flip 0/45,641)。
付録の**クロスラン flip 9.56%** はアーカイブ比較固有のノイズフロアで、本 4 セル設計
(DE 含む) には適用されない。

| 設定 | 条件 | n_incl/n_total | TE | DE | IE | IE/TE | restore | TE照合 |
|---|---|---|---|---|---|---|---|---|
| gemma-3-1b-it_gsm8k | LXT-4 | 255/1318 | 37.6% | 1.6% | 37.3% | 0.99 | 96.9% | 95.3% |
| gemma-3-1b-it_gsm8k | Random-4 | 269/1319 | 32.3% | 3.0% | 33.1% | 1.02 | 94.3% | 95.8% |
| gemma-3-1b-it_mmlu | LXT-4 | 758/2850 | 35.1% | 18.6% | 28.5% | 0.81 | 64.7% | 93.4% |
| gemma-3-1b-it_mmlu | Random-4 | 749/2850 | 26.3% | 15.4% | 19.8% | 0.75 | 54.8% | 93.1% |
| gemma-3-1b-it_mmlu_pro ※採択基準未達(淡色・付録送り) | LXT-4 | 152/1400 | 34.9% | 15.8% | 34.9% | 1.00 | 73.6% | 84.0% |
| gemma-3-1b-it_mmlu_pro ※採択基準未達(淡色・付録送り) | Random-4 | 153/1400 | 32.0% | 11.1% | 27.5% | 0.86 | 69.4% | 83.4% |
| gemma-3-1b-it_arc | LXT-4 | 386/1172 | 29.5% | 15.8% | 19.4% | 0.66 | 54.4% | 95.6% |
| gemma-3-1b-it_arc | Random-4 | 381/1172 | 24.4% | 15.7% | 16.0% | 0.66 | 46.2% | 96.0% |
| gemma-3-1b-it_commonsense_qa | LXT-4 | 461/1221 | 42.5% | 25.2% | 22.8% | 0.54 | 50.0% | 95.8% |
| gemma-3-1b-it_commonsense_qa | Random-4 | 457/1221 | 30.9% | 22.3% | 18.2% | 0.59 | 45.4% | 97.1% |
| gemma-3-4b-it_gsm8k | LXT-4 | 1042/1319 | 9.8% | 1.3% | 9.6% | 0.98 | 93.1% | 96.7% |
| gemma-3-4b-it_gsm8k | Random-4 | 1050/1319 | 5.7% | 1.3% | 5.8% | 1.02 | 90.0% | 97.3% |
| gemma-3-4b-it_mmlu | LXT-4 | 1654/2850 | 18.2% | 5.9% | 15.7% | 0.86 | 77.4% | 95.0% |
| gemma-3-4b-it_mmlu | Random-4 | 1680/2850 | 12.3% | 3.4% | 10.8% | 0.87 | 81.2% | 95.2% |
| gemma-3-4b-it_mmlu_pro | LXT-4 | 456/1400 | 23.5% | 4.8% | 21.1% | 0.90 | 82.2% | 83.4% |
| gemma-3-4b-it_mmlu_pro | Random-4 | 467/1400 | 14.1% | 2.8% | 13.7% | 0.97 | 84.8% | 82.6% |
| gemma-3-4b-it_arc | LXT-4 | 948/1172 | 11.2% | 4.5% | 7.0% | 0.62 | 67.9% | 99.5% |
| gemma-3-4b-it_arc | Random-4 | 949/1172 | 6.5% | 2.7% | 4.6% | 0.71 | 67.7% | 99.3% |
| gemma-3-4b-it_commonsense_qa | LXT-4 | 858/1221 | 19.2% | 5.9% | 12.7% | 0.66 | 73.9% | 98.9% |
| gemma-3-4b-it_commonsense_qa | Random-4 | 874/1221 | 14.4% | 4.8% | 9.2% | 0.63 | 69.8% | 99.5% |
| Llama-3.2-1B-Instruct_gsm8k | LXT-4 | 465/1319 | 30.3% | 0.9% | 30.3% | 1.00 | 97.9% | 94.8% |
| Llama-3.2-1B-Instruct_gsm8k | Random-4 | 465/1319 | 29.0% | 0.9% | 28.6% | 0.99 | 99.3% | 95.0% |
| Llama-3.2-1B-Instruct_mmlu | LXT-4 | 1190/2850 | 36.6% | 12.9% | 29.6% | 0.81 | 73.1% | 94.7% |
| Llama-3.2-1B-Instruct_mmlu | Random-4 | 1211/2850 | 23.6% | 10.5% | 18.9% | 0.80 | 64.0% | 95.3% |
| Llama-3.2-1B-Instruct_mmlu_pro ※採択基準未達(淡色・付録送り) | LXT-4 | 229/1400 | 35.4% | 14.0% | 31.9% | 0.90 | 67.9% | 89.6% |
| Llama-3.2-1B-Instruct_mmlu_pro ※採択基準未達(淡色・付録送り) | Random-4 | 241/1400 | 29.5% | 13.7% | 22.0% | 0.75 | 59.2% | 91.5% |
| Llama-3.2-1B-Instruct_arc | LXT-4 | 615/1172 | 35.1% | 18.5% | 26.0% | 0.74 | 58.8% | 98.4% |
| Llama-3.2-1B-Instruct_arc | Random-4 | 613/1172 | 17.9% | 10.9% | 13.7% | 0.76 | 62.7% | 98.6% |
| Llama-3.2-1B-Instruct_commonsense_qa | LXT-4 | 717/1221 | 35.8% | 16.2% | 26.2% | 0.73 | 64.6% | 98.6% |
| Llama-3.2-1B-Instruct_commonsense_qa | Random-4 | 711/1221 | 30.1% | 15.3% | 19.4% | 0.64 | 55.1% | 97.2% |
| Llama-3.2-3B-Instruct_gsm8k | LXT-4 | 904/1319 | 19.5% | 1.3% | 19.1% | 0.98 | 96.0% | 96.7% |
| Llama-3.2-3B-Instruct_gsm8k | Random-4 | 908/1319 | 17.1% | 1.1% | 17.2% | 1.01 | 95.5% | 97.0% |
| Llama-3.2-3B-Instruct_mmlu | LXT-4 | 1684/2850 | 22.8% | 4.0% | 21.1% | 0.92 | 86.5% | 95.6% |
| Llama-3.2-3B-Instruct_mmlu | Random-4 | 1713/2850 | 16.3% | 4.4% | 13.6% | 0.84 | 76.7% | 96.1% |
| Llama-3.2-3B-Instruct_mmlu_pro | LXT-4 | 464/1400 | 32.5% | 6.0% | 31.5% | 0.97 | 85.4% | 89.6% |
| Llama-3.2-3B-Instruct_mmlu_pro | Random-4 | 471/1400 | 17.0% | 4.9% | 15.1% | 0.89 | 77.5% | 89.2% |
| Llama-3.2-3B-Instruct_arc | LXT-4 | 890/1172 | 19.3% | 5.1% | 15.1% | 0.78 | 80.8% | 99.1% |
| Llama-3.2-3B-Instruct_arc | Random-4 | 895/1172 | 11.8% | 3.8% | 8.6% | 0.73 | 80.2% | 98.8% |
| Llama-3.2-3B-Instruct_commonsense_qa | LXT-4 | 867/1221 | 26.8% | 7.8% | 21.0% | 0.78 | 77.6% | 98.1% |
| Llama-3.2-3B-Instruct_commonsense_qa | Random-4 | 882/1221 | 19.2% | 8.8% | 12.5% | 0.65 | 60.4% | 98.0% |
| Mistral-7B-Instruct-v0.3_gsm8k | LXT-4 | 568/1319 | 25.9% | 1.8% | 25.5% | 0.99 | 94.6% | 96.3% |
| Mistral-7B-Instruct-v0.3_gsm8k | Random-4 | 569/1318 | 21.6% | 1.2% | 21.1% | 0.98 | 94.3% | 96.1% |
| Mistral-7B-Instruct-v0.3_mmlu | LXT-4 | 1741/2850 | 19.0% | 5.4% | 17.1% | 0.90 | 79.7% | 96.5% |
| Mistral-7B-Instruct-v0.3_mmlu | Random-4 | 1754/2850 | 12.1% | 3.4% | 10.8% | 0.89 | 83.6% | 96.4% |
| Mistral-7B-Instruct-v0.3_mmlu_pro | LXT-4 | 457/1400 | 30.4% | 9.2% | 26.5% | 0.87 | 79.1% | 92.1% |
| Mistral-7B-Instruct-v0.3_mmlu_pro | Random-4 | 459/1400 | 20.9% | 5.7% | 19.6% | 0.94 | 83.3% | 91.0% |
| Mistral-7B-Instruct-v0.3_arc | LXT-4 | 904/1172 | 14.0% | 4.2% | 9.2% | 0.65 | 76.4% | 98.8% |
| Mistral-7B-Instruct-v0.3_arc | Random-4 | 907/1172 | 8.3% | 3.1% | 6.3% | 0.76 | 66.7% | 99.0% |
| Mistral-7B-Instruct-v0.3_commonsense_qa | LXT-4 | 885/1221 | 19.4% | 6.7% | 14.7% | 0.76 | 70.3% | 99.5% |
| Mistral-7B-Instruct-v0.3_commonsense_qa | Random-4 | 897/1221 | 16.9% | 7.0% | 11.0% | 0.65 | 65.1% | 99.0% |
| **pooled (25設定)** | LXT-4 | 19550/39809 | 23.9% | 7.4% | 19.7% | 0.83 | **76.2%** | — |
| **pooled (25設定)** | Random-4 | 19725/39809 | 17.0% | 6.1% | 13.6% | 0.80 | **72.2%** | — |

**pooled 感度分析・条件付き併記 (事前登録, `outcomes.json` から再集計)**:

| pooled | 条件 | restore (主) | GSM8K / MC restore | sens TE/DE/IE (除外込み) | sens IE/TE | IE\|ROUGE<1 |
|---|---|---|---|---|---|---|
| 25設定 | LXT-4 | 76.2% | 95.8% / 73.0% | 26.0 / 8.1 / 21.8% | 0.839 | 20.5% |
| 25設定 | Random-4 | 72.2% | 95.4% / 67.5% | 18.9 / 6.5 / 15.4% | 0.816 | 15.2% |

要点: **restore (主推定量) は pooled 76.2% (LXT-4) / 72.2% (Random-4)**、GSM8K で
90〜99% (自由記述はほぼ全て CoT 経由)、多肢選択で 45〜85%。記述的比率 IE/TE は GSM8K で
≈1.0 (DE 0.9〜1.8%)、多肢選択で 0.54〜1.00。**restore 優位 (間接経路支配, IE>DE) の
分解構造は両摂動条件で 48/50 設定に成立** (修正Aの見出し論理)。唯一の例外は
**Gemma-3-1B×CommonsenseQA の両条件** (LXT-4: DE 25.2% > IE 22.8% / Random-4:
DE 22.3% > IE 18.2%) で、1B×多肢選択の DE 増大が IE を上回る (§宿題3 で規模×形式
依存として定量化; `analysis/exp1_de_refinement/de_ie_exceptions.csv`)。**選択バイアス頑健性**:
構造的除外を含めた除外込み感度分析でも記述的 IE/TE はほぼ不変 (0.83→0.84, 0.80→0.82)
であり、restore 優位の見出しは除外設計のアーティファクトではない。

## 実験3: KL divergence プロファイル (50設定)

ソース: 実験1と同じ `summary.json` の divergence 欄 (mmlu は n 加重統合)。
KL_sum = 位置別 KL(clean‖pert) の合計 (セルC forward)。prec@10 = KL 上位10 と
R_C 上位10 の precision、null = 並べ替え帰無の平均、onset率 = rank 閾値超えの
発散オンセットが検出されたサンプル率。

| 設定 | 条件 | n_ok | KL_sum | KL_sum(flip) | KL_sum(noflip) | prec@10 | null | onset率 |
|---|---|---|---|---|---|---|---|---|
| gemma-3-1b-it_gsm8k | LXT-4 | 603 | 12.07 | 13.73 | 9.74 | 0.136 | 0.327 | 47.4% |
| gemma-3-1b-it_gsm8k | Random-4 | 632 | 6.10 | 8.26 | 3.71 | 0.147 | 0.329 | 14.4% |
| gemma-3-1b-it_mmlu | LXT-4 | 1691 | 11.81 | 13.02 | 10.99 | 0.278 | 0.280 | 55.4% |
| gemma-3-1b-it_mmlu | Random-4 | 1730 | 4.87 | 5.61 | 4.46 | 0.277 | 0.280 | 19.5% |
| gemma-3-1b-it_mmlu_pro ※採択基準未達(淡色・付録送り) | LXT-4 | 871 | 10.84 | 11.68 | 9.97 | 0.255 | 0.269 | 49.9% |
| gemma-3-1b-it_mmlu_pro ※採択基準未達(淡色・付録送り) | Random-4 | 872 | 3.55 | 4.75 | 2.60 | 0.259 | 0.272 | 12.6% |
| gemma-3-1b-it_arc | LXT-4 | 826 | 8.39 | 8.71 | 8.21 | 0.264 | 0.285 | 42.6% |
| gemma-3-1b-it_arc | Random-4 | 812 | 3.63 | 4.10 | 3.42 | 0.277 | 0.283 | 16.3% |
| gemma-3-1b-it_commonsense_qa | LXT-4 | 1044 | 13.41 | 14.79 | 12.11 | 0.454 | 0.460 | 70.2% |
| gemma-3-1b-it_commonsense_qa | Random-4 | 1033 | 5.90 | 7.11 | 5.11 | 0.471 | 0.459 | 33.7% |
| gemma-3-4b-it_gsm8k | LXT-4 | 1187 | 6.87 | 10.64 | 6.22 | 0.151 | 0.334 | 31.0% |
| gemma-3-4b-it_gsm8k | Random-4 | 1201 | 2.41 | 5.77 | 1.98 | 0.156 | 0.334 | 2.7% |
| gemma-3-4b-it_mmlu | LXT-4 | 2506 | 11.66 | 16.67 | 9.94 | 0.179 | 0.213 | 38.5% |
| gemma-3-4b-it_mmlu | Random-4 | 2530 | 5.34 | 8.38 | 4.61 | 0.191 | 0.213 | 16.7% |
| gemma-3-4b-it_mmlu_pro | LXT-4 | 991 | 11.67 | 15.60 | 9.56 | 0.167 | 0.212 | 35.3% |
| gemma-3-4b-it_mmlu_pro | Random-4 | 1010 | 4.14 | 6.00 | 3.48 | 0.185 | 0.212 | 9.4% |
| gemma-3-4b-it_arc | LXT-4 | 1161 | 6.03 | 9.90 | 5.30 | 0.244 | 0.261 | 22.0% |
| gemma-3-4b-it_arc | Random-4 | 1164 | 2.89 | 4.52 | 2.69 | 0.248 | 0.262 | 7.8% |
| gemma-3-4b-it_commonsense_qa | LXT-4 | 1170 | 11.79 | 18.07 | 9.44 | 0.402 | 0.423 | 49.7% |
| gemma-3-4b-it_commonsense_qa | Random-4 | 1190 | 5.70 | 11.22 | 4.26 | 0.424 | 0.424 | 25.0% |
| Llama-3.2-1B-Instruct_gsm8k | LXT-4 | 1263 | 5.52 | 5.65 | 5.35 | 0.215 | 0.385 | 55.5% |
| Llama-3.2-1B-Instruct_gsm8k | Random-4 | 1256 | 2.10 | 2.54 | 1.61 | 0.255 | 0.384 | 10.4% |
| Llama-3.2-1B-Instruct_mmlu | LXT-4 | 2550 | 4.88 | 5.53 | 4.41 | 0.284 | 0.281 | 43.6% |
| Llama-3.2-1B-Instruct_mmlu | Random-4 | 2601 | 1.97 | 2.53 | 1.71 | 0.276 | 0.280 | 10.2% |
| Llama-3.2-1B-Instruct_mmlu_pro ※採択基準未達(淡色・付録送り) | LXT-4 | 1156 | 3.96 | 4.29 | 3.65 | 0.328 | 0.314 | 32.8% |
| Llama-3.2-1B-Instruct_mmlu_pro ※採択基準未達(淡色・付録送り) | Random-4 | 1199 | 1.41 | 1.73 | 1.22 | 0.316 | 0.311 | 8.6% |
| Llama-3.2-1B-Instruct_arc | LXT-4 | 1170 | 3.33 | 3.82 | 2.97 | 0.362 | 0.364 | 32.5% |
| Llama-3.2-1B-Instruct_arc | Random-4 | 1168 | 1.36 | 1.66 | 1.25 | 0.365 | 0.364 | 6.9% |
| Llama-3.2-1B-Instruct_commonsense_qa | LXT-4 | 1215 | 6.16 | 6.96 | 5.52 | 0.444 | 0.452 | 53.7% |
| Llama-3.2-1B-Instruct_commonsense_qa | Random-4 | 1212 | 2.46 | 2.87 | 2.23 | 0.463 | 0.453 | 16.5% |
| Llama-3.2-3B-Instruct_gsm8k | LXT-4 | 1245 | 3.55 | 4.39 | 3.13 | 0.204 | 0.338 | 31.8% |
| Llama-3.2-3B-Instruct_gsm8k | Random-4 | 1250 | 1.62 | 2.25 | 1.36 | 0.205 | 0.338 | 4.2% |
| Llama-3.2-3B-Instruct_mmlu | LXT-4 | 2587 | 4.00 | 5.09 | 3.48 | 0.240 | 0.238 | 31.2% |
| Llama-3.2-3B-Instruct_mmlu | Random-4 | 2629 | 2.11 | 3.16 | 1.76 | 0.252 | 0.239 | 9.1% |
| Llama-3.2-3B-Instruct_mmlu_pro | LXT-4 | 1067 | 3.49 | 4.44 | 2.69 | 0.242 | 0.252 | 28.4% |
| Llama-3.2-3B-Instruct_mmlu_pro | Random-4 | 1081 | 1.37 | 1.99 | 1.09 | 0.265 | 0.252 | 7.3% |
| Llama-3.2-3B-Instruct_arc | LXT-4 | 1159 | 2.90 | 3.93 | 2.53 | 0.306 | 0.311 | 21.8% |
| Llama-3.2-3B-Instruct_arc | Random-4 | 1165 | 1.60 | 2.18 | 1.47 | 0.318 | 0.312 | 6.2% |
| Llama-3.2-3B-Instruct_commonsense_qa | LXT-4 | 1184 | 6.19 | 8.00 | 5.24 | 0.383 | 0.375 | 43.8% |
| Llama-3.2-3B-Instruct_commonsense_qa | Random-4 | 1202 | 2.91 | 4.49 | 2.33 | 0.408 | 0.375 | 16.7% |
| Mistral-7B-Instruct-v0.3_gsm8k | LXT-4 | 1295 | 3.66 | 4.72 | 3.02 | 0.156 | 0.271 | 16.8% |
| Mistral-7B-Instruct-v0.3_gsm8k | Random-4 | 1291 | 2.04 | 3.05 | 1.52 | 0.159 | 0.271 | 5.6% |
| Mistral-7B-Instruct-v0.3_mmlu | LXT-4 | 2638 | 3.60 | 5.62 | 2.84 | 0.291 | 0.274 | 18.7% |
| Mistral-7B-Instruct-v0.3_mmlu | Random-4 | 2649 | 1.95 | 3.57 | 1.53 | 0.281 | 0.274 | 7.6% |
| Mistral-7B-Instruct-v0.3_mmlu_pro | LXT-4 | 1167 | 3.68 | 5.22 | 2.51 | 0.240 | 0.251 | 14.9% |
| Mistral-7B-Instruct-v0.3_mmlu_pro | Random-4 | 1170 | 1.87 | 2.94 | 1.29 | 0.244 | 0.250 | 6.2% |
| Mistral-7B-Instruct-v0.3_arc | LXT-4 | 1146 | 3.05 | 5.29 | 2.51 | 0.314 | 0.300 | 12.7% |
| Mistral-7B-Instruct-v0.3_arc | Random-4 | 1148 | 1.87 | 3.15 | 1.67 | 0.311 | 0.301 | 5.7% |
| Mistral-7B-Instruct-v0.3_commonsense_qa | LXT-4 | 1179 | 6.48 | 10.42 | 5.12 | 0.385 | 0.381 | 33.2% |
| Mistral-7B-Instruct-v0.3_commonsense_qa | Random-4 | 1196 | 3.92 | 6.35 | 3.20 | 0.392 | 0.380 | 19.8% |

要点: KL_sum は **flip 群 > noflip 群が 50/50 設定**、LXT-4 > Random-4 も全設定で
成立。prec@10 は null と同水準以下 (KL 上位位置と R_C 上位語は重ならない =
空間的に相補的)。onset 検出率は LXT-4 で高い (12.7〜70.2% vs Random-4 2.7〜33.7%)。

### 実験3 補遺: KL–R_C 空間相補性 (前半/後半分布, 宿題5 統合, 全60設定)

ソース: 本体 `analysis/exp3_kl_rc_spatial/`(quadrant_table.csv 60設定 /
complementarity_stats.json / summary.json)。CoT 内の相対位置 0–1 を前半(<0.5)/
後半(≥0.5)に二分し、KL ピーク位置と R_C 上位語位置の分布を集計。H3 修正版
Complementarity(摂動伝播フェーズ=前半 → 答え決定フェーズ=後半)の実証補強。

**大域(全60設定, 位置プール)**: KL 平均位置 **0.359** vs R_C 平均位置 **0.541**
(Mann–Whitney p≈0)、overlap 0.753、Wasserstein 0.182。

**条件別平均(設定単位)**:

| 条件 | 平均 前半KL率 | 平均 後半R_C率 | 平均 overlap | 平均 Wasserstein | 相補的(KL前半>0.5 かつ R_C後半>0.5) |
|---|---|---|---|---|---|
| importance (LXT-4) | **0.749** | 0.620 | 0.600 | **0.285** | 24/30 |
| random | 0.606 | 0.620 | 0.761 | 0.168 | 24/30 |
| 全体 | 0.678 | 0.620 | 0.680 | 0.226 | **48/60** |

**M3×B2 コア(importance, mmlu は p0/p1 を位置数加重統合)**:

| 設定 | 前半KL率 | 後半R_C率 | KL平均位置 | R_C平均位置 | Wasserstein |
|---|---|---|---|---|---|
| gemma-3-4b-it × gsm8k | 0.740 | **0.851** | 0.306 | 0.754 | 0.448 |
| gemma-3-4b-it × mmlu | 0.687 | 0.683 | 0.359 | 0.621 | 0.262 |
| Llama-3.2-3B × gsm8k | 0.731 | **0.788** | 0.303 | 0.712 | 0.408 |
| Llama-3.2-3B × mmlu | 0.767 | 0.622 | 0.278 | 0.588 | 0.311 |
| Mistral-7B × gsm8k | 0.708 | 0.498 | 0.337 | 0.496 | 0.159 |
| Mistral-7B × mmlu | 0.657 | 0.497 | 0.365 | 0.494 | 0.129 |

要点:
- **KL は前半優位・R_C は後半優位**の相補構造が過半(48/60、importance 24/30)で成立。
  分離は importance>random、GSM8K>MMLU で最大(Wasserstein: gemma/llama×gsm8k で
  0.41〜0.45)。構造化算術ほど二相分離が顕著という H3 修正版の予測と整合。
- 例外: **Mistral は R_C 後半率≈0.50**(答え決定点が CoT 中央付近に分散)で分離小
  (Wasserstein 0.13〜0.16)。Mistral のアーキテクチャ固有性(実験4 H4 の逆転と同型)。
- **H3(Complementarity)判定 = 支持(再確認)**。

## 実験2: コア対比 (25設定, 無制限top-R_C vs 層内マッチ統制, delete)

flip 率は各腕の生値 (k=1/2/4)、リスク差/CI/McNemar p は k=4 の
top_rc_unrestricted vs stratum_matched_random 対比。

| 設定 | n_paired | top flip (k=1/2/4) | 統制 flip (k=1/2/4) | リスク差(k=4) | CI95 | McNemar p |
|---|---|---|---|---|---|---|
| gemma-3-1b-it_gsm8k | 478 | 75.6%/62.0%/85.0% | 4.4%/4.4%/4.2% | +0.8033 | [+0.766, +0.837] | 7.8e-111 |
| gemma-3-1b-it_mmlu | 449 | 12.4%/17.6%/22.0% | 7.3%/7.9%/9.8% | +0.1047 | [+0.065, +0.145] | 1.5e-06 |
| gemma-3-1b-it_mmlu_pro ※採択基準未達(淡色・付録送り) | 195 | 13.2%/21.6%/25.5% | 7.5%/11.6%/13.8% | +0.1026 | [+0.051, +0.154] | 0.00018 |
| gemma-3-1b-it_arc | 491 | 9.6%/14.4%/16.6% | 6.4%/9.7%/9.8% | +0.0611 | [+0.031, +0.096] | 0.00053 |
| gemma-3-1b-it_commonsense_qa | 431 | 12.7%/17.5%/25.6% | 8.2%/11.1%/15.1% | +0.1160 | [+0.074, +0.158] | 1.1e-07 |
| gemma-3-4b-it_gsm8k | 475 | 6.4%/15.2%/66.3% | 2.2%/2.2%/1.9% | +0.6316 | [+0.587, +0.674] | 7.4e-89 |
| gemma-3-4b-it_mmlu | 436 | 10.8%/15.4%/20.7% | 1.5%/1.5%/2.1% | +0.1720 | [+0.135, +0.209] | 4e-19 |
| gemma-3-4b-it_mmlu_pro | 449 | 16.7%/20.7%/25.5% | 2.1%/2.3%/2.4% | +0.2094 | [+0.174, +0.252] | 3.1e-26 |
| gemma-3-4b-it_arc | 486 | 3.8%/5.4%/5.6% | 1.2%/1.6%/1.4% | +0.0370 | [+0.019, +0.058] | 0.00053 |
| gemma-3-4b-it_commonsense_qa | 445 | 4.4%/6.0%/6.6% | 0.8%/0.8%/0.9% | +0.0629 | [+0.038, +0.088] | 2.5e-07 |
| Llama-3.2-1B-Instruct_gsm8k | 457 | 40.1%/74.6%/96.6% | 15.8%/26.5%/38.3% | +0.5842 | [+0.532, +0.630] | 1.3e-72 |
| Llama-3.2-1B-Instruct_mmlu | 465 | 86.5%/90.5%/94.0% | 18.1%/32.7%/42.8% | +0.5118 | [+0.465, +0.559] | 3.3e-63 |
| Llama-3.2-1B-Instruct_mmlu_pro ※採択基準未達(淡色・付録送り) | 247 | 77.9%/84.5%/85.5% | 14.7%/25.1%/42.5% | +0.4251 | [+0.360, +0.490] | 1.3e-30 |
| Llama-3.2-1B-Instruct_arc | 474 | 96.2%/98.8%/99.8% | 19.3%/30.8%/48.7% | +0.5105 | [+0.466, +0.557] | 1.7e-71 |
| Llama-3.2-1B-Instruct_commonsense_qa | 449 | 97.4%/98.6%/99.4% | 15.8%/27.7%/47.7% | +0.5167 | [+0.468, +0.561] | 1e-65 |
| Llama-3.2-3B-Instruct_gsm8k | 478 | 48.8%/85.0%/96.4% | 22.2%/30.8%/41.4% | +0.5502 | [+0.504, +0.596] | 9e-78 |
| Llama-3.2-3B-Instruct_mmlu | 454 | 77.2%/92.1%/95.3% | 22.5%/31.4%/47.4% | +0.4780 | [+0.430, +0.526] | 4.6e-59 |
| Llama-3.2-3B-Instruct_mmlu_pro | 457 | 72.3%/79.4%/82.5% | 17.5%/26.3%/35.7% | +0.4683 | [+0.420, +0.519] | 2.1e-60 |
| Llama-3.2-3B-Instruct_arc | 482 | 85.8%/94.0%/97.8% | 21.8%/33.3%/41.5% | +0.5705 | [+0.527, +0.616] | 3.4e-77 |
| Llama-3.2-3B-Instruct_commonsense_qa | 489 | 89.2%/95.6%/97.6% | 17.0%/27.0%/39.7% | +0.5808 | [+0.534, +0.626] | 1.6e-76 |
| Mistral-7B-Instruct-v0.3_gsm8k | 471 | 2.6%/19.2%/35.8% | 1.6%/3.4%/8.1% | +0.2633 | [+0.214, +0.314] | 8e-22 |
| Mistral-7B-Instruct-v0.3_mmlu | 432 | 8.2%/11.5%/14.0% | 3.4%/4.5%/5.3% | +0.0764 | [+0.042, +0.111] | 2.7e-05 |
| Mistral-7B-Instruct-v0.3_mmlu_pro | 449 | 8.7%/11.7%/16.4% | 2.6%/4.1%/5.6% | +0.1024 | [+0.065, +0.140] | 9.8e-08 |
| Mistral-7B-Instruct-v0.3_arc | 477 | 2.4%/3.8%/4.0% | 1.0%/2.0%/2.5% | +0.0126 | [-0.008, +0.034] | 0.31 |
| Mistral-7B-Instruct-v0.3_commonsense_qa | 460 | 3.4%/5.8%/7.2% | 1.8%/3.0%/4.1% | +0.0348 | [+0.007, +0.063] | 0.02 |

**Mistral 5設定の復旧 (2026-07-18)**: 当初の Mistral 行の欠測 (n_paired ≤ 2) の
根本原因は、アーカイブ Mistral の `_cot.pt` において token_scores のトークン
文字列に空白マーカー (先頭スペース / ▁) が無く、`tokens_to_words` の語境界検出が
一度も発火せずに word_scores が全文結合の1語に潰れていたこと。この結果 exp2 core
の標的選定が候補と一切交差せず、全腕が insufficient_candidates でスキップされて
いた (Gemma/Llama のアーカイブは正常)。アーカイブは読み取り専用のため読み込み側で
修復: トークン列を既知テキスト (prompt + generated_text) へ貪欲整合し、CoT 領域内
トークンの relevance 合計で語ランキングを再構築 (`loo_scorer.rc_word_ranking_from_token_scores`、
Gemma アーカイブの word_scores と小数4桁一致で検証; exp-02 worktree commit
`fef3958`)。修復後の再走 (2026-07-18) で上表の実数値を取得し、旧結果は
`*_core.broken` として退避済み。Mistral は ARC のみ非有意で、他4設定は
top > 統制が有意 — 他モデルと同じ方向 (効果量は Llama 系より小さい)。

## 実験7: 校正器3段 精度 (25設定×3校正器)

| 設定 | clean | LXT-4 | spellfix | neuralfix | llmfix |
|---|---|---|---|---|---|
| gemma-3-1b-it_gsm8k | 0.404 | 0.330 | 0.356 | 0.403 | 0.389 |
| gemma-3-1b-it_mmlu_pro ※採択基準未達(淡色・付録送り) | 0.154 | 0.152 | 0.156 | 0.153 | 0.159 |
| gemma-3-1b-it_mmlu | 0.412 | 0.378 | 0.385 | 0.391 | 0.391 |
| gemma-3-1b-it_arc | 0.473 | 0.415 | 0.439 | 0.461 | 0.457 |
| gemma-3-1b-it_commonsense_qa | 0.464 | 0.358 | 0.437 | 0.447 | 0.427 |
| gemma-3-4b-it_gsm8k | 0.835 | 0.782 | 0.762 | 0.826 | 0.819 |
| gemma-3-4b-it_mmlu_pro | 0.373 | 0.355 | 0.334 | 0.346 | 0.372 |
| gemma-3-4b-it_mmlu | 0.632 | 0.586 | 0.573 | 0.593 | 0.600 |
| gemma-3-4b-it_arc | 0.816 | 0.764 | 0.770 | 0.796 | 0.795 |
| gemma-3-4b-it_commonsense_qa | 0.727 | 0.650 | 0.630 | 0.697 | 0.664 |
| Llama-3.2-1B-Instruct_gsm8k | 0.361 | 0.335 | 0.334 | 0.360 | 0.368 |
| Llama-3.2-1B-Instruct_mmlu | 0.451 | 0.386 | 0.408 | 0.434 | 0.435 |
| Llama-3.2-1B-Instruct_mmlu_pro ※採択基準未達(淡色・付録送り) | 0.190 | 0.178 | 0.181 | 0.185 | 0.193 |
| Llama-3.2-1B-Instruct_arc | 0.527 | 0.446 | 0.468 | 0.511 | 0.492 |
| Llama-3.2-1B-Instruct_commonsense_qa | 0.583 | 0.489 | 0.523 | 0.556 | 0.533 |
| Llama-3.2-3B-Instruct_gsm8k | 0.705 | 0.640 | 0.627 | 0.695 | 0.688 |
| Llama-3.2-3B-Instruct_mmlu_pro | 0.368 | 0.321 | 0.333 | 0.347 | 0.343 |
| Llama-3.2-3B-Instruct_mmlu | 0.627 | 0.564 | 0.576 | 0.578 | 0.591 |
| Llama-3.2-3B-Instruct_arc | 0.768 | 0.697 | 0.700 | 0.759 | 0.733 |
| Llama-3.2-3B-Instruct_commonsense_qa | 0.731 | 0.618 | 0.638 | 0.695 | 0.654 |
| Mistral-7B-Instruct-v0.3_gsm8k | 0.433 | 0.400 | 0.386 | 0.427 | 0.422 |
| Mistral-7B-Instruct-v0.3_mmlu | 0.644 | 0.581 | 0.581 | 0.605 | 0.608 |
| Mistral-7B-Instruct-v0.3_mmlu_pro | 0.349 | 0.324 | 0.331 | 0.324 | 0.346 |
| Mistral-7B-Instruct-v0.3_arc | 0.781 | 0.724 | 0.726 | 0.765 | 0.757 |
| Mistral-7B-Instruct-v0.3_commonsense_qa | 0.740 | 0.656 | 0.643 | 0.699 | 0.686 |

### 実験7 補遺: within-run byte-identical 検証 (75設定 = 25設定×3校正器, 2026-07-19 完走)

clean 入力と校正後入力を**同一ラン・同一バッチ** (greedy, temperature=0.0) で生成し、
byte-identical 復元ペア (プロンプト厳密一致) の flip を正式測定。ソース:
exp-07-correctors worktree `docs/dev_notes_07_correctors.md` +
`results/prod/exp7/within_run/within_run_summary.{json,md}`。

- byte-identical 総数 **45,641 ペア** (率: spellfix 10.5〜32.5% /
  llmfix 27.8〜64.4% / neuralfix 36.6〜83.8%)。
- **within-run flip 0/45,641 (0.00%)** — 全25設定×3校正器で 0。全ペアで生成
  テキスト自体が byte 一致、生成失敗 0、シャード失敗 0。
- 同一集合のクロスラン flip (本番生成 vs アーカイブ baseline、参考ノイズフロア)
  = **9.56% (4,362/45,641)** (spellfix 9.38% / neuralfix 8.80% / llmfix 10.60%)。
  ベンチ別では mmlu_pro (16〜20%) > gsm8k > mmlu > arc/csqa。
- 結論: 「byte-identical 復元 → flip 0%」が within-run で厳密に成立 (greedy の
  理論どおり)。クロスラン比較で byte-identical 集合に見える flip (〜9.6%) は
  全量が再現性ノイズ。

M3×B2 の内訳 (18設定、13,438 ペア; within-run flip はすべて 0):

| モデル | ベンチ | 校正器 | byte-identical n/N | within-run flip | クロスラン flip (参考) |
|---|---|---|---|---|---|
| Llama-3.2-3B-Instruct | gsm8k | pyspell | 196/1319 | 0/196 (0.0%) | 18/196 (9.2%) |
| Llama-3.2-3B-Instruct | gsm8k | T5-large-spell | 614/1319 | 0/614 (0.0%) | 71/614 (11.6%) |
| Llama-3.2-3B-Instruct | gsm8k | Qwen2.5-7B-Instruct | 491/1319 | 0/491 (0.0%) | 51/491 (10.4%) |
| Llama-3.2-3B-Instruct | mmlu | pyspell | 511/2850 | 0/511 (0.0%) | 53/511 (10.4%) |
| Llama-3.2-3B-Instruct | mmlu | T5-large-spell | 1413/2850 | 0/1413 (0.0%) | 99/1413 (7.0%) |
| Llama-3.2-3B-Instruct | mmlu | Qwen2.5-7B-Instruct | 1322/2850 | 0/1322 (0.0%) | 148/1322 (11.2%) |
| Mistral-7B-Instruct-v0.3 | gsm8k | pyspell | 138/1319 | 0/138 (0.0%) | 24/138 (17.4%) |
| Mistral-7B-Instruct-v0.3 | gsm8k | T5-large-spell | 679/1319 | 0/679 (0.0%) | 105/679 (15.5%) |
| Mistral-7B-Instruct-v0.3 | gsm8k | Qwen2.5-7B-Instruct | 517/1319 | 0/517 (0.0%) | 80/517 (15.5%) |
| Mistral-7B-Instruct-v0.3 | mmlu | pyspell | 393/2850 | 0/393 (0.0%) | 29/393 (7.4%) |
| Mistral-7B-Instruct-v0.3 | mmlu | T5-large-spell | 1422/2850 | 0/1422 (0.0%) | 101/1422 (7.1%) |
| Mistral-7B-Instruct-v0.3 | mmlu | Qwen2.5-7B-Instruct | 1207/2850 | 0/1207 (0.0%) | 122/1207 (10.1%) |
| gemma-3-4b-it | gsm8k | pyspell | 185/1319 | 0/185 (0.0%) | 9/185 (4.9%) |
| gemma-3-4b-it | gsm8k | T5-large-spell | 577/1319 | 0/577 (0.0%) | 24/577 (4.2%) |
| gemma-3-4b-it | gsm8k | Qwen2.5-7B-Instruct | 397/1319 | 0/397 (0.0%) | 14/397 (3.5%) |
| gemma-3-4b-it | mmlu | pyspell | 584/2850 | 0/584 (0.0%) | 55/584 (9.4%) |
| gemma-3-4b-it | mmlu | T5-large-spell | 1475/2850 | 0/1475 (0.0%) | 139/1475 (9.4%) |
| gemma-3-4b-it | mmlu | Qwen2.5-7B-Instruct | 1317/2850 | 0/1317 (0.0%) | 139/1317 (10.6%) |

注: fully_restored フラグ (空白正規化の全文一致) とプロンプト厳密一致は neural/LLM
校正器で最大 234 件/設定の不一致 (方向は全て「フラグ=True だが byte 非一致」、原因は
校正器の連続空白正規化)。within-run 検証は厳密 byte-identical 集合 (fully_restored の
部分集合) で実施。

### 実験7 補遺: R_Q 偏在 (校正ボトルネックの局在, 宿題2 統合, 2026-07-18 作成/2026-07-19 検証)

ソース: 本体 `analysis/exp7_tables/`(rq_mannwhitney.csv 100行 = 75設定×校正器 +
25プール / restoration_rates.csv / flip_rates.csv)。R_Q = 校正後生成 results.json
`perturbed_tokens[].importance_score`(語単位は貪欲・語順保存マッチの最大値)。
帰無仮説: 復元失敗語と復元語の R_Q 分布が同一。

**校正器別 平均(25設定)**:

| 校正器 | word 復元率 | flip 率 |
|---|---|---|
| spellfix | 0.663 | 0.227 |
| neuralfix | **0.886** | **0.137** |
| llmfix | 0.734 | 0.162 |

**R_Q 偏在 Mann–Whitney(プール25設定、Holm m=25)**:

| 指標 | 値 |
|---|---|
| Holm 有意(p<0.05)設定数 | **17/25** |
| AUC = P(R_Q_failed > R_Q_restored) 平均 | **0.539** |
| AUC>0.5 の設定数 | **23/25** |
| median(R_Q_failed − R_Q_restored) 平均 | **+0.050** |

**H7-4(ボトルネック R_Q 偏在)判定 = 支持(方向一貫・効果は小)**: 復元失敗語は
復元語より R_Q が高い側へ偏る(23/25 で AUC>0.5、17/25 で Holm 有意)が、AUC≈0.54 と
効果量は小さく「高 R_Q への決定的集中」ではなく確率的偏り。ボトルネックの主因は
復元率そのもの(neural 0.886 > llm 0.734 > spellfix 0.663)であり、byte-identical
復元 → flip 0%(前節 45,641 ペア)が残余ギャップの原因が補正品質にあることの決定的証拠。

## 実験9: 設定別 repair係数 (50設定, clean正解条件付き)

パース修正済 (宿題5, 2026-07-18)。ソース: exp-09-inner-repair worktree
`results/prod/exp9/regression_{設定}_{条件}.csv` の `repair_score` 行
(n は `analysis_summary.json` の n_obs、係数は両ソースで一致確認済)。
機械可読版: `analysis/exp9_per_setting/repair_coefficients.csv`。
※ = Holm (m=50) で p<0.05。要約: 係数負 47/50、raw 有意 25/50、Holm 有意 10/50
(lxt4 8 / random4 2)。

**4共変量の標準化係数比較 [宿題4, 2026-07-19]** (`analysis/exp9_covariate_comparison/`):
repair_score・split_increment・zipf_freq・r_q を z 標準化し同一 GLM (cluster-robust SE) に
投入した pooled 比較では、**14/14 の集計 (主報告 Llama+Mistral・全モデル・Gemma単独・
条件別・Qwen/MATH 拡張再pooled) で repair_score が最大の |標準化係数|** (例: Llama+Mistral
repair −0.165 vs r_q +0.123 vs zipf −0.095)。→ pooled 水準で「最強の負予測子」を支持。
ただし個別設定では repair 単独最大は 27/72 (38%) のみ (残りは zipf/split/r_q が上回る) で、
集計での優位は符号一貫性 (64/72 設定で負) による。詳細は `covariate_comparison.csv` /
`per_setting_ranks.csv`。

| 設定 | 条件 | repair係数 | SE | p | p_Holm(m=50) | n |
|---|---|---|---|---|---|---|
| Llama-3.2-1B-Instruct_arc | lxt4 | -0.0729 | 0.0538 | 0.176 | 1 | 2321 |
| Llama-3.2-1B-Instruct_arc | random4 | -0.0904 | 0.0651 | 0.165 | 1 | 2451 |
| Llama-3.2-1B-Instruct_commonsense_qa | lxt4 | -0.0602 | 0.0489 | 0.218 | 1 | 2736 |
| Llama-3.2-1B-Instruct_commonsense_qa | random4 | -0.0900 | 0.0483 | 0.0623 | 1 | 2802 |
| Llama-3.2-1B-Instruct_gsm8k | lxt4 | -0.0265 | 0.0640 | 0.679 | 1 | 1751 |
| Llama-3.2-1B-Instruct_gsm8k | random4 | +0.0888 | 0.0679 | 0.191 | 1 | 1841 |
| Llama-3.2-1B-Instruct_mmlu | lxt4 | -0.1123 | 0.0389 | 0.00384 | 0.138 | 4622 |
| Llama-3.2-1B-Instruct_mmlu | random4 | -0.0443 | 0.0412 | 0.282 | 1 | 5046 |
| Llama-3.2-1B-Instruct_mmlu_pro ※採択基準未達(淡色・付録送り) | lxt4 | -0.0084 | 0.0873 | 0.923 | 1 | 972 |
| Llama-3.2-1B-Instruct_mmlu_pro ※採択基準未達(淡色・付録送り) | random4 | -0.1095 | 0.0843 | 0.194 | 1 | 1041 |
| Llama-3.2-3B-Instruct_arc | lxt4 | -0.2450 | 0.0549 | 8.27e-06 | 0.00038 ※ | 3394 |
| Llama-3.2-3B-Instruct_arc | random4 | -0.1812 | 0.0640 | 0.00462 | 0.158 | 3564 |
| Llama-3.2-3B-Instruct_commonsense_qa | lxt4 | -0.1427 | 0.0471 | 0.00246 | 0.0912 | 3428 |
| Llama-3.2-3B-Instruct_commonsense_qa | random4 | -0.2102 | 0.0526 | 6.45e-05 | 0.00284 ※ | 3509 |
| Llama-3.2-3B-Instruct_gsm8k | lxt4 | -0.2445 | 0.0543 | 6.66e-06 | 0.000313 ※ | 3485 |
| Llama-3.2-3B-Instruct_gsm8k | random4 | -0.1705 | 0.0556 | 0.00217 | 0.0847 | 3600 |
| Llama-3.2-3B-Instruct_mmlu | lxt4 | -0.1080 | 0.0356 | 0.0024 | 0.0912 | 6455 |
| Llama-3.2-3B-Instruct_mmlu | random4 | -0.1178 | 0.0383 | 0.00208 | 0.083 | 6969 |
| Llama-3.2-3B-Instruct_mmlu_pro | lxt4 | -0.1302 | 0.0622 | 0.0363 | 0.979 | 1897 |
| Llama-3.2-3B-Instruct_mmlu_pro | random4 | -0.0914 | 0.0694 | 0.188 | 1 | 2023 |
| Mistral-7B-Instruct-v0.3_arc | lxt4 | -0.2770 | 0.0611 | 5.73e-06 | 0.000275 ※ | 3462 |
| Mistral-7B-Instruct-v0.3_arc | random4 | -0.1149 | 0.0678 | 0.0899 | 1 | 3587 |
| Mistral-7B-Instruct-v0.3_commonsense_qa | lxt4 | -0.2956 | 0.0547 | 6.42e-08 | 3.21e-06 ※ | 3455 |
| Mistral-7B-Instruct-v0.3_commonsense_qa | random4 | -0.0693 | 0.0525 | 0.186 | 1 | 3514 |
| Mistral-7B-Instruct-v0.3_gsm8k | lxt4 | -0.1740 | 0.0616 | 0.00474 | 0.158 | 2074 |
| Mistral-7B-Instruct-v0.3_gsm8k | random4 | -0.1474 | 0.0648 | 0.0228 | 0.708 | 2182 |
| Mistral-7B-Instruct-v0.3_mmlu | lxt4 | -0.0741 | 0.0378 | 0.0503 | 1 | 6628 |
| Mistral-7B-Instruct-v0.3_mmlu | random4 | -0.0846 | 0.0380 | 0.0259 | 0.778 | 7165 |
| Mistral-7B-Instruct-v0.3_mmlu_pro | lxt4 | -0.2241 | 0.0636 | 0.000427 | 0.0184 ※ | 1799 |
| Mistral-7B-Instruct-v0.3_mmlu_pro | random4 | -0.1346 | 0.0643 | 0.0364 | 0.979 | 1925 |
| gemma-3-1b-it_arc | lxt4 | -0.0524 | 0.0492 | 0.288 | 1 | 2155 |
| gemma-3-1b-it_arc | random4 | -0.1748 | 0.0497 | 0.000439 | 0.0184 ※ | 2203 |
| gemma-3-1b-it_commonsense_qa | lxt4 | -0.1105 | 0.0471 | 0.0191 | 0.61 | 2202 |
| gemma-3-1b-it_commonsense_qa | random4 | -0.0414 | 0.0456 | 0.364 | 1 | 2251 |
| gemma-3-1b-it_gsm8k | lxt4 | -0.0575 | 0.0520 | 0.269 | 1 | 1999 |
| gemma-3-1b-it_gsm8k | random4 | -0.0565 | 0.0501 | 0.26 | 1 | 2063 |
| gemma-3-1b-it_mmlu | lxt4 | -0.0628 | 0.0356 | 0.0782 | 1 | 4380 |
| gemma-3-1b-it_mmlu | random4 | -0.0397 | 0.0357 | 0.265 | 1 | 4633 |
| gemma-3-1b-it_mmlu_pro ※採択基準未達(淡色・付録送り) | lxt4 | +0.0106 | 0.0824 | 0.897 | 1 | 798 |
| gemma-3-1b-it_mmlu_pro ※採択基準未達(淡色・付録送り) | random4 | -0.0881 | 0.0794 | 0.267 | 1 | 852 |
| gemma-3-4b-it_arc | lxt4 | -0.1926 | 0.0586 | 0.00102 | 0.0417 ※ | 3729 |
| gemma-3-4b-it_arc | random4 | -0.0464 | 0.0791 | 0.558 | 1 | 3815 |
| gemma-3-4b-it_commonsense_qa | lxt4 | -0.1347 | 0.0474 | 0.00451 | 0.158 | 3439 |
| gemma-3-4b-it_commonsense_qa | random4 | -0.1146 | 0.0526 | 0.0293 | 0.822 | 3507 |
| gemma-3-4b-it_gsm8k | lxt4 | -0.2570 | 0.0553 | 3.34e-06 | 0.000163 ※ | 4265 |
| gemma-3-4b-it_gsm8k | random4 | -0.0455 | 0.0807 | 0.573 | 1 | 4335 |
| gemma-3-4b-it_mmlu | lxt4 | -0.1574 | 0.0359 | 1.19e-05 | 0.000537 ※ | 6726 |
| gemma-3-4b-it_mmlu | random4 | -0.0290 | 0.0391 | 0.458 | 1 | 7110 |
| gemma-3-4b-it_mmlu_pro | lxt4 | +0.0251 | 0.0569 | 0.659 | 1 | 1968 |
| gemma-3-4b-it_mmlu_pro | random4 | -0.1416 | 0.0645 | 0.0283 | 0.82 | 2062 |

## 実験8: activation patching 3モデル統合 (M3×B2×2条件 = 12条件, S2 KL recovery)

ソース: exp-08-patching worktree `docs/dev_notes_08_patching.md` (本番集計 2026-07-17
+ Llama 再走集計 2026-07-18)。数値の再導出は
`results/prod/exp8/<設定>/{lxt4,rnd4}/*.json` の `cells[]` から可能。主指標は
S2 KL recovery のセル median (question_span, clean→pert)。読み出しは
max_new_tokens=16 + clean CoT 強制の regime のため GSM8K の flip 系指標は使わない。

### 実行サマリ (6設定, 完了ペア数)

| 設定 | n_tasks | done | excluded | failed | セル/ペア |
|---|---|---|---|---|---|
| gemma-3-4b-it × gsm8k | 207 | 173 (lxt 109 / rnd 64) | 34 | 0 | 216 (12窓) |
| gemma-3-4b-it × mmlu | 606 | 576 (lxt 348 / rnd 228) | 30 | 0 | 216 (12窓) |
| Mistral-7B-v0.3 × gsm8k | 269 | 267 (lxt 146 / rnd 121) | 2 | 0 | 198 (11窓) |
| Mistral-7B-v0.3 × mmlu | 628 | 608 (lxt 370 / rnd 238) | 20 | 0 | 198 (11窓) |
| Llama-3.2-3B × gsm8k | 362 | 337 (lxt 183 / rnd 154) | 25 | 0 | 180 (10窓) |
| Llama-3.2-3B × mmlu | 750 | 709 (lxt 421 / rnd 288) | 41 | 0 | 180 (10窓) |

(Llama 2設定は 07-17 の HF Hub 一時障害で未取得 → `HF_HUB_OFFLINE=1` で
07-17 再走・07-18 集計。全6設定で整合性検証 PASS: ファイル数 = n_tasks、
done+excluded が run_summary.json と一致、config_hash 不一致 0。)

### residual 層窓プロファイル (lxt4, question_span, c2p, median)

窓は各モデルの層数準拠 (Gemma 34層=12窓 / Mistral 32層=11窓 / Llama 28層=10窓)。
最終2窓は全設定 ≈0 のため省略:

| 設定 | 0-3 | 3-6 | 6-9 | 9-12 | 12-15 | 15-18 | 18-21 | 21-24 | 24-27 |
|---|---|---|---|---|---|---|---|---|---|
| gemma × gsm8k | .716 | **.769** | .714 | .693 | .595 | .490 | .289 | .205 | .158 |
| gemma × mmlu | .710 | .737 | **.762** | .706 | .662 | .525 | .427 | .247 | .155 |
| mistral × gsm8k | **.585** | .561 | .439 | .364 | .226 | .151 | .052 | .041 | .026 |
| mistral × mmlu | **.698** | .631 | .575 | .580 | .451 | .295 | .156 | .076 | .059 |
| llama × gsm8k | .722 | **.723** | .591 | .495 | .349 | .227 | .195 | .144 | .062 |
| llama × mmlu | .723 | **.766** | .734 | .653 | .556 | .382 | .248 | .167 | .090 |

### 12条件の最良セル (median 基準)

| 設定 | 条件 | 最良セル | median | frac_pos | mlp\|0-3 | attn\|0-3 |
|---|---|---|---|---|---|---|
| gemma × gsm8k | lxt4 | residual\|3-6 | 0.769 | 0.90 | 0.486 | −0.279 |
| gemma × gsm8k | rnd4 | residual\|3-6 | 0.346 | 0.67 | 0.027 | −0.030 |
| gemma × mmlu | lxt4 | residual\|6-9 | 0.762 | 0.88 | 0.311 | −0.339 |
| gemma × mmlu | rnd4 | residual\|9-12 | 0.328 | 0.70 | −0.022 | −0.305 |
| mistral × gsm8k | lxt4 | residual\|0-3 | 0.585 | 0.80 | 0.525 | −0.060 |
| mistral × gsm8k | rnd4 | residual\|3-6 | 0.397 | 0.70 | 0.375 | −0.170 |
| mistral × mmlu | lxt4 | residual\|0-3 | 0.698 | 0.85 | 0.590 | −0.072 |
| mistral × mmlu | rnd4 | residual\|0-3 | 0.430 | 0.67 | 0.328 | −0.138 |
| llama × gsm8k | lxt4 | residual\|3-6 | 0.723 | 0.89 | 0.622 | −0.165 |
| llama × gsm8k | rnd4 | residual\|3-6 | 0.516 | 0.86 | 0.486 | −0.198 |
| llama × mmlu | lxt4 | residual\|3-6 | 0.766 | 0.96 | 0.606 | −0.192 |
| llama × mmlu | rnd4 | residual\|0-3 | 0.464 | 0.78 | 0.396 | −0.200 |

要点:

- **早期層 residual 局在が3モデルで再現**: 最良セルは12条件すべて
  residual\|[0,12) (うち10条件が [0,6))。深さ方向に単調減衰し最終窓 ≈0。
  mlp は早期のみ正で residual に次ぎ、attn は ≈0 かつ最早期 [0,3) が一貫して負。
- **LXT-4 は Random-4 の 4.8〜10.1 倍の分布乖離 (KL_unpatched) かつ回復率
  1.4〜2.3 倍** (gemma 9.9〜10.1×/2.2〜2.3×、mistral 4.8〜4.9×/1.5〜1.6×、
  llama 4.9〜6.2×/1.4〜1.7×)。
- **MMLU flip 逆転率** (分岐ペア限定、question_span への residual[0,3) 1窓
  パッチ): gemma 54/72 (75%, lxt4) / 29/42 (69%, rnd4)、mistral 47/67 (70%) /
  23/30 (77%)、llama 45/54 (83%) / 44/59 (75%)。GSM8K は分岐ペアが 0〜4% で
  flip 系指標は本 regime では無意味 (S2 KL recovery が主指標)。
- **主結論「質問タイポの効果は早期層の摂動語スパン residual 表現に局在し、
  そこへの1窓パッチで過半が打ち消せる」は 3モデル×2ベンチ×2摂動条件の
  全12条件で再現**。

## 実験10: スコープ拡張

### ①〜③ 新モデル・新ベンチの生成完了分 (accuracy)

ソース: exp-10-scope worktree `outputs/{baseline,perturbed}/*/summary.json`
(Qwen の B5 clean はアーカイブ baseline)。greedy, seed=42。

R1蒸留 (DeepSeek-R1-Distill-Qwen-7B, `<think>` 形式, max_new_tokens 4096/8192):

| ベンチ | clean | LXT-4 | Random-4 |
|---|---|---|---|
| gsm8k | 0.848 (1119/1319) | 0.811 (1070/1319) | 0.827 (1091/1319) |
| math | 0.734 (367/500) | 0.650 (325/500) | 0.610 (305/500) |
| mmlu | 0.697 (3972/5700) | 0.662 (3771/5700) | 0.668 (3805/5700) |

Qwen2.5-7B-Instruct (第4家族):

| ベンチ | clean | LXT-4 | Random-4 |
|---|---|---|---|
| gsm8k | 0.896 (1182/1319) | 0.863 (1138/1319) | 0.887 (1170/1319) |
| mmlu | 0.761 (4336/5700) | 0.712 (4059/5700) | 0.745 (4244/5700) |
| mmlu_pro | 0.530 (742/1400) | 0.471 (660/1400) | 0.515 (721/1400) |
| arc | 0.903 (1058/1172) | 0.877 (1028/1172) | 0.886 (1038/1172) |
| commonsense_qa | 0.831 (1015/1221) | 0.722 (882/1221) | 0.781 (954/1221) |
| math | 0.498 (249/500) | 0.418 (209/500) | 0.434 (217/500) |

MATH-500 再生成 (第2自由記述, M5 + Qwen):

| モデル | clean | LXT-4 | Random-4 |
|---|---|---|---|
| gemma-3-1b-it | 0.268 (134/500) | 0.144 (72/500) | 0.166 (83/500) |
| gemma-3-4b-it | 0.444 (222/500) | 0.368 (184/500) | 0.364 (182/500) |
| Llama-3.2-1B-Instruct | 0.222 (111/500) | 0.124 (62/500) | 0.144 (72/500) |
| Llama-3.2-3B-Instruct | 0.300 (150/500) | 0.214 (107/500) | 0.260 (130/500) |
| Mistral-7B-Instruct-v0.3 | 0.128 (64/500) | 0.086 (43/500) | 0.124 (62/500) |
| Qwen2.5-7B-Instruct | 0.498 (249/500) | 0.418 (209/500) | 0.434 (217/500) |

傾向: LXT-4 の低下 ≧ Random-4 の低下が Qwen 全6ベンチ・MATH 5/6 モデルで成立
(例外: MATH の gemma-3-4b は同水準、R1×math は Random-4 の方が低い)。

### ④ 自然typo A/B 比較 (gemma-3-4b-it, 合成 LXT-4 vs GitHub Typo Corpus 分布)

ソース: exp-10-scope worktree `analysis/exp10_natural_typo/ab_comparison.{json,md}`。
標的語は A/B で同一 (LXT-4 の標的トークンを固定)、k=4。

| 指標 | gsm8k A(合成) | gsm8k B(自然) | mmlu A(合成) | mmlu B(自然) |
|---|---|---|---|---|
| 精度 (baseline) | 0.8347 | 0.8347 | 0.6323 | 0.6323 |
| 精度 (摂動後) | 0.7824 | 0.7824 | 0.586 | 0.5937 |
| Δ精度 | -0.0523 | -0.0523 | -0.0463 | -0.0386 |
| flip率 (正→誤) | 0.1163 | 0.109 | 0.202 | 0.2003 |
| 回答変化率 | 0.2009 | 0.1827 | 0.2888 | 0.2989 |

- gsm8k: flip一致 Jaccard=0.3405 (Aのみ65 / Bのみ57 / 両方63)、
  McNemar p=0.526 (n_correct=1101)
- mmlu: flip一致 Jaccard=0.3602 (Aのみ172 / Bのみ169 / 両方192)、
  McNemar p=0.914 (n_correct=1802)
- 操作分布 B (自然): deletion ≈0.41 / insertion ≈0.23 / substitution ≈0.23 /
  transposition ≈0.13〜0.14 (A は double_typing/omission/proximity 各≈1/3)
- 判定: **flip 率・Δ精度は typo 操作分布に対して頑健** (A/B 差は両ベンチとも
  McNemar 非有意)。flip する個々のサンプルは中程度しか重ならない
  (Jaccard ≈0.34〜0.36)。内的軸相関は B 側 AttnLRP 未計算のため対象外。

## 実験6: 帰属手法ファミリー比較 (M3×B2×2条件×3手法 = 36シャード, 2026-07-18〜19)

ソース: exp-06-attribution worktree `results/attribution_family/*/summary.json`
(36シャード完了、総サンプル 10,800 中スコア済 10,569、エラー 0、align 失敗 66、
skip 165)。値は各手法 top-10 と AttnLRP R_C top-10 の **mean Jaccard@10**
(300 サンプル/シャード、IG は m=16)。

| 設定 | 条件 | G×I | IG | rollout | n |
|---|---|---|---|---|---|
| gemma-3-4b-it_gsm8k | clean | 0.308 | 0.380 | 0.154 | 300 |
| gemma-3-4b-it_gsm8k | lxt4 | 0.318 | 0.380 | 0.151 | 293 |
| gemma-3-4b-it_mmlu | clean | 0.249 | 0.256 | 0.217 | 293 |
| gemma-3-4b-it_mmlu | lxt4 | 0.249 | 0.270 | 0.228 | 288 |
| Llama-3.2-3B-Instruct_gsm8k | clean | 0.344 | 0.404 | 0.341 | 298 |
| Llama-3.2-3B-Instruct_gsm8k | lxt4 | 0.335 | 0.405 | 0.333 | 293 |
| Llama-3.2-3B-Instruct_mmlu | clean | 0.241 | 0.219 | 0.346 | 290 |
| Llama-3.2-3B-Instruct_mmlu | lxt4 | 0.262 | 0.240 | 0.360 | 285 |
| Mistral-7B-Instruct-v0.3_gsm8k | clean | 0.417 | 0.294 | 0.268 | 300 |
| Mistral-7B-Instruct-v0.3_gsm8k | lxt4 | 0.426 | 0.294 | 0.265 | 295 |
| Mistral-7B-Instruct-v0.3_mmlu | clean | 0.412 | 0.270 | 0.290 | 293 |
| Mistral-7B-Instruct-v0.3_mmlu | lxt4 | 0.422 | 0.279 | 0.297 | 295 |

要点:

- 全 36 シャードで Jaccard@10 は 0.15〜0.43 — ランダム期待値を大きく上回り、
  LOO スモーク (0.46, Gemma×GSM8K) と同水準帯。
- **clean と LXT-4 でほぼ不変** (最大差 0.021) — 手法間の overlap は摂動に安定。
- 最良手法はモデル依存: Gemma/Llama×GSM8K は IG、Llama×MMLU は rollout、
  Mistral は両ベンチとも G×I。

### (iv) LOO 再構成: 完了分

ソース: exp-06-attribution worktree `results/loo/*/summary.json` +
`docs/dev_notes_06_attribution.md` (スモーク)。LOO 定義 = 出現ごと削除→タイプ
平均集約 (案B)。

| ラン | n | mean LOO-vs-R_C Jaccard@10 | median |
|---|---|---|---|
| gemma-3-4b-it_gsm8k clean (本番, occurrence) | 300 | **0.4319** | 0.4286 |
| gemma-3-4b-it_gsm8k clean (スモーク, 改行修正後) | 16 | 0.4599 | 0.4286 |

- スモーク→本番 (n=16→300) で mean 0.46→0.43 とほぼ維持。帰属なし (LOO) でも
  AttnLRP R_C 上位集合の約4割が再構成でき、G×I/IG/rollout (0.15〜0.43) の
  上限帯に位置する。
- 案B vs 案A (全出現一括削除) の Top-10 Jaccard: mean 0.755 (スモーク) —
  定義変更でランキングは概ね保持。

### (H6) ρ(J_method@10 | R) 保持表 (宿題1, 2026-07-19 完結)

ソース: 本体 `analysis/exp6_rho_preservation/`(build_rho_preservation.py /
preservation_table.csv / README.md)。exp-06-attribution worktree の
`results/{attribution_family,loo}/*/results.json` を入力に、per-sample J@10 を
アーカイブ `k4_importance/full_results.json` の flip(=answer_changed)・ROUGE-L と
sample_id 結合し、**実験4/Step0 と同一の偏相関 partial_corr(J@10, flip | ROUGE-L)**
(残差+Pearson)で再算出。

**重要な補正**: exp-06 の既存集計(dev_notes 表2)の `ρ(J_method|R)` は実体が
**Spearman(J_method@10, ROUGE-L)**(18/18有意, 0.55〜0.85)で、実験4の ρ(J|R)
(flip 偏相関, ROUGE統制)とは別統計だった。下表は実験4と同一手続きに揃えた偏相関。
`***`=Holm(m=30)p<0.05、`(ns)`=非有意。符号は負が期待(安定性↑→flip↓)。

| 設定 | R_C(n300) | G×I | IG | rollout | LOO(occ) |
|---|---|---|---|---|---|
| gemma-3-4b-it × gsm8k | −0.421 *** | −0.123 (ns) | −0.277 *** | −0.089 (ns) | −0.433 *** |
| gemma-3-4b-it × mmlu | −0.511 *** | −0.007 (ns) | −0.055 (ns) | −0.004 (ns) | −0.206 *** |
| Llama-3.2-3B × gsm8k | −0.479 *** | −0.237 *** | −0.385 *** | +0.001 (ns) | −0.439 *** |
| Llama-3.2-3B × mmlu | −0.617 *** | −0.065 (ns) | −0.077 (ns) | −0.175 (ns) | −0.251 *** |
| Mistral-7B × gsm8k | −0.269 *** | −0.213 *** | −0.546 *** | −0.102 (ns) | −0.418 *** |
| Mistral-7B × mmlu | −0.244 *** | −0.103 (ns) | −0.108 (ns) | −0.104 (ns) | −0.177 *** |
| **符号が負** | 6/6 | 5/6 | 6/6 | 4/6 | 6/6 |
| **Holm有意(負)** | 6/6 | 2/6 | 3/6 | **0/6** | **6/6** |

n = 各セル 255〜288。参考: Spearman(J,ROUGE)版(dev_notes 表2 再現)は全手法 0.34〜
0.85 で rollout が最大だが、偏相関では rollout は 0/6(J が入力変化を反映し ROUGE と
機械的に連動するため、flip への追加説明力なし)。LOO type 感度(Gemma のみ)は occ と
ほぼ同一(gsm8k −0.440 / mmlu −0.201)。

要点:
- **LOO(帰属フリー)は全6設定で負・Holm有意**(−0.18〜−0.44)。LOO–vs–R_C
  Jaccard@10 ≈0.43〜0.46 も予測範囲(0.3〜0.5)内。R3 の leave-one-out 要求への
  最終回答 = 相関構造は AttnLRP 固有ではない。
- 勾配系(IG/G×I)は GSM8K(自由記述)で保持、MMLU(多肢選択)で減衰非有意
  (実験4 H4 の MC 汚染と整合)。rollout は本基準で棄却。
- **H6 判定 = 条件付き支持**: 符号は 21/24 セルで保持(過半)、偏相関の完全な
  符号+有意保持は LOO 6/6・勾配系 GSM8K のみ・rollout 0/6。詳細は
  hypothesis_registry.md H6。

## GLMM 最終推定 (実験1 / 実験5 pooled, 2026-07-18)

ソース: 本体 `analysis/glmm_final/` (README.md +
exp1_glmm_pooled.csv / exp5_glmm_pooled.csv)。推定器: statsmodels
`BinomialBayesMixedGLM` (変分ベイズ, fit_vb; R lme4 はホストに Rscript が無く
使用不可)。設定別 glmm 欄との照合 64/64 (max |Δcoef| = 4.7e-06)。

### 実験1: flip ~ q_typo × cot_typo + (1|item) — pooled 係数 (logit, 事後平均 ± VB SD)

n: lxt4 19,865 item / rnd4 20,087 item / all 39,952 item (× 4セル)。

| pool | model | Intercept | q_typo | cot_typo | q_typo:cot_typo | 収束 |
|---|---|---|---|---|---|---|
| lxt4 | (1\|item) | −11.851 ± 0.016 | +6.458 ± 0.021 | +8.624 ± 0.019 | −5.884 ± 0.026 | OK |
| lxt4 | +(1\|setting) | −11.734 ± 0.017 | +6.542 ± 0.021 | +8.737 ± 0.019 | −5.962 ± 0.026 | OK |
| rnd4 | (1\|item) | −11.961 ± 0.018 | +6.163 ± 0.023 | +7.846 ± 0.020 | −5.587 ± 0.028 | OK |
| rnd4 | +(1\|setting) | −11.751 ± 0.018 | +6.223 ± 0.023 | +7.925 ± 0.021 | −5.643 ± 0.028 | OK |
| all | +(1\|setting) | −12.440 ± 0.012 | +7.016 ± 0.015 | +8.975 ± 0.014 | −6.445 ± 0.019 | OK |

pooled_all の (1|item) 単独は tight 再収束でも縮退が残るため不採用
(+(1|setting) 版とクラスタロバスト marginal を正とする)。クラスタロバスト
marginal (GLM, cluster(item) SE):

| pool | C セル (q のみ) | D − C | B − C |
|---|---|---|---|
| lxt4 | −2.533 ± 0.027 | +1.114 ± 0.030 | +1.357 ± 0.026 |
| rnd4 | −2.753 ± 0.030 | +0.897 ± 0.033 | +1.152 ± 0.028 |
| all | −2.638 ± 0.020 | +1.015 ± 0.022 | +1.262 ± 0.019 |

すべて z ≫ 3。結論: **DE (Cセル) は小さく、CoT 側タイポ (D, B) が flip を支配。
q×cot 交互作用は負 (サブ加法的)** — 両摂動条件で同型。

### 実験5: error ~ condition + (1|item) + (1|setting) — pooled

39,809 item × 2条件 = 79,618 行 (誤答率 raw: LXT-4 51.7% / Matched-Rnd-4 49.5%)。

| term | coef (logit) | sd | z | OR |
|---|---|---|---|---|
| Intercept (LXT-4) | +0.126 | 0.009 | +14.5 | 1.135 |
| cond: Matched-Rnd-4 | **−0.181** | 0.014 | **−12.5** | **0.834** |

re_sd: item 2.20 / setting 1.41、収束 OK。クラスタロバスト代替:
cond = −0.0912 ± 0.0098 (z=−9.3, p=1.6e-20) で符号・有意性一致。結論:
5変数マッチングで表層特性を揃えても **LXT-4 の誤答オッズは約1.20倍
(1/0.834) 高く**、変量効果統制下で高度に有意。

## 実験1+3 / 実験9 拡張グリッド (進行中: 検証バッチ分, 2026-07-18)

第4家族 (Qwen2.5-7B) と第2自由記述 (MATH-500) への拡張のうち、検証シャード
完了分。実験1 ソース: exp-01-03-transplant worktree `results/exp01_03/
{Qwen2.5-7B-Instruct_gsm8k,gemma-3-4b-it_math}_k4_{importance,random}/summary.json`。

### 実験1 4セル分解 (検証シャード)

| 設定 | 条件 | n_incl/n_total | TE | DE | IE | restore | TE照合率 |
|---|---|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct_gsm8k | LXT-4 | 217/1319 | 3.2% | 0.5% | 3.2% | 85.7% | 96.5% |
| Qwen2.5-7B-Instruct_gsm8k | Random-4 | 250/1319 | 3.6% | 0.8% | 4.0% | 77.8% | 97.4% |
| gemma-3-4b-it_math | LXT-4 | 98/500 | 10.2% | 5.1% | 9.2% | 60.0% | 94.0% |
| gemma-3-4b-it_math | Random-4 | 112/500 | 10.7% | 2.7% | 11.6% | 91.7% | 93.8% |

- **IE 優位の分解構造 (DE 小) が新モデル・新ベンチでも保持** (IE/TE ≈ 0.9〜1.1、
  DE ≤ 5.1%)。
- 除外が多い点に注意: Qwen×GSM8K は multi_trigger (CoT 内に答え句トリガが複数)
  が主因 (863〜933件)、MATH は no_trigger (答え句トリガ不検出, 217〜235件) が主因。
  上表の n_incl はこの multi_trigger を素朴に除外した保守的版で、点推定の CI は広い
  (例: Qwen LXT-4 TE CI95 [0.9%, 5.5%])。
- **Qwen multi_trigger 除外の改修と改修前後一致 (B2, Track C 改修済み, commit `4052b2c`)**:
  Qwen の multi_trigger 除外の **98.4% (105,780/107,549 側発火)** は「The answer is X.
  The answer is X.」の**同一答えの単純反復**であり (答えが実際に変わる真の曖昧さは 1.6%
  のみ)、過剰除外だった。最初の宣言直前で切断すれば再生成は最初の宣言を復元するため、
  同一答えの反復は曖昧ではない。opt-in の `dedup_same_answer_triggers`
  (既定 False=従来挙動を完全維持) を有効化すると **n_incl が回復**する:
  gsm8k 217→**839** / mmlu 4→**2285** / mmlu_pro 1→**521** / arc 4→**621**。
  **改修前後一致の検証**: 既存5モデル74シャード + Qwen 18シャードを dedup off で
  再解析すると格納済み summary.json を **0 mismatch で完全再現**するため、48/50 の
  見出し (基底5モデル) は本改修の影響を受けない。改修は Qwen の過剰除外のみを回復する。

### 実験9 inner repair (検証シャード)

ソース: exp-09-inner-repair worktree `results/exp9/summary_{Qwen2.5-7B-Instruct_gsm8k,
gemma-3-4b-it_math}_{lxt4,random4}.json`。sanity_clean_pair 全 PASS。

| 設定 | 条件 | n | repair(flip) | repair(noflip) | lens_typo | lens_clean_self |
|---|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct_gsm8k | lxt4 | 1294 | 0.791 | 0.810 | 0.098 | 0.047 |
| Qwen2.5-7B-Instruct_gsm8k | random4 | 1282 | 0.795 | 0.800 | 0.143 | 0.039 |
| gemma-3-4b-it_math | lxt4 | 242 | 0.998 | 0.998 | 0.271 | 0.897 |
| gemma-3-4b-it_math | random4 | 243 | 0.998 | 0.998 | 0.350 | 0.916 |

- **noflip > flip の方向 (修復スコアが高いほど flip しにくい) は Qwen でも保持**。
- Gemma の repair 飽和 (≈0.998) は MATH でも再現 (モデル固有の性質)。
- Qwen は lens_clean_self が低い (0.04〜0.05) — Mistral と同型のモデル固有現象。

## Phase B / ERDC 拡張: 実験11-17 + 敵対的レビュー + 防御D1 (2026-07-19)

数値ソースは各 `analysis/` 配下の JSON/CSV から直接転記。ERDC 連鎖
(Encode–Repair–Divert–Carry) の段間リンクを閉じる実験群と、敵対的レビュー対応・防御上限。

### 実験11: 連鎖媒介 (H11, G→S2, `analysis/exp11_chain_mediation/`)

第1段 OLS `KL_sum ~ repair_min + 統制` の負有意、および媒介率
`PM = (a_total − a_direct)/a_total`(KL_sum 投入による repair 係数減衰)。flip = TE flip
(B=(typo,typo) vs A=(clean,clean)、included = not exclude & a_correct)。

| グループ | 設定数 | 第1段 負×有意 | 媒介率 中央値 | pooled 媒介率 |
|---|---|---|---|---|
| **core5 (主分析)** | 50 | **35 (70%)** | **0.523** | **GLMM 0.577 / FE 0.578** |
| core5 × MC | 30 | 28 (**93%**) | 0.635 | mean 0.710 |
| all (Qwen 検証含む) | 53 | 36 (68%) | 0.524 | GLMM 0.581 |
| 感度: repair_mean (core5) | 50 | — | — | GLMM 0.638 |
| **反例: MATH シャード** | 11 | **0 (0%)** | **−0.402** | — |

- **H11 = SUPPORTED**: 第1段負有意が過半(70%)、pooled 媒介率 ≥50%(0.577)。修復(弱リンク)の
  flip への効果の約 **58%** が分岐(KL_sum)を経由。KL_sum の flip 係数 **+0.505**(分岐大→flip大)。
  MC タスクで特に強い(第1段 93% 負有意)。→ 「修復失敗→分岐→flip」の S1/G→S2 接続を支持。
- **境界条件(反例)**: MATH では第1段負有意 0/11・媒介率中央値 **−0.40**。MATH では repair↑ が
  KL_sum を下げず媒介が成立しない=修復が分岐を介さず読み出し段へ直接効く別経路(連鎖の分岐)。
- Qwen は Track C dedup-on(`qwen_dedup_exclude.json`)で included を上書き、検証扱い(主判定は core5)。

### 実験12: R_C 組成 (H12, M2, `analysis/exp12_rc_composition/`)

各設定 clean 側 R_C top-10 を {conclusion / numeric / content / function} に分類し、
組成シェアを Δρ(top10)・削除RD と相関。Mistral は必須の再構築ローダー(word_scores 退化を
token_scores 貪欲整列で回避)。

| 家系 | conclusion シェア(平均) | Δρ(top10) 平均 | Δρ>0 割合 |
|---|---|---|---|
| Gemma | 0.169 | +0.252 | 92% (11/12) |
| Llama | 0.136 | +0.383 | 83% (10/12) |
| Mistral | **0.012** | **−0.088** | 17% (1/6) |

| 事前登録予測 | 閾値 | 実測 | 判定 |
|---|---|---|---|
| Gemma/Llama MC 結論句シェア | >0.5 | 平均 **0.130**, 0/16 通過 | **不成立(強形棄却)** |
| Mistral 結論句シェア | <0.3 | 平均 0.012, 6/6 | 成立(自明) |
| GSM8K/MATH 数値+内容 | >0.7 | 平均 0.693, 5/11 | 部分 |
| \|r(結論句, Δρ)\| 全31設定 | ≥0.7 | Pearson **0.184** | **不成立** |

- **H12 強形 = REFUTED**(結論句が top-10 の過半を占めるという前提は過大;実測シェア 0.130)。
- **機構は成立方向**: **MC 20設定に限定**すると r(結論句シェア, Δρ) = **+0.705**(p=0.0005, |r|≥0.7 を満たす)。
  削除RD も結論句シェアと **r=+0.516**(p=0.008)で連動(内容語+数値とは無相関 r=−0.001)。
  全31で薄まるのは GSM8K/MATH(答え定型が numeric に流れ結論句軸が無意味)混入のため。
  → 答え定型は少数派だが Δρ を弁別する軸として機能(閾値を修正)。

### 実験13: 読み出し集中度 (H13, M3, `analysis/exp13_readout_concentration/`)

R_C 分布の Gini(LOO 全語)と削除RD の連動。RD_content(k=4) 主・RD_all も併記。

| model | bench | LOO Gini | LOO top1 | attn Gini | RD_content | RD_all |
|---|---|---|---|---|---|---|
| gemma-3-1b-it | gsm8k | 0.934 | 0.796 | 0.792 | −0.004 | 0.803 |
| gemma-3-1b-it | mmlu | 0.773 | 0.510 | 0.758 | 0.092 | 0.105 |
| gemma-3-4b-it | gsm8k | 0.916 | 0.598 | 0.773 | 0.009 | 0.632 |
| gemma-3-4b-it | mmlu | 0.797 | 0.492 | 0.757 | 0.080 | 0.172 |
| Llama-3.2-1B-Instruct | gsm8k | 0.854 | 0.722 | 0.711 | 0.391 | 0.584 |
| Llama-3.2-1B-Instruct | mmlu | 0.774 | 0.490 | 0.651 | 0.496 | 0.512 |
| Llama-3.2-3B-Instruct | gsm8k | 0.843 | 0.649 | 0.726 | 0.332 | 0.550 |
| Llama-3.2-3B-Instruct | mmlu | 0.741 | 0.452 | 0.639 | **0.479** | 0.478 |
| Mistral-7B-Instruct-v0.3 | gsm8k | 0.868 | 0.722 | 0.670 | 0.005 | 0.263 |
| Mistral-7B-Instruct-v0.3 | mmlu | 0.713 | 0.458 | 0.610 | **0.026** | 0.076 |

- 家系 Gini: **Gemma 0.855 > Llama 0.803 > Mistral 0.790**(事前登録 Llama>Gemma>Mistral とは Gemma/Llama 逆転)。
- rank-corr(LOO Gini, RD_content) = **−0.564**(p=0.090) → 事前形棄却(scope 不一致で逆符号)。
- rank-corr(LOO Gini, RD_all[scope一致]) = **+0.782**(p=0.0075, ≥0.7 を満たす)。
  集中度が numeric/機能語由来のとき(Gemma gsm8k 等)content 削除には効かない。
- **H13 事前形 = REFUTED / scope 一致機構は成立**。
- **Mistral 二重乖離**: 観察的集中(LOO Gini 0.87/0.71、内容語質量シェア)は Llama と同程度でも
  RD_content が桁違いに低い(mmlu **0.026** vs Llama-3B **0.479**、gsm8k 0.005 vs 0.332)=因果読み出しが冗長/分散(削除に強い)。

### 実験14: no-CoT ショートカット (H14, 残差DE, `results/exp14_nocot/analysis/`)

no-CoT(空 CoT で即答強制)flip と DE の連動。設定数 72(回帰採用 n=60)。

| 層別 | rank-corr(noCoT_flip, DE) | n |
|---|---|---|
| 全設定 | **−0.036** (p=0.79) | 60 |
| MC 課題のみ | **+0.726** (p<0.001) | 40 |
| 生成課題のみ | **+0.633** | 20 |
| (参考) 全設定 noCoT_flip~IE | +0.578 | 60 |
| (参考) MC noCoT_flip~IE | +0.755 | 40 |

- サンプル OR(Mantel-Haenszel) = **8.85**(crude 10.12)。
- 事前登録判定: rank≥0.7 = False, OR>3 = True → **H14 リテラル = 不支持**。
  全設定 ρ≈0 は **Simpson 型**(MC/生成で層別すると ρ=+0.73/+0.63 と強正)→ **Simpson 機構は支持**。
- 鋭い予測(Gemma-1B×CSQA が DE>IE の唯一設定): 全設定 top25% = False(importance rank 9/60)だが
  **MC 課題内では top25% = True(importance rank 2/40, percentile 0.05)**。
- noCoT_flip は DE 特異でなく typo 感受性全般(IE とも連動)を反映 → 「DE = 直接読み出し成分だが
  タスク横断の単一指標ではない」と記録。

### 実験17: 行動修復 (H17, M1 行動形, `analysis/exp17_behavioral_repair/`, DeepSeek-R1-Distill-Qwen-7B)

自己訂正マーカーと flip の共起(OR of flip given repair marker)。included = baseline-correct。

| task | strict-cue OR [95%CI] | broad(cue\|markC) OR [95%CI] |
|---|---|---|
| MATH | **2.76 [1.76, 4.33]** | 2.63 [1.83, 3.77] |
| GSM8K | **2.96 [2.12, 4.14]** | 2.25 [1.73, 2.93] |
| MMLU | **1.98 [1.70, 2.30]** | 1.48 [1.31, 1.67] |

- **H17 = REFUTED(逆方向)**。事前登録は「訂正マーカー→flip 抑制(OR<1)」だったが、全 task で
  OR>1・CI が 1 を除外 → マーカーは flip と**共起**(=タイポに気づき解釈を彷徨う「難儀」信号、成功修復ではない)。
- R_Q 単調性なし(MATH markC は R_Q 五分位で平坦)。R1×MATH の Random>LXT 逆転も importance/random で
  マーカー率が等しく行動非対称なし → **M1 を表現レベル(実験9 隠れ状態コサイン)に一本化**。逆転は
  摂動トークンの構造的性質(Track C)に帰属。手動 FP 監査 17/20 が真陽性。

### 敵対的レビュー B 群 (`analysis/{b1_exp2_edit_balance,b4_exp3_entropy,b5_natural_typo_correctors,b6_choice_letter_bias}/`)

| 群 | 攻撃(交絡/自明化) | 主結果(ソース直接) | 判定 |
|---|---|---|---|
| **B1** | 実験2 の flip は削除操作/位置の交絡では | replace で flip 消失(Llama-3B mmlu: top **delete 0.771 → replace 0.047**, matched 0.226→0.011)。edit_pos+n_spans 統制後も is_top OR≫1 有意(content_k4 OR=10.5, p=1e-28) | 削除操作特異だが grammar 統制後も R_C 選択性は残る |
| **B4** | 実験3 の相補性は低エントロピー機械成分では | 回避の **数値タスク 64% / MC 6%** がエントロピーで説明(pooled ALL 50%)。数値タスクは entropy 統制後も残差有意(gsm8k Wilcoxon p=3.3e-94 等) | 相補性は弱まるが数値タスクで有意残差 |
| **B5** | neural 校正の優位は合成 typo 限定では | neural(T5)優位は**自然 typo でも保持**(gsm8k nat +0.335 / mmlu nat +0.229 vs pyspell)。自然/合成の word_restoration 差はほぼ 0 | 反証成功 |
| **B6** | DE(cell C)flip はランダムか | DE flip は**第1選択肢(A)へ系統偏り**(pooled 4-opt χ²=39.3, **p=1.5e-8**, A share 0.358 vs base 0.244; 16 model×bench 中 11 が most-over=A) | 位置/ラベルバイアス確認(Cramér V ≈0.12–0.16) |

A1(Mistral 監査)・A2(restore 反証)は `experiment_details.md` の実験4/3・実験1+3 節の監査段落に反映。

### 防御実験 D1: 重要語優先校正オラクル (`exp-20-defense` worktree)

因果地図の処方箋を上限として測る(手法非依存、生成のみ)。摂動語のうち上位 k 語だけ clean に戻す
oracle / 下位 k の inverse / ランダム k の random ×(k=1,2,3)、端点 k0=full-typo, k4=clean。

**データ構築完了(6設定 = M3×B2, flip サブセット × 11 条件)** — `results/d1_datasets/build_summary.json`:

| model | bench | n_flipped | n_separable | separable 保持率 |
|---|---|---|---|---|
| gemma-3-4b-it | gsm8k | 128 | 127 | 0.992 |
| gemma-3-4b-it | mmlu | 364 | 348 | 0.956 |
| Llama-3.2-3B-Instruct | gsm8k | 195 | 193 | 0.990 |
| Llama-3.2-3B-Instruct | mmlu | 445 | 424 | 0.953 |
| Mistral-7B-Instruct-v0.3 | gsm8k | 147 | 138 | 0.939 |
| Mistral-7B-Instruct-v0.3 | mmlu | 381 | 352 | 0.924 |

- 端点は「同一入力の再生成」(restore=全→clean にバイト一致)= 環境整合性チェックを兼ねる。
- スモーク(gemma-3-4b-it×gsm8k, n=16)で構築パイプライン検証(separable 保持率 1.0)。
- **生成・回復曲線は進行中**(オーケストレータ起動済み、結果未取得)。事前登録判定「律速=重要語復元精度」=
  pooled k=1 で **oracle 回復率 > random > inverse** かつ oracle vs inverse 有意。

### サイズ梯子 (進行中, `exp-19-size-ladder` worktree + `analysis/size_ladder_results/`)

- **Gemma-3-12B × GSM8K clean acc = 92.2%**(1216/1319、baseline `results.json` 実測)。12B は backward 可で R_Q を通常取得。
- **Gemma-3-27B R_Q 経路確定(2026-07-19)**: AttnLRP + `--freeze_params --grad_checkpointing`(経路 a)で
  backward 通過を確認(feasibility: gen_peak 51.7GB / rq_peak 61.4GB, rq_ok=true, rq_nonzero_relevance=64,
  rq_topk = ducks/Janet/lay/…)。梯子は主分析(25–31設定)とは統計分離した確証レプリケーション層。
