# 実験16 (P6): 家族効果の吸収テスト — 判定

**判定: REFUTED(反証)** — 家族効果は ERDC 共変量で縮小・消失せず、むしろ**増大(抑圧/suppression)**する。

全数値は `exp16_absorption.json`(ディスク上、`run_exp16_absorption.py` が生成)由来。

## 手法
- **fitter(主):** 固定効果ロジット + cluster-robust SE(cluster = `setting`, 40クラスタ)。
  BBGLM(VB) は交差ランダム効果(item ~17k水準)で非現実的なため、task 指定の
  fallback を主解析に採用。**BBGLM(1|setting) を robustness で併記し、同一パターンを確認。**
- family は 4 水準あるが、`features_partial` の `family` 列は Qwen 設定で 18701 行 NaN
  (exp12 moderator 欠損)。`model` から決定論的に family を再構成し完全化(Qwen 含む)。
- 連続共変量は complete-case 上で z 標準化。family は treatment coding(基準 = Gemma)。
- ネスト: M0 `flip~family` → M1 +carry(rouge,jaccard) → M2 +repair_min,kl_sum
  → M3 +delta_rho,rq_mean,split_mean。全段を**同一 complete-case** で fit。

## 使用 N と脱落
- included: **45,339** / complete-case(全 ERDC 特徴 non-null): **34,190** / 除外 **11,149**。
- complete-case の家族: **Gemma / Llama / Mistral の3家族のみ**。benchmark: arc/csqa/gsm8k/mmlu(math脱落)。設定: 40。
- **Qwen は M1+ に投入不可**: rouge_l_f1 / cot_jaccard_top10 が 0 件、kl_sum ≈6.5%、
  delta_rho ≈3.7% しか無く(Step0 CoT-ROUGE 未計算 + exp12 moderator 欠損)、
  carry 以降の共変量が全欠損。→ **吸収テストは Qwen を検定できない**。
- **math** は全モデルで ROUGE 欠損のため complete-case から脱落。

## 結果(family log-odds, 基準=Gemma)
| model | Llama coef (SE, p) | Mistral coef (SE, p) | family L2 | L2 縮小率 vs M0 | family block Wald p (Holm) |
|---|---|---|---|---|---|
| M0 base | +0.304 (0.201, .131) | −0.198 (0.201, .324) | 0.363 | 0% | 8.1e-3 (8.1e-3) |
| M1 +carry | +0.197 (0.262, .451) | **−3.991 (0.418, 1.2e-21)** | 3.996 | **−1000%** | 2.5e-37 (1.0e-36) |
| M2 +repair/divert | +0.438 (0.308, .156) | **−3.755 (0.437, 9.0e-18)** | 3.780 | **−941%** | 5.8e-35 (1.7e-34) |
| M3 +readout/enc | +0.322 (0.358, .368) | **−3.435 (0.448, 1.9e-14)** | 3.450 | **−850%** | 1.2e-15 (2.4e-15) |

- **縮小率が負 = 増大**。仮説は縮小(→0)を予測したが、family L2 log-odds は M0 の 0.363 から
  M1 で 3.996 へ約 11 倍に膨張。full model(M3)でも family ブロックは **極めて有意**
  (Wald p = 1.2e-15, Holm p = 2.4e-15)。**非有意化せず、逆に強化**。
- 駆動源は主に **Mistral**: carry を統制すると Mistral の flip 傾向が Gemma 比で桁違いに低下
  (odds ratio e^−3.4≈0.03)。Llama は M1 で一旦 35% 縮小するが不安定(M2で符号反転的に増大)。
- **BBGLM(1|setting) が同一パターンを再現**(Mistral: M1 −4.199, M3 −3.839, いずれも p≪1e-10)。
  → cluster-robust の数値アーティファクトではなく、2 fitter で一致する**条件付き抑圧効果**。

## 頑健性
- **M3 + gini_rc**(exp13 loo_gini, gsm8k/mmlu subset, N=19,296, Qwen 無し): family は再び増大
  (M0subset Mistral −0.099 → M3+gini −2.137)。gini_rc_z 自体は有意(coef +0.334, p=0.011)だが
  family を吸収しない。
- **記述用 4家族 M0**(全 included, N=45,339): Qwen が最大の周辺 family 効果(−0.912, p=6.9e-5)。
  しかし Qwen は共変量欠損で吸収テスト対象外。

## 解釈と正直な限界
1. **P6 は反証**。「家族差は ERDC パラメータの違いに還元できる」という主張は成り立たない。
   ERDC 共変量を統制すると family 効果は縮小するどころか**増大**し、full model でも高度に有意。
   = family は ERDC 連鎖で**説明されない残差的・条件付き差**を持つ(特に Mistral の抑圧)。
2. ただし **M0 の周辺 family 効果自体が小さい**(個別 dummy は n.s., block p=0.008 と弱め)。
   増大は「小さい周辺効果が共変量統制で顕在化する suppression」であり、絶対的な family 主効果が
   元から大きいという主張ではない。
3. **検定範囲が限定的**: 3家族 / 4 benchmark のみ。**周辺 family 効果が最大の Qwen を検定できていない**
   (carry 特徴が全欠損)。この意味で P6 の完全な検証は未達であり、反証は「検定可能な範囲での反証」。
4. Qwen を吸収テストに載せるには Step0 の CoT-ROUGE / Jaccard と exp12 moderator を Qwen で
   再計算する必要がある(本 exp16 の範囲外)。

## 未マージ項目(family-dummy 版時点)
- **exp14 noCoT_flip(per-sample)**: sample_id に結合可能な per-sample テーブルは非実在。
  `results/exp14_nocot/analysis/h14_summary.json` は設定レベル CoT×noCoT 2×2 分割表のみ、
  `settings.csv` は設定レベル `nocot_flip_rate` のみ。family-dummy 版 GLMM(sample レベル ERDC)
  には非投入。**ただし下記 H16 正式判定では設定レベル `nocot_flip_rate` を M3b として結合・使用。**

---

# H16 正式判定(設定分散 吸収)

**判定: SUPPORTED(A = 0.645 ≥ 0.5)** — 事前登録 H16(registry 712-736)の estimand
「設定レベル ERDC モデレーターが設定分散 σ²_setting を A≥0.5 吸収」を満たす。

全数値は `exp16_h16_absorption.json`(`run_exp16_h16.py` 生成)由来。

## 手法(前回の family-dummy 版とは別 estimand)
- **fitter:** `BinomialBayesMixedGLM`(VB), ランダム切片 `(1|setting)`。item RE(~17k交差水準)は
  非現実的なため `(1|setting)` のみ。σ² は VB 事後の log-SD パラメータより **σ² = exp(2·vcp_mean)**。
- base `flip ~ 1 + (1|setting)` と mod `flip ~ M1+M2+M3a+M3b + (1|setting)` を**同一 complete-case** で fit。
- **A = 1 − σ²_setting(mod)/σ²_setting(base)**。
- 設定レベルモデレーター(全て z 標準化):
  M1 = repair_min 設定平均, M2 = share_conclusion(exp12), M3a = gini_rc(exp13 loo_gini),
  M3b = nocot_flip_rate(exp14 settings.csv, no-CoT shortcut)。

## 使用設定数 / N と脱落
- **Primary(4モデレーター): 20 設定, N=20,500**(gsm8k+mmlu, 3家族 Gemma/Llama/Mistral, 両条件)。
  gini_rc が gsm8k/mmlu×5非Qwenモデルのみのため、**Qwen と arc/csqa/math 設定は gini 欠損で脱落**。
- no-gini(3モデレーター)版は 52 設定 / N=36,740(Qwen・全benchmark 含む, 広被覆)。

## 結果
| model | σ²_setting(base) | σ²_setting(mod) | **A** | 判定 |
|---|---|---|---|---|
| Primary 4-mod (20設定, N=20,500) | 0.3427 [0.184, 0.637] | 0.1215 [0.065, 0.228] | **0.645** | Supported |
| + family covariate (20設定) | 0.3427 | 0.1139 | **0.668** | Supported |
| no-gini 3-mod (52設定, N=36,740) | 0.4925 | 0.2309 | **0.531** | Supported |

**mod のモデレーター係数(z, Primary):**
- gini_rc_z: coef −0.697 (p≈0) — 読み出し集中が高い設定ほど flip 少(頑健)。
- nocot_flip_rate_z: coef +0.693 (p≈0) — no-CoT shortcut 依存の設定ほど flip 多(脆弱)。
- share_conclusion_z: coef +0.051 (p=5.6e-3) — 小さいが有意。
- repair_setting_mean_z: coef +0.014 (p=0.45) — 設定分散への寄与は n.s.。
→ 設定分散の吸収は主に **M3 の2成分(gini 集中 + no-CoT shortcut)** が担う。M1(修復)は
  設定レベルでは寄与小(no-gini 版では repair coef −0.176, p≈0 と有意になり被覆依存)。

**falsification branch は非発動**(A≥0.5)。二層レジーム分類は本 estimand では不要。

## family-dummy 版(前節)との整合(正直な解釈)
- 2つは**別 estimand**であり矛盾しない。
  - **H16(設定分散)**: 設定間の flip 異質性は ERDC 設定モデレーターで 64.5% 説明できる → Supported。
    しかも **family を足しても A は 0.645→0.668 とほぼ不変**(family は設定分散をほとんど追加説明しない)。
  - **family-dummy 版(サンプルレベル)**: 残差的な**条件付き family 効果**(特に Mistral)は ERDC 共変量で
    吸収されず、むしろ増大(suppression)。H16 の +family 版でも mod 内 Mistral coef −0.539 (p≈0) が残存し、
    この suppression 所見と整合。
- まとめ: **ERDC は設定間分散の大半を説明する(H16 Supported)が、家族(Mistral)固有の条件付き差は
  設定分散とは別次元で残る(family suppression)**。ERDC 連鎖は「どの設定が壊れるか」をよく説明するが、
  「Mistral が条件付きで極端に頑健」という家族特異性までは還元しきれない。
