# 実験11〜18 マスター計画: ERDC連鎖の因果閉鎖と異質性の統一

本書は、ユーザーが 2026-07-19 に提示した**統一仮説 ERDC連鎖(Encode–Repair–Divert–Carry)**と、
それを検証・因果閉鎖・統一するための実験11〜18の正典計画である。Phase A(実験1〜10)で確立した
事実を前提に、(1) 連鎖の各段の因果的接続を閉じ、(2) モデル×ベンチ間の異質性を単一連鎖の
3モデレーターで吸収する。各実験の {目的・手法(数式)・設定・GPU見積・依存関係・出力先・
事前登録予測・反証分岐} を詳細化する。事前登録の判定基準は `docs/hypothesis_registry.md` の
H11〜H18 と一対一で対応する。

- 実装先: `/diskthalys/ssd14tc/sfukuhata/dev/kanolab/typo_robust_analysis/projects/typo-cot`(モジュール `typo_cot`)
- Phase A の正典: `docs/experiment_plan.md`(実験1〜10)、`docs/hypothesis_registry.md`(H1〜H10 判定)
- 本計画の仮説定義: `docs/hypothesis_registry.md` §ERDC Unified Hypothesis(H11〜H18)

---

## 0. 統一仮説: ERDC連鎖(Encode–Repair–Divert–Carry)

typo による回答変化は、次の4段の因果連鎖の可視的な終点である。

- **S1(Encode)** — typo摂動語の**早期層における語彙符号化の損傷**として発生する。
  実験8で **12/12 条件が早期層局在**を示した。
- **G(Repair gate)** — **内部修復に失敗した損傷のみ**が前方へ通過する内部修復ゲート。
  実験9で修復スコアが flip の負予測子(**負方向 47/50** の診断)。**モデレーター M1 = ゲートGの通過率**。
- **S2(Divert)** — 通過した損傷が **CoT生成を少数の分岐点で逸らす**。
  実験3で KL が少数位置に集中し、**flip群 > noflip群 が 50/50**。
- **S3(Carry)** — **逸れたCoTテキストが効果の大部分を回答まで搬送する**。
  実験1で **IE/TE ≈ 0.8**、clean CoT 強制で **restore 90%+**。
- **残余(DE)** — 読み出し段の **CoT迂回ショートカット**(直接効果)。**モデレーター M3** が支配。

**異質性の3モデレーター**(連鎖は単一だが、各段の強度が設定間で異なる):

| ID | モデレーター | 定義 | 主データ源 | 検証実験 |
|---|---|---|---|---|
| **M1** | 修復能力 | ゲートGの通過率(修復スコア分布) | 実験9 repair_score | 実験11・16・17 |
| **M2** | 答え決定情報の所在 | R_C上位質量の組成(結論句/数値/内容語シェア) | 実験4 fixed R_C | 実験12・16 |
| **M3** | 読み出し集中度+ショートカット依存 | R_C分布の Gini + no-CoT ショートカット依存度 | 実験2 削除RD, 実験1 DE | 実験13・14・16 |

### 0.1 ERDC連鎖リンク図

```
                         typo(摂動4語)
                              │
                              ▼
   ┌───────────────────────────────────────────────────┐
   │ S1 Encode  早期層の語彙符号化損傷                   │  ← 実験8: 12/12 早期層局在
   └───────────────────────────────────────────────────┘
                              │
                    ╔═════════╪═════════╗
                    ║  G Repair gate     ║              ← 実験9: 負方向 47/50 / M1=通過率
                    ║  修復失敗分のみ通過 ║   ◄─ 実験11(G→S2 媒介の検定)
                    ╚═════════╪═════════╝
                              │
   ┌───────────────────────────────────────────────────┐
   │ S2 Divert  CoT生成を少数の分岐点で逸らす            │  ← 実験3: KL集中, flip>noflip 50/50
   └───────────────────────────────────────────────────┘
              ▲                                    │
              ╎  ★ 実験15: 早期窓patch→自由生成で   │
              ╎     S1→S2 を因果的に閉じる(本計画の要)│
              ╎  (S1をdoして S2のCoT分岐・flipを誘発) │
              └╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┘
                              │
   ┌───────────────────────────────────────────────────┐
   │ S3 Carry  逸れたCoTが効果を回答まで搬送            │  ← 実験1: IE/TE≈0.8, restore 90%+
   └───────────────────────────────────────────────────┘
                              │
                ┌─────────────┴──────────────┐
                ▼                             ▼
          [回答変化(TE)]          残余 DE: 読み出し段のCoT迂回  ← M3 / 実験13・14
```

**因果閉鎖の要点(強調)**: Phase A は連鎖の各段を**個別に**観測した(S1=実験8、G=実験9、
S2=実験3、S3=実験1)が、段と段の**接続**は相関的にしか結ばれていない。実験11〜18 の中核目的は
この接続を因果で閉じることであり、とりわけ **S1→S2 は実験15(早期窓 activation patching →
自由生成)で閉じる**。実験15 は「早期層の符号化(S1)を do(clean→pert / pert→clean)したとき、
下流の CoT が実験3で定義した分岐点で逸れる(S2)か」を直接測る。ここが閉じると、
「早期符号化損傷 → CoT分岐 → 回答搬送」の全経路が介入で裏づけられ、ERDC は
相関連鎖から**因果連鎖**へ格上げされる。

### 0.2 仮説 × ERDC段 × 実験 対応表

| 仮説 | 短名 | ERDC上の役割 | 検証実験 | Tier |
|---|---|---|---|---|
| H11 | Chain Mediation | G→S2 の媒介を検定(連鎖媒介) | 実験11 | 1 |
| H12 | R_C Composition | M2(答え決定情報の所在)の定量 | 実験12 | 1 |
| H13 | Read-out Concentration | M3(読み出し集中度) | 実験13 | 2 |
| H14 | no-CoT Shortcut | 残余 DE(CoT迂回ショートカット) | 実験14 | 2 |
| H15 | Patch→Free Generation | **S1→S2 の因果閉鎖** | 実験15 | 3 |
| H16 | Unified GLMM | M1・M2・M3 による異質性の統一吸収 | 実験16 | 1(骨格)+2(完成) |
| H17 | Behavioral Repair | M1 の行動的発現(明示的訂正) | 実験17 | 1 |
| H18 | Format Transplant | S3/DE の形式依存(移植実験) | 実験18 | 3 |

---

## 1. Tier構成と実行順

設計原理は Phase A と同じ2つ: **上流依存**(既存出力の再解析が先、新規生成が後)と、
**GPUと人手の並行**(Tier 1 は既存出力の再解析なので GPU をほぼ使わず今日から回る)。

| Tier | 実験(実行順) | GPU見積 | 性質 |
|---|---|---|---|
| **0** | 実験6(ρ保持)/ 実験7(R_Q偏在) | 別トラック進行中(Track A 完了) | 既存出力の CPU 集計。`docs/followup_plan_20260719.md` 参照 |
| **1** | 実験12 → 実験11 → 実験16骨格 → 実験17 | **GPU 0日** | Phase A 出力(実験4/9/3/1/10)の再解析中心。CPU 集計 |
| **2** | 実験14 → 実験13 | **各 ≈1日** | no-CoT 生成(実験14)と削除RD補完(実験13) |
| **3** | 実験15 → 実験18 | **各 1〜2日** | 介入生成: 早期窓patch→自由生成(15)、形式移植生成(18) |

- **総GPU予算: 約 4〜6 GPU日**(Tier 1=0 + Tier 2≈2 + Tier 3≈3〜4)。
- Tier 0 は本計画外の別トラックで**すでに完了**(H6=条件付き支持、H7-4=支持)。本計画はこれらの
  確定結果を M2/M3 の補助証拠として引用する。
- **「GPU0」の意味**: Tier 1 の4実験はいずれも Phase A の既存 results.json / 統合テーブルを
  再結合・再回帰するのみで新規推論を要しない(実験17 は実験10 で生成済みの R1蒸留系 CoT を再利用)。
  よって GPU 見積は 0日(全て CPU 集計)。実験10 の R1蒸留系生成が未完了の場合のみ、
  実験17 はその完了を待つ(§3.7 の依存関係参照)。

### 1.1 依存グラフ

```
Phase A 出力                           実験11〜18
─────────────                          ──────────
実験4 (fixed R_C) ─────────────┬─────► 実験12 (M2 R_C組成)  ─┐
                               └─────► 実験13 (M3 Gini)      │
実験9 (repair_score) ──────────┬─────► 実験11 (G→S2 媒介)    ├─► 実験16 (統一GLMM)
実験3 (KL_sum, onset) ─────────┘                             │        ▲
実験1 (IE/TE/DE, cell C) ──────┬─────► 実験14 (no-CoT DE)    ─┘        │
                               └─────► 実験15 (S1→S2 閉鎖) ◄── 実験8 (早期窓)
実験2 (削除RD) ────────────────────►  実験13                         │
実験10 (R1蒸留CoT) ─────────────────► 実験17 (行動的修復) ───────────┘
実験1(restore)+実験4(Δρ) ──────────► 実験18 (形式移植)
```

実験16 は Tier 1 で **M1(実験9)+M2(実験12)** を投入した**骨格**を組み、Tier 2 完了後に
**M3(実験13)** を追加して**完成**させる(2段階ビルド)。

---

## 2. 3週間スケジュール

単一GPU直列を前提とし、Tier 1 は GPU を使わないため**第1週に人手だけで前倒し完了**できる。

| 週 | GPU(直列) | 人手(並行) | マイルストーン |
|---|---|---|---|
| **第1週** | (Tier 1 は GPU 不要) | 実験12 → 11 → 16骨格 → 17 を実行(全て CPU 再解析)。実験14・15 の生成スクリプト実装 | M1・M2 の確定、H11・H12・H17 判定、H16 骨格 |
| **第2週** | 実験14(≈1日)→ 実験13(≈1日) | 実験16 に M3 を投入し**完成**。実験15・18 の介入器実装 | M3 の確定、H13・H14 判定、H16 完成(H16 の異質性吸収率を算出) |
| **第3週** | 実験15(1〜2日)→ 実験18(1〜2日) | 統一主張の執筆、ERDC 因果図の SVG 化、revision notes への H11〜H18 反映 | **S1→S2 因果閉鎖(H15)**、H18 判定、統一主張の確定 |

- **クリティカルパス**: 実験12(M2)→ 実験16骨格 → 実験13(M3)→ 実験16完成。H16 の異質性吸収は
  M1・M2・M3 が揃って初めて算出できるため、Tier 1→2 のこの鎖が全体の律速。
- **最重要成果物**: 第3週の**実験15(S1→S2 因果閉鎖)**。ここが ERDC を因果連鎖として確定する。
- **調整弁**: 締切逼迫時は Tier 3 の実験18(形式移植)を後回し(H18 は連鎖本体でなく S3/DE の
  形式依存を問う付随仮説)。実験15 は連鎖閉鎖の要なので死守。

---

## 3. 実験別詳細(実験11〜18)

各実験は `docs/hypothesis_registry.md` の同番号 H と一対一。数式・判定基準は両文書で一致させる。

### 3.1 実験11: 連鎖媒介(G→S2)[Tier 1, GPU 0日]

- **目的**: ERDC の **G(修復)→ S2(CoT分岐)** の接続を媒介分析で閉じる。修復の失敗が
  CoT分岐の増大を**媒介して**回答を変えるのか、それとも修復が読み出し段に**直接**効くのかを分離する。
- **手法(数式)**:
  1. **第1段(G→S2)**: 設定ごとに `KL_sum_i = β0 + β1·repair_min_i + ε_i`。
     `repair_min` = サンプル i の層横断の**最小修復スコア**(最も直せなかった層=損傷の残存量)、
     `KL_sum` = 実験3の位置別 KL の総和(S2 の逸れの総量)。予測: **β1 < 0 が過半の設定で有意**
     (修復が悪いほど CoT が逸れる)。
  2. **第2段(媒介)**: `flip ~ repair + (1|item)`(周辺効果 β_repair^total)と
     `flip ~ repair + KL_sum + (1|item)`(KL_sum 統制後 β_repair^direct)を比較。
     **媒介割合** `PM = (β_repair^total − β_repair^direct) / β_repair^total ≥ 0.5`
     (repair の直接効果が KL_sum 統制で **50%以上減衰**)。bootstrap で PM の 95%CI、Holm。
- **設定**: 実験9 と実験3 が両方揃う設定(M3×B2 + 修復スコアが計算できる M5×B5 の交わり)。
  サンプルは実験3・9 の共通 sample_id 結合。新規推論なし。
- **GPU見積**: **0日**(実験9 repair + 実験3 KL の per-sample 結合回帰、CPU)。
- **依存関係**: 実験9(repair_score / repair_min)、実験3(KL_sum)、実験1・2(flip)。実験12 とは独立。
- **出力先(提案)**: `analysis/exp11_chain_mediation/`(mediation_table.csv / stage1_glmm.json /
  summary.json)。
- **事前登録予測(H11)**: 第1段 `β1(KL_sum~repair_min)` が負に有意な設定が過半。
  かつ repair の直接効果が KL_sum 統制で **PM ≥ 0.5**(50%以上減衰)。
- **反証分岐**: PM が 50% に届かず repair の直接効果が残る場合 → 修復は S2 を経由せず
  **読み出し段に直接効く**経路を持つ。この場合 ERDC を**連鎖の分枝修正**として報告
  (G から S3/読み出しへの直接枝を追加。連鎖仮説の棄却ではなく分枝化)。

### 3.2 実験12: R_C組成(M2)[Tier 1, GPU 0日]

- **目的**: **M2(答え決定情報の所在)**を定量化する。R_C 上位質量が結論句・数値・内容語の
  どこに載るかがモデル/形式で異なることを示し、これが fixed-target の減衰 Δρ(実験4)を説明するか
  を検証する。
- **手法(数式)**:
  1. 実験4 の fixed-target R_C ランキング上位質量を語型別に分類(結論句 / 数値 / 内容語 / 機能語)。
     `conclusion_share = (結論句トークンへの R_C 質量) / (上位 top-k 質量の総和)`
     (数値・内容語シェアも同様)。分類は答え句検出(`evaluation/extractor.py`)+ POS タガー + 数値正規表現。
  2. 設定レベルで `r(conclusion_share, Δρ)` を算出(Δρ = ρ_fixed − ρ_default、実験4より)。
- **設定**: 実験4 が回った全設定(最大 M5+Qwen × B5+MATH)。新規推論なし。
- **GPU見積**: **0日**(実験4 の R_C を再利用 + タグ付け、CPU)。
- **依存関係**: 実験4(fixed R_C・Δρ)。実験13 と R_C 分布を共有。
- **出力先(提案)**: `analysis/exp12_rc_composition/`(composition_by_setting.csv /
  share_vs_deltarho.csv / summary.json)。
- **事前登録予測(H12)**:
  - Gemma/Llama 系 **MC で結論句シェア > 0.5**、**Mistral < 0.3**。
  - **GSM8K/MATH で数値+内容語シェア > 0.7**。
  - 設定レベル **|r(結論句シェア, Δρ)| ≥ 0.7**。
- **反証分岐**: シェアと Δρ の相関が弱い(|r| < 0.7)場合 → M2 は Δρ の主因ではなく、
  減衰は別要因(実験4 の family×format 交互作用=H4 で確認済みのアーキテクチャ差)に帰属。
  M2 を「補助モデレーター」に格下げして報告。

### 3.3 実験13: 読み出し集中度(M3)[Tier 2, GPU ≈1日]

- **目的**: **M3(読み出し集中度)**を定量化し、R_C が少数トークンに集中するモデルほど
  削除介入(実験2)の効果が大きいことを示す。読み出しの「一点依存」がショートカット依存と表裏で
  あることの証拠。
- **手法(数式)**:
  1. 各サンプルの R_C 分布に **Gini係数** `G = (Σ_i Σ_j |x_i − x_j|) / (2 n Σ_i x_i)` を適用
     (x = CoTトークン別 R_C、G→1 で一点集中)。設定別に平均 Gini。
  2. **削除RD**(risk difference)= 実験2 の top-R_C 削除 flip率 − ランダム削除 flip率。
  3. 設定横断で `rank-corr(mean Gini, 削除RD)` を算出(Spearman)。
- **設定**: 実験2 が回った設定 + R_C 分布が取れる設定。削除RD の穴埋めに一部設定で追加削除生成が
  必要な場合のみ GPU 使用。
- **GPU見積**: **≈1日**(削除RD の未取得設定を補完生成、他は再解析)。
- **依存関係**: 実験4(R_C 分布)、実験2(削除RD)。実験12 と R_C を共有。実験16 の M3 入力。
- **出力先(提案)**: `analysis/exp13_readout_gini/`(gini_by_setting.csv / gini_vs_rd.csv / summary.json)。
- **事前登録予測(H13)**: Gini の設定順位が **Llama > Gemma > Mistral**、かつ
  **rank-corr(Gini, 削除RD) ≥ 0.7**。
- **反証分岐**: 順位が崩れる/相関が弱い場合 → 読み出し集中度は削除感受性を説明せず、
  M3 を「ショートカット依存(実験14)」単独へ再定義。Gini は記述指標として付録送り。

### 3.4 実験14: no-CoTショートカット(残余DE)[Tier 2, GPU ≈1日]

- **目的**: 残余の **直接効果 DE(読み出し段の CoT迂回ショートカット)** の実体を、
  CoT を経由しない直接回答条件で測る。DE が大きい設定ほど「CoT を使わずに答えられてしまう」
  ことを示す。
- **手法(数式)**:
  1. **no-CoT条件**: 質問直後に答え句を強制し CoT を生成させない(teacher-forcing で CoT スパンを空に)。
     clean/typo で `noCoT_flip` = no-CoT 条件下の flip率。
  2. 設定横断 `rank-corr(noCoT_flip, DE)`(DE = 実験1 の直接効果 P(flip|cell C))。
  3. サンプルレベルの重なり: no-CoT で flip する集合と実験1 で DE を示す集合の
     **オッズ比 OR**(2×2 分割表)。
- **設定**: 実験1 が回った設定(7モデル×6ベンチ相当)。no-CoT 直接回答の新規生成。
- **GPU見積**: **≈1日**(答えスパンのみの短生成、backward 不要)。
- **依存関係**: 実験1(DE)。実験13 と M3 を共有(集中度⇄ショートカット依存の表裏)。
- **出力先(提案)**: `analysis/exp14_nocot_shortcut/`(nocot_flip.csv / overlap_or.csv / summary.json)。
- **事前登録予測(H14)**: `rank-corr(noCoT_flip, DE) ≥ 0.7`、サンプル重なり **OR > 3**。
  **鋭い予測**: **Gemma-3-1B × CSQA(Phase A で唯一 DE > IE の設定)が
  ショートカット依存度ランキングの最上位圏**に来る。
- **反証分岐**: 相関が弱い、または Gemma-1B×CSQA が最上位に来ない場合 → DE は CoT迂回
  ショートカットでは説明できず、DE を「読み出し段の残留符号化損傷」など別機構へ再解釈。

### 3.5 実験15: 早期窓 activation patching → 自由生成(S1→S2 の因果閉鎖)[Tier 3, GPU 1〜2日]

- **目的**: **本計画の要**。ERDC の **S1(早期層符号化)→ S2(CoT分岐)** を介入で閉じる。
  実験8 が同定した早期層窓に patch を当て、その後 CoT を**自由生成**させ、実験3 で定義した
  分岐が誘発/消失するかを直接観測する。相関(実験8×実験3)を因果に変える。
- **手法(数式)**:
  1. **denoising(修復)**: typo run の早期層窓(実験8 の 12/12 局在帯)の残差ストリームを
     clean run の値で上書き `do(h^{early} := h^{early}_clean)` → 以降を**自由生成**。
  2. **noising(破壊)**: 逆に clean run の早期窓に typo run の値を注入 → 自由生成で分岐/flip を誘発。
  3. **指標**:
     - `ΔROUGE = ROUGE(patched CoT, clean CoT) − ROUGE(unpatched pert CoT, clean CoT)`。
     - `flip率`(patched vs unpatched)。
     - **発散オンセット**(実験3 定義: clean 実トークンの rank が閾値を超えて落ちる最初の位置)の
       消失率。
     - **後期窓 patch** を対照に置き効果差を測る。
- **設定**: M3×B2(実験8 と同一の代表設定)。flip 事例 300〜500/設定。層窓は早期帯/後期帯の2種、
  方向は denoising/noising の2種。
- **GPU見積**: **1〜2日**(2-pass patching + 自由生成の forward スイープ、backward 不要)。
- **依存関係**: 実験8(早期窓の同定)、実験3(オンセット定義)、実験1(cell C 構成の位置整列)。
- **出力先(提案)**: `analysis/exp15_patch_freegen/`(patch_rouge.csv / onset_removal.csv /
  layer_window_sweep.json)。実装は実験8 の `intervention/patching.py` を自由生成に拡張。
- **事前登録予測(H15)**:
  - **早期窓 denoising patch** で `ΔROUGE ≥ +0.15`(patched CoT が clean に近づく)。
  - **flip が半減以上**。
  - **発散オンセットが過半消失**。
  - **後期窓 patch ≈ 無効果**(効果が早期に局在)。
  - **noising で CoT が分岐し flip を誘発**(逆方向の因果も成立)。
- **反証分岐**: 早期窓 patch が自由生成を clean 側に戻さない場合 → S1 の符号化損傷は
  S2 の CoT分岐を**単独では**駆動せず、途中に別経路が介在。ERDC の S1→S2 リンクを
  「必要だが不十分」に弱め、実験11 の媒介経路と併せて再定式化。

### 3.6 実験16: 統一 GLMM による異質性の吸収 [Tier 1骨格 + Tier 2完成, GPU 0日]

- **目的**: モデル×ベンチ間の異質性が、単一の ERDC 連鎖の**3モデレーター M1/M2/M3** で
  説明できることを示す。設定間のランダム勾配分散(=いま「異質性」と呼んでいるもの)を
  モデレーター投入でどれだけ吸収できるかを定量する。
- **手法(数式)**:
  1. **ベースライン GLMM**: `flip ~ Q_p*C_p + (1 + perturb | setting) + (1|item)`。
     設定ランダム勾配の分散 `σ²_slope(base)` を推定。
  2. **モデレーター投入 GLMM**: 固定効果に **M1(repair 通過率)・M2(結論句シェア等 R_C 組成)・
     M3(Gini + no-CoT ショートカット依存)** を追加し、`σ²_slope(mod)` を推定。
  3. **吸収率** `A = 1 − σ²_slope(mod)/σ²_slope(base) ≥ 0.5`(モデレーターが設定ランダム勾配分散の
     50%以上を吸収)。
- **設定**: Tier 1 骨格 = M1(実験9)+ M2(実験12)のみで暫定推定。Tier 2 完成 = M3(実験13)を
  追加した完全モデル。全設定プール。
- **GPU見積**: **0日**(GLMM 推定、CPU)。
- **依存関係**: 実験9(M1)、実験12(M2)、実験13(M3、Tier 2 後追い)、実験1(flip 応答)。
- **出力先(提案)**: `analysis/exp16_unified_glmm/`(glmm_base.json / glmm_moderated.json /
  variance_absorption.csv)。
- **事前登録予測(H16)**: モデレーター投入で設定ランダム勾配分散の **50%以上を吸収**(A ≥ 0.5)。
- **反証分岐(フォールバック)**: 吸収されない(A < 0.5)場合 → 異質性は連続モデレーターでは
  捉えきれない。この場合、単一連鎖の統一を放棄せず、**二レジーム分類学**として報告:
  **自由記述型レジーム**(S3 搬送優位・IE 支配)と **選択式型レジーム**(DE/ショートカット優位)の
  2型に設定を分類し、各レジーム内で ERDC を適用する。

### 3.7 実験17: 行動的修復(M1 の行動的発現)[Tier 1, GPU 0日]

- **目的**: **M1(修復能力)** が表現レベル(実験9 の隠れ状態 cos)だけでなく**行動レベル**でも
  発現することを示す。reasoning 特化(R1蒸留)系が typo を CoT 内で**明示的に訂正**する現象を
  定量し、これが実験10 で観測された「R1蒸留系での LXT 優位消失」を説明することを示す。
- **手法(数式)**:
  1. R1蒸留系の生成 CoT から**明示的訂正**(例: "typo", "I think they mean", 綴り直し)を
     検出(正規表現 + 語彙照合)。`corrected_i ∈ {0,1}`。
  2. `P(corrected | R_Q)` のロジスティック回帰: 高 R_Q 語 typo ほど訂正が起きやすい(正係数)。
  3. `P(flip | corrected=1) < P(flip | corrected=0)`(訂正ありは flip しにくい)を McNemar 系で検定。
  4. これにより「R1蒸留系で LXT-4(高 R_Q 標的)の優位が消える(実験10)」= 高 R_Q ほど訂正され
     flip が抑制される、という機構的説明を与える。
- **設定**: 実験10 で生成済みの R1蒸留系 CoT(実験1・3 参加分)を再利用。R_Q は実験7 由来。
- **GPU見積**: **0日**(実験10 の既存 R1 生成の再解析 + 訂正検出、CPU)。
  実験10 の R1蒸留系生成が未完なら、その完了が前提(§1 注記)。
- **依存関係**: 実験10(R1蒸留 CoT)、実験7(R_Q)。実験16 の M1 を行動側から補強。
- **出力先(提案)**: `analysis/exp17_behavioral_repair/`(correction_detection.csv /
  rq_correction_logit.json / flip_by_correction.csv)。
- **事前登録予測(H17)**: 高 R_Q 語 typo ほど明示的訂正が起きやすく、訂正ありサンプルは
  flip しにくい → **R1蒸留系の LXT 優位消失を説明**。
- **反証分岐**: 訂正頻度が R_Q と無関係、または訂正が flip を抑制しない場合 → LXT 優位消失は
  行動的修復では説明できず、表現レベル修復(実験9 の M1)の R1系での強化として再解釈。

### 3.8 実験18: 形式移植(S3/DE の形式依存)[Tier 3, GPU 1〜2日]

- **目的**: S3(搬送)と DE(ショートカット)の均衡が**問題の内容ではなく回答形式**に駆動される
  ことを、形式を入れ替える移植実験で示す。GSM8K を多肢選択化、MMLU を自由記述化し、
  DE・restore・Δρ の反転を測る。
- **手法(数式)**:
  1. **MC化GSM8K**: GSM8K の数値解答を選択肢化(正解 + 距離マッチした 3〜4 ディストラクタ)。
     `DE_MC`(実験1 の直接効果)、`restore_MC`(clean CoT 強制の復帰率)、`Δρ_MC`(実験4)を測る。
  2. **自由記述化MMLU**: MMLU の選択肢を隠し自由記述で解答させ、`DE_free`・`Δρ_free` を測る。
  3. 元形式との対比(GSM8K free ↔ MC、MMLU MC ↔ free)。
- **設定**: 代表 M3 × {MC化GSM8K, 自由記述化MMLU}。実験1・4 と同一プロトコルで新規生成。
- **GPU見積**: **1〜2日**(形式変換後の clean/typo 生成 + cell C 強制 + 一部 R_C 再帰属)。
- **依存関係**: 実験1(DE/restore の基準)、実験4(Δρ の基準)。連鎖本体でなく付随仮説。
- **出力先(提案)**: `analysis/exp18_format_transplant/`(mc_gsm8k.csv / free_mmlu.csv / summary.json)。
- **事前登録予測(H18)**:
  - **MC化GSM8K で DE↑・restore↓・Δρ 膨張**(選択式にすると読み出しショートカットが増え搬送が減る)。
  - **自由記述化MMLU で DE↓・Δρ→0**(自由記述にすると搬送優位・目標依存が消える)。
- **反証分岐**: 形式を変えても DE/Δρ が動かない場合 → S3/DE の均衡は形式ではなく**内容
  (算術 vs 常識推論)**に駆動される。M2/M3 の形式依存の主張を「内容依存」に修正。

---

## 4. 統一主張ドラフト

> **確定文面(ユーザー考察 2026-07-19 の「統一主張のドラフト」ブロックを verbatim 収録)。**
> 実験群が通った場合の主張。強さは、各リンクが相関でなく介入で支持されていること
> (実験1・2・8・15)と、異質性が「説明済みの変動」に変わること(実験12・13・14・16)の
> 2点に懸かる。実験16の吸収検定が失敗したときのフォールバック(二レジーム報告)まで
> 事前登録済み(`hypothesis_registry.md` H16)。

### 4.1 日本語(verbatim)

> typoによる回答変化は、(1)摂動語の**早期層における語彙符号化の損傷**として発生し、(2)**内部修復**
> (表現的、reasoningモデルでは行動的)を逃れた損傷だけがCoT生成を少数の分岐点で逸らし、(3)逸れた
> **CoTテキストが効果の約8割を回答まで搬送**する。残余はCoTを迂回する**読み出しショートカット**であり、
> その割合は形式とモデルの読み出し構造で決まる。モデル・ベンチマーク間の見かけの異質性は、機構の違いでは
> なく、修復能力(M1)・答え決定情報の所在(M2)・読み出し集中度(M3)という3つの測定可能なパラメータの
> 違いとして説明される。

### 4.2 English(paste-ready draft)

> Typo-induced answer changes are the visible endpoint of a four-stage causal chain we name
> **ERDC (Encode–Repair–Divert–Carry)**. A perturbation first manifests as damage to the lexical
> encoding of the perturbed word in the early layers (**Encode**; Exp. 8 localizes this to early
> layers in 12/12 conditions). This damage is filtered by an internal repair gate: only damage the
> model fails to repair in its inner lexicon propagates forward (**Repair**; Exp. 9, the repair
> score is a negative predictor of flips in 47/50 diagnoses). Surviving damage diverts
> chain-of-thought generation at a small number of branch points (**Divert**; Exp. 3, KL divergence
> concentrates and is larger for flipped than non-flipped samples, 50/50). The diverted CoT text
> then carries most of the effect all the way to the answer (**Carry**; Exp. 1, IE/TE ≈ 0.8 and
> forcing the clean CoT restores the answer in 90%+). A residual direct effect (DE) that bypasses
> the CoT at the read-out stage accounts for the remainder. The heterogeneity across models and
> benchmarks reduces to three moderators of this single chain: **M1**, repair capacity (the pass
> rate of gate G); **M2**, where answer-determining information sits (the composition of the top
> R_C mass); and **M3**, read-out concentration together with dependence on the CoT-bypass
> shortcut. Experiments 11–18 close the links between stages causally — Exp. 15 in particular
> closes Encode→Divert by patching the early-layer window and observing the induced CoT divergence
> under free generation — and Exp. 16 absorbs the heterogeneity into the three moderators, thereby
> promoting ERDC from a chain of correlations to a single causal chain.

---

## 5. 規約

- **GPU 使用**: Tier 1(実験11・12・16骨格・17)は GPU 0日(CPU 再解析)。Tier 2〜3 のみ GPU を使用
  (MEMORY の GPU 割り当て指示に従い、nvidia-smi ガードの上で確保)。
- **数値の出所**: 全て results.json / 統合テーブル / archive full_results.json から直接転記。
  予測値は事前登録であり実測ではない(判定は実行後に `hypothesis_registry.md` へ追記)。
- **統計の定型**: McNemar + リスク差CI / GLMM(必ず +(1|item))/ Holm / paired bootstrap
  (Phase A と同一)。
- **事前登録の完全性**: 各 H の反証分岐を実行前に本書と `hypothesis_registry.md` に凍結済み。
  実測が予測と乖離した場合は分岐に従って修正し、修正過程を透明化する(Phase A の H3/H4/H9 と同じ運用)。
- **他 worktree**: 読み取りのみ。コミットは本体チェックアウトで実施、push なし。

---

## 6. 対応表(H ↔ 実験 ↔ データソース)

| H | 実験 | ERDC段/モデレーター | 主データソース | Tier / GPU |
|---|---|---|---|---|
| H11 | 実験11 | G→S2 連鎖媒介 | 実験9 repair_min + 実験3 KL_sum + 実験1/2 flip | 1 / 0日 |
| H12 | 実験12 | M2 R_C組成 | 実験4 fixed R_C + Δρ | 1 / 0日 |
| H13 | 実験13 | M3 読み出し集中 | 実験4 R_C(Gini)+ 実験2 削除RD | 2 / ≈1日 |
| H14 | 実験14 | 残余 DE ショートカット | 実験1 DE + no-CoT 生成 | 2 / ≈1日 |
| H15 | 実験15 | **S1→S2 因果閉鎖** | 実験8 早期窓 + 実験3 onset + 実験1 cell C | 3 / 1〜2日 |
| H16 | 実験16 | M1・M2・M3 統一吸収 | 実験9/12/13 + 実験1 flip | 1骨格+2完成 / 0日 |
| H17 | 実験17 | M1 行動的発現 | 実験10 R1蒸留 CoT + 実験7 R_Q | 1 / 0日 |
| H18 | 実験18 | S3/DE 形式依存 | 実験1 DE/restore + 実験4 Δρ + 形式移植生成 | 3 / 1〜2日 |

---

## Version History

| Date | Change |
|------|--------|
| 2026-07-19 | 初版作成。ERDC連鎖(S1/G/S2/S3+DE、モデレーター M1/M2/M3)、実験11〜18 の詳細計画、Tier表、3週間スケジュール、S1→S2 因果閉鎖(実験15)の強調、統一主張ドラフトを収録。H11〜H18 は `hypothesis_registry.md` に事前登録。 |
