# ARR August 2026 Resubmission: FB対応 改善計画

本書は、完了済み7実験の結果に対するフィードバック(FB1: 予測と実際の対照分析、FB2: 査読者目線の採点)に基づき、残りの実験実行と論文構成を最適化するための具体的なアクションプランである。

**指導原理**: 「最悪のシナリオは都合の良い結果だけ報告し都合の悪い結果を隠すこと」(FB2)。Mistralの逆方向・precision@10のnull以下・Gemma飽和は全て本文で正面から報告し、解釈の枠組みを調整することで論文を強化する。

---

## 対処が必要な3件の概要

| # | 問題 | 実験 | 深刻度 | 対処の方向 |
|---|---|---|---|---|
| 1 | Mistral全設定で$\Delta\rho < 0$(fixed-targetで相関強化） | 実験4 | 高 | 形式二分法の放棄、family x format交互作用として再定式化 |
| 2 | precision@10 = 0.151 < null mean 0.334（KLピークとR_C上位が「避け合う」） | 実験3 | 高 | 「帰属と行動の一致」撤回、「相補性」フレーミングへ転換 |
| 3 | Gemma修復スコア0.99飽和（flip/noflip差が消失） | 実験9 | 高 | 修復仮説の適用範囲をLlama/Mistralに限定、Gemma飽和を独立知見として報告 |

---

## Part 1: 即座に実行可能な分析の追加（GPU不要、既存データの再分析）

### 1-1. 実験4: family x format 交互作用分析

**背景**: 事前予測は「自由記述 $\Delta\rho \approx 0$、多肢選択 $\Delta\rho > 0$」だったが、実際のデータは:
- Gemma/Llama の多肢選択: $\Delta\rho > 0$（大幅減衰、予測通り）
- Gemma/Llama の GSM8K: $\Delta\rho \approx 0$（予測通り、一部例外）
- **Mistral は全設定で $\Delta\rho < 0$**（GSM8K: -0.162、MMLU: -0.072、ARC: -0.082、CSQA: -0.076）

形式（自由記述 vs 多肢選択）だけでは説明できない。**モデル家族が交絡している**。

**スクリプト仕様: `scripts/analysis/exp4_family_interaction.py`**

- **入力**: `delta_rho_table.json`（75行 = 25設定 x 3 k値）
- **引数**: `--k top10`（主分析のk値）、`--output-dir <path>`
- **計算内容**:
  1. k=top10 の25行を抽出
  2. family列を付与: Gemma (gemma-3-1b-it, gemma-3-4b-it) / Llama (Llama-3.2-1B, Llama-3.2-3B) / Mistral (Mistral-7B)
  3. format列を付与: free-form (gsm8k) / MC (mmlu, mmlu_pro, arc, commonsense_qa)
  4. 2x3 の family x format pooled集計: 各セルの $\Delta\rho$ の平均・中央値・95% CI (bootstrap B=10,000)
  5. 二元配置 ANOVA（不均衡セルに対応、Type III SS）: $\Delta\rho \sim \text{family} + \text{format} + \text{family} \times \text{format}$
  6. Mistral単独のpooled $\Delta\rho$（全5ベンチ）の bootstrap CI とゼロからの乖離検定
  7. Gemma/Llama pooled（Mistral除外）での format 主効果の再検定
- **出力**:
  - `interaction_table.csv`: 6セル (3 family x 2 format) の集計
  - `anova_results.json`: F値、p値、効果量 ($\eta^2$)
  - `heatmap_delta_rho.pdf`: family x format の $\Delta\rho$ ヒートマップ（Fig.3差替候補）
  - `mistral_pooled.json`: Mistral単独の検定結果
- **期待される効果**: 「形式二分法」を「family x format 交互作用」に精密化。Mistralの逆方向を「異常値として除外」ではなく「交互作用の一端として包含」できる。論文上は「Gemma/Llamaでは内生性が多肢選択に集中、Mistralでは固定により選択バイアスのノイズが除去され真の関連が露出する」と解釈。Fig.3をheatmapに差し替え、形式のみの棒グラフから脱却。
- **所要時間**: 実装 0.5日、実行 数分（CPU）
- **依存関係**: なし（delta_rho_table.json は完了済み）

### 1-2. 実験3: KLピークとR_C上位の空間分布の深堀（「相補性」パターンの構造化）

**背景**: precision@10 = 0.151（null mean = 0.334、p = 0.885）。KL divergenceが大きい位置と $R_C$ 上位トークンが同じ場所にいないどころか、**ランダムより避け合っている**。これは「帰属と行動が一致する」仮説の否定だが、別の解釈が可能: **KLピーク = 「モデルが書き直したくなる場所」（摂動の初期伝播点）、R_C上位 = 「答えを決定する場所」（下流の因果的決定点）。この2つが空間的に分離していること自体が情報**。

**スクリプト仕様: `scripts/analysis/exp3_spatial_complementarity.py`**

- **入力**: `exp01_03/*/divergence/*.json`（68,462プロファイル）
- **引数**: `--model <model>`, `--benchmark <bench>`, `--condition {importance,random}`, `--output-dir <path>`
- **計算内容**:
  1. 各サンプルの divergence JSON から: `kl` (位置別KL配列)、`precision_at_k`、`null`、`onset`、`n_positions` を読み込み
  2. KLピーク位置の分布: CoT内の相対位置 (0=CoT先頭, 1=CoT末尾) に正規化したKL上位10位置のヒストグラム
  3. R_C上位10の相対位置分布（Step 0のmaster tableからR_Cランキングを取得し、CoT内の出現位置を同様に正規化）
  4. 2分布の重なり指標: overlap coefficient、Wasserstein距離
  5. **象限分析**: CoT位置を「前半(0-0.5) vs 後半(0.5-1.0)」に二分し、KL上位とR_C上位の分布を2x2で集計
     - 仮説: KLピークは「CoT前半（問題の言い換え・初期推論）」に集中、R_C上位は「答え直近（最終推論ステップ）」に集中 → 空間的分離が「相補性」を構造的に説明
  6. flip群 vs noflip群での空間分布の比較
  7. 発散オンセットとR_C上位初出位置の差（gap）の分布
- **出力**:
  - `spatial_distribution.pdf`: KLピーク vs R_C上位の相対位置ヒストグラム（重ね描き）
  - `quadrant_table.csv`: 前半/後半 x KL/R_C の2x2表
  - `complementarity_stats.json`: overlap coefficient、Wasserstein、gap統計
  - `flip_vs_noflip_spatial.pdf`: flip状態別の空間比較
- **期待される効果**: precision@10がnull以下という「失敗」を「KLピークとR_C上位はCoTの異なる機能段階に対応する」という**肯定的知見**に変換。論文の「帰属と行動の一致」をDrop、代わりに「帰属(R_C)は答え決定点を捉え、KL乖離は摂動伝播の起点を捉える -- 両者は相補的である」と再フレーム。FB2の「相補性フレーミング」を実装。
- **所要時間**: 実装 1日、実行 10-20分（CPU、68K JSONの読み込みがボトルネック）
- **依存関係**: Step 0 master table（R_C位置参照）、exp01_03 divergence（完了済み）

### 1-3. 実験9: cos以外の指標によるGemma飽和の原因診断

**背景**: Gemmaの修復スコア(max cos) = 0.99近傍に飽和。flip = 0.998, noflip = 0.998で差がない。これはcos類似が天井効果で弁別力を失っていることを意味する。一方、Llama (0.68-0.76) とMistral (0.67-0.80) では十分な分散がある。

word_rows のスキーマには `cos_curve` (層別cos配列)、`lens_min_rank`、`lens_first_hit_layer_top5` が既に計算済み。追加指標はこれらから派生計算可能。

**スクリプト仕様: `scripts/analysis/exp9_gemma_diagnosis.py`**

- **入力**: `exp9/word_rows_*.jsonl`（全60シャード、約164K行）
- **引数**: `--output-dir <path>`, `--models gemma-3-1b-it,gemma-3-4b-it`（診断対象）
- **計算内容**:
  1. 全word_rowsを読み込み、`clean_correct == true` でフィルタ
  2. Gemmaのみ抽出し、以下の追加指標を計算:
     - **cos_curve の分散・range**: `max(cos_curve) - min(cos_curve)`。飽和しているのはmaxだけで、curveの形状には差があるかもしれない
     - **cos_curve の「修復勾配」**: 入力層(layer 0-2)のcos平均 vs 中間層(layer L/3 - 2L/3)のcos平均 vs 出力層(最終3層)のcos平均。3点の勾配パターンで分類
     - **cos_curve の「修復速度」**: cos > 0.95（ほぼ修復）に初めて達する層（Gemmaでは非常に早い層で到達するはず）
     - **logit lens乖離**: `lens_min_rank` のflip/noflip差。cos飽和でもlogit lens rankには差がある可能性
     - **cos_curve 末尾の降下**: Gemmaのcos_curveは最終層で急落する傾向（サンプルデータから確認済み: layer 25で0.987 → layer 26で0.598）。最終層降下幅のflip/noflip比較
  3. Llama/Mistralとの対比: 同じ指標を全モデルで計算し、Gemma特有の構造を浮き彫りに
  4. **仮説検証**: Gemma飽和の原因候補
     - (a) layer normalization後の表現空間が小さい → cos_curve の初期値（layer 0）を比較
     - (b) 埋め込み近接性 → `cos_curve[0]`（入力埋め込み直後のcos）がGemmaで既に高いか
     - (c) 修復が超早期に完了 → 修復速度のモデル間比較
  5. 条件付き回帰: cos以外の指標（lens_min_rank、cos_range、修復速度）を予測子とした `flip ~ X + (1|item)` をGemma限定で実行
- **出力**:
  - `gemma_diagnosis.json`: 各指標のflip/noflip統計、モデル間比較
  - `cos_curve_profiles.pdf`: モデル別の平均cos_curve（flip/noflip別、95% CI帯付き）
  - `alternative_metrics_regression.json`: cos以外の指標での回帰結果
  - `saturation_cause.md`: 原因診断のサマリ（人間可読）
- **期待される効果**: Gemma飽和が「修復が早すぎる（layer normの正規化で早期に収束）」ことが原因であれば、「Gemmaは修復が速すぎてcos指標がfloor/ceilingで潰れる → 修復の質ではなく再構成段階の問題」という解釈が可能。これはFB2の(b)+(c)戦略と合致: Gemma飽和は「修復後の再構成段階の問題」として再解釈。
- **所要時間**: 実装 1日、実行 5-10分（CPU、164K行JSONL読み込み）
- **依存関係**: exp9 word_rows（完了済み）

### 1-4. 実験9: Gemma除外版 pooled 回帰の主報告化

**背景**: FB2の方針(b)に従い、Gemma飽和の影響を受けない Llama/Mistral 限定の pooled 回帰を主報告にする。analysis_summary.json のデータ:
- pooled_Llama-3.2-1B: repair_coef = -0.102, p = 4.3e-9
- pooled_Llama-3.2-3B: repair_coef = -0.189, p = 4.0e-32
- pooled_Mistral-7B: repair_coef = -0.155, p = 8.9e-19
- pooled_gemma-3-1b-it: repair_coef = -0.088, p = 4.5e-9 (飽和にもかかわらず有意 -- 微小な差が大Nで検出)
- pooled_gemma-3-4b-it: repair_coef = -0.146, p = 9.1e-19 (同上)
- pooled_all: repair_coef = -0.128, p = 1.5e-36

**スクリプト仕様: `scripts/analysis/exp9_pooled_llama_mistral.py`**

- **入力**: `exp9/word_rows_*.jsonl`（Llama + Mistral のシャードのみ）
- **引数**: `--output-dir <path>`
- **計算内容**:
  1. Llama-1B, Llama-3B, Mistral-7B の word_rows を結合（clean_correct = true）
  2. pooled 回帰: `flip ~ repair_score + split_increment + zipf_freq + r_q + (1|sample_id)`、クラスタロバストSE
  3. 条件別回帰: LXT-4 と Random-4 で分けて安定性確認
  4. Gemma全体との効果量比較: Cohen's d での修復スコア flip/noflip 差
  5. 修復スコアの分布可視化: Llama/Mistral vs Gemma（violin plot）
- **出力**:
  - `llama_mistral_pooled_regression.json`: 主回帰結果
  - `effect_size_comparison.csv`: モデル家族別の効果量
  - `repair_distribution.pdf`: violin plot
- **期待される効果**: 本文の主報告を「Llama/Mistralでは修復スコアが flip の最強の負予測子（coef = -0.15 to -0.19, p < 1e-18）」とし、Gemma飽和を独立した知見として分離。「5モデル全体で有意（coef = -0.128, p = 1.5e-36）」は補足として残す。Gemma飽和は Discussion の「モデル家族効果」節で議論。
- **所要時間**: 実装 0.5日、実行 5分（CPU）
- **依存関係**: exp9 word_rows（完了済み）

---

## Part 2: 進行中実験の焦点調整

### 2-1. 実験8: flipペア選定とGemma以外での再現性

**現状**: スモーク完了（Gemma-3-4B x GSM8K, n=16, 0.4 GPU日見積り）。

**焦点調整**:
- flipペアをLXT-4/Random-4半々に（修正A、コスト増なし）→ 実施済みか確認が必要
- 結果解釈の優先順位: **Llama-3.2-3B と Mistral-7B での再現が最重要**。Gemmaではcos飽和と同じ問題（表現空間の圧縮で層間差が小さい）が起きうるため、Gemma結果は参考扱いにする事前方針を明文化
- 「MLP早期層が質問摂動の媒介を支配」がGemma以外で再現するかが、Representation層の柱の成否を決める

**期待される効果**: 実験8が3家族で走ることで、Part 1-1のfamily x format交互作用と合わせて「モデル家族効果」の議論が立体化する
- **所要時間**: 本番実行 0.4-1.1 GPU日（見積り済み）
- **依存関係**: 実験1（flipペア、完了済み）

### 2-2. 実験6: LOO本番でのGemma飽和迂回

**現状**: LOOスモーク完了（Gemma-3-4B x GSM8K, n=16, LOO vs R_C Jaccard@10 = 0.460）。

**焦点調整**:
- LOO重要度 $\rho(J|R)$ の符号・有意性がGemma飽和を**迂回する別経路**になりうる: LOOは行動的定義（log-prob低下）なのでcos類似の天井効果に影響されない
- 本番 M3 x B2 で Gemma-3-4B の $\rho(J_{\text{LOO}}|R)$ が有意であれば、「cos指標は飽和してもLOO重要度による内的軸の再構成は可能」と書ける
- **最優先で走らせるべき設定**: Gemma-3-4B x MMLU（実験4でfixed-target相関が最も減衰した設定で、LOOが残ればAttribution軸の防御に直結）

**期待される効果**: 実験9のGemma飽和問題への保険。LOOが生きれば「cosは飽和するがLOOは弁別する → 帰属なしの行動指標がrobust」
- **所要時間**: LOO本番 8-10 GPU時間
- **依存関係**: 実験4（fixed-targetプロトコル、完了済み）

### 2-3. 実験7: neural/LLM段の本番完走

**現状**: pyspell完了、neural(T5)進行中、LLM(Qwen)一部完了。

**焦点調整**:
- neuralの優位が予想外に鮮明（accuracy: LXT-4 → neuralfix でほぼclean復帰）。この結果はFB2の「批判3: 防御」への強い回答
- 本番完走後に必須の成果物: **byte-identical → flip 0% のベンチ別全数表**
- 「保守的校正の優位」として強調: neural(T5)は文脈ありだが保守的（typo修正に特化）、LLMは汎用だが誤修正リスクあり → neuralが最適点
- 3面図の作成を急ぐ（校正器強度 x {復元率, accuracy回復, 高R_Q語集中}）

**期待される効果**: 批判3のスコアを確実に固める。neuralの優位は「表層修復は可能だが、修復精度のボトルネックが答え関連語に集中する」というボトルネック論の核心を実証
- **所要時間**: 残り 2-3 GPU日
- **依存関係**: なし（独立実行可能）

---

## Part 3: 論文構成の再設計

### 3-1. 本文4本柱の再評価

FB1/FB2を踏まえた各柱の強度評価:

| 柱 | 実験 | 強度 | 変更点 |
|---|---|---|---|
| 1. 因果分解 | 実験1 | **最強** | 変更なし。IE優位の分解構造が両摂動条件で一致。見出し数字も予測範囲内 |
| 2. 削除介入 | 実験2 | **強い+新発見** | 数値 vs 内容の層別が予想外に鮮明。GSM8K の R_C 上位がほぼ数値語という発見を前面に |
| 3. 測定の診断 | 実験4 | **要再定義** | 「改善」→「診断ツール」に再定義。Fig.3 を family x format heatmap に差替 |
| 4. 防御の実証 | 実験7 | **方向は良い** | neural の優位を「保守的校正」として強調。完走待ち |

**Fig.3 差替の決定**: $\Delta\rho$ の family x format heatmap。行 = 3 family (Gemma/Llama/Mistral)、列 = 2 format (free-form/MC)。セル内に pooled $\Delta\rho$ と 95% CI。カラースケールで正(減衰)=赤、負(強化)=青。Mistral列が青一色になることで「モデル家族効果」が視覚的に即座に伝わる。

### 3-2. Discussion の新項: 「モデル家族効果」

以下3件を統合した1段落(約200語)の Discussion 節を新設:

1. **Mistralの逆方向 (実験4)**: fixed-target化で相関が強化される。解釈: Mistralではdefault条件の帰属が選択バイアスノイズを含んでおり、固定により真の関連が露出。logit lens でMistralの `clean_self_min_rank` が異常に高い (0.27-0.50 vs Llama/Gemma 0.86-0.97) こととの関連を議論
2. **KLとR_Cの相補性 (実験3)**: KLピーク（摂動伝播の起点）と R_C上位（答え決定点）が空間的に分離。摂動の初期効果と最終的な因果効果は異なるCoT段階に局在する
3. **Gemmaの修復飽和 (実験9)**: cos類似が0.99で天井に張り付き弁別力を失う。原因として layer normalization 後の表現空間の圧縮を議論。Llama/Mistralでの修復仮説の有効性との対比

**構成案**: 「3件の予測外結果は、いずれもモデルアーキテクチャの家族レベルの差異に帰着する。Gemmaの強い正規化、Mistralの logit lens 異常、Llama の中間的挙動は、typo に対する内部処理パイプラインがアーキテクチャ依存であることを示唆する。」

### 3-3. H1〜Hn レジストリの更新

事前登録済みの分岐条件のうち、発動した分岐の記録:

| 仮説ID | 事前登録内容 | 発動条件 | 実際 | 採択する分岐 |
|---|---|---|---|---|
| H4-format | 自由記述 $\Delta\rho \approx 0$、MC $\Delta\rho > 0$ | format主効果のみ | family x format交互作用 | **分岐発動**: 形式二分法を交互作用に修正 |
| H3-precision | precision@10 > null → 帰属と行動の一致 | precision > null mean | precision < null mean | **分岐発動**: 一致論を撤回、相補性に転換 |
| H9-repair | 修復スコアが flip の最強負予測子 (全モデル) | 全5モデルで再現 | Gemma で天井効果 | **分岐発動**: 範囲を Llama/Mistral に限定 |
| H1-pattern | パターン X (IE優位) or Y (DE優位) | IE/TE > 0.7 | IE/TE = 0.80-0.83 | パターン X 採択（予測通り） |
| H2-deletion | top-R_C > random x3倍 | flip率比 > 3 | 66.7% vs 4.3% (x15倍) | 予測を大幅に上回る効果（予測通りの方向） |

### 3-4. 三層章立ての調整

```
Section 3: Surface Layer
  3.1 ROUGE-L の二軸分析（既存、変更なし）
  3.2 校正器3段ラダー（実験7）← neuralの優位を前面に

Section 4: Attribution Layer
  4.1 CoT移植 2x2: 因果分解（実験1）← 最強の柱、変更なし
  4.2 重要トークン削除（実験2）← 数値 vs 内容の層別を新発見として
  4.3 fixed-target 診断（実験4）← 「改善」→「診断ツール」、heatmap
  4.4 マッチド統制（実験5）← 22/25有意、簡潔に

Section 5: Representation Layer
  5.1 activation patching（実験8）← Llama/Mistral優先

Section 6: Why do some typos succeed?
  6.1 内部修復分析（実験9）← Llama/Mistral限定主報告
  6.2 KL発散プロファイル（実験3）← 相補性フレーミング

Section 7: Discussion
  7.1 モデル家族効果（新設: Mistral逆方向 + Gemma飽和 + KL-R_C相補性）
  7.2 Limitations（faithfulness、D層のB2限定、Gemma飽和の解釈限界）
```

---

## Part 4: 追加実験の要否判断

### 判断ツリー

```
Part 1-3 (Gemma cos以外の指標)
  |
  +-- lens_min_rank 等で flip/noflip 差が有意
  |     --> Gemma飽和は「cos指標の限界」として説明可能
  |         --> 追加実験不要。本文は条件付き
  |             (Llama/Mistral 有意なら主報告1段落、Gemma は付録注)
  |
  +-- cos以外でも差なし
        |
        +-- Part 2-2 (LOO) で Gemma の rho(J_LOO|R) が有意
        |     --> LOO が Gemma での修復仮説の代替
        |         --> 追加実験不要
        |
        +-- LOO でも非有意
              |
              +-- (a) Gemma の layer-norm 前の中間表現で再計算
              |     --> 追加 GPU 0.5-1日、word_rows パイプラインに
              |       pre-norm hook を追加して cos_curve_prenorm を計算
              |
              +-- (b) 実験3の KL 群間差を「なぜ」への主回答に昇格
                    --> 実験9 を降格（付録）、実験3 + 実験5 で
                      「何が効く摂動を決めるか」を再構成
                      （KL onset の早さ x マッチド統制の表層特性）
```

**最も可能性の高いシナリオ**: Part 1-3 で cos_curve の形状分析（修復速度・末尾降下幅）に flip/noflip 差が見つかり、「cos の max は飽和するが修復の質的パターンには差がある」という形で解決する。根拠: Gemma のサンプルデータで cos_curve の末尾降下幅が大きく（layer 25: 0.987 → layer 26: 0.598）、この降下パターンに flip 依存性がある可能性。

---

## 実行順序とタイムライン

### Phase A: 即座に実行（GPU不要、CPU分析のみ）[2日]

| 優先度 | タスク | 所要時間 | 依存 |
|---|---|---|---|
| **P0** | 1-1: exp4 family x format 交互作用分析 | 0.5日 | なし |
| **P0** | 1-4: exp9 Llama/Mistral pooled回帰の主報告化 | 0.5日 | なし |
| **P1** | 1-2: exp3 空間分布の深堀 | 1日 | Step 0 master table |
| **P1** | 1-3: exp9 Gemma飽和の原因診断 | 1日 | なし |

### Phase B: 進行中実験の焦点調整 [並行実行]

| 優先度 | タスク | 所要時間 | 依存 |
|---|---|---|---|
| **P0** | 2-3: 実験7 neural/LLM完走 + 3面図 | 2-3 GPU日 | なし |
| **P1** | 2-1: 実験8 本番（3家族 x B2） | 0.4-1.1 GPU日 | 実験1 (完了) |
| **P1** | 2-2: 実験6 LOO本番（M3 x B2） | 8-10 GPU時間 | 実験4 (完了) |
| **P2** | 実験2 本番キュー継続 | 進行中 | 実験4 (完了) |

### Phase C: 論文構成の再設計 [Phase A完了後]

| 優先度 | タスク | 所要時間 | 依存 |
|---|---|---|---|
| **P0** | 3-1: Fig.3 heatmap作成 | 0.5日 | 1-1 完了 |
| **P0** | 3-3: H1-Hn レジストリ更新 | 0.5日 | なし |
| **P1** | 3-2: Discussion「モデル家族効果」節の執筆 | 1日 | 1-1, 1-3 完了 |
| **P1** | 3-4: 章立て確定 + 本文スケルトン | 2日 | Phase B の主要結果 |

### Phase D: 追加実験の判断 [Phase A完了後に判断]

Part 1-3 の結果を見てから Part 4 の判断ツリーを実行。最悪でも +0.5-1 GPU日。

---

## リスク管理

### 最大のリスク: Gemma飽和が全経路で解決しない場合

**対策**: 実験9を「Llama/Mistral限定のWhy回答」として本文に置き（1段落）、Gemma飽和を付録の独立知見として報告。これでも pooled_all の p = 1.5e-36 は維持されるので、主結論は崩れない。FB2の「都合の悪い結果を隠さない」原則に従い、Gemma飽和のcos天井効果を明示的に Limitations に記載。

### 次点のリスク: Mistralの逆方向がレビューアに「測定の妥当性への疑問」と読まれる

**対策**: Part 1-1のANOVAで交互作用項が有意であれば、「形式 x 家族の交互作用は既知のアーキテクチャ差（Mistralのlogit lens異常）と整合する」と積極的に議論。Mistralの `clean_self_min_rank` データ（0.27-0.50、他モデルは0.86-0.97）を根拠として提示。fixed-targetを「改善ツール」ではなく「診断ツール」に再定義することで、Mistralの逆方向は「この診断がモデル家族の内部処理差を検出した」というポジティブな解釈になる。

### 三番目のリスク: precision@10 < null の「相補性」解釈がレビューアに受け入れられない

**対策**: Part 1-2の象限分析で物理的な空間分離パターンが明確に出れば、「KLピークはCoT前半（問題解釈段階）に集中し、R_C上位はCoT後半（解答導出段階）に集中する」という構造的説明が可能。これが出なければ、実験3を「帰属の行動的裏づけは得られなかった」として正直にLimitationsに記載し、実験2（削除による因果的裏づけ）を代替として前面に出す。実験3は付録に降格。

---

## チェックリスト: FB2の各批判へのスコア寄与

| 批判 | 対応実験 | FB2評価 | 本計画での追加効果 | 目標到達 |
|---|---|---|---|---|
| 1. 因果 | 実験1 | 0.5-1.0動かせる | 変更なし（最強のまま） | 確実 |
| 2. 測定 | 実験4+5 | 実験4のMistralが複雑に | family x format交互作用で積極活用 | 射程内 |
| 3. 防御 | 実験7 | neuralの優位が予想外に鮮明 | 完走+3面図で確定 | 確実 |
| 4. なぜ | 実験9 | Gemma飽和が最大危険信号 | Llama/Mistral限定+cos以外指標 | 条件付き |

**総合判断**: Findings圏（3名平均3.0+）は射程内。本計画のPhase A-Cを実行すれば、3件の「都合の悪い結果」が全て「モデル家族効果」として統合的に説明可能になり、隠蔽ではなく積極的な知見として提示できる。
