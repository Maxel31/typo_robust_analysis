# Dev notes — 実験8-fine (1層分解 activation patching)

粗い窓 (幅3) の実験8 で確定した最良窓 **residual[0,6) が 10/12 条件** を、**1 層解像度**に
精密化する。Fig.5 を「窓の棒グラフ」から「相対深さ l/L × 回復率の連続プロファイル」に格上げする。

## 主張の精密化 (敵対的レビュー A3 対応)

深さプロファイルは「**注入の座 (injection site)**」ではなく、**摂動語スパンからの読み出し完了時点
(read-out completion point)** を局在させるものとして報告する。patch は摂動語スパンに限定されるため、
後期層で効かないのは「情報が既に他位置へ伝播した後」である可能性を排除できない。ただし**防御含意**
(早期にスパンを直せば推論は間に合って回復する) は不変であり、Fig.5 の価値は保たれる。

## 実験設定

- 部位: **residual のみ** (粗い窓で最良確定済み)。位置種別: **摂動語スパン (question_span) のみ**。
- スイープ: 第0〜11層を幅1・stride1 で 12 点 + 検証点 第14/20/26層 3 点 (計 15 点)。
- 方向: **denoising (clean→pert)** が主。**noising (pert→clean)** は早期帯 (0–7) を回し、判定は
  最良層±1 の 3 点で行う (十分性 H8f-5)。
- 指標: 主 = **S2 KL 回復率** `s2_kl_recovery` (最初の CoT 語 c1 分布)。
  副 = 分岐ペアの **flip 逆転** `answer_matches_donor` (MMLU のみ; GSM8K は分岐 0–4% で無意味)。
- n = 150/設定 (LXT-4 : Random-4 半々)。モデル: Gemma-3-4B(34層)/Llama-3.2-3B(28層)/Mistral-7B(32層)。

### 統制 (1層解像度で初めて意味を持つ)

1. **sham patch**: recipient (pert) 自身の値を摂動語スパンに書き戻すダミー。効果ゼロのはず
   (hook アーチファクト検出)。S2 KL の基準は常に **target (clean)** 分布に取るため、sham は
   `s2_kl_recovery ≈ 0` になる (自分を基準にすると 1.0 に化ける実装バグを修正済み)。
2. **累積 vs 単層**: 単層 (各層の限界寄与) に加え、累積 patch (第0層〜第l層を全差替, l=0..11)。
   単層 max ≪ 累積 max なら分散書き込みの証拠。
3. **相対深さ整列**: モデル間比較は絶対層番号でなく **l/L** で行い重ね描き (Fig.5 差替候補 PNG)。

## A3 敵対的レビュー統制 (相乗り)

**攻撃**: 「patch が摂動語スパン限定なので後期層で効かないのは情報が他位置へ漏出したため。深さ単調
減衰はどんな入力摂動でも出る = 早期層局在は過剰解釈」。

同じ hook・同じ flip ペアで以下を追加セルとして回す:

- **A3(a) other_span** (特異性): 無摂動語 (スパンから +offset の下流位置; donor/recipient を同一
  offset で 1 対 1 対応) を clean 値で patch。全層で回復 ≈ 0 のはず。漏出した下流コピーではなく
  スパンそのものが座であることの証明。判定: `|mean| < 0.2` を全早期層で満たすか。
- **A3(b) all_positions** (枠組みサニティ, 恒等性の逆): 全プロンプト位置を clean 値で patch。
  任意層で **完全回復 (≈1)** のはず (数学的保証: 全位置が clean → c1 も clean)。単層スイープの
  部分回復が hook 不全ではなく真の部分効果であることを示す。トークン数一致ペアのみ。
- **A3(c) semantic** (意味置換対照): typo でなく**同義でない実語**に標的語を置換した摂動で同じ
  1 層 denoising スイープを回す (`--perturb-mode semantic`; 生成不要 = clean CoT を teacher-forcing)。
  - **同形** (Pearson>0.8 かつピーク±1 一致) → 「読み出しダイナミクスは入力摂動一般の性質であり、
    **typo 固有寄与は LXT vs Random の倍率差 (4.8–10.1倍) に局在**する」と正直に書く。
  - **異形** → typo 固有の深さ局在を主張できる。
  結果は `results/prod/exp8_fine_semantic/` に分離出力し、`analyze_fine.py --semantic-results-dir` で
  typo プロファイルと比較 (`profile_isomorphism`)。
- **A3(d) attn 最早期の負値**: 粗い run で attn 部位の最早期窓に小さな負の回復が出るが、現状**無説明**。
  「**観察として報告・解釈は保留**」と明記する (fine は residual のみ; attn は粗い付録から引き継ぎ)。

## 実装

- `src/typo_cot/intervention/patching.py`: `single_layer_windows` / `cumulative_windows` /
  `relative_depth` / `align_by_relative_depth` (単層/累積の位置指定・相対深さ整列; ユニットテスト済)。
- `scripts/exp8/run_patching_fine.py`: 粗い `run_patching.py` の `prepare_pair` / `_generate` /
  `_c1_logits` を再利用。`run_pair_fine` が single/cumulative/noising/sham + A3(a)(b) セルを生成。
  `--perturb-mode semantic` で A3(c) (単層のみ)。捕捉は residual 全位置 (統制で任意位置の donor 値が要る)。
- `src/typo_cot/intervention/semantic_control.py`: `make_semantic_pair` (標的語→実語ランダム置換, 決定論的)。
- `src/typo_cot/intervention/fine_analysis.py`: 層プロファイル集計・bootstrap CI・最良層・プラトー判定・
  累積飽和・`judge_h8f1..5` (反証も透明化)。
- `scripts/exp8/analyze_fine.py`: 設定別プロファイル CSV・相対深さ重ね描き PNG (Fig.5 候補)・
  判定 JSON (H8f-1..5 + A3 特異性ブロック + semantic 同形性)。
- `scripts/exp8/prod_fine/{smoke_fine,run_fine_queue}.sh`: GPU ロックヘルパー経由・冪等・typo+semantic 2 パス。

## 判定 (事前登録)

- **H8f-1** ピーク l/L<0.2 / **H8f-2** プラトー vs スパイク / **H8f-3** 累積飽和 (≥単層max×1.2) /
  **H8f-4** 検証点 (14/20/26) ≈0 / **H8f-5** noising 最良層±1 で KL 過半再現。
- **A3**: (a) other_span≈0 / (b) all_positions≈1 / (c) typo vs semantic の同形性 (同形なら typo 固有寄与は
  倍率差に局在)。反証は各 `supported` フラグと補助量で正直に記録する。

## 出力

`analysis/exp8_fine/` — 設定別 層プロファイル CSV、相対深さ重ね描き PNG (Fig.5 候補)、判定 JSON。
