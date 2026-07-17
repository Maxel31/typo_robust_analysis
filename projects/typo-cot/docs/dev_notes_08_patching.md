# 実験8 (activation patching) 開発メモ

ブランチ: `exp/08-patching` / 担当: 実験8 専任エージェント
正典: `docs/experiment_plan.md` §4 実験8 (+§7)。
上流依存: 実験1の `intervention/cell_builder.py` (セルC構成) — `exp/01-03-transplant`
ブランチをマージして流用 (コミット 67ad400)。

## モジュール構成

| モジュール | 役割 |
|---|---|
| `src/typo_cot/intervention/patching.py` | hook 管理 (捕捉/注入)・位置整列・層窓/部位/方向スイープ計画・パッチ付き生成・Δlogit/KL 指標。GPU 非依存 (モデルは引数注入) |
| `scripts/exp8/run_patching.py` | CLI。flip ペア選定 (LXT-4/Random-4 半々)・シャード対応・冪等 (ペア単位 JSON、既存はスキップ) |

## hook の取り付け点 (transformers 4.57.6 で確認済み)

デコーダ層の取得は `model.get_decoder().layers` を第一候補とし、失敗時は
「`self_attn` と `mlp` を持つ要素からなる nn.ModuleList を名前に 'vision' /
'visual' を含むパスを除いて探索」にフォールバックする。

| 家族 | クラス | 層パス | n_layers |
|---|---|---|---|
| Gemma-3-4B-it | `Gemma3ForConditionalGeneration` | `model.language_model.layers` (`get_decoder()` = `Gemma3TextModel`) | 34 |
| Llama-3.2-3B-Instruct | `LlamaForCausalLM` | `model.layers` | 28 |
| Mistral-7B-Instruct-v0.3 | `MistralForCausalLM` | `model.layers` | 32 |

部位 (site) 3種 = 計画書の {residual stream / attention 出力 / MLP 出力}:

| site | hook 対象 | 出力形式 (4.57) | 意味 |
|---|---|---|---|
| `residual` | `layers[l]` (層モジュール自体の forward hook) | Llama/Mistral: Tensor、Gemma3: tuple の [0] | 第 l 層通過後の残差ストリーム |
| `attn` | `layers[l].self_attn` | tuple `(attn_output, attn_weights)` の [0] | attention 出力 (o_proj 後、残差加算前。Gemma3 は post_attention_layernorm 前) |
| `mlp` | `layers[l].mlp` | Tensor | MLP 出力 (down_proj 後、残差加算前。Gemma3 は post_feedforward_layernorm 前) |

出力の tuple/Tensor 差は `_get_hidden` / `_replace_hidden` で吸収する。
注入は forward hook の返り値置換で行い、値は donor 捕捉値 (CPU 保持) を
実行時に device/dtype キャストしてコピーする。捕捉→再注入は bit 同一なので
恒等パッチ (no-op) は greedy 生成を一切変えないはず — これをスモーク合格判定
(a) に使う。

## 2 run 構成と位置整列 (セルC流用)

- **donor/recipient の 2 run**: clean run = 実験1セルA入力、pert run = セルC入力。
  両者は質問のみ異なり、CoT 以降は**同一文字列を teacher-forcing** する。
- **強制テキスト** (answer 読み出しモード):
  `prompt(Q_side) + C_c_prefix + trigger + common_answer_prefix`
  - `C_c_prefix` / `trigger`: `cell_builder.truncate_before_answer` の切断結果
    (trigger は clean CoT 中の実文字列 "The answer is" 等)。
  - `common_answer_prefix`: clean 側答え継続 (clean CoT の trigger 以降) と
    typo 側答え継続 (typo CoT の trigger 以降、無ければ `" " + answer_typo`) を
    継続トークン列にしたときの共通先頭部。最初に分岐するトークン対
    `(a_clean, a_typo)` が Δlogit 読み出しの対象。16 トークン以内に分岐が
    無いペアは除外 (`no_answer_divergence`)。
- **suffix 整列**: `full_ids[prompt_len:]` が両 run で完全一致することを要求
  (実験3 `align_cot_targets` と同じ規約)。不一致は `token_alignment_mismatch`
  で除外。一致すれば CoT 以降の位置対応は質問長差 Δ のオフセット補正のみ。
- **部位別のパッチ位置**:
  - (a) `question_span`: アーカイブ `perturbed_tokens` の各語 (k=4) について、
    clean 質問中の original_token / typo 質問中の perturbed_token の文字スパンを
    プロンプト末尾の質問領域から特定し (最後の出現)、offset_mapping で
    **スパン末尾トークン** 1 点に対応させる (計画書の「スパン末尾 or 平均プーリング」
    のうち前者を採用)。語 i ↔ 語 i で整列 (4 点)。
  - (b) `cot_suffix`: プロンプト直後〜強制末尾の全位置 (CoT+答え句 suffix)。
  - (c) `answer_span`: trigger 開始トークン〜強制末尾。trigger 開始オフセットは
    run ごとに計算し、両 run で一致しない場合は除外 (`trigger_offset_mismatch`)。
- **S2 モード** (CoT 開始段): 入力はプロンプトのみ。最終位置の next-token 分布
  (= c1 の分布) を読み出し、質問位置パッチ (site=a のみ) の KL 回復を層別に測る。
  指標: KL(p_donor ‖ p_patched) と KL(p_donor ‖ p_recipient) から
  recovery = 1 − KL_patched/KL_recipient。

## スイープの軸と 1 ペアあたりの forward 数

**「部位」の 2 解釈と実装**: 計画書 §4 実験8 手法3 の「部位3種」は
{(a)質問の摂動語スパン/(b)CoT+答えのsuffix/(c)答え位置} =
**注入する位置の種別** (`site_kind`) であり、介入は
「do(第l層の残差ストリーム := clean実行の値)」= residual が既定。
一方タスク指示は部位を {residual / attention出力 / MLP出力} =
**hook 先モジュール** (`site`) とも読める。実装は両軸を直交して持ち
(`--sites` × `--site-kinds`)、スイープは
`site (hook部位) × site_kind (位置種別) × 層窓 × 方向` の全交差。
本番でどちらを見出しの「部位3」とするかは open question (計画書に忠実なら
`--sites residual` 固定で site_kind×方向×層窓)。

- 方向 2: `clean→pert` (denoising: recipient=pert run, donor=clean run) /
  `pert→clean` (noising: 逆)。
- 層窓: 幅 w=3、stride は既定 w (非重複)。Gemma-3-4B (34層) → 12 窓
  ([0,3) … [33,34))。--window-stride 1 でスライディングに変更可。
- 1 セル = 1 回の patched generate (prefill 1 + ≤16 decode)。
  step-0 scores から Δlogit、生成テキストから flip を同時取得。
  site_kind=question_span のセルでは同じ forward の最終層 hook から
  c1 位置の hidden を取り、final norm + lm_head で c1 分布を復元して
  S2 の KL 回復も同時に読み出す (追加 forward ゼロ)。
- 1 ペア×1 摂動条件あたり (Gemma-3-4B, w=s=3):
  - 捕捉 2 forward + 基準 2 forward + 基準 2 generate (clean/pert 両 run)
  - no-op 検証 6 (3 hook部位 × 2 方向、中央窓、cot_suffix 位置)
  - 本スイープ: 3 hook部位 × 3 位置種別 × 12 窓 × 2 方向 = 216 patched generate
    (計画書忠実の residual 限定なら 72)
  - 計 ≈ 228 forward/ペア/条件 (residual 限定なら ≈ 84)。
    バッチは 1 固定 (プレフィル長がペアごとに異なるため)。

## 指標 (セルごと)

- `delta_logit` = logit(a_clean) − logit(a_typo) @ 分岐位置 (step-0 scores)。
- `recovery` = (Δ_patched − Δ_recipient) / (Δ_donor − Δ_recipient)。
  |Δ_donor − Δ_recipient| < 1e−3 のペアは recovery を null にする
  (セルCで DE≈0 のペアでは未定義 — 実験1本番の DE 規模で解釈が決まる)。
- `answer`: 生成継続 (trigger+common_prefix+生成) から extractor で抽出。
  `matches_donor` / `matches_recipient` / flip 解消 (denoising:
  recipient≠donor だったペアで patched==donor)。
- S2: `kl_recovery` (上記)。
- 統計 (本番): flip 解消/悪化のセルごと McNemar + リスク差 CI、セル間 Holm
  (共通規約 `analysis/stats.py` — Step 0 側で整備予定の共通定型に接続)。
  スモークは生データ JSON 保存まで。

## flip ペア選定 (修正A)

アーカイブの baseline/perturbed `results.json` の `is_correct` から
flip = (clean 正解 ∧ 摂動誤答) を判定 (analysis/full_results.json の
pattern="correct→incorrect" と同値)。LXT-4 (`k4_importance`) と Random-4
(`k4_random`) から半々。sample_id ソート後 seed=42 でシャッフルし先頭 n/2 件
ずつ。gemma-3-4b-it × gsm8k: LXT-4 flip 86 / Random-4 flip 53 (n=16 は充足)。

## CLI 仕様

```
uv run python scripts/exp8/run_patching.py \
  --model google/gemma-3-4b-it --benchmark gsm8k \
  --baseline-dir  <archive>/outputs/baseline/gemma-3-4b-it_gsm8k \
  --perturbed-dir-lxt <archive>/outputs/perturbed/gemma-3-4b-it_gsm8k_k4_importance \
  --perturbed-dir-rnd <archive>/outputs/perturbed/gemma-3-4b-it_gsm8k_k4_random \
  --n-pairs 16 --window-size 3 --mode both --noop-check \
  --output-dir results/exp8/smoke_gemma3-4b_gsm8k \
  [--num-shards N --shard-index i]
```

- 冪等性: ペア×条件ごとに `results/<cond>/<sample_id>.json` を書き、
  既存かつ config ハッシュ一致ならスキップ。`--force` で再計算。
- シャード: sample_id ソート後 `pairs[i::N]`。
- GPU 実行は必ず `tmp/gpu-locks/run_with_gpu.sh` 経由
  (`setup_device` は呼ばず、ヘルパーの CUDA_VISIBLE_DEVICES を尊重)。

## 設計判断 (技術的なもの)

1. **層窓の既定は非重複 (stride=w=3)**: スライディング (stride=1) はコスト×3。
   本番でピーク層の解像度が要る場合のみ --window-stride 1 で再走査する。
2. **site=(a) はスパン末尾トークン 1 点**: 平均プーリング注入は
   「捕捉値の平均を全スパン位置に書き込む」となり非対称が出るため初版では
   採用しない (open question に記載)。
3. **common_answer_prefix の強制**: flip 判定は「共通接頭辞に条件付けた生成」
   になる。GSM8K の数値答えでは共通部は通常空か 1 トークンで影響軽微だが、
   本番前に条件付けなし版 (trigger 直後から生成、Δlogit は別 forward) との
   感度チェックを検討 (open question)。
4. **KL の向き**: S2 は KL(donor ‖ ·) に固定 (実験3 の KL(clean‖typo) と整合)。
5. **decode ステップの安全性**: 注入 hook は「出力 seq_len > max(dst)」の
   ときのみ発火。generate の decode ステップ (seq_len=1) では自動的に無効。
6. **捕捉は必要位置のみ**: donor 側 forward で {span 末尾, suffix 全位置} だけを
   CPU に複製 (bf16 のまま)。34層×3部位×(suffix≈300)×2560 ≈ 160MB/ペア (一時)。

## 本番 GPU 見積 (スモーク実測 2026-07-15, RTX PRO 6000 Blackwell)

実測: Gemma-3-4B × GSM8K で 1 forward (prefill ≈1100 tok + 16 decode) ≈ 0.175 s
(decode 支配的)。1 ペア (228 forward, 3部位交差) ≈ 40 s。

アーカイブ flip 全数 (M3×B2, LXT-4+Random-4: gemma 662 / llama-3B 870 /
mistral 729 ペア) を使う場合:

| 構成 | 概算 |
|---|---|
| 計画書忠実 (--sites residual, w=s=3) | ≈ 9〜10 h ≈ **0.4 GPU日** |
| hook部位 3 交差 (residual/attn/mlp) | ≈ 25 h ≈ **1.1 GPU日** |
| 上記 + スライディング窓 (stride=1, 窓数×3) | ≈ 1.2 / 3.1 GPU日 |

いずれも計画書の 4〜6 GPU日を大きく下回る (Mistral の decode 速度を
0.28 s/fwd と仮定。MMLU はプロンプト長同程度・CoT 短めでほぼ同等)。
実験1本番で flip ペアを再判定・増員した場合はペア数に比例してスケール。

## スモーク検証集計 (2026-07-16, gemma-3-4b-it x gsm8k, n_pairs=16)

### 実行サマリ

| 項目 | 値 |
|---|---|
| 全タスク数 | 16 (lxt4: 8, rnd4: 8) |
| 完了 (patching 実行) | 11 |
| 冪等スキップ | 2 |
| 除外 | 3 (no_typo_answer: 2, no_answer_divergence: 1) |
| 失敗 | 0 |

除外ペア (lxt4 のみ): gsm8k_00107 (no_typo_answer), gsm8k_00152 (no_typo_answer),
gsm8k_00757 (no_answer_divergence), gsm8k_00959 (no_typo_answer)。
lxt4 は 4/8 完了、rnd4 は 8/8 完了。

### 判定 (a): 恒等パッチ (no-op) テスト

**PASS**: 72/72 の no-op 検証がすべて合格。6 組合せ (3 hook部位 x 2 方向) x
12 ペアで、全て generation_unchanged=true かつ answer_unchanged=true。
恒等パッチ (donor=recipient) で greedy 生成が bit 同一であることを確認。

### 判定 (b): clean→pert (denoising) パッチの効果

2592 セル (1296 clean_to_pert + 1296 pert_to_clean) の全数解析。

**答え flip**: 0/2592 セルで答え flip なし (0.0%)。全ペアで clean/pert の答えが
もとから一致 (flip ペアではなく同答ペアが選定された) ため、Dlogit 回復のみが
有意義な指標。

**S2 KL 回復 (question_span, clean_to_pert 方向)**:

| hook部位 | mean recovery | max | n_positive/total |
|---|---|---|---|
| residual | -0.045 | 0.989 | 84/144 (58%) |
| attn | -0.095 | 0.783 | 74/144 (51%) |
| mlp | +0.045 | 0.936 | 78/144 (54%) |

residual の早期層窓 [3,6) で平均 0.075 のピーク。mlp は [0,3) で 0.355
と顕著に高い (入力埋め込み直後の MLP が質問摂動の影響を支配的に媒介)。
後半層窓 [24,27)~[30,33) では residual/attn ともに recovery 負値
(KL がむしろ増大)、これは後半層の残差が答え句方向に特化しているため
質問スパンのパッチが有害に作用する解釈と整合。

### 判定 (c): スキーマ網羅性

**PASS**: 12/12 完了ペアで site*site_kind*direction = 18/18 全カバー。
各ペア 216 セル (3 hook x 3 位置種別 x 12 窓 x 2 方向)。
cell_exclude_reasons が付いたペアは 2 件 (gsm8k_00507: multi_trigger_typo,
gsm8k_00464: multi_trigger_clean + no_trigger_typo) だが、
セル生成自体はエラーなく完了。

### 最終判定

スモーク **PASS**。(a) no-op 恒等性 100%、(b) S2 KL 回復の方向は mlp
早期層で明瞭 (+0.355)、(c) スキーマ全セル完走。

**注意点**: 今回の n=16 ペアでは答え flip ペアが 0 件だったため、
denoising による答え flip 逆転は未検証。本番で flip ペアを明示的に
選定する必要がある (open question 4 で議論済み)。

## Open questions (ユーザー判断待ち)

0. **見出しの「部位3」の解釈**: 計画書 (§4 実験8 手法3) の部位 =
   位置種別 {質問スパン/CoT+答えsuffix/答え位置} で介入は residual 固定、
   タスク指示の部位 = hook 先 {residual/attn/mlp}。実装は両軸対応済み。
   本番は計画書解釈 (--sites residual、コスト 1/3) を既定とする想定でよいか。
1. 本番の層窓 stride (非重複 w=3 か、スライディング s=1 で解像度優先か)。
2. site=(a) の平均プーリング変種を追加するか (計画書は「または」表記)。
3. flip 判定の common_answer_prefix 条件付け (設計判断3) の感度チェック要否。
4. 本番の flip ペア数: 計画書は 300〜500/設定だが、例えば gemma-3-4b×gsm8k の
   アーカイブ flip は LXT-4 86 / Random-4 53 (計 139)。全数使用でも計画数に
   届かない設定が出うる — 実験1本番の再判定 flip (全ペア再生成) で補うか、
   設定横断でプールするかの判断が必要。
5. S2 の測定点は実験3本番のオンセット分布確定後に再調整 (現状は c1 固定)。
6. 実験1本番で DE≈0 (パターンX) が確定した場合、answer モードの主指標を
   Δlogit 回復から S2 KL 回復へ重心移動するか。

## 本番集計 (2026-07-17, M3×B2 全 flip ペア, 3 hook部位交差, w=s=3)

キュー: `scripts/exp8/prod/run_prod_queue.sh` (GPU 0 専用, 2026-07-16 10:27 開始,
07-17 00:31 QUEUE FINISHED)。出力: `results/prod/exp8/<model>_<bench>/{lxt4,rnd4}/`。

### 実行サマリと整合性検証

| 設定 | n_tasks | done | excluded | failed | 除外内訳 | セル/ペア |
|---|---|---|---|---|---|---|
| gemma-3-4b-it × gsm8k | 207 | 173 (lxt 109 / rnd 64) | 34 | 0 | no_typo_answer 32, no_answer_divergence 1, no_trigger_clean 1 | 216 (12窓) |
| gemma-3-4b-it × mmlu | 606 | 576 (lxt 348 / rnd 228) | 30 | 0 | no_trigger_clean 23, no_typo_answer 7 | 216 (12窓) |
| Llama-3.2-3B × gsm8k | — | **0 (FAIL rc=1)** | — | — | — | — |
| Llama-3.2-3B × mmlu | — | **0 (FAIL rc=1)** | — | — | — | — |
| Mistral-7B-v0.3 × gsm8k | 269 | 267 (lxt 146 / rnd 121) | 2 | 0 | no_typo_answer 1, no_answer_divergence 1 | 198 (11窓) |
| Mistral-7B-v0.3 × mmlu | 628 | 608 (lxt 370 / rnd 238) | 20 | 0 | no_trigger_clean 18, no_typo_answer 2 | 198 (11窓) |

- **整合性 PASS (完走 4 設定)**: 各設定でファイル数 = n_tasks、done+excluded の
  実数が run_summary.json と完全一致。config_hash 不一致 0。全 done ペアで
  セル数 = 3部位 × 3位置種別 × 2方向 × 層窓数 (Gemma 216 / Mistral 198。
  「216 セル」は 34 層モデル固有で、32 層の Mistral は 11 窓 = 198 セルが正)、
  site×site_kind×direction 18/18 カバー。
- **Llama 2 設定は未実施 (要再走)**: 17:16 / 17:23 に両設定とも起動直後の
  `AutoTokenizer.from_pretrained` で失敗。根本原因は transformers の
  `_patch_mistral_regex` → `is_base_mistral` が**ローカルキャッシュ有でも**
  HF Hub へ `model_info` を照会し、HF 側 504 Gateway Timeout (当時の一時障害)
  で例外化したこと (`logs/exp8/prod/Llama-3.2-3B-Instruct_*.log`)。
  出力ディレクトリは空でキューは冪等なので、`HF_HUB_OFFLINE=1` を付けて
  当該 2 設定のみ再実行すればよい。
- **出力ディレクトリ共有バグの退避検証 PASS**: 修正前スクリプトはルート直下
  `results/prod/exp8/{lxt4,rnd4}/` と `run_summary.json` を全設定で共有していた。
  gemma_gsm8k (12:28 完了時) と gemma_mmlu (restart watcher が 17:08 に退避) の
  バックアップを md5 で全数照合: ルート直下 lxt4 492 + rnd4 321 = **813 ファイル
  すべて設定別ディレクトリのコピーと bit 一致**、欠損 0。ルートの
  run_summary.json は gemma_mmlu (hash cc0d22cbed4e08dd) と同一。
  → **設定別ディレクトリが正**。ルート直下の lxt4/ rnd4/ run_summary.json は
  完全な重複残骸であり削除可 (未削除のまま保持)。

### S2 KL recovery プロファイル (主指標, question_span, clean→pert)

セル単位の median (mean は KL_recipient≈0 ペアの比の発散で外れ値支配になるため
median / frac_pos を主とする)。residual の層窓プロファイル (lxt4):

| 設定 (lxt4) | 0-3 | 3-6 | 6-9 | 9-12 | 12-15 | 15-18 | 18-21 | 21-24 | 24-27 | 27-30 | 30- |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma × gsm8k | .716 | **.769** | .714 | .693 | .595 | .490 | .289 | .205 | .158 | .089 | ≈0 |
| gemma × mmlu | .710 | .737 | **.762** | .706 | .662 | .525 | .427 | .247 | .155 | .112 | ≈0 |
| mistral × gsm8k | **.585** | .561 | .439 | .364 | .226 | .151 | .052 | .041 | .026 | .014 | ≈0 |
| mistral × mmlu | **.698** | .631 | .575 | .580 | .451 | .295 | .156 | .076 | .059 | .036 | ≈0 |

各設定×条件の最良セル (median 基準) と部位ごとの早期層の値:

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

**モデル間再現性: 仮説「MLP 早期層 or residual 早期〜中期層が質問摂動の媒介を
支配」は完走 4 設定すべて (2 モデル × 2 ベンチ × 2 摂動条件) で再現**:

1. residual 早期〜中期層 ([0,3)〜[12,15)) が最大 (median 0.33〜0.77)、
   深さ方向に単調減衰して最終層で ≈0。ピークは Gemma が [3,9)、
   Mistral が [0,6) とやや浅い。
2. mlp は早期層 ([0,3)〜[6,9)) のみ正 (最良 0.31〜0.59)、residual に次ぐ。
3. attn は全層でほぼ 0、しかも最早期 [0,3) は一貫して**負** (−0.06〜−0.34;
   質問スパン位置の attention 出力単独の移植は c1 分布をむしろ乱す)。
   スモーク時の mean ベース所見 (mlp[0,3) 優位) は外れ値の影響で、
   median ベースでは residual 優位が正しい描像。

### LXT-4 vs Random-4 の効果比

| 設定 | KL_unpatched 平均 (lxt / rnd) | 比 | 最良セル median 回復 (lxt / rnd) | 比 |
|---|---|---|---|---|
| gemma × gsm8k | 0.470 / 0.047 | **10.1×** | 0.769 / 0.346 | **2.2×** |
| gemma × mmlu | 1.257 / 0.127 | **9.9×** | 0.762 / 0.328 | **2.3×** |
| mistral × gsm8k | 0.612 / 0.128 | **4.8×** | 0.585 / 0.397 | **1.5×** |
| mistral × mmlu | 0.737 / 0.150 | **4.9×** | 0.698 / 0.430 | **1.6×** |

LXT-4 (重要語摂動) は Random-4 より c1 分布を 5〜10 倍強く乱し (KL_unpatched)、
かつ早期層パッチによる回復率も 1.5〜2.3 倍高い。つまり重要語の摂動効果は
早期層の質問スパン表現に**より集中して**書き込まれており、そこを clean 値に
戻すだけで大部分が打ち消せる。Random-4 の摂動は絶対量が小さいうえ回復も
部分的 (残りは層窓・位置に分散) で、Gemma でこの対比がより鮮明。

### 答え flip (副指標) と max_new_tokens=16 の制約 【重要な注意】

本番はスモークと同じ `max_new_tokens=16` + clean CoT teacher-forcing の読み出し
regime であり、**アーカイブ選定時の flip がこの regime では大半再現しない**
(監視報告済み)。baseline で clean/pert の答えが分岐したペア (flip 再現):

| 設定 | lxt4 分岐/done | rnd4 分岐/done |
|---|---|---|
| gemma × gsm8k | 0/109 (0%) | 0/64 (0%) |
| gemma × mmlu | 72/348 (21%) | 42/228 (18%) |
| mistral × gsm8k | 4/146 (3%) | 5/121 (4%) |
| mistral × mmlu | 67/370 (18%) | 30/238 (13%) |

- **影響範囲**: GSM8K では flip 逆転がほぼ観測不能 (CoT を clean 側で
  teacher-forcing するため、答え分岐の主経路である「CoT 自体の逸脱」が
  遮断される)。GSM8K の flip 系指標 (flip 解消率・McNemar) は本 regime では
  **無意味**であり報告しない。MMLU は選択肢ラベル 1 トークンの読み出しで
  13〜21% のペアが分岐を保持し、副指標として使用可。
- **したがって本実験の主指標は S2 KL recovery である** (上記プロファイル。
  Δlogit recovery も分岐位置が定義できる範囲で補助に使える)。
- MMLU の分岐ペアに限れば flip 逆転は明瞭: question_span への
  residual[0,3) パッチだけで donor 答えへの flip が lxt4 69〜75% /
  rnd4 69〜77% (gemma 54/72・29/42, mistral 47/67・23/30)。flip の層窓分布も
  KL recovery と同じく早期 residual/mlp に集中し、attn は僅少。
  一方 cot_suffix / answer_span への中後期 residual パッチは分岐ペアの
  97〜100% を flip させるが、これは答え表現の直接上書きに近く
  (positive control)、質問摂動の媒介の証拠としては question_span 系のみを使う。

### 結論 (論文向けサマリ)

1. **質問タイポの効果は早期層の摂動語スパン表現に局在する**: 摂動語位置の
   residual stream を第 0〜12 層で clean 値に戻すだけで、CoT 開始分布 (c1) の
   KL 乖離が中央値 59〜77% (LXT-4) 回復し、MMLU では答え flip の約 7 割が
   逆転する。この profile は Gemma-3-4B / Mistral-7B × GSM8K / MMLU の
   4 設定で一貫して再現し、寄与は residual ≫ MLP (早期のみ) ≫ attn (≈0)。
2. **重要語摂動 (LXT-4) はランダム摂動の 5〜10 倍の分布乖離を生むが、
   その効果はより「パッチ可能」**: 早期層 1 窓の介入で過半が打ち消せる。
   ランダム摂動は小さく分散的で単一窓の回復率が低い (1.5〜2.3 倍差)。
3. 制約: 読み出しは max_new_tokens=16 + clean CoT 強制の条件付き regime で
   あり、GSM8K の flip 系指標は使えない (S2 KL recovery が主指標)。
   Llama-3.2-3B の 2 設定は HF Hub 一時障害で未取得 (再走で補完予定) のため、
   現時点のモデル間一般化主張は 2 モデルに限る。

集計スクリプト: セッションスクラッチパッドの ad-hoc 集計 (読み取りのみ、
median/frac_pos は s2_kl_recovery の null 除外後)。数値の再導出は
`results/prod/exp8/<設定>/{lxt4,rnd4}/*.json` の `cells[]` から可能。
