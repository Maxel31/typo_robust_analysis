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
