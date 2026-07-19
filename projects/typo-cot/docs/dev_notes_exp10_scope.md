# 実験10 (スコープ拡張) dev notes — 推論基盤と生成

担当: exp/10-scope ブランチ。ユーザー指示「DeepSeek-R1蒸留とMATH-500についてはまだ推論結果が存在しないため、推論から行なってください。使用するgpuは現段階では3,4とします。」

## 重要な事実確認 (2026-07-14)

- **MATH-500 は M5+Qwen ではアーカイブに既存**。`archive_baseline` に
  `{gemma-3-1b/4b/12b-it, Llama-3.2-1B/3B-Instruct, Mistral-7B-Instruct-v0.3, Qwen2.5-0.5B/1.5B/3B/7B-Instruct}_math`
  (各500件、importance_scores 完備、config: max_new_tokens=512, seed=42, greedy)。
  摂動側も `outputs/perturbed` に gemma/Llama 系 6条件、Mistral/Qwen0.5/1.5 は 4条件既存。
  → 本タスクで再生成せず、**真に欠けている R1蒸留系の生成に集中**(open_questions に記録)。
  - アーカイブで欠けている MATH 摂動: Qwen2.5-3B/7B の全条件、Mistral/Qwen0.5/1.5 の k1/k2 系。
- **DeepSeek-R1-Distill-Qwen-7B はアーカイブに一切ない**(モデルも HF キャッシュに無し→DL済み)。
- lxt 2.1 は qwen2 パッチで R1-Distill-Qwen-7B (Qwen2ForCausalLM) に**適用可能なことを GPU スモークで確認済み**(R_Q 計算成功)。R_C(長大CoTのbackward)は計画書 §4-実験10③ の通り非計算。

## 実装 (TDD)

- `src/typo_cot/models/reasoning.py`: R1蒸留サポート
  - ゼロショット+チャットテンプレート(DeepSeek公式推奨。テンプレートは
    `<｜begin▁of▁sentence｜><｜User｜>{msg}<｜Assistant｜><think>\n` を生成=CoTが本文から始まる)
  - 答えトリガー: gsm8k/mmlu は既存 canonical「The answer is …」、math は \boxed{}
  - `split_reasoning_output` (<think>/</think> 分離)、`think_prefix_end` (R_Q ターゲット位置)
  - 抽出チェーン: ベンチ抽出器 → $記号除去リトライ → boxed フォールバック → 切断時 CoT 救済
  - max_new_tokens 既定: gsm8k/mmlu 4096, math 8192 (レジストリの512はreasoning系に不適合のため拡張)
- `ModelWrapper.ALLOWED_MODELS` に deepseek-ai/DeepSeek-R1-Distill-Qwen-7B 追加
- `scripts/run_inference_reasoning.py`: シャード生成(--start/--end)+--merge 統合。
  出力スキーマはアーカイブ `outputs/baseline/<model>_<bench>/` と互換
  (results.json に cot_text / answer_text / has_think_close / truncated /
  num_generated_tokens / extraction_method を追加。question_top_k_words は R_Q 由来)。
  `--compute_rq` で PerturbedDatasetCreator 互換の importance_scores/*.pt を保存
  (token_scores=質問のみフィルタ済み、token_scores_with_choices=選択肢込みフィルタ済み)。
- `tests/test_reasoning.py` (19件) / `tests/test_math500.py` (10件)。全体 172 passed, 24 skipped。
- 環境: exp/04 から cherry-pick (setup_device の CUDA_VISIBLE_DEVICES 尊重 / transformers<5 / torch==2.9.1)。

## 生成規約

- greedy (do_sample=False), seed=42 (Step 0 レジストリと同一)、bf16、batch=16(mathは12)、左パディング、add_special_tokens=False(テンプレートがBOS含有)。
- サンプル集合はアーカイブと同一: gsm8k=test全1319件、mmlu=57サブセット×100件 seed42=5700件、math=全500件。
- 出力先: `projects/typo-cot/outputs/baseline/DeepSeek-R1-Distill-Qwen-7B_{gsm8k,math,mmlu}/`
  (シャード: `shards/results_{start:05d}_{end:05d}.json`、毎バッチ上書き保存=中断復帰可)

## 実行ログ (2026-07-14)

- 19:39 GPUスモーク (gsm8k n=4, R_Q付き): 3/4 正解、think_close 4/4、抽出は $記号ケースを修正。
- 19:43 本番開始。driver A (GPU片方): gsm8k [0,220)(220,440)(440,660) → math [0,250) → mmlu [0,2700) 300刻み。
  driver B: gsm8k [660,880)(880,1100)(1100,1319) → math [250,500) → mmlu [2700,5700) 300刻み。
  ドライバは 1 ヘルパー呼び出し=1 シャード(シャード間でロック解放)。
  ログ: scratchpad/driverA.log, driverB.log。ペース ~45s/バッチ16件。

### シャード進捗

(完了時に追記)

## 再開コマンド

```bash
cd /diskthalys/ssd14tc/sfukuhata/dev/kanolab/typo_robust_analysis/.claude/worktrees/exp-10-scope/projects/typo-cot
H=/diskthalys/ssd14tc/sfukuhata/dev/kanolab/typo_robust_analysis/tmp/gpu-locks/run_with_gpu.sh
# 未完了シャードの再実行 (例: gsm8k [220,440)):
bash $H uv run python scripts/run_inference_reasoning.py --benchmark gsm8k --start 220 --end 440 --batch_size 16 --compute_rq
# 全シャード完了後の統合:
uv run python scripts/run_inference_reasoning.py --benchmark gsm8k --merge
uv run python scripts/run_inference_reasoning.py --benchmark math --merge
uv run python scripts/run_inference_reasoning.py --benchmark mmlu --merge
```

## 次ステップ (摂動側)

```bash
# Random-4 データセット (R_Q 上位4語を除外して乱択; importance_scores 必須=生成済み)
uv run python scripts/run_perturbation.py --baseline_dir outputs/baseline/DeepSeek-R1-Distill-Qwen-7B_gsm8k -k 4 --random_perturbation
# LXT-4 データセット (R_Q 上位4語)
uv run python scripts/run_perturbation.py --baseline_dir outputs/baseline/DeepSeek-R1-Distill-Qwen-7B_gsm8k -k 4
# 摂動側生成 (シャード分割で)
bash $H uv run python scripts/run_inference_reasoning.py --benchmark gsm8k \
  --perturbed_data datasets/perturbed/DeepSeek-R1-Distill-Qwen-7B_gsm8k_k4_random_with_choices/perturbed_dataset.json \
  --start 0 --end 220 --batch_size 16
```

## 既知の注意点

- word 集約表示で質問末尾トークンがテンプレートトークンと結合して
  `take?<｜Assistant｜><think>` のような表示になる(token_scores/offset は正しく
  範囲フィルタ済みなので摂動標的選定には影響なし。表示のみの問題)。
- DeepSeek 公式推奨は temp 0.6/top_p 0.95 だが、Step 0 凍結(greedy/seed42)と
  アーカイブ整合を優先して greedy。greedy の反復ループは truncation_rate で監視。
- mmlu の質問スパンは選択肢埋め込み時(摂動データ)は最初の改行まで(既存規約と同一)。

## R1蒸留×MATH の LXT<Random 逆転の解剖 (Track C, 2026-07-19)

再解析のみ(CPU, 生成再利用)。`outputs/{baseline,perturbed}/DeepSeek-R1-Distill-Qwen-7B_math*`
の results.json を突き合わせて flip 集合と摂動語カテゴリを集計。

### 事実

- 精度: baseline 0.734 / LXT-4(importance) 0.650 / Random-4 0.610。
  Random の方が有害 = **逆転**(通常は importance 標的の LXT が有害)。
- flip 集合 (baseline 正解 367 → 摂動で誤答): LXT 68 (18.5%) < Random 87 (23.7%)。
  重複 31、LXT のみ 37、Random のみ 56。
- 摂動語カテゴリ (全摂動): LXT = 自然語 84.8% / latex 9.9% / 変数(単文字) 2.8% / 記号 2.5%。
  Random = 自然語 79.6% / latex 10.4% / **変数 7.6%** / 記号 2.4%。
  → 当初仮説「MATH の R_Q 上位語は LaTeX に偏る」は**棄却**(LXT 標的の大半は自然語)。
- 差の源泉は**単文字の数式変数・区切り記号**。Random は全問の 28.4% でこれらを破壊
  (LXT 18.4%; ~1.5倍)。flip 集合では Random 43.7% vs LXT 36.8% が変数/記号を含む。
  例: `x→c`(math_00074), `k→m`(math_00108), `x→s`×2(math_00114), `$→nn`(math_00035)。
  これらは正解を別問題に silent に書き換え、R1 は破壊後問題を忠実に解いて誤答。
- 難易度プロキシ: Random-only flip は生成トークン長が大 (~3004 vs base-correct 2533)、
  質問文字数は短め (133) = 記号密度の高い問題を Random が壊しやすい傾向。
- 一般性: 逆転は R1蒸留で顕著、gemma-4B は誤差内 (0.368 vs 0.364)、
  Qwen/Llama/Mistral は通常序列 → 長大 CoT の自己訂正が関与する現象。

### 考察草稿 (experiment_details.md 実験10節へ追記予定, 4–5文)

> R1蒸留×MATH では Random-4 (acc 0.610) が LXT-4 (0.650) より有害という逆転が生じた
> (baseline 0.734、flip は Random 87 > LXT 68)。摂動語カテゴリの集計から当初仮説
> 「R_Q 上位語が LaTeX 記法に偏る」は棄却され、LXT-4 の標的の 84.8% はむしろ自然語で、
> 差を生むのは単文字の数式変数・区切り記号だった — Random は全問の 28.4% でこれらを破壊し
> (LXT は 18.4%)、flip 集合でも変数/記号の関与率が高い (43.7% vs 36.8%)。変数改名
> (x→c, k→m) や `$`→`nn` の破壊は言語的冗長性を持たず R1 の長大 CoT でも復元されないため、
> モデルは破壊後の別問題を忠実に解いて誤答する一方、高重要度の自然語 typo は R1 の言語
> 事前分布で自己訂正され吸収される。したがって重要度標的化の優位は、(a) 標的トークン型に
> 対しモデルが強い誤り訂正能力を持ち(長大 CoT × 自然語の冗長性)、かつ (b) 真の脆弱性が
> 低重要度だが構造的に不可欠なトークン(変数・記号)に宿るとき崩れる。MATH では saliency と
> typo 脆弱性が符号反転しており、この逆転が R1蒸留で顕著・gemma-4B で誤差内・他モデルで非出現
> なのは長大 CoT 自己訂正の関与を示唆する。
