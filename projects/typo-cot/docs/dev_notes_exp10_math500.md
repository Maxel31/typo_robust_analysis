# 実験10② dev notes — MATH-500 全面新規再生成 (M5+Qwen)

担当: exp/10-scope ブランチ(R1蒸留の dev_notes_exp10_scope.md とは別トラック)。

## ユーザー決定 (2026-07-14)

- MATH-500 はアーカイブの既存生成(2026-05-21)を採用せず、**新リポジトリのコードで全て新規再生成**(既存アーカイブとの一致検証も行う)。
- デコーディングは全モデル統一 greedy / seed=42(Step 0 凍結レジストリ準拠)。GPU は 3・4 のみ。

## スコープ

- モデル6: gemma-3-1b-it / gemma-3-4b-it / Llama-3.2-1B-Instruct / Llama-3.2-3B-Instruct /
  Mistral-7B-Instruct-v0.3 / Qwen2.5-7B-Instruct(計画書 §4 実験10②「M5+Qwen×MATH-500」)。
  R1蒸留は別ドライバ(dev_notes_exp10_scope.md)が生成中のため対象外。
- ベンチ: MATH-500(HuggingFaceH4/MATH-500 test 全500件、sample_id=math_00000..math_00499)。
- 摂動: LXT-4(k=4 importance)と Random-4(計画書 §2.2 修正A)。
- 生成規約: greedy(temperature=0.0/do_sample=False)、seed=42、max_new_tokens=512、
  batch_size=1(アーカイブ config と同一)、bf16。PYTHONHASHSEED=42
  (Random-4 の乱択が hash() を使うため)。

## 実装 (TDD、コミット)

- d8a35a8 test / 8ed3633 feat: `src/typo_cot/sharding.py`
  (shard_results_path / load_shard_rows / merge_shard_results / build_summary_from_results。
  命名・統合規約は run_inference_reasoning.py と同一、summary はアーカイブ互換スキーマ)
- 909d30d feat: `scripts/run_inference.py` に --start/--end/--merge を追記型で追加
  (シャード= shards/results_{start:05d}_{end:05d}.json 毎バッチ上書き=シャード内復帰可、
  完了済みシャードはモデルロード前に早期スキップ。引数なしは従来挙動)
- 4879e6d feat: `scripts/exp10_math500/run_queue.sh`(キュー本体)+
  `scripts/exp10_math500/verify_vs_archive.py`(アーカイブ照合)
- テスト: 183 passed / 24 skipped(既存 172+11 新規、既存を壊していない)

## キュー構成

- driver A: gemma-3-1b → gemma-3-4b → Mistral-7B / driver B: Llama-1B → Llama-3B → Qwen-7B。
- モデルごとに直列: clean シャード → merge → (最初のモデルのみ検証ゲート)
  → LXT-4/Random-4 データセット作成(CPU) → 摂動側生成シャード ×2条件 → merge。
- シャード境界: 1B=250×2 / 3-4B=125×4 / 7B=84×6(1シャード 20〜40分目安、実測で調整可:
  run_queue.sh の MODELS 配列を編集)。
- 総シャード数: GPU 72(clean 24 + 摂動 48)+ CPU merge 18 + データセット作成 12。
- 1シャード=1ヘルパー呼び出し(run_with_gpu.sh、GPU 3/4 flock 排他、GPU_LOCK_TIMEOUT=86400)。
  SMOKE_PAUSED 検出(exit 86)時はキューを停止する(再開は下記コマンド)。

## 起動・監視・再開

```bash
cd /diskthalys/ssd14tc/sfukuhata/dev/kanolab/typo_robust_analysis/.claude/worktrees/exp-10-scope/projects/typo-cot
# 起動(再開も同じコマンド。完了済みシャードは自動スキップ)
nohup bash scripts/exp10_math500/run_queue.sh A >> logs/exp10_math500/driverA.log 2>&1 &
nohup bash scripts/exp10_math500/run_queue.sh B >> logs/exp10_math500/driverB.log 2>&1 &
# 監視
tail -f logs/exp10_math500/driverA.log logs/exp10_math500/driverB.log
cat logs/exp10_math500/progress_A.json logs/exp10_math500/progress_B.json
```

## 検証ゲート (最初のモデルの clean 完了時)

```bash
uv run --no-sync python scripts/exp10_math500/verify_vs_archive.py --model_short gemma-3-1b-it
uv run --no-sync python scripts/exp10_math500/verify_vs_archive.py --model_short Llama-3.2-1B-Instruct
# 両方 OK なら:
touch scripts/exp10_math500/VERIFY_OK
```

- 判定: 精度差・抽出成功率差が ±3pt 以内 & n=500 一致で OK。
  乖離時は logs/exp10_math500/verify_*.json の per_sample_agreement を見て原因調査
  (プロンプト差 / transformers バージョン差 / ハードウェア数値差の順に疑う)。
- アーカイブ精度(参考): gemma-1b 26.8% / gemma-4b 44.4% / Llama-1B 22.2% /
  Llama-3B 30.0% / Mistral-7B 12.8% / Qwen-7B 49.8%。

### 検証結果

(完了時に追記)

## 出力先 (アーカイブ互換)

- clean: `outputs/baseline/<model>_math/`(results.json / summary.json / config.json /
  importance_scores/*.pt / shards/)
- 摂動データセット: `datasets/perturbed/<model>_math_k4_with_choices/` と
  `<model>_math_k4_random_with_choices/`
- 摂動側生成: `outputs/perturbed/<model>_math_k4_importance/` と `<model>_math_k4_random/`

## GPU 時間見込み

- clean 合計 ~13-16 GPU時間、摂動側 ×2 で総計 ~40-48 GPU時間 ≒ 2 GPU日。
  GPU 3/4 は R1蒸留ドライバ・実験4/5/7 キューと共有のため wall-clock はその 1.5〜3倍。

## open questions

(報告参照)
