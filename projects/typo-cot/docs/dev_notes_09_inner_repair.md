# 実験9 (inner lexicon 修復スコア) 開発メモ

ブランチ: `exp/09-inner-repair`。experiment_plan.md §4「実験9」および §7 の
実装マッピング (`repair/lexicon_probe.py`) に対応する。
共有ドキュメント (experiment_plan.md / work_items.md / README.md) は編集しない。

## 実装構成

| モジュール | 役割 |
|---|---|
| `src/typo_cot/repair/span_align.py` | clean/typo テキストの difflib 整列 -> 摂動語スパン対の復元、文字スパン -> スパン末尾トークン変換 |
| `src/typo_cot/repair/lexicon_probe.py` | 層別 hidden 抽出 (1 forward, output_hidden_states=True)、層別 cos、修復スコア = 最大層 cos (埋め込み層除外)、logit lens (最終 norm + unembed、Gemma-2 softcap 対応、悲観的 tie-break) |
| `src/typo_cot/repair/features.py` | 分割数増分・Zipf 頻度 (wordfreq)・先頭トークン ID |
| `src/typo_cot/repair/archive_access.py` | アーカイブ読み出しの薄い隔離層 (lxt4/random4)。**Step 0 master table 完成時は `load_condition_records()` の差し替えのみで移行** |
| `src/typo_cot/repair/pipeline.py` | HF モデル名対応、clean/typo プロンプト対の構築、語レベル行の組み立て |
| `src/typo_cot/repair/regression.py` | flip ~ 修復スコア + 分割増分 + Zipf + R_Q、GLM Binomial + クラスタロバスト SE (item=sample_id; (1\|item) の近似) |
| `scripts/exp9/run_inner_repair.py` | 計測ランナー (GPU)。`--dry-run` で CPU 整列検証のみ |
| `scripts/exp9/analyze_inner_repair.py` | 回帰係数表 CSV・flip 群別の層別 cos カーブ図・サマリ JSON (CPU) |
| `scripts/exp9/smoke.sh` | スモーク一式 (GPU ヘルパー経由で呼ぶ) |

## 設計上の要点 (実データ検証で判明した事項を含む)

1. **プロンプト整列**: 生成時 (`scripts/run_inference.py`) と同一テンプレートで
   clean/typo の完全プロンプトを構築し、プロンプト全体を difflib で整列する。
   few-shot 文脈が共通接頭辞になるため差分は typo 編集のみ。
2. **MMLU 系の選択肢**: アーカイブの `perturbed_question` は選択肢行
   "(A) ..." を **埋め込み済み** で `perturbed_choices` は None
   (Phase 2 の include_choices=True の仕様)。typo 側プロンプトに clean 選択肢を
   再付加してはならない (生成時と同じく choices=None で渡す)。
3. **壊れた perturbed_token メタデータ**: アーカイブには offset ずれで
   `perturbed_token` が実テキストと一致しないエントリがある (例: ' field' -> 'n fi')。
   包含照合で突合できない残余は、差分領域と摂動トークンを出現順で zip して救済。
   gemma-3-4b-it n=32 の dry-run で整列率: gsm8k/lxt4 98.4%, gsm8k/random4 100%,
   mmlu/lxt4 99.2%, mmlu/random4 100% (救済前は 76.6〜93.8%)。
4. **修復スコア**: `repair_score = max_{l>=1} cos(h_clean^l, h_typo^l)`
   (スパン末尾トークン、層0=入力埋め込みは除外; トークン自体が違うため)。
   副定義 (復号一致層) は `lens_first_hit_layer_top5` (typo 側 hidden から
   clean 語先頭トークンが top-5 に入る最初の層) として同時に出力。
5. **logit lens**: 最終 norm + unembed。`find_decoder_backbone()` が
   `layers`+`norm` を持つモジュールを探すため Gemma-3 マルチモーダル構成でも動く。
   Gemma-2 系の final_logit_softcapping は config から自動適用 (Gemma-3 は None)。
   rank の tie は標的の不利に数える (退化ケースを復号成功と誤判定しない)。
6. **R_Q**: `perturbed_tokens[].importance_score` (アーカイブ由来) を語レベルの
   R_Q としてそのまま使用。回帰の統制変数に入る。
7. **flip 定義**: baseline と perturbed の `extracted_answer` の不一致
   (span_extract_ok の語のみ)。主推定量 (clean 正解限定) は
   `--clean-correct-only` で対応。

## テスト

`tests/test_repair_*.py` (span_align / features / lexicon_probe /
archive_access / regression / pipeline)、GPU 不要。
既存テストと合わせ 192 passed / 24 skipped。

## スモーク (pending)

ユーザー指示 (2026-07-14 18:05) により GPU スモークは一時停止中
(GPU 3/4 他ユーザー占有)。実行手順:

```bash
cd projects/typo-cot
bash /diskthalys/ssd14tc/sfukuhata/dev/kanolab/typo_robust_analysis/tmp/gpu-locks/run_with_gpu.sh \
    bash scripts/exp9/smoke.sh
uv run python scripts/exp9/analyze_inner_repair.py \
    --input-dir results/smoke/exp9 --output-dir results/smoke/exp9/analysis
```

完了条件の確認先:
- (a) clean 同一語対 cos≈1: 各 summary_*.json の `sanity_clean_pair.pass`
- (b) logit lens が clean 語自身を復号: `lens_hit_rate_clean_self` (高いこと) と
  word_rows の `clean_self_min_rank`
- (c) 修復スコアと flip の負の符号: analysis_summary.json の
  `mean_repair_flip < mean_repair_noflip` と `repair_coef < 0`
  (n=32 では検出力不足の可能性 -> 方向のみ報告)

CPU 側は検証済み: dry-run (整列率上記) と、合成 word_rows での
analyze_inner_repair.py の end-to-end 動作 (負の repair 係数を回収)。

## 本番実行の見積り

M5+Qwen (6モデル) x B5 (5ベンチ) x 2条件、全サンプル。forward 2 回/サンプル
(+サニティ 2 回/設定)。プロンプト 2〜4k トークン、バッチ 1 で
1 設定 (~1300 サンプル) あたり 20〜60 分 -> 計画の GPU 1.5〜2 日と整合。
高速化するなら extract_span_hiddens の複数サンプルバッチ化が第一候補。
