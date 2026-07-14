# 実験1 (CoT transplant 2×2) + 実験3 (forced-decoding divergence) 開発メモ

ブランチ: `exp/01-03-transplant` / 担当: 実験1・実験3 専任エージェント

## モジュール構成 (src/typo_cot/intervention/)

| モジュール | 役割 |
|---|---|
| `records.py` | `PairRecord` — clean×typo のサンプル対。Step 0 master table に差し替えやすいフィールド構成 |
| `archive_loader.py` | アーカイブ baseline/perturbed `results.json` を sample_id で結合 → `PairRecord`。**アーカイブ依存はここだけ** |
| `cell_builder.py` | 答え句 (`The answer is`) 直前での CoT 切断 + 4セル (A/B/C/D) の teacher-forcing 入力構築 |
| `runner.py` | `run_cells(pairs, generate_fn, ...)` — バッチ生成・抽出・TE 照合。モデルは generate_fn として注入 (ユニットテストはモック) |
| `analysis.py` | flip 表 (TE/DE/IE)・見出し復帰率・IE\|CoT変化・bootstrap CI・GLMM (BinomialBayesMixedGLM, VB) |
| `divergence.py` | 位置別 KL/log-prob/rank のチャンク計算・オフセット補正付き位置対応・発散オンセット・precision@k + シャッフル帰無 |

CLI: `scripts/exp01_03/run_transplant.py` (実験3 は `--dump-divergence` フラグ)。

## セル定義

- A = (clean 質問, clean CoT) … 基準
- B = (typo 質問, typo CoT) … 総効果 TE (アーカイブ再現の検証にも使用)
- C = (typo 質問, clean CoT) … 直接効果 DE
- D = (clean 質問, typo CoT) … 間接効果 IE

## 設計判断 (技術的なもの)

1. **teacher-forcing は単純連結**: アーカイブの生成 (`scripts/run_inference.py`) は
   chat template を使わないプレーンテキスト few-shot completion
   (`system_prompt + "\n\n" + user_prompt`) なので、切断済み CoT を
   プロンプト末尾に連結するのが忠実な prefill 実装。将来 DeepSeek-R1-Distill 等の
   新規生成でチャットテンプレートを使う場合は `continue_final_message` 相当が必要。
2. **除外フラグ** (`cell_builder`): `no_trigger_*` / `multi_trigger_*` /
   `early_trigger_*` (先頭25%以内) / `residual_fragment_*`
   (prefix に `Answer:` 等の変種が残留)。主分析は全フラグなしに限定し、
   `flip_rate_sensitivity` に除外込みの値を併記。
3. **CUDA_VISIBLE_DEVICES を触らない**: `models/wrapper.setup_device` は
   CUDA_VISIBLE_DEVICES を上書きするため使用しない。`ModelWrapper` を直接
   構築し、run_with_gpu.sh が設定したデバイスをそのまま使う。
4. **divergence の位置対応**: prompt 単体と full input を別々にトークナイズし、
   suffix のトークン ID 列が clean/typo run で完全一致した場合のみ計算
   (`token_alignment_mismatch` でフラグして除外、件数は summary に記録)。
5. **GLMM**: A セルは構造的に flip=0 で切片が縮退するが、
   BinomialBayesMixedGLM のベイズ事前分布が正則化するため有限に推定される。
   全条件同時推定 (交互作用込み) という仕様通りの形。
6. **cot_changed**: 空白正規化した切断後 prefix の不一致で判定
   (ROUGE-L<1 の代理。master table 到来後に CoT:ROUGE-L 列へ差し替え可)。

## アーカイブ CPU 検証 (2026-07-14)

gemma-3-4b-it × gsm8k × k4_importance: 1319 ペア結合、除外 132 件
(no_trigger_typo 69 / no_trigger_clean 35 / multi_trigger 46 / residual 12 / early 1)。
mmlu 側も 2850 ペア結合、選択肢インライン再構成が clean 側テンプレート整形と一致。

## 本番実行の想定コマンド

```bash
bash /diskthalys/ssd14tc/sfukuhata/dev/kanolab/typo_robust_analysis/tmp/gpu-locks/run_with_gpu.sh \
  uv run python scripts/exp01_03/run_transplant.py \
  --model google/gemma-3-4b-it --benchmark gsm8k \
  --baseline-dir  $ARCHIVE/outputs/baseline/gemma-3-4b-it_gsm8k \
  --perturbed-dir $ARCHIVE/outputs/perturbed/gemma-3-4b-it_gsm8k_k4_importance \
  --dump-divergence --output-dir results/exp01_03/gemma-3-4b-it_gsm8k_k4_importance
```

Random-4 は `--perturbed-dir ..._k4_random`。全 25 設定は docs/v1_run_manifest.md 参照。

## 事前登録: 分岐別の結論文 (実行前に固定, 2026-07-14)

- **パターンX (CoT 媒介優位)**: IE が TE の大部分 (目安 7 割以上) を占め、
  DE 条件で clean CoT を強制すると大半の flip が元の答えに復帰する場合 —
  「Transplanting the clean CoT under the perturbed question restores the
  original answer in the majority of flipped cases (headline restore rate),
  while transplanting the perturbed CoT under the clean question reproduces
  most flips. Typo-induced errors are therefore causally mediated by the
  CoT text itself, not merely correlated with it: the pathway
  typo → CoT change → answer change carries the effect.」
- **パターンY (直接経路優位)**: DE が TE の大きな割合 (目安 4 割以上) を占め、
  clean CoT を強制しても flip が残る場合 —
  「Even when the clean CoT is forced verbatim, a substantial fraction of
  flips persists (DE), revealing a direct pathway from the perturbed input
  to the answer that bypasses the generated reasoning text. This motivates
  the internal-state analysis of Experiment 8, and implies that
  CoT-level defenses alone cannot fully recover accuracy.」

どちらの結果でも執筆可能。GLMM の交互作用項が大きい場合は
「質問 typo と CoT typo の効果は加法的でない」ことを明記する。

## 残タスク / 注意

- DeepSeek-R1-Distill-Qwen-7B はアーカイブに生成ログがないため、
  まず clean/typo の CoT 生成から必要 (別途; trigger_pattern も要調整)。
- MATH-500 / 拡張モデルの一部もアーカイブ有無を確認のこと。
- precision@10 の語タイプ対応はトークン近似 (KL 側) vs 単語 (R_C 側)。
  本実行前に offset_mapping ベースの単語集約に精緻化する余地あり。
- LOO ランキング比較 (修正B) は未実装 (実験8 側と共有予定のため保留)。
