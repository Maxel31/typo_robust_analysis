# 実験7 (校正器3段ラダー) 開発メモ

ブランチ `exp/07-correctors`。実験計画は docs/experiment_plan.md §4「実験7」を正典とする。
本メモは実装上の決定事項と検証結果のみを記録する (共有ドキュメントは編集しない)。

## 実装物

| パス | 役割 |
|---|---|
| `src/typo_cot/defense/correctors.py` | 校正器3段の統一インターフェース `correct(text)->text` |
| `src/typo_cot/defense/restoration.py` | 語単位の復元/非復元/誤修正分類 (rebuttal スクリプトのライブラリ化) |
| `src/typo_cot/defense/analysis.py` | flip サブセット集計・R_Q 偏在 Mann-Whitney |
| `scripts/exp7/make_corrected_dataset.py` | 摂動データセット→校正済みデータセット (15 テキストジョブ用) |
| `scripts/exp7/verify_pyspell_parity.py` | pyspell 段の rebuttal アーカイブ一致検証 |
| `scripts/exp7/analyze_correction.py` | baseline/摂動/校正後の3条件突き合わせ集計 (73 評価ラン用) |
| `scripts/exp7/smoke_correct_and_eval.py` | 校正→復元判定→評価生成のスモーク1周 |

テスト: `tests/test_defense_{correctors,restoration,analysis}.py` (GPU 不要、
校正モデルは generate_fn 注入でモック)。

## 校正器の選定と決定事項

1. **pyspell 段 (弱)**: `pyspellchecker==0.9.0` に固定 (辞書同梱のため訂正結果が
   バージョン依存)。rebuttal の `make_spellfix_dataset.py:correct_text` と同一ロジック。
   **唯一の差分**: pyspellchecker の `correction()` は頻度同点の候補を set 反復順
   (PYTHONHASHSEED 依存) で選び再現不能なため、`(最高頻度, 辞書順最小)` の決定的
   選択に置換した (`PySpellCorrector._correction`)。
2. **neural 段 (中)**: neuspell は**導入不可**と判定 →
   `ai-forever/T5-large-spell` (prefix `"grammar: "`) で代替。
   - neuspell==1.0.0 は pip 解決・インストール可能だが、
     `BertChecker.from_pretrained()` がダウンロードする model.pth.tar が
     Google Drive の virus-scan 警告 HTML (2430 bytes) で壊れている
     (torch.load: `invalid load key '<'`)。加えて torch>=2.6 の
     `weights_only=True` 既定とも非互換。証跡: `results/smoke/neuspell_feasibility.json`
   - 複数行入力 (選択肢行) は行単位で校正し改行構造を保存する。
3. **LLM 段 (強)**: `Qwen/Qwen2.5-7B-Instruct`、温度0、保守的プロンプト
   (「typo のみ修正・他は一切変更禁止」)。出力は `<corrected>...</corrected>` から
   抽出し、パース失敗時はフォーマット再指示を付けて1回リトライ
   (greedy では同一プロンプト→同一出力のためプロンプトを変える)。
   2回失敗時は原文を返し `parse_failed` を記録。

## スモーク検証結果 (2026-07-14)

### ① pyspell 段: rebuttal アーカイブとの出力一致 (CPU)

| 設定 | byte 一致 | 一致率 | 不一致の内訳 |
|---|---|---|---|
| gemma-3-4b-it × GSM8K (n=1319) | 1291 | 97.9% | 28件すべて同点/辞書差で説明可能 |
| gemma-3-4b-it × MMLU (n=2850) | 2772 | 97.3% | 78件すべて説明可能 (語単位: 同点76・辞書差6) |

不一致はすべて pyspellchecker 上流の非決定性 (頻度同点候補の PYTHONHASHSEED
依存選択; 例: 'mph' の候補 eph/kph/ph は全て頻度50) と rebuttal 実行時の
旧辞書との頻度表差分に起因し、**ロジック差による不一致は0件**。
検証: seed 0-3 で correction('mph') が eph/ph/kph と変動することを確認。
rebuttal 生成時の正確な pyspellchecker バージョンは特定不能
(archive .venv は現在 0.9.0 だが生成後に更新された形跡; 0.8.4 が最も近い挙動)。

### ①' pyspell 段: 復元統計の再現 (CPU、GSM8K 全 1319 件)

`make_corrected_dataset.py --corrector pyspell` の restoration_stats が
rebuttal のアーカイブ値を再現: word_restored 3135/5195 (60.3%; archive 3134
=同点1語差)、**fully_restored 185/1319 (14.0%) は完全一致** (論文の
byte-identical n=185 に対応)。collateral_changes のみ 395 vs 900 と異なるが、
これは archive 側 make_spellfix_dataset.py の変更ログベース集計が
regex語 vs 空白トークンの不整合で過大計上していたため。本実装は
analyze_spellfix.py と同じ位置ベース定義 (論文の誤修正率 15.6%/34.1% の出所)
に統一した。

### ② neural 段 / ③ LLM 段 (GPU, n=16, GSM8K, 評価=Gemma-3-4B-it)

`results/smoke/smoke_cycle_{neural,llm}.json` (校正→復元判定→ clean/校正後の
両方を同一ランで greedy 生成→ byte-identical 集合の flip 0% 検算):

| 指標 | ② neural (T5-large-spell) | ③ LLM (Qwen2.5-7B-Instruct) |
|---|---|---|
| 語復元率 | 0.871 | 0.710 |
| byte-identical 率 | 9/16 (0.563) | 5/16 (0.313) |
| 誤修正サンプル率 | 0.125 | 0.125 |
| LLM パース失敗 | — | 0 |
| accuracy clean / corrected | 0.75 / 0.75 | 0.75 / 0.75 |
| flip 率 (対 clean・同一ラン) | 0.063 (1/16) | 0.125 (2/16) |
| **byte-identical 集合の flip** | **0/9 (PASS)** | **0/5 (PASS)** |

n=16 の予備値であり傾向の解釈には使わない (スモークは配管の検証が目的)。
byte-identical → flip 0 は両段で成立 (greedy の理論通り)。

## 本番実行の手順 (参考)

1. 15 テキストジョブ: 各ベンチマーク×校正器で `make_corrected_dataset.py`
   (LXT-4 データセットは `configs/paths.yaml` の archive_perturbed_datasets から)
2. 73 評価ラン: `scripts/rebuttal/run_generation_only.py --perturbed_data <校正済み>`
   (AttnLRP 不要・バッチ推論可。pyspell×gemma-3-4b-it×{GSM8K,MMLU} の2ランは
   rebuttal 済みのため除外)
3. 集計: `analyze_correction.py` → R_Q 偏在はモデル別 Mann-Whitney
4. 修正A: LOO 重要度 (実験6-iv の副産物) での再集計は Step 0 / 実験6 の
   成果物待ち (token_rq_comparison は importance_score の出所に依存しないので
   LOO スコアを流し込むだけ)

## 本番実行 (2026-07-14 開始)

### 校正グリッドの粒度決定 (前段 open question への回答)

校正ジョブの粒度は **「ベンチ×校正器×摂動元モデル」= LXT-4 摂動データセット
ごと** (25×3 = 75 ジョブ) とする。ユーザー決定 (2026-07-14)。理由: LXT-4 の
標的語は摂動元モデルの AttnLRP 帰属に依存するため、同じベンチでもモデルごとに
摂動文が異なる。校正器は決定的でも入力が違えば出力が違う——「ベンチ×校正器」
の 15 ジョブに縮約すると、どのモデルの摂動文を校正したのかが混ざり、
モデル別の R_Q 偏在分析 (Mann-Whitney) や byte-identical 検算が成立しない。
これは科学的要請であり実装都合ではない。実測で全 25 データセットのサンプル数は
ベンチ内で全モデル一致 (gsm8k 1319 / mmlu 2850 / mmlu_pro 1400 / arc 1172 /
commonsense_qa 1221; 計 7,962/モデル)。

### rebuttal 2設定の統一再生成 (pyspellchecker==0.9.0 + 決定的タイブレーク)

ユーザー決定 (2026-07-14): rebuttal 掲載の pyspell×gemma-3-4b-it×{GSM8K,MMLU}
は決定的実装で**再生成し、評価も再実行**する (掲載値と訂正語の 2〜3% が変わる
ことは承認済み。revision notes に「手続き統一のため再計算」と明記する前提)。
検証レポート: `results/prod/exp7/rebuttal_regen_diff_{gsm8k,mmlu}.json`
(scripts/exp7/compare_pyspell_regen.py)。

| 指標 | GSM8K | MMLU |
|---|---|---|
| 2回実行の byte 一致 (再現性) | PASS | PASS |
| 校正後テキストが旧 rebuttal と異なるサンプル | 28/1319 (2.1%) | 78/2850 (2.7%) |
| 摂動語位置の訂正語変化率 | 14/5195 (0.27%) | 37/10565 (0.35%) |
| fully_restored (byte-identical) 新/旧 | 185/185 (14.03%, 不変) | 584/585 (20.49%/20.53%, −1) |
| 語復元率 新/旧 | 60.35% / 60.33% (+1語) | 71.69% / 71.69% (同数) |

差分はいずれも旧実装の非決定的タイブレーク (PYTHONHASHSEED 依存) と rebuttal
実行時の旧辞書差に由来 (スモーク①の parity 検証と同じ 28/78 サンプル)。
訂正語の変化はユーザー承認済みの 2〜3% の範囲内。

### 本番の実行体制 (シャードキュー)

- CPU: `scripts/exp7/prod/run_pyspell_grid.sh` — 25 設定を nohup ループで処理
  (冪等: restoration_stats.json があればスキップ)。
- GPU (起動は exp4→5→7 の実行順により **exp4/exp5 のキューが掃けた後**):
  - `scripts/exp7/prod/run_correction_queue.sh neural|llm` — 校正段。
  - `scripts/exp7/prod/run_eval_generation_queue.sh` — 評価生成 75 ラン
    (rebuttal 2設定も統一再実行に含める)。
  - 1 シャード = 1 回の `run_with_gpu.sh` 呼び出し (GPU 3/4, flock 排他,
    GPU_LOCK_TIMEOUT=86400)。シャード出力→ merge スクリプトで非シャード実行と
    同一スキーマに結合 (shard==whole の同値を ARC 7件で検証済み)。
- 想定シャード数と GPU 時間 (概算; 実測で更新):
  - neural 段: シャード 1500 件 → 30 シャード。T5-large ~0.5-1.5s/サンプル →
    約 8〜17 GPU 時間。
  - LLM 段: シャード 500 件 → 90 シャード。Qwen2.5-7B greedy ~2-4s/サンプル →
    約 22〜44 GPU 時間。
  - 評価生成: シャード 800 件 → 180 シャード (75 ラン)。batch 4, 512 tokens,
    ~1-2s/サンプル (1B〜7B 混合) → 約 35〜80 GPU 時間。
  - 合計 ~65〜140 GPU 時間 (GPU 3/4 の 2 枚を実験10 と共有)。

## 未決事項 (open questions)

- LLM 校正の「同一モデル版は参考掲載」(評価モデル自身で校正) の実施範囲。
- neural 段の縮退弁 (GSM8K/MMLU のみ) を引くかどうか。
