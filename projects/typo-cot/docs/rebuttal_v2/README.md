# Rebuttal v2：監査・限定対照実験

Status: protocol revision 2.0.0; PR-01 intake, PR-02 answer audit, and PR-03 input audit / two-stage annotation implemented with CPU acceptance fixtures; planning/generation/reduction not implemented; real-model smoke not run; formal experiments not run.

本書はPR-00で固定した科学的契約である。フィールド・入出力参照・状態遷移は [schemas.md](schemas.md)、既知の結果と判断記録は [decision_log.md](decision_log.md)、機械可読設定は [protocol.json](../../configs/rebuttal_v2/protocol.json) を参照する。PR-01の [intake](intake.md)、PR-02の [answer-audit](answer_audit.md)、PR-03の [input-audit・二段階注釈](input_audit.md) を実装した。ここに示す計画・生成・集計CLIは未実装であり、機能ごとのPRで追加する。

## 1. 目的と範囲

3日間のRebuttalで回答する問いは、①回答抽出が報告値を変えていないか、②実編集箇所と意味保存を確認した例でも回復と対照差が残るか、③同じ実行環境で対照差が成立し、別の一設定でも確認できるか、である。

January計画の語同定L／対象・属性への結び付けBの識別、学習、全層探索、自然誤字benchmark新設、content-aware prefix圧縮は含めない。これらは `ARR_January_2027_lexical_binding_plan.md` の後続研究として扱う。Rebuttalでは書込座標で回復できることを測り、語情報やbindingの機構を既に同定したとは書かない。

2026-09-16に確認した[ARR author response規定](https://aclrollingreview.org/authors#author-response)では、査読への直接回答となる小規模な追加対照は許されるが、大幅な新研究は対象外で、回答はテキストのみ・外部リンクなし。本READMEとPR資料は開発用であり、そのまま回答へリンクするための文書ではない。投稿時は当該cycleの案内も確認する。

## 2. 確認した出典と、まだない入力

- Repository: https://github.com/Maxel31/typo_robust_analysis/tree/main/projects/typo-cot
- GitHub mainとローカル読取snapshotの一致を確認したcommit: `f3c91f8e662a2b60cec30cc84ca62ba715d5469a`。
- 設計原稿に記載されたPDF SHA-256（`experiments/catalog.py` の値と一致。PR-00で論文ファイルを新たに取得・再検証したものではない）: `2cfb736e4636ee8db8dc6a92a6004c6e36914538a9acadcd66073289580a39d0`。
- 設計原稿はユーザー提示の3レビューに基づく。PR-00では設計原稿を入力とし、レビュー原文やrevisions PDFを新たに取得したとは扱わない。
- 公開repoの `results/` に実験結果一式は含まれない。原172例等のexact ID、当時のclean/typo全文、全arm生成、model revisionを本作業では取得できていない。以下はこれらを受け入れる仕様で、数値再現やGPU動作確認済みという意味ではない。

開始時に必要な入力は `archive_index.json` に明示する。所在が未確定の値はnullにして不足一覧を出せるが、それを既知のprovenanceとして扱わない。

| 入力 | 必要な内容 | 不足時の扱い |
|---|---|---|
| 提出時pair一覧 | task、model、元問題ID、target rule、exact clean/typo text、pair identity | 172件を新しく選び直して原cohortとは呼ばない |
| 提出時生成 | clean/typo/correct/offset/crossのraw text、可能ならtoken IDs・停止理由 | 文字列がなければparser再評価不可。既存集計だけから補完しない |
| モデル・tokenizer | model IDとimmutable revision、tokenizer revision、prompt template | 不明なら当時との完全比較を主張しない。新revision実験は別provenance |
| prompt・編集情報 | few-shotを含む全文、意図token、実編集span、実際に書いた位置 | 不明なtarget情報はunknown。実spanから独立に再構成可能な部分を監査 |
| RQ2原記録 | 完全prefix、削除span、質問、各armの生成cap | optional R5を行う場合だけ必須 |
| donor bank | recipientの元問題と重複しない、同settingのclean入力と編集対応 | なければcrossの適格性と推論範囲を限定。都合のよい循環割当へ黙って変更しない |

## 3. 既存実装の再利用と変更理由

パスはすべて `projects/typo-cot/` 相対。READMEのimplemented表示や既存test通過を、論文の数値の正しさの証拠とは扱わない。

| 既存箇所 | 確認した挙動 | v2の扱い |
|---|---|---|
| `experiments/build_rebuttal_manifest/protocol.py`, `runner.py` | 各cellの成功数と全体1,241/800一致を要求。全6設定を前提 | 旧アーカイブ照合用として保持。新観測用manifestは別実装で、成功数一致を要求しない |
| `evaluation/extractor.py`, `fallback.py` | cap時のpositional fallback無効化が一次抽出器の行末数値等には効かない。選択肢の語境界にも問題 | 旧parserをversion付きで保持し、新parserを独立実装。全armを両方で採点 |
| `experiments/targeting_fidelity_audit/runner.py` | 現行producerの編集を再生し、tokenizer再読込なしで構造を監査 | 提出時exact textの監査とは区別。pinned tokenizerによる独立再構成と人手意味監査を追加 |
| `experiments/patch_coordinate_controls/` | GSM8Kと[0,6)固定。過去correctを参照。baseline replay/self-copyの保護あり | protected部分を参考に、fresh全armを一体で測るrunnerを新設 |
| `experiments/six_setting_patch_controls/` | 過去correctを借用し新offset/crossを生成。全6設定契約 | 新しい単一setting／少数settingのrunへ直接流用しない |
| `experiments/fixed_window_answer_patching/` | actual offsets、word-final、prefill block-output patchの部品 | independent no-opと実model smokeを通った部品を再利用 |
| `experiments/build_rebuttal_manifest/planning.py` | 全座標が有効なstrict offset計画 | 独立span検証の後に利用。旧coordinate plannerのdrop方式と混同しない |
| `experiments/patch_harm_audit/` | 保存されたtypo正解例に新patchのみ生成 | harmを測る場合もfresh baselineを必要とする。今回は本実験のcohort遷移として記録 |
| `experiments/answer_line_deletion/`, 同docs | 最終非空行削除、空prefix、16/256 token protocol差の記述 | その差の実データ上の確認を先に行い、旧数値の意味を断定しない |

設計原稿が挙げる既存抽出の反例と、PR-02で検証する受入例（実cohortでの影響件数は未確認）：

| 入力 | 既存抽出 | v2で要求する挙動 |
|---|---|---|
| `We compute 2 + 2 = 4\nStill calculating` | `allow_positional=False`でも4 | 明示最終回答なし |
| `Intermediate = 4\nFinal calculation = 9` | 最初の行末の4 | 明示最終回答なし／人手監査対象。都合で9を選ばない |
| MMLU: `The answer is definitely B.` | definitely先頭のD | Dを返さない。固定した構文外ならunextractable |
| MMLU-Pro: `The answer is incorrect; continue reasoning.` | incorrect先頭のI | Iを返さない |

これらは実験cohortの影響件数を示さない。R1で保存生成に適用して測る。

## 4. 実験一覧と優先順位

| ID | 優先度 | 問い | 対象 | 新しいGPU計算 |
|---|---|---|---|---|
| R0 | 必須 | 元データ・仕様・分母を追跡できるか | 利用可能な全archive、主172例を優先 | 不要 |
| R1 | 必須 | 回答抽出が成功数・cohortを変えるか | 保存された全arm生成。172例を最優先、可能なら1,241例 | 不要 |
| R2 | 必須 | 実編集・patch位置・意味保存を確認した例でも効果が残るか | 主172例全件。第二settingはR4実施時に追加 | tokenizer CPU＋人手。既存生成再採点のみならGPU不要 |
| R3 | 主GPU実験 | 同一実行条件でcorrectと対照に差があるか | Gemma-3-4B-IT／GSM8K、原172例 | 6条件／pair、最大1,032生成 |
| R4 | 次順位 | early優位がなかった第二設定でも座標・donor対照が効くか | Qwen2.5-3B-Instruct／MMLU-Pro、原97例 | 10条件／pair、最大970生成 |
| R5 | 余力時のみ | answer-line削除の低下に形式・生成予算が寄与するか | Gemma／GSM8K原CoT-swap cohortから固定32例まで | 4条件×2cap、最大256生成 |

R3はデータとsmoke検証の合格後に実施する。R4/R5はR0〜R3を圧迫しない。R4の入力がないから肯定的な別settingへ置換することはしない。別案を採る場合は、その新結果を見る前にv2.1として理由を記録する。

R4でQwenを選ぶ理由は、論文の[0,6)優位が支持されなかった条件を直接確認するため。元6設定poolの一つを増やす実験ではなく、同じtaskでmodelだけを変える比較でもないことを明記する。

論文への対応は、R0/R1が§3.2（p4）・Appendix A generation/extraction（p11）・Table3（p12）、R2が§3.1（p3–4）・Appendix A targeting/gold-option（p11–12）、R3/R4が§3.3（p4）・§4.1（p6）・Tables6–7（p15）、R5が§3.4（p5）・Table1（p7）である。Appendix Bのgold-intact/target-faithful解析（p13）はKL分布のsubsetであり、主回答回復cohortの意味保存監査を代替しない。

## 5. R0：アーカイブ・分母・仕様監査

### 目的

新しい実装・parser・環境の結果を、過去の結果と混ぜていないかを確かめる。投稿時と現行public producerの違いを追跡する。

### 手順

1. 入力ファイルのSHA、schema、元commit、prompt、model/tokenizer revision、generation設定を収集。
2. frozen pair IDとexact textを固定。ID衝突、同一IDの異なるtext、重複元問題を検出。
3. 提出時集計と保存出力からの集計を別列で比較。差異を報告し、値を合わせるための再選別をしない。
4. `submitted-archive`, `public-regeneration`, `fresh-audit` を `source_kind` で区別。
5. provenance欠落を一覧化。hashがあることだけで、当時の意味・抽出・位置の妥当性が確認されたとはしない。

原報告の参照値は、Gemma/GSM8K correct129/172、offset44/172、cross42/172。Table6全体は800/1,241。これらは比較対象のreference metadataであり、新観測の受入条件ではない。

出力：`source_audit.json`, `archive_inventory.jsonl`, `pair_manifest.jsonl`, `pair_manifest.meta.json`, `archive_generation_records.jsonl`, `historical_count_comparison.csv`, `missing_inputs.csv`, `run.json`。manifestのmetadataは0行の場合もprotocolと入力監査への参照を保持する。

完了条件：取得した記録から出せる表と出せない表が明確で、各値の出典が追える。原ID不明ならexact replicationは未完として残す。新たなpublic cohortを作る場合は別研究対象として明示し、原172例を再現したとは呼ばない。

## 6. R1：回答抽出の感度分析

### 目的

途中計算、選択肢の部分一致、停止理由やfallbackが、誤答・回復の判定を変えていないかを調べる。

### Parser契約

- `archive_parser`: 当時のコードが特定できる場合だけarchive版と呼ぶ。不明なら `public_v1_proxy` と記録。
- `explicit_answer_v2`: goldを入力に使わず、固定した明示回答構文と完全なtoken境界で抽出する。
- 全arm・clean/typo baselineへ同じparserを適用。新promptやanswer-only生成へ変更しない。
- 選択肢はtaskの有効label集合内の独立文字だけ。英単語の頭文字を採用しない。
- 数値は符号・小数・桁区切り・分数・指数表記の完全な構文を固定して正規化。float誤差を避け、Decimal/Fraction等の正確な比較を使う。単位を含む未対応構文は先頭数値だけ採用せず、unextractableとして人手監査へ回す。
- 明示markerは `The answer is ...`, `Answer: ...`, `Final answer: ...`, 妥当な `boxed{...}` を対象とする。marker・句読点・単位の許容形をfixtureで固定し、受入後に成功数を見て増やさない。
- 複数の明示回答が同じ値なら採用。異なる値がある場合は保守的に `ambiguous` とし、人手監査へ回す。引用・否定等の構文上疑義も別statusで保持する。
- cap到達それ自体を自動的な誤答にはしない。完結した明示回答があれば通常の抽出規則を適用し、cap flagを併記。明示回答のない途中数値・last_letterは採用しない。
- EOSが512番目に出た例はEOS終了。長さ512だけからcapとは推定しない。停止情報がないarchiveは `termination=unknown`。

v2の機械採点は保守的に、最終非空行にある明示markerと回答全体をfull matchする。末尾の一つの終止符と空白以外に未対応の文字があれば部分一致させない。選択肢は単独labelまたは丸括弧付きlabel。数値は符号付き整数／桁区切り整数／小数／有限な指数表記／整数同士の分数で、分母0を拒否する。boxedは対応する括弧を解析し、中身全体に同じ規則を適用する。引用符やMarkdown引用の内側、途中行のmarkerは最終回答として採用しない。複数の異なる明示回答がある場合は先述のambiguous規則を優先する。

PR-00に固定する追加fixture: `Answer: 4 or 5`はambiguous、`Answer: 4/5`は4/5、`Answer: 1e3`は1000、`Answer: 4 followed by more reasoning`はunextractable、引用された`"Answer: 4"`はunextractable。`Answer: 4 dollars`はこのv2では未対応としてunextractable。表には「固定抽出規則下の回答成功」と表示し、parser未対応をモデルの明確な誤答と同一視しない。人手監査による回答判定を別列で示す。

coverageが低くてもprimary parserや分母を差し替えず、人手判定を結果後にprimary列へ昇格しない。各比較の同じ固定集合上でarm別の抽出成功・unextractable・ambiguous件数を必ず併記する。paired比較だけでcoverage差が相殺されるとは仮定しない。完全な独立人手採点がない場合、parserに依存しない意味的正答率・介入効果は `inconclusive` とし、機械表の主張を固定抽出規則下の回答成功に限定する。人手標本の結果はその標本の補助解析であり、未監査例を補完しない。構文を広げる場合は新revisionとし、元規則の結果も保持する。

### 評価

固定した元cohort上で、parserごとの抽出回答、正否、回復数、差、unextractable/ambiguous率、EOS/cap/unknown率を報告する。clean/typo判定の変更によるcohort遷移も別表にする。

全parser不一致・疑義例を、arm名・model名・goldを隠した生成文のみで人手確認し、回答の有無と内容を判定する。その後にgold比較を行う。全件が時間内に難しい場合は、生成結果の正否で選ばず不一致種別内のhash順で固定数を抽出し、未監査件数とsamplingを明記する。両parser一致例も固定した小標本を監査し、両者共通の誤りを見逃していないか確認する。

主要出力：`answer_audit_records.jsonl`, `parser_transition_table.csv`, `baseline_transition_table.csv`, `parser_disagreement_blind.jsonl`, `parser_audit_summary.json`。

結果の読み方：差が小さければこの感度分析の範囲で抽出による影響が限定的。差が大きければ訂正値と影響範囲を提示する。新parserも人間の意味判断の完全なoracleとは扱わない。

## 7. R2：実編集・座標・意味保存の監査

### 目的

原論文の30.2% target-token missと、patch座標の誤りを区別し、意味が変わった例が主回復率を支えていないかを調べる。21.5% gold-option edit率を別cohortへ転用しない。

### 自動監査

- exact clean/typo textの文字差分を計算。複数の同一語があってもsubstring検索で位置を決めない。
- intended token hit、actual changed word、patch write endpointを別列にする。
- frozen tokenizerとexact full promptからtoken offsetを独立再構成し、保存されたclean/typo座標を照合。
- Unicode、先頭空白、複数token語、few-shot中の同語、編集によるtoken数増減を検証。
- 元の最大4編集／targeting方式を保つ。Januaryの一箇所編集へ作り直さない。
- 問題文、選択肢内容、選択肢label、数値、数量表現、単位、否定、対象名への変更をflag化。
- risk flagは意味保存の自動正解ではない。target-token missも、それだけで主解析から除外しない。

### 人手監査

主172例は全件を対象とし、patch結果を見て注釈対象を選ばない。

Stage A: 訂正元、意図語、model、出力、armを隠し、typo質問と必要な文脈から解釈候補・曖昧性を判定。

Stage B: clean原文を開示し、課題・対象・属性・goldの意味が保存されているか判定。ラベルは `unique_preserved`, `ambiguous`, `task_changed`, `unassessable`。

2名独立判定と裁定を基本とする。担当者を確保できない場合は単独監査と明記し、二名判定と称さない。R4へ進む場合はその固定cohortも同じ規則で監査する。選択肢編集は自動的にtask_changedとせず、選択肢無変更の感度分析を追加する。

### 解析集合

- `C_archive`: 保存された原cohort。意味・新parserで再選別せず旧結果との比較用に保持。
- `C_semantic`: unique_preservedが確認された集合。
- `C_aligned`: 独立に実編集語とtoken endpointを対応できた集合。
- `C_correct`: C_semantic ∩ C_aligned ∩ correctの事前実行条件が有効。
- `C_offset`: C_correct ∩ offsetの事前座標規則が有効。
- `C_cross`: C_correct ∩ crossの事前donor規則が有効。
- `C_three`: C_offset ∩ C_cross。三条件を同一分母で並べる表に用いる。
- `C_fresh_failure`: fresh baselineでclean正解かつtypo不正解の集合。生成後に定まる条件付き解析として、各適格集合全体とは別に示す。

以後の主要paired比較はcorrect−offsetをC_offset、correct−crossをC_crossで測る。両比較の分母が異なり得るのでnを併記し、三armの率を直接比較する表はC_threeへ揃える。donor不足時でもself/offset比較を失わない。旧文書のC_common相当はC_threeとする。

R2の低コスト主成果は、保存された全armを同じparserで採点し直し、同一の意味保存集合上でcorrect/offset/crossの差を再集計すること。保存offsetが新strict規則を満たさない例は、旧定義の記述値と新適格集合を分ける。新規則で実行した結果へ読み替えない。

出力：`input_audit_records.jsonl`, `annotation_stage_a.jsonl`, `annotation_stage_b.jsonl`, `semantic_labels.jsonl`, `annotation_agreement.json`, `cohort_flow.csv`, `semantic_strata_table.csv`。

## 8. R3：主172例の同一runによる対照確認

### 確かめること

正しいclean stateを実編集語に書く操作の優位が、過去correctと新controlの実行差やparser差によるものではないかを調べる。新strict offsetと独立donorを使う場合は、歴史的129/44/42の完全再現ではなく、その制限を改めた追加対照と明記する。

### 固定設定

| 項目 | 設定 |
|---|---|
| Model/task | `google/gemma-3-4b-it` / GSM8K |
| 対象 | 提出時原172pair。ID・全文が回収できた集合を一覧化し、欠落を補充しない |
| Prompt | 保存されたexact promptと8-shot。新しいsystem/answer-only指示を追加しない |
| Generation | greedy、BF16、非量子化、left padding、max_new_tokens=512、do_sample=false、num_beams=1 |
| Model実行 | revision・tokenizer・backend・library・deviceを固定。開始時はbatch size 1 |
| Window | decoder block index 0,1,2,3,4,5。半開区間[0,6) |
| Patch位置 | 各actual aligned edited wordの最終token。全block出力のresidual state |
| Patch時点 | prefillのみ。後続decodeに繰り返し書き込まない |
| 主要評価 | 最終回答正解と同一cohortでのpaired差 |

### Arm

1. clean：fresh clean baseline。
2. typo：fresh typo baseline。
3. self：typo stateを同じtypo位置へ戻す。
4. correct：same-item clean stateを対応するtypo実編集語endpointへ書く。
5. offset：clean sourceとtypo writeの両方を対応endpointから+2tokenへ移す。
6. cross：別の原問題のclean stateをrecipientの元の編集語endpointへ書く。

fresh結果ではclean/typo/correct/controlを同じruntime、同じ生成設定で生成する。archiveのcorrectを新controlと主要比較に混ぜない。

### Offset

両側の全offsetがprompt内部（最初・最後のprompt tokenを除く）、全編集spanの外、位置重複なしであることを要求。1つでも不適格ならoffset arm全体をinvalidとし、無効な位置だけ落とさない。correctと同じ介入位置数を保つ。旧offsetがsourceとwriteを同時に動かすため、source内容とwrite位置の効果を単独には分解できないことも明記する。

### Cross donor

同じmodel/task、target rule、aligned edited word数の固定donor bankを用い、recipientの元問題とは分離する。clean stateの配列順とrecipientの対応順は各prompt内の編集語順で固定。gold labelやpatch成否をdonor選択に使わない。

donor bankと各割当を全run前に保存する。revision 2.0.0は一対一・非復元割当とし、donorを再使用しない。bank不足・word数不一致はcross invalidとして残す。推論は固定bank条件付きであり、独立donorへの一般化はしない。再使用を許す変更には新protocol revisionが必要。旧循環donorをそのまま使う記述的な歴史比較は可能だが、iid pairを前提とする強い有意性の根拠に追加しない。

`--donor-bank`はoptional。未提供時はcrossをnot_availableとして計画に残し、C_cross/C_threeの表をNAにする。correct−offsetとself-copyは継続できるが、matching donor特異性を新しく確認したとは書かない。

### 実行前の合格条件

- 小さい独立参照計算でhookの位置・write値・回数・cleanupが正しい。
- 各model 4〜8例のsmokeで、非介入とself-copyの生成token列が一致。logit差の数値許容条件は同backendの反復を使って結果生成前に記録する。
- source captureとgeneration prefillのbackend/KV設定差を検査。
- 各層1回だけ適用され、decodeには適用されない。
- batch/padding差、EOS-at-cap、token数変化の対応を検査。

self-copy不一致を科学的harmと数えない。原因を調べ、該当runtimeの結果公開を止める。差異を無視できるよう許容幅を結果後に広げない。

## 9. R4：Qwen／MMLU-Proの限定追加対照

この追加実験はR3完了後の次順位。原論文Table7のQwen n97では[0,6)52.6%、[6,12)55.7%と報告され、early優位は示されていない。これを前提に、同じwindow内でcorrectが対照を上回るかを確認する。

- Model: `Qwen/Qwen2.5-3B-Instruct`。task MMLU-Pro、exact原prompt、5-shot。
- 原97pairを対象。全ID・全文・prompt等が回収できることが開始条件。
- GenerationはR3と同じ512token。選択肢は各itemの実際の有効label集合を使う。
- Windowは[0,6)と[6,12)の二つだけ。新たな最良層探索は行わない。
- clean/typo baselineを共通に、各windowのself/correct/offset/crossを生成。計10条件。
- 監査、donor bank、strict offset、parser、common-validはR3と同じ規則。
- 時間不足なら実施しない。Nを効果量やp値で増減しない。事前のthroughputで縮小する場合は、生成前にhash順のsubset IDと数を別protocol revisionへ固定し、元97例全体の結果とは呼ばない。

評価はwindowごとのcorrect−offset、correct−cross、および同一pairのcorrect([0,6))−correct([6,12))。後者でearlyが良くなくても、前者の座標特異性は成立し得る。どちらも早期層の唯一性や残り計算量の交絡解消を示さない。原800/1,241の全6設定へ対照を追加したとは書かない。

## 10. R5：RQ2の形式・生成予算対照（optional）

本節は後続PR-07の設計要件であり、revision 2.0.0の選択可能な実験ではない。必須成果後に専用optional protocol revisionでmodel/runtime、cap別generation設定、全arm、選択IDとhash規則を固定するまで、`plan --experiments R5` は拒否する。

まずCPUで、削除した行が実際に回答行か、prefixが空になったか、原生成capが16/256等のどれかを監査する。既存docsの歴史的差を、raw記録未確認で実測済みとはしない。

新GPU実験はGemma/GSM8Kの原CoT-swap error cohortから、full-text成功に条件付けずhash順で最大32例を固定。最終回答行が特定でき、削除後も非空prefixが残る集合を主内容比較とし、空になった例の割合は元集合で別途報告する。

四条件はfull、deleted、full＋共通cue、deleted＋共通cue。Cueは全例同一の答えを含まない `Continue from the supplied text and give your final answer.` とし、質問やgoldは追加しない。双方へ同じ区切りとcueを付ける。

生成cap16を論文記載条件、512を予算感度分析として四条件とも実施する。これにより削除条件だけ長く生成して有利にする比較を避ける。最大32×4×2=256生成。時間的に両cap・全armを揃えられなければGPU対照は見送り、既存のtruncation confoundを認める。

評価：capごとのfull−deleted、cueによる変化、EOS/cap/unextractable率、削除後prefix長、明示回答の有無。cueで回復すれば形式・継続条件の寄与と整合的。cue後も差が残っても、当該cueだけでは説明できないという範囲で、reasoningそのものの効果や全answer leakage除去とは解釈しない。

## 11. 指標と統計契約

### 分母を混ぜない

行動評価は固定cohort全体、介入比較は事前の座標・意味監査適格集合、回復率はfresh clean-correct/typo-wrongへの条件付き値、と区別する。実行成功後に不適切にgood casesだけ選ばない。主要correct−offsetはC_offset全体、correct−crossはC_cross全体の回答成功率差。回復率と回復率差は各集合∩C_fresh_failureでの二次解析とする。

Primary parser v2で確定したC_fresh_failureのIDを固定し、その同じID上で旧新parserの採点を比較する。parser自身でfailure集合を作り直した表も補助として出すが、その集合遷移による差を純粋な採点差と呼ばない。主適格集合C_offset/C_cross/C_threeはparserによって変更しない。

モデルが生成したunextractable/ambiguousは、固定parserでの回答不成功として分母に残す。一方、OOM、source欠落、hook違反は実行statusでありモデル誤答へ変換しない。未完了runはpartialとし、主要表にcompleteと表示しない。

### 主要指標

各setting・window・parserについて以下を出す。

1. n_archive、n_text_available、n_semantic、n_alignment_valid、n_offset、n_cross、n_three、n_runtime_complete、n_fresh_failure。
2. 原cohortでのfresh clean/typo正答率とarchiveとの一致・不一致。
3. C_three全体での三arm回答成功率と、C_correct全体のcorrect−typoのpaired差。
4. C_three∩C_fresh_failureでの三arm回復率。各対照固有の分母による二次解析も別に示す。
5. C_offsetでのcorrect−offset、C_crossでのcorrect−crossのpaired risk differenceと95% CI。
6. self-copy token一致率、hook適用回数違反、logit差。
7. parser遷移、cap、unextractable、ambiguous、missingの件数。
8. R4では二window間のpaired差。first-CoT-token KLは任意の補助診断に留める。

成功数kと分母nを必ず併記。表示丸め前の値をJSONへ保存。比率の差はpercentage pointsで表示する。

元問題groupを単位に10,000回、seed42でpaired bootstrapし、同じ問題の別target rule・摂動・全armを同時に再抽出する。settingは固定した対象であり、settingそのものを再標本化してモデル母集団へ一般化しない。

主要結果は効果量とCI。補助的なexact McNemarは一問題一pair等の独立性が成立するときだけ出す。R3の二対照比較を一familyとしてHolm補正。cross未実施なら未実施と表示し、検定枠は二つのまま補正計算上のみ欠測pを1として扱う。R4/R5は事前に指定した追加診断としてCIと生のdiscordanceを中心に報告し、R3の検定familyに結果後追加しない。旧循環donor依存を無視したp値を再強調しない。

donorは固定bank条件付き。共通適格例が少ないときはCIが広いことを報告し、非有意を同等性と解釈しない。Nは原cohortと時間予算で決め、検出力を保証したとは書かない。

## 12. データ契約

新規packageの配置先：`src/typo_cot/experiments/rebuttal_v2/`。既存v1の意味を黙って変えない。旧parser・旧結果はread-onlyで保持する。

以下はartifactの役割の要約であり、完全なfield一覧ではない。field・null許可・identity・参照・状態の正規定義は [schemas.md](schemas.md) のみとし、実装・検証はその定義を使う。

### pair_manifest.jsonl

出典、pairと元問題groupのidentity、protocolとexact textのhashを保持する。完全なfield定義はschemas.md §4を参照。

内容：exact clean/typo text、full promptまたはその参照とhash、goldとcanonicalization版、保存されたintended/actual editとtoken位置、historical membership、入力availabilityの各statusと理由。独立に再構成した編集span・token位置・監査適格性は後段の `input_audit_records.jsonl` にmanifest参照とともに保存し、intake manifestへ書き戻さない。

nullはunknownとして許可する項目をschemaで列挙し、GPUに必要なunknownが残ればplan preflightで拒否する。`pair_id`だけで内容を同一とみなさずtext hashも検証する。

### annotation

配布用blind IDと非公開mappingを分離し、stage、判定者、解釈、label、根拠と裁定を記録する（schemas.md §6）。Stage A exportにclean、gold、model、answer、patch outcomeが混入しない検査を置く。原問題に実際に含まれる選択肢は隠さず、gold labelだけ隠す。

### plan.jsonl

各armの出典・座標・window・計算設定と適格性を生成前に固定する（schemas.md §7）。

正式GPU planは対象の元問題group identityを解決した後に作る。未解決なら不足一覧を出してpreflightで停止し、shard未割当のvalid jobを作らない。CPU監査にはそのrecordを残す。preflight通過後の実行適格性はarm単位で、clean/typoはtextとruntimeが利用可能な原cohort全例へwindow=nullで一回ずつ計画する。self/correctはalignment適格、offset/crossはさらに各規則適格な例へwindowごとに計画する。意味保存は解析ラベルとして保持し、意味監査不適格例をbaseline計画から消さない。invalid armはeligibility行として残して生成jobには含めず、reducerのexpected generation gridもvalid armだけから構成する。R4のbaselineは二windowで共有するため最大10生成となる。

### generation_records.jsonl

各jobの一つの成功生成と生の出力・停止情報・runtime integrityを保持し、再試行履歴は `attempt_records.jsonl` に分離する（schemas.md §8）。答え抽出は別の `score_records.jsonl` にgeneration参照・parser版・抽出status・`gold_match`を保存する。`archive_parse` / `audit_parse` はその結合viewであり、raw generationを上書きせず再採点する。出力やparserが欠ける場合は採点行を作らず `scoring_coverage.jsonl` に記録する。

### run.json

protocol/code/input/model/tokenizer/runtime/planのhash、起動引数、開始終了、complete/partial/failed、expected/completed/failed IDs、出力hash。観測した低い成功率をfailed条件にしない。科学的結果の不足と、実行契約違反を区別する。

GPU間で一致すべき `scientific_config_sha256`（model/tokenizer/dtype/backend/generation/code/device model等）と、worker固有の `worker_telemetry`（physical GPU番号、serial、host、時刻等）を分離する。同一GPU型の別physical番号は許容する。異機種や計算設定差は別runとして扱い、共通hashへ混ぜない。

## 13. CLIと実行順（PR実装後の契約）

既存の `typo-cot experiments` 等を改名せず、argparseに `rebuttal-v2` namespaceを追加する。CPU commandを読むだけでGPUモデルをimportしない。

以下はrepo rootからのコマンド契約。`SOURCE_ROOT` と `REBUTTAL_V2_ROOT` はユーザーの実データ位置へ設定する変数。`intake`、`answer-audit`、`input-audit`、`annotation-export`、`annotation-import` は実装済みであり、`plan`・`run`・`reduce` はまだ実行できない受入仕様である。

```bash
uv run --project projects/typo-cot typo-cot rebuttal-v2 intake \
  --archive-index "${SOURCE_ROOT}/archive_index.json" \
  --protocol projects/typo-cot/configs/rebuttal_v2/protocol.json \
  --output-dir "${REBUTTAL_V2_ROOT}/intake"

uv run --project projects/typo-cot typo-cot rebuttal-v2 answer-audit \
  --manifest "${REBUTTAL_V2_ROOT}/intake/pair_manifest.jsonl" \
  --output-dir "${REBUTTAL_V2_ROOT}/answer-audit"

uv run --project projects/typo-cot typo-cot rebuttal-v2 input-audit \
  --manifest "${REBUTTAL_V2_ROOT}/intake/pair_manifest.jsonl" \
  --tokenizer-lock "${SOURCE_ROOT}/tokenizer_lock.json" \
  --output-dir "${REBUTTAL_V2_ROOT}/input-audit"

uv run --project projects/typo-cot typo-cot rebuttal-v2 annotation-export \
  --audit "${REBUTTAL_V2_ROOT}/input-audit/input_audit_records.jsonl" \
  --stage a --output-dir "${REBUTTAL_V2_ROOT}/annotation-a"

uv run --project projects/typo-cot typo-cot rebuttal-v2 annotation-export \
  --audit "${REBUTTAL_V2_ROOT}/input-audit/input_audit_records.jsonl" \
  --stage b --stage-a-labels "${SOURCE_ROOT}/stage_a_labels.jsonl" \
  --annotation-batch "${REBUTTAL_V2_ROOT}/annotation-a/annotation_batch.json" \
  --output-dir "${REBUTTAL_V2_ROOT}/annotation-b"

uv run --project projects/typo-cot typo-cot rebuttal-v2 annotation-import \
  --audit "${REBUTTAL_V2_ROOT}/input-audit/input_audit_records.jsonl" \
  --labels "${SOURCE_ROOT}/adjudicated_labels.jsonl" \
  --annotation-batch "${REBUTTAL_V2_ROOT}/annotation-b/annotation_batch.json" \
  --output-dir "${REBUTTAL_V2_ROOT}/semantic-audit"

uv run --project projects/typo-cot typo-cot rebuttal-v2 plan \
  --manifest "${REBUTTAL_V2_ROOT}/intake/pair_manifest.jsonl" \
  --input-audit "${REBUTTAL_V2_ROOT}/input-audit/input_audit_records.jsonl" \
  --labels "${REBUTTAL_V2_ROOT}/semantic-audit/semantic_labels.jsonl" \
  --donor-bank "${SOURCE_ROOT}/donor_bank.jsonl" \
  --protocol projects/typo-cot/configs/rebuttal_v2/protocol.json \
  --runtime-lock "${SOURCE_ROOT}/runtime_lock.json" \
  --experiments R3 --output-dir "${REBUTTAL_V2_ROOT}/plan"

CUDA_VISIBLE_DEVICES=0 uv run --project projects/typo-cot --extra lrp \
  typo-cot rebuttal-v2 run \
  --plan "${REBUTTAL_V2_ROOT}/plan/plan.jsonl" \
  --shard-index 0 --num-shards 2 --output-dir "${REBUTTAL_V2_ROOT}/shard-0"

uv run --project projects/typo-cot typo-cot rebuttal-v2 reduce \
  --intake-run "${REBUTTAL_V2_ROOT}/intake/run.json" \
  --answer-audit-run "${REBUTTAL_V2_ROOT}/answer-audit/run.json" \
  --input-audit-run "${REBUTTAL_V2_ROOT}/input-audit/run.json" \
  --semantic-audit-run "${REBUTTAL_V2_ROOT}/semantic-audit/run.json" \
  --plan "${REBUTTAL_V2_ROOT}/plan/plan.jsonl" \
  --shard-root "${REBUTTAL_V2_ROOT}" \
  --output-dir "${REBUTTAL_V2_ROOT}/report"
```

CPU監査だけを報告する場合、`reduce` は `--intake-run` と利用可能な監査runを受け取り、`--plan` と `--shard-root` を省略する。T0〜T3の作成可能な表を出し、未実施の実験は `not_run`、途中まで実行された実験は `partial` とする。入力がないだけで未実施とは断定せず、実施状況不明はcoverageのunknownとして残す。GPU集計では `--plan` と `--shard-root` を両方指定する。run参照とplan metadataの入力hashが異なる場合は混合を拒否する。出力artifactの探索・参照解決は [schemas.md](schemas.md) に従う。

`--runtime-lock` は新実行のimmutable model/tokenizer revisionと計算設定を固定する入力であり、archiveの不明なrevisionを書き換えない。`--donor-bank` は省略できる。annotation-exportは配布用JSONLと非公開mappingを別artifactへ出力し、後者を注釈者へ配布しない。

`run --smoke` は独立smoke planを使い、正式集計へ混ぜない。`--resume` はpairの全armとinput/config/code hash一致を確認し、成功したarmだけ残して不都合なarmを再抽選しない。

複数GPUではglobal planを一度だけ作り、元問題group単位で全armを同じshardへ割り当てる。donorはshard外の固定bankから読み込める。各workerでcohort/donorを作り直さない。`--limit`を分散実行の代用品にしない。

## 14. protocol.jsonで固定する最小設定

```json
{
  "schema_version": "rebuttal-protocol/v2",
  "protocol_revision": "2.0.0",
  "analysis_kind": "reviewer-requested-post-hoc-audit",
  "seed": 42,
  "cohort_policy": "exact-archived-ids-no-outcome-count-acceptance",
  "primary_experiments": [
    "R0",
    "R1",
    "R2",
    "R3"
  ],
  "optional_experiments": [
    "R4"
  ],
  "settings": [
    {
      "id": "R3",
      "model": "google/gemma-3-4b-it",
      "task": "gsm8k",
      "historical_n_reference": 172,
      "windows": [
        [
          0,
          6
        ]
      ],
      "shots": 8
    },
    {
      "id": "R4",
      "model": "Qwen/Qwen2.5-3B-Instruct",
      "task": "mmlu-pro",
      "historical_n_reference": 97,
      "windows": [
        [
          0,
          6
        ],
        [
          6,
          12
        ]
      ],
      "shots": 5
    }
  ],
  "generation": {
    "dtype": "bfloat16",
    "quantization": false,
    "do_sample": false,
    "num_beams": 1,
    "max_new_tokens": 512,
    "padding_side": "left",
    "batch_size": 1
  },
  "arms": [
    "clean",
    "typo",
    "self",
    "correct",
    "offset",
    "cross"
  ],
  "patch": {
    "site": "complete-decoder-block-output",
    "phase": "prefill-only",
    "coordinates": "actual-edited-word-final",
    "offset_tokens": 2,
    "offset_policy": "all-valid-all-edited-spans-excluded",
    "donor_policy": "disjoint-original-problem-fixed-bank",
    "donor_assignment": "hash-ordered-one-to-one-without-replacement",
    "donor_bank_group_disjoint": true
  },
  "parser": {
    "primary": "explicit_answer_v2",
    "sensitivity": "archive_or_labeled_public_proxy",
    "conflicting_explicit_answers": "ambiguous"
  },
  "estimands": {
    "primary_offset": "C_offset-all-eligible",
    "primary_cross": "C_cross-all-eligible",
    "three_arm_table": "C_three",
    "secondary_recovery": "primary-parser-fixed-fresh-failure-ids",
    "missing_donor_bank": "cross-not-available-offset-retained"
  },
  "statistics": {
    "cluster": "original_problem_group_id",
    "bootstrap_replicates": 10000,
    "seed": 42,
    "confidence_level": 0.95,
    "primary_test_family": "R3-correct-vs-offset-and-cross",
    "setting_resampling": false,
    "bootstrap_interval": "percentile",
    "bootstrap_quantile_method": "linear",
    "primary_test_family_size": 2,
    "missing_primary_p": 1
  },
  "runtime_lock_required": [
    "model_revision",
    "tokenizer_revision",
    "prompt_sha256",
    "library_versions",
    "attention_backend",
    "device_identity",
    "effective_eos_ids"
  ]
}
```

`historical_n_reference` は期待する元cohortのreferenceで、成功数制約ではない。回収件数の不足はcoverageとして残し、原cohort完全再現とは表示しない。runtime lockの値は実hostで解決し、未解決のまま正式runを許可しない。JSONに書かれていないR5は別のoptional設定PRで全条件を固定してから実行する。

## 15. 結果表とRebuttalへの接続

必須表：

- T0 出典・欠落・provenance差分。
- T1 parser別の成功数、抽出不能、baseline/cohort遷移。
- T2 意図token／実編集／実patch位置／意味保存の別々の件数。
- T3 original→semantic→alignment→control-valid→runtime→fresh-failureの分母表。
- T4 R3のarm別率とpaired差。archive旧定義とfresh新定義を別panel。
- T5 R4を行った場合のみ、window内対照差とwindow間差。
- T6 R5を行った場合のみ、cue・cap・prefix条件別表。

機械出力 `claims.jsonl` は `claim`, `supporting_table`, `n`, `effect`, `ci`, `scope`, `limitation`, `status` を持つ。statusはsupported/not_supported/inconclusive/not_run。新しい成功率が予想より低くても生成を拒否しない。

レビューへの対応：R1意味保存・再現性→R0/R1/R2。R2位置・donor対照の範囲→R3/R4。R3選択条件・data-adaptive window→分母表と固定windowの限定。R3 RQ2形式交絡→CPU監査とoptional R5。新規性、自然誤字一般化、深さ交絡、RQ3圧縮はRebuttalで解決済みとせず、主張の限定とJanuary研究方針で回答する。

## 16. 三日間の実行順と停止条件

| 時間帯 | 開発・実験 | 完了基準 |
|---|---|---|
| Day1前半 | PR-00設計確定、R0入力照合、PR-01 intake | 原cohortと必要入力の所在・欠落を確定 |
| Day1後半 | PR-02 parser、PR-03 span/注釈、R1/R2開始 | CPU再集計と盲検監査が進む。既知の抽出反例を検出 |
| Day2前半 | PR-04 plan、PR-05 runner、smoke | no-op、座標、全arm再生成が正しい |
| Day2後半 | R3実行、R2裁定、PR-06集計。余力を事前計測してR4開始判断 | 主172例の証拠を優先。未解決の不具合があればR4へ進まない |
| Day3前半 | R3集計、可能ならR4完了、疑義例の監査 | paired表、欠測、parser差の説明が完成 |
| Day3後半 | Rebuttal本文へ数値・分母・限定を反映 | 外部リンクなしで読める短いテキスト表。未実施は未実施 |

開発者1名ならPRは順に実装し、並行開発を前提に3日を保証しない。注釈担当者が別にいる場合のみ人手監査を並行する。R4/R5よりR0〜R3の完成を優先。R3自体が期限内に完了しない場合は、R1/R2の保存出力の監査と訂正を主要成果として返す。

最大6GPUの利用はPR-05のshard検証後。例：R3へ2〜3台、R4へ2台、残りをsmoke/再実行用。1台あたりの実測速度とI/Oから見積もり、6台のVRAMを共有メモリと仮定しない。最大R3+R4=2,002生成は上限計画であり、実時間の保証ではない。

## 17. Januaryへ引き継ぐもの

exact text・編集span・盲検監査・gold非依存parser・全arm共通のidentity・欠測管理・prefill検証・immutable plan/reduceを引き継ぐ。Rebuttal特有の172/97というcohort、[0,6)というwindow、cross donorの意味、selected failure分母をJanuaryの語同定L/B研究へそのまま持ち込まない。新RQは別READMEで操作校正と独立testを定義する。

## 18. PR-00で固定する実装境界と受入例

### Parser文法と判定順

`explicit_answer_v2` の文法は次のとおり。goldはこの処理に渡さない。
UTF-8原文を保持し、行ごとの前後空白を除く以外にUnicode正規化や文の書換えを行わない。

- 行全体に対して `The answer is`（後続に一つ以上の空白）、`Answer:`、
  `Final answer:` をASCII大小文字を区別せず認識する。
  colonの後の空白は省略可能。`The final answer is`、Markdown強調、引用内のmarkerは未対応。
- 回答は有効label一文字（ASCII大小文字を同一視）、その丸括弧付き表現、
  またはtaskに対応する数値一個。数値taskに選択肢labelを返さない。
  markerのない単独数値・単独labelは最終行でも採用しない。
- 数値の数字はASCII `0-9`。任意の先頭符号、整数、整数部付き小数または `.5` 型小数、
  正しい3桁区切り（例 `1,250.5`）、`e/E` と符号付き整数指数を許容する。
  整数同士の分数は `-4/5` のようにslashの前後に空白なしとし、分母0を拒否する。
  指数・分数は正確な有理数へ正規化し、floatを介さず比較する。
  `NaN`、`Infinity`、百分率、通貨記号、単位、桁区切り不正は未対応。
- 対応するbraceを持つ `boxed{...}` と `\boxed{...}` は、単独のmarkerまたは
  上記markerの回答部として受け付ける。内部全体が一つの有効な回答でなければ不採用。
  dollarによる数式囲みや入れ子のLaTeX命令は未対応。
- 回答の後は空白と一つまでの終止符 `.` のみ許す。残りの文字列を切り捨てて
  先頭数値・先頭labelを採用しない。

引用領域内の候補は矛盾検出からも除く。行全体が引用符で囲まれた例、
Markdown blockquote、backtick/tildeによるfenced codeを受入fixtureに含める。
引用領域の境界を確定できない場合は疑義理由を残してunextractableとし、
領域内の候補を最終回答として採用しない。

まず全行から引用でない完全な明示回答を集め、正規化値の異なる候補が複数あれば
`ambiguous` とする。同値の `4/5` と `0.8` は矛盾ではない。
最終行の `Answer: 4 or 5` のように、` or ` で結んだ複数の有効な異なる候補も
`ambiguous` とする。これは曖昧性の検出専用であり、候補を一つ選ぶ規則ではない。
矛盾がなければ、最終非空行が上記文法に完全一致するときだけ `extracted`、
それ以外は `unextractable` とする。途中行のmarkerは矛盾検出には使えても、
最終行が推論途中であれば抽出成功の根拠にはしない。

引用・否定・構文外文字などの疑義は `reasons` に残す。
抽出statusは `extracted / unextractable / ambiguous`、
停止状態は `eos / length-cap / unknown` として独立させる。
raw生成が存在しない場合は抽出自体が未実施であり、unextractableなモデル出力を捏造しない。

| task / 入力 | 抽出status | 正規化回答 |
|---|---|---|
| GSM8K: `We compute 2 + 2 = 4\nStill calculating` | unextractable | null |
| GSM8K: `Answer: -1,250.5` | extracted | -2501/2 |
| GSM8K: `Answer: 4/5` | extracted | 4/5 |
| GSM8K: `Answer: 1e3` | extracted | 1000/1 |
| GSM8K: `Answer: 4/5\nFinal answer: 0.8.` | extracted | 4/5 |
| GSM8K: `Answer: 4\nFinal answer: 5` | ambiguous | null |
| GSM8K: `Answer: 4 or 5` | ambiguous | null |
| GSM8K: `Answer: 4 followed by more reasoning` | unextractable | null |
| GSM8K: `"Answer: 4"` / `> Answer: 4` | unextractable | null |
| GSM8K: `Answer: 4 dollars` / `Answer: 4/0` | unextractable | null |
| GSM8K: `\boxed{-4/5}` | extracted | -4/5 |
| MMLU: `The answer is definitely B.` | unextractable | null |
| MMLU-Pro: `The answer is incorrect; continue reasoning.` | unextractable | null |
| MMLU-Pro: `Answer: (J).`、有効label A〜J | extracted | J |
| MMLU: `Answer: J`、有効label A〜D | unextractable | null |

明示回答を含む同一textはEOSでもcapでも同じ抽出結果になる。
512番目のtokenがeffective EOSなら停止状態はeos。
token列も停止記録もないarchiveはtext長から停止状態を推測しない。
PR-02はこの表とgold不変性を独立fixtureで検証する。

### 座標・donor・統計の具体化

文字spanはexact text中のUnicode code pointの半開区間、token位置は
special tokenを含むexact full promptの0始まり位置とする。
正規化で文字位置を変えない。差分の曖昧性で編集語・endpointを一意に決められない場合は
理由付きで保持し、保存座標を使って独立監査の成功に置き換えない。

revision 2.0.0のcrossは、recipient全体と元問題groupが重複しないbankを使う。
同model/task/target rule/編集語数のstratum内で、seed42とidentityから得るhash順の
recipientとdonorを一対一で対応させ、donorを再使用しない。
不足分はcross invalid、bank未提供はcross not_available。
hash材料と同順位の規則は [schemas.md](schemas.md) に定義する。
再使用を許す後続案はprotocol revisionを変え、固定bank条件付きという推論範囲を残す。

bootstrapは元問題groupを復元抽出し、選ばれたgroup内の全pairと全armを同時に複製する。
各反復ではpair単位で重みを持つ成功率差を計算し、groupの平均値を等重み平均しない。
10,000反復の2.5%・97.5% percentileを線形補間して95% CIを求める。
点推定、成功数、分母、group数を併記し、空集合はNAとする。
R3のHolm familyは二比較に固定し、未実施crossのp=1は補正計算専用。
未実施比較の観測p値や効果量として1や0を表示しない。

### PRの進め方と状態

各機能のブランチは最新mainから `ARR2026_August_Rebuttal/NN-feature` として作成する。
PR-00〜06を順に実装・検証・PR作成し、全レビュー指摘を解消してmain反映を確認してから
次へ進む。PR-07は必須成果後のoptional。PRを積み重ねたブランチを次の起点にしない。
PR本文にはREADMEの受入条件、検証結果、比較可能性への影響、未実行項目を書く。

`implementation`、`cpu_fixtures`、`real_model_smoke`、`formal_results` は別々に記録する。
PR-00の完了は仕様の固定だけを表し、後続CLIの成功stubや実験カタログの
implemented登録は追加しない。文書の文言一致だけを検証する新規testは追加せず、
既存の公開設定一覧には新しいprotocol.jsonを登録する。
