# LXP-Perturbation 実装リファレンス

AttnLRPによる寄与トークン特定と文字レベル摂動を組み合わせた、LLMのCoT推論過程への影響分析フレームワークの詳細実装ドキュメント。

---

## 1. ディレクトリ構造

```
JSAI2026/
├── src/attn_perturbation/          # コアライブラリ
│   ├── __init__.py
│   ├── config.py                   # Pydantic設定管理
│   ├── models/
│   │   ├── __init__.py
│   │   ├── wrapper.py              # モデルラッパー（lxt統合）
│   │   └── prompts.py              # ベンチマーク別プロンプトテンプレート
│   ├── data/
│   │   ├── __init__.py
│   │   └── loader.py               # ベンチマーク別データローダー
│   ├── lrp/
│   │   ├── __init__.py
│   │   └── analyzer.py             # AttnLRP重要度分析
│   ├── perturbation/
│   │   ├── __init__.py
│   │   ├── generator.py            # 文字レベル摂動生成器
│   │   └── dataset.py              # 摂動データセット作成
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── extractor.py            # 回答抽出器
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── analyzer.py             # 比較分析（Phase 4メイン）
│   │   ├── metrics.py              # 統計メトリクス
│   │   └── appendix_analyzer.py    # Appendix分析（WordCloud/品詞/位置）
│   └── visualization/
│       ├── __init__.py
│       └── heatmap.py              # PDFヒートマップ生成
├── scripts/                         # CLIエントリポイント
│   ├── run_inference.py            # Phase 1 & 3: 推論 + AttnLRP
│   ├── run_perturbation.py         # Phase 2: 摂動データセット作成
│   ├── run_analysis.py             # Phase 4: 比較分析
│   ├── run_appendix_analysis.py    # Appendix分析
│   ├── generate_example_heatmaps.py # ヒートマップ生成
│   └── upload_to_wandb.py          # W&Bアップロード
├── tests/                           # テスト
│   ├── test_config.py
│   ├── test_data_loader.py
│   ├── test_extractor.py
│   ├── test_lrp_analyzer.py
│   ├── test_model_wrapper.py
│   ├── test_package.py
│   ├── test_perturbation.py
│   └── test_prompts.py
├── outputs/                         # 実験結果（実行時生成）
│   ├── baseline/                   #   Phase 1出力
│   │   └── {model}_{benchmark}/
│   │       ├── results.json        #     サンプル別結果
│   │       ├── summary.json        #     全体メトリクス
│   │       ├── config.json         #     実験設定
│   │       ├── importance_scores/  #     重要度スコア（.pt）
│   │       └── heatmaps/           #     PDFヒートマップ
│   ├── perturbed/                  #   Phase 3出力
│   │   └── {model}_{benchmark}_k{N}_{mode}/
│   └── analysis/                   #   Phase 4出力
│       └── {benchmark}/{model}/k{N}_{mode}/
│           ├── full_results.json
│           └── analysis_results.json
├── datasets/perturbed/              # Phase 2出力（摂動データセット）
├── pyproject.toml
└── README.md
```

### 出力ディレクトリの命名規則

| Phase | ディレクトリ名パターン | 例 |
|-------|----------------------|---|
| Phase 1 | `outputs/baseline/{model_short}_{benchmark}` | `gemma-3-4b-it_mmlu` |
| Phase 2 | `datasets/perturbed/{model_short}_{benchmark}_k{N}_{mode}` | `gemma-3-4b-it_mmlu_k4_importance` |
| Phase 3 | `outputs/perturbed/{model_short}_{benchmark}_k{N}_{mode}` | `gemma-3-4b-it_mmlu_k4_importance` |
| Phase 4 | `outputs/analysis/{benchmark}/{model_short}/k{N}_{mode}` | `mmlu/gemma-3-4b-it/k4_importance` |

`mode` は `importance`（LXP）/ `random` / `bottom_k`（Anti-LXP）のいずれか。

---

## 2. モデル・ベンチマークの読み込み

### 2.1 モデルの読み込み (`models/wrapper.py`)

#### 対応モデル

`ALLOWED_MODELS` リストで明示的に管理:

| モデルファミリー | 対応サイズ |
|----------------|-----------|
| Llama-3.2 | 1B, 3B |
| Llama-3.1 | 70B |
| Gemma-3 | 1B, 4B, 12B, 27B |
| Qwen2.5 | 32B |
| Mistral | 7B (v0.3) |

#### `ModelWrapper` クラスの構造

```python
class ModelWrapper:
    def __init__(self, model_name: str, gpu_id: str = "0", ...):
        self._model_name = model_name
        self._model = None        # 遅延ロード
        self._tokenizer = None    # 遅延ロード
        self._lxt_wrapped = False
```

**遅延ロード**: `model` / `tokenizer` はプロパティアクセス時に初回のみロードされる。

#### lxt統合（最重要ポイント）

```python
def wrap_for_lxt(self) -> None:
    """lxtのmonkey_patchをモデルモジュールに適用.

    重要: model生成前（_load_model()呼び出し前）に実行する必要がある。
    lxtはモデルの各レイヤーモジュールをパッチして、
    勾配ベースの寄与度計算を可能にする。
    """
    from lxt.utils import monkey_patch
    # モデル名からモデルタイプを判定
    model_type = self._detect_model_type()
    # モデルタイプ別のpatch_mapをインポート
    if model_type == "llama":
        from lxt.models.llama import get_patch_map
    elif model_type == "gemma":
        from lxt.models.gemma import get_patch_map
    # ...

    # transformersのモデルモジュールにパッチを適用
    import transformers
    model_module = getattr(transformers, model_class_name)
    monkey_patch(model_module, get_patch_map())
```

**制約**: `monkey_patch()` は **モデルの `from_pretrained()` 呼び出しよりも前** に実行しなければならない。パッチはtransformersのモジュール自体を書き換えるため、ロード後に適用しても効果がない。

#### テキスト生成

```python
def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.0) -> GenerationResult:
    """単一プロンプトの推論."""

def generate_batch(self, prompts: list[str], max_new_tokens: int = 512, ...) -> list[GenerationResult]:
    """バッチ推論（left-paddingを使用）."""
```

バッチ推論時は `tokenizer.padding_side = "left"` に切り替え、推論後に元に戻す。

#### GPU設定

```python
def setup_device(gpu_id: str = "0") -> tuple[torch.device, bool]:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    # gpu_idにカンマがあればマルチGPU
    use_multi_gpu = "," in gpu_id
    return device, use_multi_gpu
```

マルチGPU時は `device_map="auto"` でモデルをロード。

#### ファクトリ関数

```python
def create_model_wrapper(model_name: str, gpu_id: str = "0", wrap_for_lxt: bool = False) -> ModelWrapper:
    wrapper = ModelWrapper(model_name=model_name, gpu_id=gpu_id)
    if wrap_for_lxt:
        wrapper.wrap_for_lxt()  # モデルロード前にパッチ適用
    return wrapper
```

### 2.2 ベンチマークの読み込み (`data/loader.py`)

#### 対応ベンチマーク

| ベンチマーク | HuggingFaceデータセット | Split | 回答形式 | サンプリング |
|------------|----------------------|-------|---------|------------|
| MMLU | `cais/mmlu` | test | 4択 (A-D) | サブセット別（57科目）×N件 |
| MMLU-Pro | `TIGER-Lab/MMLU-Pro` | test | 10択 (A-J) | カテゴリ別（14）×N件 |
| GSM8K | `openai/gsm8k` | test | 数値 | 全件（1319件） |
| SQuAD v2 | `rajpurkar/squad_v2` | validation | テキスト | N件ランダム |
| ARC-Challenge | `allenai/ai2_arc` (ARC-Challenge) | test | 4択 (A-D) | N件ランダム |
| CommonsenseQA | `tau/commonsense_qa` | validation | 5択 (A-E) | N件ランダム |

#### `Sample` データクラス

```python
@dataclass
class Sample:
    sample_id: str
    question: str
    choices: list[str] | None = None
    correct_answer: str = ""
    context: str | None = None     # SQuAD v2用
    subset: str | None = None
    answer_start: int | None = None  # SQuAD v2用
    answer_end: int | None = None    # SQuAD v2用
```

#### ローダーの共通インターフェース

```python
class BaseBenchmarkLoader(ABC):
    @abstractmethod
    def load(self) -> list[Sample]: ...

    @abstractmethod
    def get_subsets(self) -> list[str]: ...
```

#### MMLU/MMLU-Proのサンプリング

- `samples_per_subset` パラメータでサブセット/カテゴリごとのサンプル数を制御
- シード固定でサブセット内からランダムサンプリング
- 例: MMLU 57科目 × 100件 = 最大5700件

#### CommonsenseQAの注意点

- **テストセットにラベルがない** ため、`validation` splitを使用
- 選択肢数は5択 (A-E)

### 2.3 プロンプトテンプレート (`models/prompts.py`)

#### 共通インターフェース

```python
class BasePromptTemplate(ABC):
    @abstractmethod
    def generate(self, question: str, choices: list[str] | None = None, ...) -> tuple[PromptResult, str]:
        """プロンプトを生成し、(PromptResult, フルプロンプト文字列)を返す."""
```

#### `PromptResult` — 文字位置の追跡

```python
@dataclass
class PromptResult:
    question_start_in_full: int      # 質問文の開始文字位置（few-shot除外）
    question_end_in_full: int        # 質問文の終了文字位置
    question_with_choices_end: int   # 質問文+選択肢の終了文字位置
    context_start_in_full: int | None = None  # SQuAD用: コンテキスト開始
    context_end_in_full: int | None = None    # SQuAD用: コンテキスト終了
```

**文字位置追跡の目的**: AttnLRP分析時に、few-shot例を除外して **実際の質問文のみ** の寄与トークンを抽出するため。

#### ベンチマーク別テンプレート

| ベンチマーク | few-shot数 | 回答形式 | 継承関係 |
|------------|-----------|---------|---------|
| MMLU | 5-shot CoT | "The answer is (X)" | `MMLUPromptTemplate` |
| MMLU-Pro | 5-shot CoT | "The answer is (X)" | `MMLUProPromptTemplate` |
| GSM8K | 8-shot CoT | "The answer is [数値]." | `GSM8KPromptTemplate` |
| SQuAD v2 | 0-shot | テキスト/回答不可 | `SQuADv2PromptTemplate` |
| ARC | 5-shot CoT | "The answer is (X)" | `ARCPromptTemplate(MMLUPromptTemplate)` |
| CommonsenseQA | 5-shot CoT | "The answer is (X)" | `CommonsenseQAPromptTemplate(MMLUPromptTemplate)` |

ARC / CommonsenseQA は `MMLUPromptTemplate` を継承し、few-shot例と`subject`のみ異なる。

### 2.4 回答抽出器 (`evaluation/extractor.py`)

```python
class BaseAnswerExtractor(ABC):
    def extract(self, generated_text: str) -> ExtractionResult: ...
    def is_correct(self, extracted: str, correct_answer: str) -> bool: ...
```

| ベンチマーク | 抽出器クラス | パターン例 | 選択肢範囲 |
|------------|------------|-----------|-----------|
| MMLU | `MMLUAnswerExtractor` | "The answer is (A)" | A-D |
| MMLU-Pro | `MMLUProAnswerExtractor` | "The answer is (A)" | A-J |
| GSM8K | `GSM8KAnswerExtractor` | "The answer is 123" | 数値 |
| SQuAD v2 | `SQuADv2AnswerExtractor` | テキスト/unanswerable | テキスト |
| ARC | `MMLUAnswerExtractor` | "The answer is (A)" | A-D |
| CommonsenseQA | `CommonsenseQAAnswerExtractor` | "The answer is (A)" | A-E |

SQuADv2は追加で `compute_em()`, `compute_f1()`, `compute_scores()` メソッドを持つ。

---

## 3. パラメタ管理 (`config.py`)

Pydanticの `BaseModel` を使った型安全な設定管理:

```python
class ExperimentConfig(BaseModel):
    model: ModelConfig
    benchmark: BenchmarkConfig
    perturbation: PerturbationConfig
    lrp: LRPConfig
    output: OutputConfig
```

### 各設定クラスの詳細

#### `ModelConfig`

| フィールド | 型 | デフォルト | 説明 |
|-----------|---|----------|------|
| `name` | `str` | 必須 | HuggingFaceモデル名 |
| `gpu_id` | `str` | `"0"` | 使用するGPU ID |
| `max_new_tokens` | `int` | `512` | 最大生成トークン数 |
| `batch_size` | `int` | `1` | バッチサイズ |

#### `BenchmarkConfig`

| フィールド | 型 | デフォルト | 説明 |
|-----------|---|----------|------|
| `name` | `Literal["mmlu","gsm8k","squad"]` | 必須 | ベンチマーク名 |
| `num_samples` | `int | None` | `None` | サンプル数 |
| `seed` | `int` | `42` | ランダムシード |

※ 実際のコードでは `arc`, `commonsense_qa`, `mmlu_pro` も対応しているが、config.pyの型定義は未更新。

#### `PerturbationConfig`

| フィールド | 型 | デフォルト | 説明 |
|-----------|---|----------|------|
| `num_perturbations` | `int` | `4` | 摂動するトークン数 (k) |
| `perturbation_types` | `list[str]` | `["delete","replace","insert"]` | 使用する摂動タイプ |
| `random_seed` | `int` | `42` | ランダムシード |

#### `LRPConfig`

| フィールド | 型 | デフォルト | 説明 |
|-----------|---|----------|------|
| `importance_threshold` | `float` | `0.01` | 重要度の閾値 |
| `aggregation_method` | `Literal["sum","mean","max"]` | `"sum"` | サブワード集約方法 |
| `top_k_percent` | `float` | `10.0` | 上位k% |

#### `OutputConfig`

| フィールド | 型 | デフォルト | 説明 |
|-----------|---|----------|------|
| `base_dir` | `str` | `"./outputs"` | 出力ベースディレクトリ |
| `save_heatmaps` | `bool` | `True` | ヒートマップ保存 |
| `save_importance_scores` | `bool` | `True` | 重要度スコア保存 |

### CLI引数（run_inference.py）

実際の実験制御は主にCLI引数で行う:

```bash
python scripts/run_inference.py \
    --model google/gemma-3-4b-it \
    --benchmark mmlu \
    --num_samples 100 \
    --batch_size 4 \
    --gpu_id 0 \
    --output_dir ./outputs/baseline \
    --top_k 10 \
    --max_new_tokens 512 \
    --seed 42 \
    --heatmap_interval 50 \
    --max_retries 2 \
    --retry_delay 5
```

Phase 3（摂動後推論）では `--perturbed_data` 引数で摂動データセットのパスを指定:

```bash
python scripts/run_inference.py \
    --model google/gemma-3-4b-it \
    --benchmark mmlu \
    --perturbed_data datasets/perturbed/.../perturbed_dataset.json \
    --output_dir ./outputs/perturbed
```

---

## 4. 摂動の加え方

### 4.1 文字レベル摂動生成 (`perturbation/generator.py`)

#### 3つの摂動タイプ

```python
class PerturbationType(Enum):
    PROXIMITY = "proximity"        # 隣接キー置換
    DOUBLE_TYPING = "double_typing"  # 二重打鍵
    OMISSION = "omission"          # 打ち忘れ
```

| 摂動タイプ | 操作 | 例 |
|-----------|-----|---|
| PROXIMITY | アルファベット文字をQWERTY配列の隣接キーに置換 | `hello` → `hwllo` |
| DOUBLE_TYPING | アルファベット文字の直後に同じ文字を挿入 | `hello` → `heello` |
| OMISSION | アルファベット文字を1文字削除 | `hello` → `hllo` |

#### QWERTY隣接キーマップ

```python
KEYBOARD_NEIGHBORS = {
    "a": ["q", "w", "s", "z"],
    "b": ["v", "g", "h", "n"],
    "c": ["x", "d", "f", "v"],
    # ... 全26文字分
}
```

#### 摂動対象の選択

1. トークン文字列内の **アルファベット文字のみ** を摂動対象とする
2. 対象文字位置をランダムに1つ選択
3. 3つの摂動タイプからランダムに1つ選択して適用
4. 最初のタイプが失敗した場合、残りのタイプも順に試行

### 4.2 摂動データセット作成 (`perturbation/dataset.py`)

#### 3つの摂動モード

```
┌─────────────────────────────────────────────────────┐
│ 摂動モード                                            │
│                                                       │
│ 1. importance（デフォルト）:                            │
│    重要度スコア上位k個のトークンに摂動               │
│    → AttnLRPが「重要」と判定した箇所を壊す          │
│                                                       │
│ 2. random:                                            │
│    重要度上位k個を「除外」し、残りからランダムにk個選択  │
│    → 重要でない箇所を壊す（対照実験用）               │
│                                                       │
│ 3. bottom_k (Anti-LRP):                               │
│    重要度スコア下位k個のトークンに摂動               │
│    → AttnLRPが「最も不要」と判定した箇所を壊す       │
└─────────────────────────────────────────────────────┘
```

#### データフロー

```
Phase 1 出力
├── results.json         → 元の質問文・選択肢を取得
├── config.json          → モデル名・ベンチマーク情報
└── importance_scores/
    └── {sample_id}.pt   → token_scores（トークン, スコア）ペアのリスト
            ↓
    PerturbedDatasetCreator
            ↓
    perturbed_dataset.json
    ├── metadata (model, benchmark, k, mode, seed, ...)
    └── samples[]
        ├── sample_id
        ├── original_question
        ├── perturbed_question
        ├── perturbed_tokens[]     # どのトークンをどう摂動したか
        │   ├── token_index
        │   ├── original_token
        │   ├── perturbed_token
        │   ├── importance_score
        │   └── perturbation_type
        ├── choices
        └── correct_answer
```

#### トークン選択の制約 (`_should_skip_token`)

以下のトークンは摂動対象から **除外** される:

- 数字のみで構成されるトークン
- 選択肢記号: `(A)` ～ `(J)` の括弧付きまたは裸の文字
- 括弧のみのトークン: `(`, `)`

#### `_get_question_tokens()` — 摂動対象トークンの取得

```python
def _get_question_tokens(self, sample_id: str, ...) -> list[tuple[str, float, int]]:
    """Phase 1の.ptファイルからトークンとスコアを読み込み、
    質問文範囲内（選択肢含む/含まないを制御）のトークンのみ返す."""

    # include_choices=True の場合: token_scores_with_choices を使用
    # include_choices=False の場合: token_scores を使用
```

#### `_apply_perturbations()` — 摂動適用のフロー

```python
def _apply_perturbations(self, question: str, token_scores_list, num_perturbations: int):
    # 1. スキップ対象を除外
    # 2. モードに応じてトークンを選択:
    #    - importance: スコア降順でソート → 上位k個
    #    - random: 上位k個を除外 → 残りからランダムk個
    #    - bottom_k: スコア昇順でソート → 上位k個（=最低スコアk個）
    # 3. 各トークンに文字レベル摂動を適用
    # 4. offset_adjustmentで位置ずれを追跡
```

`offset_adjustment` は、摂動（特にDOUBLE_TYPINGやOMISSION）で文字数が変わった場合の累積オフセットを管理する。

---

## 5. AttnLRPを使用した寄与トークンの特定方法 (`lrp/analyzer.py`)

### 5.1 AttnLRPの基本原理

lxt（LRP eXplains Transformers）ライブラリを使用し、**勾配×入力** ベースの寄与度計算を行う。

### 5.2 `compute_relevance()` — 核心の計算

```python
def compute_relevance(
    self,
    input_ids: torch.Tensor,      # (1, seq_len)
    target_position: int,          # 注目するトークンの位置
) -> torch.Tensor:                 # (seq_len,) の寄与度テンソル

    # 1. 入力埋め込みを取得
    input_embeds = model.get_input_embeddings()(input_ids)
    input_embeds = input_embeds.detach().requires_grad_(True)

    # 2. 順伝播（use_cache=False が必須）
    outputs = model(inputs_embeds=input_embeds, use_cache=False)
    logits = outputs.logits  # (1, seq_len, vocab_size)

    # 3. ターゲット位置の最大ロジットから逆伝播
    target_logits = logits[0, target_position, :]
    max_logit = target_logits.max()
    max_logit.backward()

    # 4. 勾配×入力でトークンごとの寄与度を計算
    relevance = (input_embeds.grad * input_embeds).sum(dim=-1)  # (1, seq_len)
    return relevance.squeeze(0)  # (seq_len,)
```

**ポイント**:
- `use_cache=False`: lxtのパッチはKV-cacheに対応していないため必須
- `target_position`: 「どのトークンの生成に寄与したか」を指定
- `.sum(dim=-1)`: 埋め込み次元方向に集約してトークン単位のスコアにする

### 5.3 2パス分析 (`analyze_combined()`)

```
┌─────────────────────────────────────────────────────────────┐
│ 入力テキスト構造                                              │
│                                                               │
│ [few-shot例][質問文][選択肢][CoT推論過程][回答パターン]        │
│                                                               │
│ Pass 1: Question → CoT の寄与度                              │
│   target_position = cot_token_start（CoT最初のトークン）     │
│   → 質問文のどのトークンがCoT生成開始に寄与したか           │
│                                                               │
│ Pass 2: CoT → Answer の寄与度                                │
│   target_position = answer_choice_position（回答選択肢の位置）│
│   → CoTのどのトークンが最終回答選択に寄与したか             │
└─────────────────────────────────────────────────────────────┘
```

#### Pass 1 詳細: Question → CoT

```python
# 1. プロンプト + 生成テキストをトークン化
full_text = prompt + generated_text
full_input_ids = tokenizer.encode(full_text, return_tensors="pt")

# 2. プロンプトのトークン数を特定
prompt_token_count = len(tokenizer.encode(prompt))

# 3. CoTの最初のトークン位置をターゲットに
cot_token_start = prompt_token_count  # プロンプト直後

# 4. 寄与度を計算
relevance = compute_relevance(full_input_ids, target_position=cot_token_start)

# 5. 質問文範囲のみ抽出（few-shot除外）
#    offset_mappingを使用して文字位置→トークン位置のマッピング
question_relevance = filter_by_char_range(
    relevance, offset_mapping,
    question_char_start, question_char_end
)
```

#### Pass 2 詳細: CoT → Answer

```python
# 1. 回答パターンを検出（正規表現）
answer_pattern = _find_answer_pattern(generated_text)
# "The answer is (A)" → answer_choice_position を特定

# 2. 回答選択肢トークンの位置をターゲットに
relevance = compute_relevance(full_input_ids, target_position=answer_choice_position)

# 3. CoT範囲のみ抽出
#    prompt_token_count ～ answer_token_start の範囲
cot_relevance = relevance.clone()
cot_relevance[:prompt_token_count] = 0    # プロンプト部分をゼロ化
cot_relevance[answer_token_start:] = 0     # 回答部分をゼロ化
```

#### 回答パターン検出 (`_find_answer_pattern()`)

```python
# 検出パターン（優先度順）:
# 1. "The answer is (X)" / "the answer is X"
# 2. "#### [数値]"（GSM8K用）
# 3. "Answer: ..."（SQuAD用）
```

### 5.4 SQuAD用分析 (`analyze_squad()`)

SQuADでは回答が単一トークンではないため、**全生成トークンに対する寄与度を平均化**:

```python
def analyze_squad(self, prompt, generated_text, ...):
    full_input_ids = tokenizer.encode(prompt + generated_text)
    prompt_tokens = len(tokenizer.encode(prompt))
    generated_tokens = len(full_input_ids) - prompt_tokens

    # 各生成トークンに対する寄与度を計算して平均
    total_relevance = torch.zeros(len(full_input_ids))
    for i in range(generated_tokens):
        target = prompt_tokens + i
        rel = compute_relevance(full_input_ids, target_position=target)
        total_relevance += rel

    avg_relevance = total_relevance / generated_tokens
```

### 5.5 offset_mappingのフォールバック

一部のトークナイザー（Gemma等）は `return_offsets_mapping=True` をサポートしないため、手動でoffset_mappingを構築:

```python
def _compute_offset_mapping_fallback(self, text: str, tokens: list[str]) -> list[tuple[int, int]]:
    """各トークンの元テキスト上の(start, end)文字位置を手動計算."""
    offset_mapping = []
    current_pos = 0
    for token in tokens:
        clean_token = token.lstrip("▁Ġ ")  # サブワードマーカー除去
        idx = text.find(clean_token, current_pos)
        if idx >= 0:
            offset_mapping.append((idx, idx + len(clean_token)))
            current_pos = idx + len(clean_token)
        else:
            offset_mapping.append((current_pos, current_pos))
    return offset_mapping
```

### 5.6 出力データ形式

各サンプルの重要度スコアは `.pt` ファイルとして保存:

```python
# Question重要度 ({sample_id}.pt)
{
    "type": "question",
    "token_scores": [(token_str, score), ...],          # 質問文のみ
    "token_scores_with_choices": [(token_str, score), ...],  # 選択肢含む
    "word_scores": [{"word": str, "score": float, "token_indices": [int]}, ...],
    "top_k_words": [...],
    "raw_relevance": torch.Tensor,  # (seq_len,) 全体の寄与度
    "tokens": [str, ...],           # 全トークン文字列
    "offset_mapping": [(int, int), ...],  # 文字位置マッピング
    "question_char_start": int,
    "question_char_end": int,
    "question_with_choices_end": int,
}

# CoT重要度 ({sample_id}_cot.pt)
{
    "type": "cot",
    "token_scores": [(token_str, score), ...],
    "word_scores": [...],
    "raw_relevance": torch.Tensor,
    "cot_token_start": int,
    "cot_token_end": int,
}
```

---

## 6. 分析方法 (Analysis)

### 6.1 Phase 4 比較分析 (`analysis/analyzer.py`)

#### 分析フロー

```
inputs:
  before_dir (Phase 1結果)
  after_dir  (Phase 3結果)
    ↓
1. results.json をロードし、sample_id でマッチング
2. 各サンプルペアについて:
   a. 正解パターンを判定（4パターン）
   b. 質問文重要度スコアを比較（Q指標）
   c. CoT重要度スコアを比較（CoT指標）
   d. 生成テキストを比較（ROUGE-L）
3. 全体統計・相関分析・統計的検定を実行
4. full_results.json / analysis_results.json として保存
```

#### 正解パターン（4パターン）

| パターン | 摂動前 | 摂動後 | 略称 |
|---------|--------|--------|-----|
| correct→correct | 正解 | 正解 | C→C |
| correct→incorrect | 正解 | 不正解 | C→I |
| incorrect→correct | 不正解 | 正解 | I→C |
| incorrect→incorrect | 不正解 | 不正解 | I→I |

#### Q指標（質問文への影響）

| 指標 | 計算方法 | 意味 |
|-----|---------|------|
| ΔToken Num | `after_count - before_count` | 摂動によるトークン数の変化 |
| Q:Jaccard@k | `top_k_jaccard(before, after, k)` | 上位kトークンの重なり (k=3,5,10) |
| Q:Spearman-ρ | `spearmanr(before_scores, after_scores)` | 全トークンの順位相関 |
| Q:Entropy | `shannon_entropy(scores)` | 重要度分布のエントロピー |
| Q:JS-Divergence | `js_divergence(before, after)` | 重要度分布の差異 |
| Q:Concentration@k | `top_k_concentration(scores, k)` | 上位kトークンへの集中度 |

**トークンアライメント**: 摂動によりトークン化結果が変わるため、`PerturbedToken` の情報を用いて摂動前後のトークン位置を対応付ける。`_create_token_alignment()` で文字位置ベースの堅牢なマッピングを構築。

#### CoT指標（CoT推論過程への影響）

| 指標 | 計算方法 | 意味 |
|-----|---------|------|
| CoT:ROUGE-L | `rouge_l_score(before_text, after_text)` | CoTテキストの文字レベル類似度 |
| CoT:Jaccard@k | `top_k_jaccard_by_token(before, after, k)` | 上位kトークンの重なり (k=3,5,10,15,20) |
| CoT:Entropy | `shannon_entropy(scores)` | CoT重要度分布のエントロピー |
| CoT:Concentration@k | `top_k_concentration(scores, k)` | 上位kトークンへの集中度 |

**CoTのJaccardはトークンベース**: 摂動前後でCoTの長さが異なるため、インデックスベースではなくトークン文字列ベースで比較。同一トークンが複数回出現する場合は最大スコアのものを残す。

#### 統計的検定

- **Mann-Whitney U検定**: C→I vs C→C, 回答変化あり vs なし の群間比較
- **Cohen's d**: 効果量の算出
- **Spearman相関**: Q指標とCoT指標の相関分析
- **偏相関**: CoT指標と回答変化の関係（他の変数を統制）

#### 相関分析の構造

```python
# 分析対象グループ:
groups = {
    "all": 全サンプル,
    "correct→correct": C→Cサンプル,
    "correct→incorrect": C→Iサンプル,
    "answer_changed": 回答変化ありサンプル,
    "answer_unchanged": 回答変化なしサンプル,
}

# 相関ペア:
# 1. Q:Spearman-ρ vs CoT:ROUGE-L
# 2. Q:Spearman-ρ vs CoT:Jaccard@k
# 3. Q:Jaccard@k vs CoT:ROUGE-L
# 4. CoT:Jaccard@k vs CoT:ROUGE-L（メイン分析項目）
# 5. Q:ΔEntropy vs CoT:ROUGE-L
# 6. Q:JS-Divergence vs CoT:ROUGE-L
```

### 6.2 メトリクス関数 (`analysis/metrics.py`)

#### 正規化

```python
def normalize_distribution(scores: list[float]) -> list[float]:
    """負の値をシフトし、合計1の確率分布に正規化."""
    min_val = min(scores)
    if min_val < 0:
        scores = [s - min_val for s in scores]
    total = sum(scores)
    return [s / total for s in scores] if total > 0 else scores
```

#### Shannon Entropy

```python
def shannon_entropy(scores: list[float], normalize: bool = True) -> float:
    """正規化エントロピー（0-1の範囲）."""
    # H = -Σ p_i * ln(p_i) / ln(n)
```

#### JS-Divergence

```python
def js_divergence(scores1: list[float], scores2: list[float]) -> float:
    """Jensen-Shannon Divergence. 長さが異なる場合はゼロパディング."""
```

#### Top-k Jaccard（インデックスベース）

```python
def top_k_jaccard(scores1: list[float], scores2: list[float], k: int = 10) -> float:
    """上位kトークンのインデックス集合のJaccard係数."""
    top_k_1 = set(sorted_indices_1[:k])
    top_k_2 = set(sorted_indices_2[:k])
    return len(top_k_1 & top_k_2) / len(top_k_1 | top_k_2)
```

#### Top-k Jaccard（トークンベース）

```python
def top_k_jaccard_by_token(
    tokens1, scores1, tokens2, scores2, k: int = 10
) -> float:
    """トークン文字列ベースのJaccard. 重複トークンは最大スコアで代表."""
    # 1. (token, score) を重複排除（同一tokenは最大scoreを保持）
    # 2. スコア降順で上位k個を選択
    # 3. トークン文字列集合のJaccard
```

#### ROUGE-L

```python
def rouge_l_score(reference: str, hypothesis: str) -> dict:
    """文字レベルLCSに基づくROUGE-L. {precision, recall, f1} を返す."""
```

### 6.3 Appendix分析 (`analysis/appendix_analyzer.py`)

C→C / C→I パターン別に以下の可視化を生成:

| 分析 | 対象 | 出力 |
|-----|-----|------|
| A1 / B1 | WordCloud | IDF重み付きWordCloud画像 (PNG) |
| A2 / B2 | 品詞分布 | spaCyによるPOS分布棒グラフ (PNG) |
| A3 / B3 | ポジション散布図 | トークン位置 vs 重要度スコア散布図 (PNG) |

- A系列 = 質問トークン（Question → CoT の寄与度上位k）
- B系列 = CoTトークン（CoT → Answer の寄与度上位k）

**サブワードマージ**: `merge_subwords()` でSentencePiece (▁) / BPE (Ġ) のサブワードを単語単位に結合してから分析。

---

## 7. W&Bとの連携 (`scripts/upload_to_wandb.py`)

### 7.1 概要

`outputs/` 配下の実験結果を読み込み、6つの実験カテゴリに整理してW&Bにアップロードする。

### 7.2 実験カテゴリ

| 実験 | 内容 | データソース | W&Bキー |
|-----|------|------------|---------|
| 実験1 | 推論性能（Accuracy/EM） | `baseline/*/summary.json` + `perturbed/*/summary.json` | `exp1/` |
| 実験2-a | 質問文への影響指標 | `analysis/*/full_results.json` | `exp2a/` |
| 実験2-b | CoT推論過程への影響指標 | `analysis/*/full_results.json` | `exp2b/` |
| 実験3 | Q指標とCoT指標の相関 | 上記から計算 | `exp3/` |
| 実験4-a | CoT指標と回答変化の偏相関 | 上記から計算 | `exp4a/` |
| 実験4-b | CoT指標と不正解転落の偏相関 | 上記から計算 | `exp4b/` |

### 7.3 データ収集 (`ExperimentDataCollector`)

```python
class ExperimentDataCollector:
    exp1_data: list[dict]         # Accuracy/EM変化データ
    full_results_data: list[dict] # サンプルレベル分析結果
    loaded_files: set[str]        # 読み込み済みファイル追跡（重複防止）
```

#### Exp1データの読み込み

```python
def load_exp1_data(self, outputs_dir: Path):
    # 1. outputs/baseline/*/summary.json → baseline Accuracy
    # 2. outputs/perturbed/*/summary.json → perturbed Accuracy
    #    ディレクトリ名から正規表現で k, perturbation_type を抽出:
    #    パターン: r"_k(\d+)_(importance|random|bottom_k)$"
```

#### サンプルレベルデータの読み込み

```python
def load_directory(self, analysis_dir: Path):
    # outputs/analysis/**/full_results.json を再帰探索
```

### 7.4 データ集約 (`aggregate_sample_data()`)

```python
def aggregate_sample_data(full_results_data):
    """サンプルレベルデータを {benchmark → model → k → metrics} に集約.

    注意: importance (LXP) モードのみを対象。
    bottom_k (Anti-LXP) はExp1のみで使用。
    """
    # 各サンプルから以下のメトリクスを抽出:
    # - token_diff, q_jaccard_{3,5,10}, q_spearman_r
    # - cot_rouge_l, cot_jaccard_{3,5,10,15,20}
    # - pattern, answer_changed（偏相関用）
```

### 7.5 アップロード内容

#### 実験1: テーブル + 折れ線グラフ

- **テーブル**: Benchmark × Model の性能比較表
  - カラム: Baseline, Random-4, LXP-{1,2,4,8}, Anti-LXP-{1,2,4,8}
- **グラフ**: ベンチマーク別の性能推移（k vs Accuracy）
  - LXP: 実線+丸マーカー
  - Anti-LXP: 点線+ダイヤマーカー
  - Baseline: 破線
  - Random-4: ×マーカー

#### 実験2-a: Q指標テーブル + グラフ

- メトリクス: ΔToken Num, Q:Jaccard@{3,5,10}, Q:Spearman-ρ
- 各メトリクスの mean±std を k={1,2,4,8} で表示

#### 実験2-b: CoT指標テーブル + グラフ

- メトリクス: CoT:ROUGE-L, CoT:Jaccard@{3,5,10,15,20}

#### 実験3: 相関テーブル + ヒートマップ

- Q指標 × CoT指標 のSpearman相関行列
- モデル別・k別のサブプロットでヒートマップを生成

#### 実験4-a/4-b: 偏相関テーブル

- **4-a**: CoT指標と回答変化（answer_changed）の偏相関
  - ROUGE-L|Jaccard@m: Jaccard@mを統制した偏相関
  - Jaccard@m|ROUGE-L: ROUGE-Lを統制した偏相関
  - Q指標と回答変化の偏相関も計算
- **4-b**: CoT指標と不正解転落（C→C vs C→I）の偏相関
  - 4-aと同構造、対象をC→Iに限定
- **偏相関計算**: `pingouin.partial_corr()` を使用（Spearman法）
- k=average: 全kの平均相関も計算

### 7.6 Run管理

```python
class WandbUploader:
    def __init__(self, project: str = "lxp-perturbation-analysis"):
        self.project = project

    def start_run(self, resume: bool = True):
        # .wandb_run_id ファイルからRun IDを読み込み、既存Runを再開
        # resume=False の場合は新規Run作成
```

### 7.7 監視モード

```bash
# バックグラウンドで実行し、JSONファイルの追加/変更を自動検出してアップロード
nohup uv run python scripts/upload_to_wandb.py --analysis_dir outputs/analysis --watch &
```

`watchdog` ライブラリでファイルシステムを監視し、5秒間隔でデバウンスして再読み込み→再アップロードを行う。

### 7.8 環境設定

- `.env` ファイルに `WANDB_API_KEY` を設定
- `python-dotenv` で読み込み
- プロジェクト名: `lxp-perturbation-analysis`（デフォルト）

---

## 8. 実験フロー全体像

```
Phase 1: ベースライン推論 + AttnLRP
  scripts/run_inference.py --model ... --benchmark ...
  出力: outputs/baseline/{model}_{benchmark}/
        ├── results.json, summary.json, config.json
        ├── importance_scores/{sample_id}.pt, {sample_id}_cot.pt
        └── heatmaps/*.pdf
    ↓
Phase 2: 摂動データセット作成
  scripts/run_perturbation.py --baseline_dir ... -k 4
  出力: datasets/perturbed/{model}_{benchmark}_k4_importance/
        └── perturbed_dataset.json
    ↓
Phase 3: 摂動後推論 + AttnLRP
  scripts/run_inference.py --perturbed_data ... --output_dir outputs/perturbed
  出力: outputs/perturbed/{model}_{benchmark}_k4_importance/
        ├── results.json, summary.json, config.json
        ├── importance_scores/{sample_id}.pt, {sample_id}_cot.pt
        └── heatmaps/*.pdf
    ↓
Phase 4: 比較分析
  scripts/run_analysis.py --outputs_dir outputs
  出力: outputs/analysis/{benchmark}/{model}/k4_importance/
        ├── full_results.json
        └── analysis_results.json
    ↓
W&Bアップロード
  scripts/upload_to_wandb.py --analysis_dir outputs/analysis
    ↓
（オプション）Appendix分析
  scripts/run_appendix_analysis.py --outputs_dir outputs
  出力: outputs/appendix_analysis/{benchmark}/{model}/k4_importance/
        ├── wordcloud_A_{cc,ci}.png, wordcloud_B_{cc,ci}.png
        ├── pos_A_{cc,ci}.png, pos_B_{cc,ci}.png
        ├── position_A_{cc,ci}.png, position_B_{cc,ci}.png
        ├── sample_tokens_{cc,ci}.json
        └── results.json
```

---

## 9. 設計パターン・共通方針

### ファクトリパターン

全モジュールで `create_*()` ファクトリ関数を使用:

```python
create_model_wrapper(model_name, gpu_id, wrap_for_lxt)  # models/wrapper.py
create_prompt_template(benchmark)                        # models/prompts.py
create_loader(benchmark, samples_per_subset, seed, ...)  # data/loader.py
create_analyzer(model, tokenizer, top_k, device)         # lrp/analyzer.py
create_extractor(benchmark)                              # evaluation/extractor.py
create_perturbed_dataset(baseline_dir, num_perturbations, ...)  # perturbation/dataset.py
```

### ABCによる抽象化

各モジュールの基底クラスは `ABC` を継承:
- `BaseBenchmarkLoader` → 各ベンチマークローダー
- `BasePromptTemplate` → 各テンプレート
- `BaseAnswerExtractor` → 各抽出器

### データ保存形式

- **結果データ**: JSON (`results.json`, `summary.json`, `full_results.json`)
- **重要度スコア**: PyTorch `.pt` ファイル（`torch.save` / `torch.load`）
- **ヒートマップ**: PDF（lxtの`pdf_heatmap`関数使用）
- **Appendix可視化**: PNG（matplotlib）

### 主要外部依存

| ライブラリ | 用途 |
|-----------|------|
| `lxt` | AttnLRP（monkey_patch, pdf_heatmap） |
| `transformers` | モデル・トークナイザーのロード |
| `datasets` | HuggingFaceデータセットのロード |
| `torch` | テンソル演算・GPU計算 |
| `pydantic` | 設定管理 |
| `scipy` | 統計検定（Spearman, Mann-Whitney U） |
| `pingouin` | 偏相関計算 |
| `wandb` | 実験結果の可視化・管理 |
| `plotly` | W&B用グラフ生成 |
| `watchdog` | ファイルシステム監視 |
| `spacy` | 品詞タグ付け（Appendix分析） |
| `wordcloud` | WordCloud生成（Appendix分析） |
| `matplotlib` | グラフ生成（Appendix分析） |
