"""ベンチマークデータローダーモジュール.

MMLU、MMLU-Pro、GSM8K、SQuAD v2のデータを読み込む機能を提供する。
"""

import logging
import random
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass

from datasets import load_dataset

logger = logging.getLogger(__name__)


@dataclass
class Sample:
    """ベンチマークサンプルのデータクラス.

    Attributes:
        sample_id: サンプルの一意識別子
        question: 質問文
        choices: 選択肢リスト（選択式の場合）
        correct_answer: 正解
        context: コンテキスト（SQuADの場合）
        subset: サブセット名（MMLU/MMLU-Proの場合）
        answer_start: 回答の開始位置（SQuAD v2の場合）
        answer_end: 回答の終了位置（SQuAD v2の場合）
    """

    sample_id: str
    question: str
    choices: list[str] | None
    correct_answer: str
    context: str | None = None
    subset: str | None = None
    answer_start: int | None = None
    answer_end: int | None = None


class BaseBenchmarkLoader(ABC):
    """ベンチマークローダーの基底クラス."""

    @abstractmethod
    def load(self) -> list[Sample]:
        """データを読み込んでSampleのリストを返す."""
        pass

    @abstractmethod
    def get_subsets(self) -> list[str]:
        """利用可能なサブセット名のリストを返す."""
        pass


class MMLULoader(BaseBenchmarkLoader):
    """MMLUデータローダー.

    各サブセット（57種）から指定数のサンプルをランダム抽出する。
    """

    # MMLUの全57サブセット
    SUBSETS: list[str] = [
        "abstract_algebra",
        "anatomy",
        "astronomy",
        "business_ethics",
        "clinical_knowledge",
        "college_biology",
        "college_chemistry",
        "college_computer_science",
        "college_mathematics",
        "college_medicine",
        "college_physics",
        "computer_security",
        "conceptual_physics",
        "econometrics",
        "electrical_engineering",
        "elementary_mathematics",
        "formal_logic",
        "global_facts",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_computer_science",
        "high_school_european_history",
        "high_school_geography",
        "high_school_government_and_politics",
        "high_school_macroeconomics",
        "high_school_mathematics",
        "high_school_microeconomics",
        "high_school_physics",
        "high_school_psychology",
        "high_school_statistics",
        "high_school_us_history",
        "high_school_world_history",
        "human_aging",
        "human_sexuality",
        "international_law",
        "jurisprudence",
        "logical_fallacies",
        "machine_learning",
        "management",
        "marketing",
        "medical_genetics",
        "miscellaneous",
        "moral_disputes",
        "moral_scenarios",
        "nutrition",
        "philosophy",
        "prehistory",
        "professional_accounting",
        "professional_law",
        "professional_medicine",
        "professional_psychology",
        "public_relations",
        "security_studies",
        "sociology",
        "us_foreign_policy",
        "virology",
        "world_religions",
    ]

    def __init__(
        self,
        samples_per_subset: int = 50,
        split: str = "test",
        seed: int = 42,
        subsets: list[str] | None = None,
    ) -> None:
        """初期化.

        Args:
            samples_per_subset: 各サブセットから抽出するサンプル数
            split: データセットのsplit（"test", "validation", "dev"）
            seed: ランダムシード
            subsets: 使用するサブセットのリスト（Noneの場合は全57種）
        """
        self.samples_per_subset = samples_per_subset
        self.split = split
        self.seed = seed
        self.subsets = subsets if subsets is not None else self.SUBSETS

    def get_subsets(self) -> list[str]:
        """利用可能なサブセット名のリストを返す."""
        return self.subsets

    def load(self) -> list[Sample]:
        """全サブセットからデータを読み込む."""
        all_samples: list[Sample] = []
        for subset in self.subsets:
            samples = self.load_subset(subset)
            all_samples.extend(samples)
        return all_samples

    def load_subset(self, subset: str) -> list[Sample]:
        """指定サブセットからサンプルを読み込む.

        Args:
            subset: サブセット名

        Returns:
            Sampleのリスト
        """
        dataset = load_dataset("cais/mmlu", subset, split=self.split)

        # ランダムサンプリング
        random.seed(self.seed)
        indices = list(range(len(dataset)))
        if len(indices) > self.samples_per_subset:
            indices = random.sample(indices, self.samples_per_subset)

        samples: list[Sample] = []
        for idx in indices:
            item = dataset[idx]
            sample_id = f"mmlu_{subset}_{idx:04d}"
            choices = [item["choices"][i] for i in range(len(item["choices"]))]
            # 正解はインデックス（0-3）→ A-D に変換
            correct_answer = chr(ord("A") + item["answer"])

            samples.append(
                Sample(
                    sample_id=sample_id,
                    question=item["question"],
                    choices=choices,
                    correct_answer=correct_answer,
                    subset=subset,
                )
            )

        return samples

    def iter_subsets(self) -> Iterator[tuple[str, list[Sample]]]:
        """サブセットごとにイテレートする.

        Yields:
            (サブセット名, Sampleのリスト) のタプル
        """
        for subset in self.subsets:
            yield subset, self.load_subset(subset)


class MMLUProLoader(BaseBenchmarkLoader):
    """MMLU-Proデータローダー.

    MMLU-Proは14カテゴリから構成され、10個の選択肢を持つ。
    各カテゴリから指定数のサンプルをランダム抽出する。
    """

    # MMLU-Proの全14カテゴリ
    CATEGORIES: list[str] = [
        "biology",
        "business",
        "chemistry",
        "computer science",
        "economics",
        "engineering",
        "health",
        "history",
        "law",
        "math",
        "other",
        "philosophy",
        "physics",
        "psychology",
    ]

    def __init__(
        self,
        samples_per_category: int = 50,
        split: str = "test",
        seed: int = 42,
        categories: list[str] | None = None,
    ) -> None:
        """初期化.

        Args:
            samples_per_category: 各カテゴリから抽出するサンプル数
            split: データセットのsplit（"test", "validation"）
            seed: ランダムシード
            categories: 使用するカテゴリのリスト（Noneの場合は全14種）
        """
        self.samples_per_category = samples_per_category
        self.split = split
        self.seed = seed
        self.categories = categories if categories is not None else self.CATEGORIES

    def get_subsets(self) -> list[str]:
        """利用可能なカテゴリ名のリストを返す."""
        return self.categories

    def load(self) -> list[Sample]:
        """全カテゴリからデータを読み込む."""
        dataset = load_dataset("TIGER-Lab/MMLU-Pro", split=self.split)

        # カテゴリごとにグループ化
        category_samples: dict[str, list[dict]] = {cat: [] for cat in self.categories}
        for item in dataset:
            cat = item["category"]
            if cat in category_samples:
                category_samples[cat].append(item)

        # 各カテゴリからサンプリング
        random.seed(self.seed)
        all_samples: list[Sample] = []

        for category in self.categories:
            items = category_samples[category]
            if len(items) > self.samples_per_category:
                items = random.sample(items, self.samples_per_category)

            for idx, item in enumerate(items):
                sample_id = f"mmlu_pro_{category.replace(' ', '_')}_{idx:04d}"
                # MMLU-Proのoptionsはリスト形式
                choices = item["options"]
                # 正解はインデックス → A-J に変換
                correct_answer = chr(ord("A") + item["answer_index"])

                all_samples.append(
                    Sample(
                        sample_id=sample_id,
                        question=item["question"],
                        choices=choices,
                        correct_answer=correct_answer,
                        subset=category,
                    )
                )

        return all_samples

    def iter_categories(self) -> Iterator[tuple[str, list[Sample]]]:
        """カテゴリごとにイテレートする.

        Yields:
            (カテゴリ名, Sampleのリスト) のタプル
        """
        dataset = load_dataset("TIGER-Lab/MMLU-Pro", split=self.split)

        # カテゴリごとにグループ化
        category_samples: dict[str, list[dict]] = {cat: [] for cat in self.categories}
        for item in dataset:
            cat = item["category"]
            if cat in category_samples:
                category_samples[cat].append(item)

        random.seed(self.seed)
        for category in self.categories:
            items = category_samples[category]
            if len(items) > self.samples_per_category:
                items = random.sample(items, self.samples_per_category)

            samples: list[Sample] = []
            for idx, item in enumerate(items):
                sample_id = f"mmlu_pro_{category.replace(' ', '_')}_{idx:04d}"
                choices = item["options"]
                correct_answer = chr(ord("A") + item["answer_index"])

                samples.append(
                    Sample(
                        sample_id=sample_id,
                        question=item["question"],
                        choices=choices,
                        correct_answer=correct_answer,
                        subset=category,
                    )
                )

            yield category, samples


class GSM8KLoader(BaseBenchmarkLoader):
    """GSM8Kデータローダー.

    全件を使用する。
    """

    def __init__(self, split: str = "test") -> None:
        """初期化.

        Args:
            split: データセットのsplit（"train", "test"）
        """
        self.split = split

    def get_subsets(self) -> list[str]:
        """GSM8Kにはサブセットがないため空リストを返す."""
        return []

    def load(self) -> list[Sample]:
        """データを読み込む."""
        dataset = load_dataset("openai/gsm8k", "main", split=self.split)

        samples: list[Sample] = []
        for idx, item in enumerate(dataset):
            sample_id = f"gsm8k_{idx:05d}"
            # GSM8Kの正解は "#### 数値" の形式
            answer_text = item["answer"]
            # 最終的な数値を抽出
            correct_answer = answer_text.split("####")[-1].strip()

            samples.append(
                Sample(
                    sample_id=sample_id,
                    question=item["question"],
                    choices=None,  # GSM8Kは選択式ではない
                    correct_answer=correct_answer,
                )
            )

        return samples


class SQuADv2Loader(BaseBenchmarkLoader):
    """SQuAD v2データローダー.

    指定されたサンプル数をランダムサンプリングする。
    参照箇所のアノテーション情報（answer_start, answer_end）を保存する。
    """

    def __init__(
        self,
        split: str = "validation",
        num_samples: int | None = 2000,
        seed: int = 42,
    ) -> None:
        """初期化.

        Args:
            split: データセットのsplit（"train", "validation"）
            num_samples: サンプル数（Noneの場合は全件）
            seed: ランダムシード
        """
        self.split = split
        self.num_samples = num_samples
        self.seed = seed

    def get_subsets(self) -> list[str]:
        """SQuAD v2にはサブセットがないため空リストを返す."""
        return []

    def load(self) -> list[Sample]:
        """データを読み込む."""
        dataset = load_dataset("rajpurkar/squad_v2", split=self.split)

        # ランダムサンプリング
        if self.num_samples is not None and len(dataset) > self.num_samples:
            import random

            random.seed(self.seed)
            indices = random.sample(range(len(dataset)), self.num_samples)
            dataset = dataset.select(indices)
            logger.info(f"SQuAD v2: {self.num_samples}件をランダムサンプリング")
        else:
            logger.info(f"SQuAD v2: 全{len(dataset)}件を使用")

        samples: list[Sample] = []
        for item in dataset:
            sample_id = f"squad_v2_{item['id']}"
            # SQuAD v2は回答不可能な質問がある
            answers = item["answers"]["text"]
            answer_starts = item["answers"]["answer_start"]

            if answers:
                correct_answer = answers[0]
                answer_start = answer_starts[0]
                answer_end = answer_start + len(correct_answer)
            else:
                # 回答不可能な場合
                correct_answer = ""
                answer_start = None
                answer_end = None

            samples.append(
                Sample(
                    sample_id=sample_id,
                    question=item["question"],
                    choices=None,
                    correct_answer=correct_answer,
                    context=item["context"],
                    answer_start=answer_start,
                    answer_end=answer_end,
                )
            )

        return samples


class ARCLoader(BaseBenchmarkLoader):
    """ARC-Challengeデータローダー.

    AI2 Reasoning Challenge (ARC) の難易度の高い問題セット。
    4択の科学問題で構成される。
    """

    def __init__(
        self,
        num_samples: int | None = None,
        split: str = "test",
        seed: int = 42,
    ) -> None:
        """初期化.

        Args:
            num_samples: 抽出するサンプル数（Noneの場合は全件）
            split: データセットのsplit
            seed: ランダムシード
        """
        self.num_samples = num_samples
        self.split = split
        self.seed = seed

    def get_subsets(self) -> list[str]:
        """利用可能なサブセット名のリストを返す."""
        return ["ARC-Challenge"]

    def load(self) -> list[Sample]:
        """データを読み込む."""
        dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=self.split)

        # ランダムサンプリング
        if self.num_samples is not None and len(dataset) > self.num_samples:
            random.seed(self.seed)
            indices = random.sample(range(len(dataset)), self.num_samples)
            dataset = dataset.select(indices)
            logger.info(f"ARC-Challenge: {self.num_samples}件をランダムサンプリング")
        else:
            logger.info(f"ARC-Challenge: 全{len(dataset)}件を使用")

        samples: list[Sample] = []
        for item in dataset:
            sample_id = f"arc_{item['id']}"
            # choices は {'label': [...], 'text': [...]} 形式
            choices = item["choices"]["text"]
            correct_answer = item["answerKey"]  # "A", "B", "C", "D"

            samples.append(
                Sample(
                    sample_id=sample_id,
                    question=item["question"],
                    choices=choices,
                    correct_answer=correct_answer,
                    subset="ARC-Challenge",
                )
            )

        return samples


class CommonsenseQALoader(BaseBenchmarkLoader):
    """CommonsenseQAデータローダー.

    常識推論の5択問題で構成される。
    """

    def __init__(
        self,
        num_samples: int | None = None,
        split: str = "validation",
        seed: int = 42,
    ) -> None:
        """初期化.

        Args:
            num_samples: 抽出するサンプル数（Noneの場合は全件）
            split: データセットのsplit
            seed: ランダムシード
        """
        self.num_samples = num_samples
        self.split = split
        self.seed = seed

    def get_subsets(self) -> list[str]:
        """利用可能なサブセット名のリストを返す."""
        return ["commonsense_qa"]

    def load(self) -> list[Sample]:
        """データを読み込む."""
        dataset = load_dataset("tau/commonsense_qa", split=self.split)

        # ランダムサンプリング
        if self.num_samples is not None and len(dataset) > self.num_samples:
            random.seed(self.seed)
            indices = random.sample(range(len(dataset)), self.num_samples)
            dataset = dataset.select(indices)
            logger.info(f"CommonsenseQA: {self.num_samples}件をランダムサンプリング")
        else:
            logger.info(f"CommonsenseQA: 全{len(dataset)}件を使用")

        samples: list[Sample] = []
        for item in dataset:
            sample_id = f"csqa_{item['id']}"
            # choices は {'label': [...], 'text': [...]} 形式
            choices = item["choices"]["text"]
            correct_answer = item["answerKey"]  # "A", "B", "C", "D", "E"

            samples.append(
                Sample(
                    sample_id=sample_id,
                    question=item["question"],
                    choices=choices,
                    correct_answer=correct_answer,
                    subset="commonsense_qa",
                )
            )

        return samples


class BBHLoader(BaseBenchmarkLoader):
    """BIG-Bench Hard (BBH) データローダー.

    23 のサブタスクから成る CoT 推論ベンチマーク. 各サンプルは
    "input" (質問テキスト) と "target" (正解) を持つ.

    本実装ではマルチタスク横断で `samples_per_subset` 件ずつ抽出する.
    Dataset: lukaemon/bbh （23 config）
    """

    DEFAULT_SUBTASKS = (
        "boolean_expressions",
        "causal_judgement",
        "date_understanding",
        "disambiguation_qa",
        "formal_fallacies",
        "geometric_shapes",
        "hyperbaton",
        "logical_deduction_five_objects",
        "logical_deduction_seven_objects",
        "logical_deduction_three_objects",
        "movie_recommendation",
        "multistep_arithmetic_two",
        "navigate",
        "object_counting",
        "penguins_in_a_table",
        "reasoning_about_colored_objects",
        "ruin_names",
        "salient_translation_error_detection",
        "snarks",
        "sports_understanding",
        "temporal_sequences",
        "tracking_shuffled_objects_five_objects",
        "tracking_shuffled_objects_seven_objects",
        "tracking_shuffled_objects_three_objects",
        "web_of_lies",
        "word_sorting",
    )

    def __init__(
        self,
        samples_per_subset: int = 50,
        split: str = "test",
        seed: int = 42,
        subsets: list[str] | None = None,
    ) -> None:
        self.samples_per_subset = samples_per_subset
        self.split = split
        self.seed = seed
        self.subsets = subsets or list(self.DEFAULT_SUBTASKS)

    def get_subsets(self) -> list[str]:
        return list(self.DEFAULT_SUBTASKS)

    def load(self) -> list[Sample]:
        samples: list[Sample] = []
        for subtask in self.subsets:
            try:
                ds = load_dataset("lukaemon/bbh", subtask, split=self.split)
            except Exception as e:
                logger.warning(f"BBH subtask '{subtask}' を読み込めません: {e}")
                continue
            n = min(self.samples_per_subset, len(ds))
            if n < len(ds):
                random.seed(self.seed)
                idx = random.sample(range(len(ds)), n)
                ds = ds.select(idx)
            logger.info(f"BBH/{subtask}: {len(ds)} 件")
            for i, item in enumerate(ds):
                samples.append(
                    Sample(
                        sample_id=f"bbh_{subtask}_{i:04d}",
                        question=item["input"],
                        choices=None,
                        correct_answer=str(item["target"]).strip(),
                        subset=subtask,
                    )
                )
        return samples


class MATHLoader(BaseBenchmarkLoader):
    """MATH (Hendrycks et al., 2021) データローダー.

    高校数学コンテスト問題. 答えは \\boxed{...} 形式の LaTeX 文字列.
    Dataset: HuggingFaceH4/MATH-500（500件の評価サブセット）
    """

    def __init__(
        self,
        num_samples: int | None = None,
        split: str = "test",
        seed: int = 42,
    ) -> None:
        self.num_samples = num_samples
        self.split = split
        self.seed = seed

    def get_subsets(self) -> list[str]:
        return [
            "Algebra",
            "Counting & Probability",
            "Geometry",
            "Intermediate Algebra",
            "Number Theory",
            "Prealgebra",
            "Precalculus",
        ]

    def load(self) -> list[Sample]:
        ds = load_dataset("HuggingFaceH4/MATH-500", split=self.split)
        if self.num_samples is not None and len(ds) > self.num_samples:
            random.seed(self.seed)
            idx = random.sample(range(len(ds)), self.num_samples)
            ds = ds.select(idx)
            logger.info(f"MATH-500: {self.num_samples}件をランダムサンプリング")
        else:
            logger.info(f"MATH-500: 全{len(ds)}件を使用")

        samples: list[Sample] = []
        for i, item in enumerate(ds):
            subject = str(item.get("subject", "")).strip() or None
            samples.append(
                Sample(
                    sample_id=f"math_{i:05d}",
                    question=item["problem"],
                    choices=None,
                    correct_answer=str(item["answer"]).strip(),
                    subset=subject,
                )
            )
        return samples


class StrategyQALoader(BaseBenchmarkLoader):
    """StrategyQA データローダー.

    多段階常識推論の Yes/No QA. 質問文だけが与えられ、解は yes/no.
    Dataset: ChilleD/StrategyQA
    """

    def __init__(
        self,
        num_samples: int | None = None,
        split: str = "test",
        seed: int = 42,
    ) -> None:
        self.num_samples = num_samples
        self.split = split
        self.seed = seed

    def get_subsets(self) -> list[str]:
        return ["strategy_qa"]

    def load(self) -> list[Sample]:
        try:
            ds = load_dataset("ChilleD/StrategyQA", split=self.split)
        except Exception:
            # フォールバック: HF Hub 上の別ミラー
            ds = load_dataset("voidful/StrategyQA", split=self.split)
        if self.num_samples is not None and len(ds) > self.num_samples:
            random.seed(self.seed)
            idx = random.sample(range(len(ds)), self.num_samples)
            ds = ds.select(idx)
            logger.info(f"StrategyQA: {self.num_samples}件をランダムサンプリング")
        else:
            logger.info(f"StrategyQA: 全{len(ds)}件を使用")

        samples: list[Sample] = []
        for i, item in enumerate(ds):
            qid = item.get("qid") or item.get("id") or f"{i:05d}"
            ans_raw = item.get("answer", item.get("answer_text", ""))
            if isinstance(ans_raw, bool):
                ans = "yes" if ans_raw else "no"
            else:
                ans = str(ans_raw).strip().lower()
                if ans in ("true", "1"):
                    ans = "yes"
                elif ans in ("false", "0"):
                    ans = "no"
            samples.append(
                Sample(
                    sample_id=f"strategyqa_{qid}",
                    question=item["question"],
                    choices=None,
                    correct_answer=ans,
                    subset="strategy_qa",
                )
            )
        return samples


def create_loader(
    benchmark: str,
    split: str | None = None,
    samples_per_subset: int = 50,
    seed: int = 42,
    subsets: list[str] | None = None,
    num_samples: int | None = None,
) -> BaseBenchmarkLoader:
    """ベンチマーク名からローダーを作成するファクトリ関数.

    Args:
        benchmark: ベンチマーク名 (mmlu / mmlu_pro / gsm8k / squad_v2 / arc /
            commonsense_qa / bbh / math / strategy_qa)
        split: データセットのsplit
        samples_per_subset: MMLU/MMLU-Pro/BBH の各サブセットから抽出するサンプル数
        seed: ランダムシード
        subsets: MMLU/MMLU-Pro/BBH で使用するサブセット/カテゴリのリスト
        num_samples: サンプル数（Noneの場合はデフォルト値を使用）

    Returns:
        対応するローダーインスタンス

    Raises:
        ValueError: 不明なベンチマーク名の場合
    """
    if benchmark == "mmlu":
        return MMLULoader(
            samples_per_subset=samples_per_subset,
            split=split or "test",
            seed=seed,
            subsets=subsets,
        )
    elif benchmark == "mmlu_pro":
        return MMLUProLoader(
            samples_per_category=samples_per_subset,
            split=split or "test",
            seed=seed,
            categories=subsets,
        )
    elif benchmark == "gsm8k":
        return GSM8KLoader(split=split or "test")
    elif benchmark == "squad_v2":
        return SQuADv2Loader(
            split=split or "validation",
            num_samples=num_samples if num_samples is not None else 2000,
            seed=seed,
        )
    elif benchmark == "arc":
        return ARCLoader(
            num_samples=num_samples,
            split=split or "test",
            seed=seed,
        )
    elif benchmark == "commonsense_qa":
        return CommonsenseQALoader(
            num_samples=num_samples,
            split=split or "validation",
            seed=seed,
        )
    elif benchmark == "bbh":
        return BBHLoader(
            samples_per_subset=samples_per_subset,
            split=split or "test",
            seed=seed,
            subsets=subsets,
        )
    elif benchmark == "math":
        return MATHLoader(
            num_samples=num_samples,
            split=split or "test",
            seed=seed,
        )
    elif benchmark == "strategy_qa":
        return StrategyQALoader(
            num_samples=num_samples,
            split=split or "test",
            seed=seed,
        )
    else:
        raise ValueError(
            f"不明なベンチマーク: {benchmark}. 利用可能: "
            "mmlu, mmlu_pro, gsm8k, squad_v2, arc, commonsense_qa, "
            "bbh, math, strategy_qa"
        )
