"""摂動モジュールのテスト."""

import json
import tempfile
from pathlib import Path

import pytest
import torch

from typo_cot.perturbation.generator import (
    CharacterPerturbationGenerator,
    PerturbationResult,
    PerturbationType,
)
from typo_cot.perturbation.dataset import (
    PerturbedDataset,
    PerturbedDatasetCreator,
    PerturbedSample,
    PerturbedToken,
)


class TestCharacterPerturbationGenerator:
    """CharacterPerturbationGeneratorのテストクラス."""

    @pytest.fixture
    def generator(self) -> CharacterPerturbationGenerator:
        """シード固定の摂動生成器."""
        return CharacterPerturbationGenerator(seed=42)

    def test_get_char_class_lowercase(
        self, generator: CharacterPerturbationGenerator
    ) -> None:
        """小文字の文字種判定."""
        assert generator._get_char_class("a") == "lowercase"
        assert generator._get_char_class("z") == "lowercase"

    def test_get_char_class_uppercase(
        self, generator: CharacterPerturbationGenerator
    ) -> None:
        """大文字の文字種判定."""
        assert generator._get_char_class("A") == "uppercase"
        assert generator._get_char_class("Z") == "uppercase"

    def test_get_char_class_digit(
        self, generator: CharacterPerturbationGenerator
    ) -> None:
        """数字の文字種判定."""
        assert generator._get_char_class("0") == "digit"
        assert generator._get_char_class("9") == "digit"

    def test_get_char_class_other(
        self, generator: CharacterPerturbationGenerator
    ) -> None:
        """その他の文字（記号等）の文字種判定."""
        assert generator._get_char_class(" ") is None
        assert generator._get_char_class("!") is None
        assert generator._get_char_class("あ") is None

    def test_delete_char_basic(
        self, generator: CharacterPerturbationGenerator
    ) -> None:
        """基本的な文字削除テスト."""
        result = generator.delete_char("hello")
        assert result is not None
        assert len(result.perturbed) == 4
        assert result.perturbation_type == PerturbationType.DELETE
        assert result.original == "hello"

    def test_delete_char_single_char(
        self, generator: CharacterPerturbationGenerator
    ) -> None:
        """1文字の場合は削除しない."""
        result = generator.delete_char("a")
        assert result is None

    def test_delete_char_empty(
        self, generator: CharacterPerturbationGenerator
    ) -> None:
        """空文字列の場合は削除しない."""
        result = generator.delete_char("")
        assert result is None

    def test_delete_char_only_symbols(
        self, generator: CharacterPerturbationGenerator
    ) -> None:
        """記号のみの場合は削除しない."""
        result = generator.delete_char("!@#")
        assert result is None

    def test_replace_char_preserves_lowercase(
        self, generator: CharacterPerturbationGenerator
    ) -> None:
        """小文字の置換で文字種が保持される."""
        # 複数回実行して文字種が保持されることを確認
        for _ in range(10):
            result = generator.replace_char("abc")
            if result is not None:
                # 置換後の文字が小文字であることを確認
                assert result.new_char is not None
                assert result.new_char.islower()

    def test_replace_char_preserves_uppercase(
        self, generator: CharacterPerturbationGenerator
    ) -> None:
        """大文字の置換で文字種が保持される."""
        for _ in range(10):
            result = generator.replace_char("ABC")
            if result is not None:
                assert result.new_char is not None
                assert result.new_char.isupper()

    def test_replace_char_preserves_digit(
        self, generator: CharacterPerturbationGenerator
    ) -> None:
        """数字の置換で文字種が保持される."""
        for _ in range(10):
            result = generator.replace_char("123")
            if result is not None:
                assert result.new_char is not None
                assert result.new_char.isdigit()

    def test_replace_char_empty(
        self, generator: CharacterPerturbationGenerator
    ) -> None:
        """空文字列の場合は置換しない."""
        result = generator.replace_char("")
        assert result is None

    def test_insert_char_preserves_lowercase(
        self, generator: CharacterPerturbationGenerator
    ) -> None:
        """小文字に隣接する挿入で文字種が保持される."""
        for _ in range(10):
            result = generator.insert_char("abc")
            if result is not None:
                assert result.new_char is not None
                assert result.new_char.islower()
                assert len(result.perturbed) == 4

    def test_insert_char_preserves_uppercase(
        self, generator: CharacterPerturbationGenerator
    ) -> None:
        """大文字に隣接する挿入で文字種が保持される."""
        for _ in range(10):
            result = generator.insert_char("ABC")
            if result is not None:
                assert result.new_char is not None
                assert result.new_char.isupper()
                assert len(result.perturbed) == 4

    def test_insert_char_empty(
        self, generator: CharacterPerturbationGenerator
    ) -> None:
        """空文字列の場合は挿入しない."""
        result = generator.insert_char("")
        assert result is None

    def test_perturb_excludes_delete_for_single_char(
        self, generator: CharacterPerturbationGenerator
    ) -> None:
        """1文字の場合は削除が除外される."""
        # 複数回実行して削除が行われないことを確認
        for _ in range(20):
            result = generator.perturb("a")
            if result is not None:
                assert result.perturbation_type != PerturbationType.DELETE
                # 置換または挿入のいずれか
                assert result.perturbation_type in [
                    PerturbationType.REPLACE,
                    PerturbationType.INSERT,
                ]

    def test_perturb_returns_valid_result(
        self, generator: CharacterPerturbationGenerator
    ) -> None:
        """perturbが有効な結果を返す."""
        result = generator.perturb("hello")
        assert result is not None
        assert isinstance(result, PerturbationResult)
        assert result.original == "hello"
        assert result.perturbed != "hello"

    def test_perturb_empty_returns_none(
        self, generator: CharacterPerturbationGenerator
    ) -> None:
        """空文字列の場合はNoneを返す."""
        result = generator.perturb("")
        assert result is None

    def test_perturb_token_in_text(
        self, generator: CharacterPerturbationGenerator
    ) -> None:
        """テキスト内のトークンに摂動を適用."""
        text = "The quick brown fox"
        token = "quick"
        token_start = 4

        perturbed_text, result = generator.perturb_token_in_text(
            text, token, token_start
        )

        assert result is not None
        # トークン部分が変更されている
        assert perturbed_text != text
        # トークン前後は保持されている
        assert perturbed_text.startswith("The ")
        assert perturbed_text.endswith(" brown fox") or " brown fox" in perturbed_text

    def test_reproducibility_with_seed(self) -> None:
        """シードによる再現性の確認."""
        gen1 = CharacterPerturbationGenerator(seed=123)
        gen2 = CharacterPerturbationGenerator(seed=123)

        results1 = [gen1.perturb("hello") for _ in range(5)]
        results2 = [gen2.perturb("hello") for _ in range(5)]

        for r1, r2 in zip(results1, results2):
            if r1 is not None and r2 is not None:
                assert r1.perturbed == r2.perturbed
                assert r1.perturbation_type == r2.perturbation_type


class TestPerturbedDataset:
    """PerturbedDatasetのテストクラス."""

    @pytest.fixture
    def sample_dataset(self) -> PerturbedDataset:
        """サンプルの摂動データセット."""
        metadata = {
            "source_model": "test-model",
            "benchmark": "test-benchmark",
            "num_perturbations": 3,
            "seed": 42,
        }
        samples = [
            PerturbedSample(
                sample_id="sample_001",
                original_question="What is the capital of France?",
                perturbed_question="What is the captial of France?",
                perturbed_tokens=[
                    PerturbedToken(
                        token_index=5,
                        original_token="capital",
                        perturbed_token="captial",
                        importance_score=0.85,
                        perturbation_type="replace",
                        char_position=3,
                    )
                ],
                choices=["Paris", "London", "Berlin", "Madrid"],
                correct_answer="A",
                subset="geography",
            )
        ]
        return PerturbedDataset(metadata=metadata, samples=samples)

    def test_save_and_load(self, sample_dataset: PerturbedDataset) -> None:
        """保存と読み込みのテスト."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_dataset.json"
            sample_dataset.save(save_path)

            # ファイルが作成されたことを確認
            assert save_path.exists()

            # 読み込み
            loaded = PerturbedDataset.load(save_path)

            # メタデータの確認
            assert loaded.metadata["source_model"] == "test-model"
            assert loaded.metadata["num_perturbations"] == 3

            # サンプルの確認
            assert len(loaded.samples) == 1
            assert loaded.samples[0].sample_id == "sample_001"
            assert loaded.samples[0].perturbed_question == "What is the captial of France?"

            # 摂動トークンの確認
            assert len(loaded.samples[0].perturbed_tokens) == 1
            assert loaded.samples[0].perturbed_tokens[0].original_token == "capital"

    def test_save_creates_parent_dirs(self, sample_dataset: PerturbedDataset) -> None:
        """親ディレクトリが自動作成される."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "nested" / "dir" / "test_dataset.json"
            sample_dataset.save(save_path)
            assert save_path.exists()


class TestPerturbedDatasetCreator:
    """PerturbedDatasetCreatorのテストクラス."""

    @pytest.fixture
    def mock_baseline_dir(self) -> Path:
        """モックのベースラインディレクトリを作成."""
        with tempfile.TemporaryDirectory() as tmpdir:
            baseline_dir = Path(tmpdir)

            # config.jsonを作成
            config = {
                "model": "test-model",
                "benchmark": "mmlu",
            }
            with open(baseline_dir / "config.json", "w") as f:
                json.dump(config, f)

            # results.jsonを作成
            results = [
                {
                    "sample_id": "sample_001",
                    "question": "What is the capital of France?",
                    "choices": ["Paris", "London", "Berlin", "Madrid"],
                    "correct_answer": "A",
                    "subset": "geography",
                }
            ]
            with open(baseline_dir / "results.json", "w") as f:
                json.dump(results, f)

            # importance_scoresディレクトリを作成
            scores_dir = baseline_dir / "importance_scores"
            scores_dir.mkdir()

            # 重要度スコアを作成
            importance_data = {
                "tokens": ["What", " is", " the", " capital", " of", " France", "?"],
                "token_scores": [
                    ("What", 0.1),
                    (" is", 0.05),
                    (" the", 0.02),
                    (" capital", 0.85),
                    (" of", 0.03),
                    (" France", 0.75),
                    ("?", 0.01),
                ],
                "offset_mapping": [
                    (0, 4),    # What
                    (4, 7),    # is
                    (7, 11),   # the
                    (11, 19),  # capital
                    (19, 22),  # of
                    (22, 29),  # France
                    (29, 30),  # ?
                ],
                "question_char_start": 0,
                "question_char_end": 30,
            }
            torch.save(importance_data, scores_dir / "sample_001.pt")

            yield baseline_dir

    def test_creator_initialization(self, mock_baseline_dir: Path) -> None:
        """初期化のテスト."""
        creator = PerturbedDatasetCreator(
            baseline_dir=mock_baseline_dir,
            num_perturbations=2,
            seed=42,
        )
        assert creator.num_perturbations == 2
        assert creator.seed == 42
        assert len(creator.results) == 1

    def test_creator_missing_results_file(self) -> None:
        """results.jsonがない場合のエラー."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError, match="results.json"):
                PerturbedDatasetCreator(
                    baseline_dir=Path(tmpdir),
                    num_perturbations=2,
                )

    def test_load_importance_scores(self, mock_baseline_dir: Path) -> None:
        """重要度スコアの読み込み."""
        creator = PerturbedDatasetCreator(
            baseline_dir=mock_baseline_dir,
            num_perturbations=2,
        )
        scores = creator._load_importance_scores("sample_001")

        assert scores is not None
        assert "tokens" in scores
        assert "token_scores" in scores
        assert "offset_mapping" in scores

    def test_load_importance_scores_missing(self, mock_baseline_dir: Path) -> None:
        """存在しないサンプルの重要度スコア."""
        creator = PerturbedDatasetCreator(
            baseline_dir=mock_baseline_dir,
            num_perturbations=2,
        )
        scores = creator._load_importance_scores("nonexistent")
        assert scores is None

    def test_get_question_tokens(self, mock_baseline_dir: Path) -> None:
        """質問文内のトークン取得."""
        creator = PerturbedDatasetCreator(
            baseline_dir=mock_baseline_dir,
            num_perturbations=2,
        )
        importance_data = creator._load_importance_scores("sample_001")
        assert importance_data is not None

        question_tokens = creator._get_question_tokens(importance_data)

        # スコアが0より大きいトークンのみ
        assert len(question_tokens) > 0
        # (index, token, score)の形式
        for idx, token, score in question_tokens:
            assert isinstance(idx, int)
            assert isinstance(token, str)
            assert isinstance(score, float)
            assert score > 0

    def test_create_dataset(self, mock_baseline_dir: Path) -> None:
        """データセット作成のテスト."""
        creator = PerturbedDatasetCreator(
            baseline_dir=mock_baseline_dir,
            num_perturbations=2,
            seed=42,
        )
        dataset = creator.create()

        assert dataset is not None
        assert len(dataset.samples) == 1
        assert dataset.metadata["num_perturbations"] == 2

        # 摂動が適用されている
        sample = dataset.samples[0]
        assert sample.original_question == "What is the capital of France?"
        # 摂動後の質問は元と異なる可能性がある
        assert sample.perturbed_tokens is not None

    def test_should_skip_token_numbers(self, mock_baseline_dir: Path) -> None:
        """数値トークンがスキップされる."""
        creator = PerturbedDatasetCreator(
            baseline_dir=mock_baseline_dir,
            num_perturbations=2,
        )
        # 数値のみのトークン
        assert creator._should_skip_token("123") is True
        assert creator._should_skip_token("42") is True
        assert creator._should_skip_token("1,000") is True
        assert creator._should_skip_token("3.14") is True

    def test_should_skip_token_choice_symbols(self, mock_baseline_dir: Path) -> None:
        """選択肢記号がスキップされる."""
        creator = PerturbedDatasetCreator(
            baseline_dir=mock_baseline_dir,
            num_perturbations=2,
        )
        # (A), (B), ... 形式
        assert creator._should_skip_token("(A)") is True
        assert creator._should_skip_token("(B)") is True
        assert creator._should_skip_token("(J)") is True

        # A., B., ... 形式
        assert creator._should_skip_token("A.") is True
        assert creator._should_skip_token("B.") is True

        # A), B), ... 形式
        assert creator._should_skip_token("A)") is True
        assert creator._should_skip_token("B)") is True

        # A:, B:, ... 形式
        assert creator._should_skip_token("A:") is True
        assert creator._should_skip_token("B:") is True

    def test_should_skip_token_normal_words(self, mock_baseline_dir: Path) -> None:
        """通常の単語はスキップされない."""
        creator = PerturbedDatasetCreator(
            baseline_dir=mock_baseline_dir,
            num_perturbations=2,
        )
        # 通常の単語
        assert creator._should_skip_token("capital") is False
        assert creator._should_skip_token("France") is False
        assert creator._should_skip_token("What") is False
        assert creator._should_skip_token("the") is False

    def test_should_skip_token_empty(self, mock_baseline_dir: Path) -> None:
        """空白・空のトークンはスキップされる."""
        creator = PerturbedDatasetCreator(
            baseline_dir=mock_baseline_dir,
            num_perturbations=2,
        )
        assert creator._should_skip_token("") is True
        assert creator._should_skip_token("   ") is True

    def test_random_perturbation_mode(self, mock_baseline_dir: Path) -> None:
        """ランダム摂動モードのテスト."""
        creator = PerturbedDatasetCreator(
            baseline_dir=mock_baseline_dir,
            num_perturbations=2,
            seed=42,
            random_perturbation=True,
        )
        assert creator.random_perturbation is True

        dataset = creator.create()
        assert dataset is not None
        assert dataset.metadata["perturbation_mode"] == "random"

    def test_importance_perturbation_mode(self, mock_baseline_dir: Path) -> None:
        """重要度ベース摂動モードのテスト."""
        creator = PerturbedDatasetCreator(
            baseline_dir=mock_baseline_dir,
            num_perturbations=2,
            seed=42,
            random_perturbation=False,
        )
        assert creator.random_perturbation is False

        dataset = creator.create()
        assert dataset is not None
        assert dataset.metadata["perturbation_mode"] == "importance"

    def test_same_token_same_perturbation(self, mock_baseline_dir: Path) -> None:
        """同じトークンには同じ摂動が適用される（再現性テスト）."""
        # 同じシードで2回実行
        creator1 = PerturbedDatasetCreator(
            baseline_dir=mock_baseline_dir,
            num_perturbations=2,
            seed=42,
        )
        creator2 = PerturbedDatasetCreator(
            baseline_dir=mock_baseline_dir,
            num_perturbations=2,
            seed=42,
        )

        dataset1 = creator1.create()
        dataset2 = creator2.create()

        # 同じサンプルに対して同じ摂動が適用される
        assert len(dataset1.samples) == len(dataset2.samples)
        for s1, s2 in zip(dataset1.samples, dataset2.samples):
            assert s1.perturbed_question == s2.perturbed_question
            assert len(s1.perturbed_tokens) == len(s2.perturbed_tokens)
            for pt1, pt2 in zip(s1.perturbed_tokens, s2.perturbed_tokens):
                assert pt1.perturbed_token == pt2.perturbed_token
