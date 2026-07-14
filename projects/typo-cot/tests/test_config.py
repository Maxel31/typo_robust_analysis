"""設定モジュールのテスト."""

from pathlib import Path

import pytest

from typo_cot.config import (
    BenchmarkConfig,
    ExperimentConfig,
    LRPConfig,
    ModelConfig,
    OutputConfig,
    PerturbationConfig,
)


class TestModelConfig:
    """ModelConfigのテスト."""

    def test_default_values(self) -> None:
        """デフォルト値が正しく設定されることを確認."""
        config = ModelConfig(name="meta-llama/Llama-2-7b-hf")
        assert config.name == "meta-llama/Llama-2-7b-hf"
        assert config.device == "cuda"
        assert config.gpu_id == "0"
        assert config.torch_dtype == "float16"

    def test_custom_values(self) -> None:
        """カスタム値が正しく設定されることを確認."""
        config = ModelConfig(
            name="gpt2",
            device="cpu",
            gpu_id="1",
            torch_dtype="float32",
        )
        assert config.name == "gpt2"
        assert config.device == "cpu"
        assert config.gpu_id == "1"
        assert config.torch_dtype == "float32"


class TestBenchmarkConfig:
    """BenchmarkConfigのテスト."""

    def test_default_values(self) -> None:
        """デフォルト値が正しく設定されることを確認."""
        config = BenchmarkConfig(name="mmlu")
        assert config.name == "mmlu"
        assert config.num_samples == 100
        assert config.num_shots == 5
        assert config.split == "test"
        assert config.subset is None

    def test_validation_num_samples(self) -> None:
        """num_samplesのバリデーションを確認."""
        with pytest.raises(ValueError):
            BenchmarkConfig(name="mmlu", num_samples=0)

    def test_validation_num_shots(self) -> None:
        """num_shotsのバリデーションを確認."""
        with pytest.raises(ValueError):
            BenchmarkConfig(name="mmlu", num_shots=-1)


class TestPerturbationConfig:
    """PerturbationConfigのテスト."""

    def test_default_values(self) -> None:
        """デフォルト値が正しく設定されることを確認."""
        config = PerturbationConfig()
        assert config.num_perturbations == 3
        assert config.perturbation_types == ["delete", "replace", "insert"]
        assert config.random_seed == 42


class TestLRPConfig:
    """LRPConfigのテスト."""

    def test_default_values(self) -> None:
        """デフォルト値が正しく設定されることを確認."""
        config = LRPConfig()
        assert config.importance_threshold == 0.8
        assert config.aggregation_method == "sum"
        assert config.top_k_percent is None

    def test_validation_threshold(self) -> None:
        """importance_thresholdのバリデーションを確認."""
        with pytest.raises(ValueError):
            LRPConfig(importance_threshold=1.5)
        with pytest.raises(ValueError):
            LRPConfig(importance_threshold=-0.1)


class TestOutputConfig:
    """OutputConfigのテスト."""

    def test_default_values(self) -> None:
        """デフォルト値が正しく設定されることを確認."""
        config = OutputConfig()
        assert config.output_dir == Path("./outputs")
        assert config.save_intermediate is True
        assert config.save_visualizations is True

    def test_path_conversion(self) -> None:
        """文字列からPathへの変換を確認."""
        config = OutputConfig(output_dir="/tmp/test")
        assert config.output_dir == Path("/tmp/test")
        assert isinstance(config.output_dir, Path)


class TestExperimentConfig:
    """ExperimentConfigのテスト."""

    def test_minimal_config(self) -> None:
        """最小限の設定で作成できることを確認."""
        config = ExperimentConfig(
            model=ModelConfig(name="gpt2"),
            benchmark=BenchmarkConfig(name="mmlu"),
        )
        assert config.model.name == "gpt2"
        assert config.benchmark.name == "mmlu"
        assert config.perturbation.num_perturbations == 3
        assert config.lrp.importance_threshold == 0.8

    def test_get_experiment_name_auto(self) -> None:
        """実験名の自動生成を確認."""
        config = ExperimentConfig(
            model=ModelConfig(name="meta-llama/Llama-2-7b-hf"),
            benchmark=BenchmarkConfig(name="gsm8k", num_samples=50),
        )
        assert config.get_experiment_name() == "Llama-2-7b-hf_gsm8k_n50"

    def test_get_experiment_name_custom(self) -> None:
        """カスタム実験名を確認."""
        config = ExperimentConfig(
            model=ModelConfig(name="gpt2"),
            benchmark=BenchmarkConfig(name="mmlu"),
            experiment_name="my_experiment",
        )
        assert config.get_experiment_name() == "my_experiment"

    def test_get_output_path(self) -> None:
        """出力パスの取得を確認."""
        config = ExperimentConfig(
            model=ModelConfig(name="gpt2"),
            benchmark=BenchmarkConfig(name="mmlu"),
            output=OutputConfig(output_dir=Path("/tmp/outputs")),
        )
        expected = Path("/tmp/outputs/gpt2_mmlu_n100")
        assert config.get_output_path() == expected
