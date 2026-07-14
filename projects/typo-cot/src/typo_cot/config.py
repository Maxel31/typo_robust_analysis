"""実験設定を管理するモジュール.

Pydanticを使用して型安全な設定管理を提供する。
すべての実験パラメータは引数で指定可能。
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ModelConfig(BaseModel):
    """モデル設定.

    Attributes:
        name: HuggingFaceモデル名（例: "meta-llama/Llama-2-7b-hf"）
        device: 使用するデバイス（"cuda" or "cpu"）
        gpu_id: 使用するGPU ID
        torch_dtype: モデルのデータ型（"float16", "bfloat16", "float32"）
    """

    name: str = Field(description="HuggingFaceモデル名")
    device: Literal["cuda", "cpu"] = Field(default="cuda", description="使用するデバイス")
    gpu_id: str = Field(default="0", description="使用するGPU ID")
    torch_dtype: Literal["float16", "bfloat16", "float32"] = Field(
        default="float16", description="モデルのデータ型"
    )


class BenchmarkConfig(BaseModel):
    """ベンチマーク設定.

    Attributes:
        name: ベンチマーク名（"mmlu", "gsm8k", "squad"）
        num_samples: 評価サンプル数
        num_shots: Few-shotの例示数
        split: データセットのsplit（"test", "validation"）
        subset: MMLUのサブセット（Noneの場合は全体）
    """

    name: Literal["mmlu", "gsm8k", "squad"] = Field(description="ベンチマーク名")
    num_samples: int = Field(default=100, ge=1, description="評価サンプル数")
    num_shots: int = Field(default=5, ge=0, description="Few-shotの例示数")
    split: str = Field(default="test", description="データセットのsplit")
    subset: str | None = Field(default=None, description="MMLUのサブセット")


class PerturbationConfig(BaseModel):
    """摂動設定.

    Attributes:
        num_perturbations: 摂動を加える単語数
        perturbation_types: 適用する摂動タイプのリスト
        random_seed: 乱数シード
    """

    num_perturbations: int = Field(default=3, ge=1, description="摂動を加える単語数")
    perturbation_types: list[Literal["delete", "replace", "insert"]] = Field(
        default=["delete", "replace", "insert"],
        description="適用する摂動タイプ（各摂動でランダム選択）",
    )
    random_seed: int = Field(default=42, description="乱数シード")


class LRPConfig(BaseModel):
    """AttnLRP設定.

    Attributes:
        importance_threshold: 重要単語判定の閾値
        aggregation_method: トークン→単語集約方法（"sum", "mean", "max"）
        top_k_percent: IoU計算時の上位K%（閾値の代わりに使用可能）
    """

    importance_threshold: float = Field(
        default=0.8, ge=0.0, le=1.0, description="重要単語判定の閾値"
    )
    aggregation_method: Literal["sum", "mean", "max"] = Field(
        default="sum", description="トークン→単語集約方法"
    )
    top_k_percent: float | None = Field(
        default=None, ge=0.0, le=100.0, description="IoU計算時の上位K%"
    )


class OutputConfig(BaseModel):
    """出力設定.

    Attributes:
        output_dir: 結果出力先ディレクトリ
        save_intermediate: 中間結果を保存するか
        save_visualizations: 可視化結果を保存するか
    """

    output_dir: Path = Field(default=Path("./outputs"), description="結果出力先ディレクトリ")
    save_intermediate: bool = Field(default=True, description="中間結果を保存するか")
    save_visualizations: bool = Field(default=True, description="可視化結果を保存するか")

    @field_validator("output_dir", mode="before")
    @classmethod
    def convert_to_path(cls, v: str | Path) -> Path:
        """文字列をPathに変換."""
        return Path(v) if isinstance(v, str) else v


class ExperimentConfig(BaseModel):
    """実験全体の設定.

    Attributes:
        model: モデル設定
        benchmark: ベンチマーク設定
        perturbation: 摂動設定
        lrp: AttnLRP設定
        output: 出力設定
        experiment_name: 実験名（自動生成可能）
    """

    model: ModelConfig
    benchmark: BenchmarkConfig
    perturbation: PerturbationConfig = Field(default_factory=PerturbationConfig)
    lrp: LRPConfig = Field(default_factory=LRPConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    experiment_name: str | None = Field(default=None, description="実験名")

    def get_experiment_name(self) -> str:
        """実験名を取得（未設定の場合は自動生成）."""
        if self.experiment_name:
            return self.experiment_name
        # モデル名の最後の部分とベンチマーク名から自動生成
        model_short = self.model.name.split("/")[-1]
        return f"{model_short}_{self.benchmark.name}_n{self.benchmark.num_samples}"

    def get_output_path(self) -> Path:
        """実験出力パスを取得."""
        return self.output.output_dir / self.get_experiment_name()
