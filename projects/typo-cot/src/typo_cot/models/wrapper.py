"""モデルラッパーモジュール.

lxtライブラリと統合したモデルラッパーを提供する。
AttnLRPによる重要度計算に対応。
"""

import logging
import os
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)


def setup_device(gpu_id: str = "0") -> tuple[torch.device, bool]:
    """GPUの可用性をチェックし、適切なデバイスを返す.

    Args:
        gpu_id: 使用するGPU ID（デフォルト: "0"、複数の場合はカンマ区切り: "0,1,2"）

    Returns:
        (torch.device, use_multi_gpu): デバイスと複数GPU使用フラグのタプル
    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # 外部 (run_with_gpu.sh 等の GPU 排他ヘルパー) が既に CUDA_VISIBLE_DEVICES を
    # 設定している場合はそちらを優先し、上書きしない
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        gpu_id = os.environ["CUDA_VISIBLE_DEVICES"]
        logger.info(f"外部設定の CUDA_VISIBLE_DEVICES={gpu_id} を優先します")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    # 複数GPUかどうかを判定
    gpu_ids = [g.strip() for g in gpu_id.split(",")]
    use_multi_gpu = len(gpu_ids) > 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"デバイス: {device} (GPU ID: {gpu_id})")

    if device.type == "cuda":
        num_gpus = torch.cuda.device_count()
        logger.info(f"利用可能なGPU数: {num_gpus}")
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            memory_gb = torch.cuda.get_device_properties(i).total_memory / 1024**3
            logger.info(f"  GPU {i}: {gpu_name} ({memory_gb:.1f} GB)")

    return device, use_multi_gpu


@dataclass
class GenerationResult:
    """生成結果のデータクラス.

    Attributes:
        input_text: 入力テキスト
        output_text: 生成されたテキスト（入力を含む）
        generated_text: 生成されたテキストのみ（入力を除く）
        input_ids: 入力のトークンID
        output_ids: 出力のトークンID（入力を含む）
        input_tokens: 入力のトークン文字列リスト
        output_tokens: 出力のトークン文字列リスト
    """

    input_text: str
    output_text: str
    generated_text: str
    input_ids: torch.Tensor
    output_ids: torch.Tensor
    input_tokens: list[str]
    output_tokens: list[str]


class ModelWrapper:
    """lxtライブラリと統合したモデルラッパー.

    AttnLRPによる重要度計算に対応したHuggingFaceモデルのラッパー。
    """

    # 使用可能なモデルリスト（Instruct版 + PT版）
    ALLOWED_MODELS: list[str] = [
        # Llama 3.2 Instruct
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        # Llama 3.2 PT (Pretrained)
        "meta-llama/Llama-3.2-1B",
        "meta-llama/Llama-3.2-3B",
        # Llama 3.1 Instruct（大規模モデル）
        "meta-llama/Llama-3.1-70B-Instruct",
        # Gemma 3 Instruct
        "google/gemma-3-1b-it",
        "google/gemma-3-4b-it",
        "google/gemma-3-12b-it",
        "google/gemma-3-27b-it",
        # Gemma 3 PT (Pretrained)
        "google/gemma-3-1b-pt",
        "google/gemma-3-4b-pt",
        "google/gemma-3-12b-pt",
        # Qwen 2.5 Instruct
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-32B-Instruct",
        # DeepSeek-R1 蒸留 (reasoning特化、実験10③。Qwen2アーキテクチャ)
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        # Mistral Instruct
        "mistralai/Mistral-7B-Instruct-v0.3",
        # Mistral PT (Pretrained)
        "mistralai/Mistral-7B-v0.3",
    ]

    # lxtでサポートされているモデルファミリー（内部判定用）
    _SUPPORTED_FAMILIES: list[str] = [
        "llama",
        "gemma",
        "mistral",
        "qwen",
    ]

    def __init__(
        self,
        model_name: str,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
        trust_remote_code: bool = False,
        use_multi_gpu: bool = False,
    ) -> None:
        """初期化.

        Args:
            model_name: HuggingFaceモデル名またはローカルパス
            device: 使用するデバイス（Noneの場合は自動検出）
            dtype: モデルのデータ型（lxtでは勾配オーバーフロー防止のためbfloat16推奨）
            trust_remote_code: リモートコードを信頼するか
            use_multi_gpu: 複数GPUを使用するか（device_map="auto"）
        """
        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        self.use_multi_gpu = use_multi_gpu

        self._model: PreTrainedModel | None = None
        self._tokenizer: PreTrainedTokenizer | None = None
        self._is_lxt_wrapped: bool = False

    @property
    def model(self) -> PreTrainedModel:
        """モデルを取得（遅延ロード）."""
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        """トークナイザーを取得（遅延ロード）."""
        if self._tokenizer is None:
            self._load_tokenizer()
        return self._tokenizer

    def _load_model(self) -> None:
        """モデルをロード."""
        logger.info(f"モデルをロード中: {self.model_name}")

        # 複数GPUの場合はdevice_map="auto"を使用
        if self.use_multi_gpu:
            logger.info("複数GPUモード: device_map='auto'を使用")
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
                device_map="auto",
                trust_remote_code=self.trust_remote_code,
            )
        else:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
                device_map=self.device,
                trust_remote_code=self.trust_remote_code,
            )
        self._model.eval()

        logger.info(f"モデルロード完了: {self.model_name}")

    def _load_tokenizer(self) -> None:
        """トークナイザーをロード."""
        logger.info(f"トークナイザーをロード中: {self.model_name}")

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
        )

        # パディングトークンの設定
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        logger.info(f"トークナイザーロード完了: {self.model_name}")

    @classmethod
    def get_allowed_models(cls) -> list[str]:
        """使用可能なモデルリストを取得.

        Returns:
            使用可能なモデル名のリスト
        """
        return cls.ALLOWED_MODELS.copy()

    def is_allowed_model(self) -> bool:
        """モデルが使用可能リストに含まれているかチェック.

        Returns:
            使用可能な場合はTrue
        """
        return self.model_name in self.ALLOWED_MODELS

    def is_supported_for_lxt(self) -> bool:
        """モデルがlxtでサポートされているかチェック.

        Returns:
            サポートされている場合はTrue
        """
        model_name_lower = self.model_name.lower()
        return any(family in model_name_lower for family in self._SUPPORTED_FAMILIES)

    def wrap_for_lxt(self) -> "ModelWrapper":
        """lxtのAttnLRP用にモデルをラップする.

        lxtクイックスタートに準拠した実装。
        重要: monkey_patchはモデルロード**前に**適用する必要がある。
        https://lxt.readthedocs.io/en/latest/quickstart.html

        Returns:
            ラップされたModelWrapper（自身を返す）

        Raises:
            ImportError: lxtがインストールされていない場合
            ValueError: モデルがlxtでサポートされていない場合
        """
        if self._is_lxt_wrapped:
            logger.warning("モデルは既にlxtでラップされています")
            return self

        if not self.is_allowed_model():
            raise ValueError(
                f"モデル '{self.model_name}' は使用可能リストに含まれていません。"
                f"使用可能なモデル: {self.ALLOWED_MODELS}"
            )

        if not self.is_supported_for_lxt():
            raise ValueError(
                f"モデル '{self.model_name}' はlxtでサポートされていません。"
                f"サポートされているファミリー: {self._SUPPORTED_FAMILIES}"
            )

        # モデルタイプに応じたlxtのpatch_mapとモジュールをインポート
        # lxt.efficient.models にサポートされているモデル: llama, gpt2, gemma3, qwen2, qwen3
        model_name_lower = self.model_name.lower()
        patch_map = None
        model_module = None
        import_error = None
        module_name = None

        try:
            from lxt.efficient.core import monkey_patch

            if "llama" in model_name_lower or "mistral" in model_name_lower:
                import transformers.models.llama.modeling_llama as model_module
                from lxt.efficient.models.llama import attnLRP as patch_map  # noqa: N813

                module_name = "llama"
            elif "gpt2" in model_name_lower:
                import transformers.models.gpt2.modeling_gpt2 as model_module
                from lxt.efficient.models.gpt2 import attnLRP as patch_map  # noqa: N813

                module_name = "gpt2"
            elif "gemma" in model_name_lower:
                import transformers.models.gemma3.modeling_gemma3 as model_module
                from lxt.efficient.models.gemma3 import attnLRP as patch_map  # noqa: N813

                module_name = "gemma3"
            elif "qwen" in model_name_lower:
                import transformers.models.qwen2.modeling_qwen2 as model_module
                from lxt.efficient.models.qwen2 import attnLRP as patch_map  # noqa: N813

                module_name = "qwen2"
        except ImportError as e:
            import_error = e

        if patch_map is None or model_module is None:
            if import_error:
                raise ImportError(
                    f"lxtのインポートに失敗しました。\n"
                    f"モデル: {self.model_name}\n"
                    f"エラー詳細: {import_error}\n"
                    f"'uv add lxt' または 'pip install lxt' を実行してください。"
                ) from import_error
            else:
                raise ValueError(
                    f"モデル '{self.model_name}' に対応するlxt関数が見つかりません。\n"
                    f"lxtがサポートするモデル: llama, gpt2, gemma3, qwen2, mistral"
                )

        # 重要: monkey_patchはモデルロード**前に**適用する必要がある
        # lxtはモジュールレベルでクラス/関数を置き換えるため
        logger.info(f"lxtでモジュールをパッチ中... (モジュール: {module_name})")
        monkey_patch(model_module, patch_map, verbose=True)
        logger.info("lxtパッチ適用完了")

        # モデルをロード（パッチ適用後）
        if self._model is None:
            self._load_model()

        self._is_lxt_wrapped = True
        logger.info("lxtラップ完了")

        return self

    def tokenize(self, text: str, return_tensors: str = "pt") -> dict[str, Any]:
        """テキストをトークナイズ.

        Args:
            text: 入力テキスト
            return_tensors: 戻り値のテンソル形式

        Returns:
            トークナイズ結果
        """
        return self.tokenizer(text, return_tensors=return_tensors)

    def get_tokens(self, text: str) -> list[str]:
        """テキストをトークンのリストに変換.

        Args:
            text: 入力テキスト

        Returns:
            トークン文字列のリスト
        """
        input_ids = self.tokenizer.encode(text)
        return [self.tokenizer.decode([token_id]) for token_id in input_ids]

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        do_sample: bool = False,
        **kwargs: Any,
    ) -> GenerationResult:
        """テキストを生成.

        Args:
            prompt: 入力プロンプト
            max_new_tokens: 生成する最大トークン数
            temperature: サンプリング温度
            do_sample: サンプリングを行うか
            **kwargs: その他の生成パラメータ

        Returns:
            生成結果
        """
        inputs = self.tokenize(prompt)
        input_ids = inputs["input_ids"].to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else None,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs,
            )

        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        generated_text = output_text[len(prompt) :]

        # トークン文字列を取得
        input_tokens = self.get_tokens(prompt)
        output_tokens = [self.tokenizer.decode([tid]) for tid in output_ids[0].tolist()]

        return GenerationResult(
            input_text=prompt,
            output_text=output_text,
            generated_text=generated_text,
            input_ids=input_ids,
            output_ids=output_ids,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def generate_batch(
        self,
        prompts: list[str],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        do_sample: bool = False,
        **kwargs: Any,
    ) -> list[GenerationResult]:
        """複数プロンプトをバッチで推論.

        Args:
            prompts: 入力プロンプトのリスト
            max_new_tokens: 生成する最大トークン数
            temperature: サンプリング温度
            do_sample: サンプリングを行うか
            **kwargs: その他の生成パラメータ

        Returns:
            生成結果のリスト
        """
        if len(prompts) == 0:
            return []

        if len(prompts) == 1:
            return [self.generate(prompts[0], max_new_tokens, temperature, do_sample, **kwargs)]

        # バッチトークナイズ（左パディング）
        self.tokenizer.padding_side = "left"
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else None,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs,
            )

        # 各サンプルの結果を作成
        results: list[GenerationResult] = []
        for i, prompt in enumerate(prompts):
            # 出力からパディング部分を除去
            sample_output_ids = output_ids[i]

            output_text = self.tokenizer.decode(sample_output_ids, skip_special_tokens=True)
            # プロンプト部分を除いた生成テキスト
            generated_text = output_text[len(prompt) :]

            # トークン文字列を取得
            input_tokens = self.get_tokens(prompt)
            output_tokens = [
                self.tokenizer.decode([tid])
                for tid in sample_output_ids.tolist()
                if tid != self.tokenizer.pad_token_id
            ]

            results.append(
                GenerationResult(
                    input_text=prompt,
                    output_text=output_text,
                    generated_text=generated_text,
                    input_ids=input_ids[i : i + 1],  # 元のバッチ形状を維持
                    output_ids=output_ids[i : i + 1],
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
            )

        return results

    def forward_with_cache(
        self,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[Any, ...]]:
        """キャッシュ付きでフォワードパスを実行.

        Args:
            input_ids: 入力トークンID

        Returns:
            (logits, past_key_values) のタプル
        """
        input_ids = input_ids.to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                use_cache=True,
                output_attentions=False,
            )

        return outputs.logits, outputs.past_key_values


def create_model_wrapper(
    model_name: str,
    gpu_id: str = "0",
    dtype: torch.dtype = torch.bfloat16,
    wrap_for_lxt: bool = True,
) -> ModelWrapper:
    """モデルラッパーを作成するファクトリ関数.

    Args:
        model_name: HuggingFaceモデル名またはローカルパス
        gpu_id: 使用するGPU ID（複数の場合はカンマ区切り: "0,1,2"）
        dtype: モデルのデータ型（lxtでは勾配オーバーフロー防止のためbfloat16推奨）
        wrap_for_lxt: lxtでラップするか

    Returns:
        ModelWrapperインスタンス
    """
    device, use_multi_gpu = setup_device(gpu_id)

    wrapper = ModelWrapper(
        model_name=model_name,
        device=device,
        dtype=dtype,
        use_multi_gpu=use_multi_gpu,
    )

    if wrap_for_lxt:
        wrapper.wrap_for_lxt()

    return wrapper
