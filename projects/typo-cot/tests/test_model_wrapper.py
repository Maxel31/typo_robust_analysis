"""モデルラッパーモジュールのテスト."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from typo_cot.models.wrapper import (
    GenerationResult,
    ModelWrapper,
    create_model_wrapper,
    setup_device,
)


class TestSetupDevice:
    """setup_device関数のテスト."""

    @patch("torch.cuda.is_available")
    def test_setup_device_with_cuda(self, mock_cuda_available: MagicMock) -> None:
        """CUDAが利用可能な場合のデバイス設定を確認."""
        mock_cuda_available.return_value = True

        with (
            patch("torch.cuda.get_device_name", return_value="NVIDIA A100"),
            patch("torch.cuda.get_device_properties") as mock_props,
        ):
            mock_props.return_value.total_memory = 40 * 1024**3  # 40GB

            device = setup_device("0")

            assert device.type == "cuda"

    @patch("torch.cuda.is_available")
    def test_setup_device_without_cuda(self, mock_cuda_available: MagicMock) -> None:
        """CUDAが利用できない場合のデバイス設定を確認."""
        mock_cuda_available.return_value = False

        device = setup_device("0")

        assert device.type == "cpu"


class TestGenerationResult:
    """GenerationResultデータクラスのテスト."""

    def test_generation_result_creation(self) -> None:
        """GenerationResultが正しく作成されることを確認."""
        result = GenerationResult(
            input_text="Hello",
            output_text="Hello, world!",
            generated_text=", world!",
            input_ids=torch.tensor([[1, 2, 3]]),
            output_ids=torch.tensor([[1, 2, 3, 4, 5]]),
            input_tokens=["Hello"],
            output_tokens=["Hello", ",", " world", "!"],
        )

        assert result.input_text == "Hello"
        assert result.output_text == "Hello, world!"
        assert result.generated_text == ", world!"
        assert result.input_ids.shape == (1, 3)
        assert result.output_ids.shape == (1, 5)


class TestModelWrapper:
    """ModelWrapperのテスト."""

    def test_allowed_models_list(self) -> None:
        """使用可能なモデルリストが定義されていることを確認."""
        allowed = ModelWrapper.get_allowed_models()
        assert "meta-llama/Llama-3.2-1B" in allowed
        assert "gpt2" in allowed
        assert "google/gemma-3-1b-pt" in allowed
        assert "mistralai/Mistral-7B-v0.3" in allowed

    def test_is_allowed_model_true(self) -> None:
        """許可されたモデルが正しく判定されることを確認."""
        wrapper = ModelWrapper(model_name="meta-llama/Llama-3.2-1B")
        assert wrapper.is_allowed_model() is True

    def test_is_allowed_model_false(self) -> None:
        """許可されていないモデルが正しく判定されることを確認."""
        wrapper = ModelWrapper(model_name="meta-llama/Llama-2-7b-hf")
        assert wrapper.is_allowed_model() is False

    def test_is_supported_for_lxt_llama(self) -> None:
        """LLaMAモデルがlxtでサポートされていることを確認."""
        wrapper = ModelWrapper(model_name="meta-llama/Llama-3.2-1B")
        assert wrapper.is_supported_for_lxt() is True

    def test_is_supported_for_lxt_gpt2(self) -> None:
        """GPT-2モデルがlxtでサポートされていることを確認."""
        wrapper = ModelWrapper(model_name="gpt2")
        assert wrapper.is_supported_for_lxt() is True

    def test_is_supported_for_lxt_unsupported(self) -> None:
        """サポートされていないモデルを確認."""
        wrapper = ModelWrapper(model_name="bert-base-uncased")
        assert wrapper.is_supported_for_lxt() is False

    @patch("typo_cot.models.wrapper.AutoModelForCausalLM.from_pretrained")
    @patch("typo_cot.models.wrapper.AutoTokenizer.from_pretrained")
    def test_lazy_loading(
        self,
        mock_tokenizer: MagicMock,
        mock_model: MagicMock,
    ) -> None:
        """遅延ロードが正しく動作することを確認."""
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance

        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<eos>"
        mock_tokenizer.return_value = mock_tokenizer_instance

        wrapper = ModelWrapper(model_name="gpt2", device=torch.device("cpu"))

        # モデルとトークナイザーはまだロードされていない
        mock_model.assert_not_called()
        mock_tokenizer.assert_not_called()

        # モデルにアクセスするとロードされる
        _ = wrapper.model
        mock_model.assert_called_once()

        # トークナイザーにアクセスするとロードされる
        _ = wrapper.tokenizer
        mock_tokenizer.assert_called_once()

    @patch("typo_cot.models.wrapper.AutoTokenizer.from_pretrained")
    def test_tokenize(self, mock_tokenizer: MagicMock) -> None:
        """トークナイズが正しく動作することを確認."""
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.pad_token = "<pad>"
        mock_tokenizer_instance.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_tokenizer.return_value = mock_tokenizer_instance

        wrapper = ModelWrapper(model_name="gpt2", device=torch.device("cpu"))
        result = wrapper.tokenize("Hello")

        assert "input_ids" in result

    @patch("typo_cot.models.wrapper.AutoTokenizer.from_pretrained")
    def test_get_tokens(self, mock_tokenizer: MagicMock) -> None:
        """トークン取得が正しく動作することを確認."""
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.pad_token = "<pad>"
        mock_tokenizer_instance.encode.return_value = [1, 2, 3]
        mock_tokenizer_instance.decode.side_effect = lambda x: f"token_{x[0]}"
        mock_tokenizer.return_value = mock_tokenizer_instance

        wrapper = ModelWrapper(model_name="gpt2", device=torch.device("cpu"))
        tokens = wrapper.get_tokens("Hello world")

        assert len(tokens) == 3
        assert tokens[0] == "token_1"

    @patch("typo_cot.models.wrapper.AutoModelForCausalLM.from_pretrained")
    @patch("typo_cot.models.wrapper.AutoTokenizer.from_pretrained")
    def test_generate(
        self,
        mock_tokenizer: MagicMock,
        mock_model: MagicMock,
    ) -> None:
        """テキスト生成が正しく動作することを確認."""
        # モックモデルの設定
        mock_model_instance = MagicMock()
        mock_model_instance.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model.return_value = mock_model_instance

        # モックトークナイザーの設定
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.pad_token = "<pad>"
        mock_tokenizer_instance.pad_token_id = 0
        mock_tokenizer_instance.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_tokenizer_instance.encode.return_value = [1, 2, 3]
        mock_tokenizer_instance.decode.side_effect = lambda x, **kwargs: (
            "Hello world!" if isinstance(x, torch.Tensor) else f"t{x[0]}"
        )
        mock_tokenizer.return_value = mock_tokenizer_instance

        wrapper = ModelWrapper(model_name="gpt2", device=torch.device("cpu"))
        result = wrapper.generate("Hello", max_new_tokens=10)

        assert isinstance(result, GenerationResult)
        assert result.input_text == "Hello"

    def test_wrap_for_lxt_not_allowed_model(self) -> None:
        """許可リストにないモデルでlxtラップがエラーになることを確認."""
        wrapper = ModelWrapper(model_name="meta-llama/Llama-2-7b-hf", device=torch.device("cpu"))

        with pytest.raises(ValueError, match="使用可能リストに含まれていません"):
            wrapper.wrap_for_lxt()

    def test_wrap_for_lxt_unsupported_family(self) -> None:
        """lxtでサポートされていないモデルファミリーでエラーになることを確認."""
        # ALLOWED_MODELSに追加されていないがlxtでサポートされていないモデルをテスト
        # この場合、まずis_allowed_modelでエラーになる
        wrapper = ModelWrapper(model_name="bert-base-uncased", device=torch.device("cpu"))

        with pytest.raises(ValueError, match="使用可能リストに含まれていません"):
            wrapper.wrap_for_lxt()


class TestCreateModelWrapper:
    """create_model_wrapperファクトリ関数のテスト."""

    @patch("typo_cot.models.wrapper.setup_device")
    @patch.object(ModelWrapper, "wrap_for_lxt")
    def test_create_with_lxt_wrap(
        self,
        mock_wrap: MagicMock,
        mock_setup_device: MagicMock,
    ) -> None:
        """lxtラップ付きでモデルラッパーが作成されることを確認."""
        mock_setup_device.return_value = torch.device("cpu")
        mock_wrap.return_value = MagicMock()

        wrapper = create_model_wrapper(
            model_name="meta-llama/Llama-3.2-1B",
            gpu_id="0",
            wrap_for_lxt=True,
        )

        assert isinstance(wrapper, ModelWrapper)
        mock_wrap.assert_called_once()

    @patch("typo_cot.models.wrapper.setup_device")
    def test_create_without_lxt_wrap(
        self,
        mock_setup_device: MagicMock,
    ) -> None:
        """lxtラップなしでモデルラッパーが作成されることを確認."""
        mock_setup_device.return_value = torch.device("cpu")

        wrapper = create_model_wrapper(
            model_name="gpt2",
            gpu_id="0",
            wrap_for_lxt=False,
        )

        assert isinstance(wrapper, ModelWrapper)
        assert wrapper._is_lxt_wrapped is False
