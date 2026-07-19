"""実験8: activation patching の中核 (hook 管理・捕捉・注入・スイープ計画).

forward hooks による 2-pass 実行器:

    pass 1 (donor):     指定位置の活性化を全層×全部位で捕捉 (CPU 保持)
    pass 2 (recipient): 指定の {部位 × 層窓 × 方向} で donor 値を注入して
                        forward / generate し、答えトークンの logit と flip を測る

部位 (site) は計画書の 3 種:

    residual … デコーダ層モジュール出力 (第 l 層通過後の残差ストリーム)
    attn     … layers[l].self_attn 出力 (o_proj 後、残差加算前)
    mlp      … layers[l].mlp 出力 (down_proj 後、残差加算前)

このモジュールは GPU 非依存 (モデル・テンソルは呼び出し側が用意する)。
設計の詳細は docs/dev_notes_08_patching.md を参照。
"""

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

SITES: tuple[str, ...] = ("residual", "attn", "mlp")

DIRECTIONS: tuple[str, ...] = ("clean_to_pert", "pert_to_clean")


# ---------------------------------------------------------------------------
# デコーダ層の発見と部位モジュールの解決
# ---------------------------------------------------------------------------


def find_decoder_layers(model: nn.Module) -> list[nn.Module]:
    """テキストデコーダ層の ModuleList を返す.

    第一候補は `model.get_decoder().layers` (transformers 4.57 で
    Llama/Mistral/Gemma3 いずれも有効。Gemma3ForConditionalGeneration は
    get_decoder() が Gemma3TextModel を返す)。失敗時は
    「self_attn と mlp を持つ要素からなる nn.ModuleList」を、パス名に
    vision/visual を含むもの (マルチモーダルの画像塔) を除いて探索する。
    """
    get_decoder = getattr(model, "get_decoder", None)
    if callable(get_decoder):
        try:
            decoder = get_decoder()
            layers = getattr(decoder, "layers", None)
            if isinstance(layers, nn.ModuleList) and len(layers) > 0:
                first = layers[0]
                if hasattr(first, "self_attn") and hasattr(first, "mlp"):
                    return list(layers)
        except Exception:  # noqa: BLE001 - フォールバック探索に進む
            pass

    for name, module in model.named_modules():
        lowered = name.lower()
        if "vision" in lowered or "visual" in lowered:
            continue
        if isinstance(module, nn.ModuleList) and len(module) > 0:
            first = module[0]
            if hasattr(first, "self_attn") and hasattr(first, "mlp"):
                return list(module)

    raise ValueError(f"デコーダ層が見つかりません: {type(model).__name__}")


def get_site_module(layer: nn.Module, site: str) -> nn.Module:
    """部位名を hook 対象モジュールに解決する."""
    if site == "residual":
        return layer
    if site == "attn":
        return layer.self_attn
    if site == "mlp":
        return layer.mlp
    raise ValueError(f"未知の部位: {site!r} (有効: {SITES})")


def _get_hidden(output) -> torch.Tensor:
    """モジュール出力 (Tensor / tuple / dataclass) から hidden states を取り出す."""
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        return output[0]
    # BaseModelOutput 系
    hidden = getattr(output, "last_hidden_state", None)
    if hidden is not None:
        return hidden
    raise TypeError(f"hidden states を取り出せない出力型: {type(output).__name__}")


def _replace_hidden(output, new_hidden: torch.Tensor):
    """モジュール出力の hidden states を差し替えた同型の出力を返す."""
    if isinstance(output, torch.Tensor):
        return new_hidden
    if isinstance(output, tuple):
        return (new_hidden,) + output[1:]
    if isinstance(output, list):
        return [new_hidden] + output[1:]
    output.last_hidden_state = new_hidden
    return output


# ---------------------------------------------------------------------------
# 捕捉 (pass 1)
# ---------------------------------------------------------------------------


@dataclass
class ActivationCache:
    """donor run の捕捉結果.

    Attributes:
        positions: 捕捉した絶対トークン位置 (昇順である必要はない)
        data: (site, layer_idx) → Tensor [len(positions), hidden] (CPU)
    """

    positions: list[int]
    data: dict[tuple[str, int], torch.Tensor]

    def values(self, site: str, layer_idx: int, positions: Sequence[int]) -> torch.Tensor:
        """絶対位置 positions に対応する捕捉値 [len(positions), hidden] を返す.

        Raises:
            KeyError: 未捕捉の位置・部位・層を要求した場合
        """
        index = {p: i for i, p in enumerate(self.positions)}
        rows = [index[p] for p in positions]  # KeyError = 未捕捉位置
        tensor = self.data[(site, layer_idx)]
        return tensor[rows]


def capture_activations(
    model: nn.Module,
    input_ids: torch.Tensor,
    positions: Sequence[int],
    sites: Sequence[str] = SITES,
    layers: Sequence[int] | None = None,
) -> ActivationCache:
    """1 回の forward で指定位置の活性化を捕捉する (batch=1 前提).

    Args:
        model: CausalLM モデル
        input_ids: [1, T]
        positions: 捕捉する絶対トークン位置
        sites: 捕捉する部位 (既定: 全 3 部位)
        layers: 捕捉する層 index (None なら全層)

    Returns:
        ActivationCache (テンソルは CPU に複製)
    """
    if input_ids.shape[0] != 1:
        raise ValueError("capture_activations は batch=1 のみ対応")

    decoder_layers = find_decoder_layers(model)
    layer_indices = list(layers) if layers is not None else list(range(len(decoder_layers)))
    pos_list = list(positions)

    data: dict[tuple[str, int], torch.Tensor] = {}
    handles = []

    def make_hook(site: str, layer_idx: int):
        def hook(_module, _inputs, output):
            hidden = _get_hidden(output)
            data[(site, layer_idx)] = hidden[0, pos_list, :].detach().to("cpu")
            return None  # 出力は変更しない

        return hook

    try:
        for site in sites:
            for li in layer_indices:
                module = get_site_module(decoder_layers[li], site)
                handles.append(module.register_forward_hook(make_hook(site, li)))
        with torch.no_grad():
            model(input_ids=input_ids)
    finally:
        for h in handles:
            h.remove()

    return ActivationCache(positions=pos_list, data=data)


# ---------------------------------------------------------------------------
# 注入 (pass 2)
# ---------------------------------------------------------------------------


class PatchInjector:
    """forward hook で donor 活性化を注入するコンテキストマネージャ.

    dst_positions は recipient run の絶対トークン位置。values[layer_idx] は
    [len(dst_positions), hidden] の donor 値。出力の系列長が
    max(dst_positions) に満たない forward (generate の decode ステップ等)
    ではパッチを適用しない。
    """

    def __init__(
        self,
        layers: Sequence[nn.Module],
        site: str,
        layer_indices: Sequence[int],
        dst_positions: Sequence[int],
        values: dict[int, torch.Tensor],
    ) -> None:
        self.layers = list(layers)
        self.site = site
        self.layer_indices = list(layer_indices)
        self.dst_positions = list(dst_positions)
        self.values = values
        self._handles: list = []
        if not self.dst_positions:
            raise ValueError("dst_positions が空です")
        for li in self.layer_indices:
            if li not in values:
                raise ValueError(f"層 {li} の注入値がありません")
            if values[li].shape[0] != len(self.dst_positions):
                raise ValueError(
                    f"層 {li}: 注入値の行数 {values[li].shape[0]} が "
                    f"dst_positions の数 {len(self.dst_positions)} と不一致"
                )
        self._min_seq_len = max(self.dst_positions) + 1

    def _make_hook(self, layer_idx: int):
        def hook(_module, _inputs, output):
            hidden = _get_hidden(output)
            if hidden.shape[1] < self._min_seq_len:
                return None  # prefill 外 (decode ステップ / 短系列) では何もしない
            patched = hidden.clone()
            donor = self.values[layer_idx].to(device=hidden.device, dtype=hidden.dtype)
            patched[0, self.dst_positions, :] = donor
            return _replace_hidden(output, patched)

        return hook

    def __enter__(self) -> "PatchInjector":
        for li in self.layer_indices:
            module = get_site_module(self.layers[li], self.site)
            self._handles.append(module.register_forward_hook(self._make_hook(li)))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


# ---------------------------------------------------------------------------
# スイープ計画 (層窓 × 部位 × 方向)
# ---------------------------------------------------------------------------


def layer_windows(n_layers: int, width: int, stride: int) -> list[tuple[int, int]]:
    """層窓 [(start, end), ...] を列挙する (end は排他的、末尾は部分窓可)."""
    if n_layers <= 0 or width <= 0 or stride <= 0:
        raise ValueError("n_layers / width / stride は正の整数")
    windows: list[tuple[int, int]] = []
    for start in range(0, n_layers, stride):
        windows.append((start, min(start + width, n_layers)))
    return windows


def single_layer_windows(layers: Sequence[int]) -> list[tuple[int, int]]:
    """各層 l を幅1窓 (l, l+1) にする (実験8-fine の単層スイープの位置指定).

    Args:
        layers: 単層でパッチする層 index (例: 0..11 + 検証点 14/20/26)

    Returns:
        [(l, l+1), ...] (入力順を保持)
    """
    windows: list[tuple[int, int]] = []
    for layer in layers:
        if layer < 0:
            raise ValueError(f"層 index は非負: {layer}")
        windows.append((layer, layer + 1))
    return windows


def cumulative_windows(layers: Sequence[int]) -> list[tuple[int, int]]:
    """各終端層 l を累積窓 (0, l+1) にする (第0層から第l層まで全差替).

    単層スイープ (各層の限界寄与) に対し、累積は「第0層からここまで直せば
    何%戻るか」を測る。単層 max ≪ 累積 max なら分散書き込みの証拠。

    Args:
        layers: 累積窓の終端層 index (例: 0..11)

    Returns:
        [(0, l+1), ...] (入力順を保持)
    """
    windows: list[tuple[int, int]] = []
    for layer in layers:
        if layer < 0:
            raise ValueError(f"層 index は非負: {layer}")
        windows.append((0, layer + 1))
    return windows


def relative_depth(layer_idx: int, n_layers: int) -> float:
    """層 index の相対深さ l/L を返す (モデル間比較のための整列軸).

    Args:
        layer_idx: 層 index (0 起点)
        n_layers: そのモデルの総層数 L

    Returns:
        layer_idx / n_layers (0.0〜)
    """
    if n_layers <= 0:
        raise ValueError(f"n_layers は正の整数: {n_layers}")
    return layer_idx / n_layers


def align_by_relative_depth(
    profiles: dict[str, dict[int, float]],
    n_layers: dict[str, int],
) -> dict[str, list[tuple[float, float]]]:
    """各モデルの {層 index → 値} を相対深さ軸 (l/L) で整列する.

    Fig.5 差替候補 (相対深さ×回復率の重ね描き) のための整列プリミティブ。

    Args:
        profiles: model 名 → {層 index → 値}
        n_layers: model 名 → 総層数 L

    Returns:
        model 名 → [(rel_depth, value), ...] を rel_depth 昇順にしたリスト

    Raises:
        KeyError: profiles にあるモデルの総層数が n_layers に無い場合
    """
    aligned: dict[str, list[tuple[float, float]]] = {}
    for model_name, layer_values in profiles.items():
        if model_name not in n_layers:
            raise KeyError(f"n_layers に {model_name!r} の総層数がありません")
        L = n_layers[model_name]
        points = [(relative_depth(li, L), v) for li, v in layer_values.items()]
        points.sort(key=lambda dv: dv[0])
        aligned[model_name] = points
    return aligned


@dataclass(frozen=True)
class PatchCell:
    """スイープの 1 セル (部位 × 層窓 × 方向)."""

    site: str
    window: tuple[int, int]
    direction: str


def iter_patch_cells(
    n_layers: int,
    window_size: int,
    window_stride: int,
    sites: Sequence[str] = SITES,
    directions: Sequence[str] = DIRECTIONS,
) -> Iterator[PatchCell]:
    """部位 × 層窓 × 方向の全セルを列挙する."""
    windows = layer_windows(n_layers, window_size, window_stride)
    for site in sites:
        for window in windows:
            for direction in directions:
                yield PatchCell(site=site, window=window, direction=direction)


# ---------------------------------------------------------------------------
# 位置整列ユーティリティ
# ---------------------------------------------------------------------------


@dataclass
class FirstDivergence:
    """2 つの答え継続トークン列が最初に分岐する位置.

    Attributes:
        common: 分岐前の共通接頭辞トークン列
        token_a: 分岐位置の a 側トークン (clean 側)
        token_b: 分岐位置の b 側トークン (typo 側)
    """

    common: list[int]
    token_a: int
    token_b: int


def first_divergence(
    ids_a: Sequence[int],
    ids_b: Sequence[int],
    limit: int = 16,
) -> FirstDivergence | None:
    """先頭 limit トークン以内で最初に分岐するトークン対を返す.

    分岐が無い (同一・一方が他方の接頭辞・limit 超過) 場合は None。
    """
    n = min(len(ids_a), len(ids_b), limit)
    for i in range(n):
        if ids_a[i] != ids_b[i]:
            return FirstDivergence(common=list(ids_a[:i]), token_a=ids_a[i], token_b=ids_b[i])
    return None


def span_end_token(
    offsets: Sequence[tuple[int, int]],
    char_start: int,
    char_end: int,
) -> int | None:
    """文字スパン [char_start, char_end) に重なる最後のトークン index を返す.

    幅 0 の offset (special token) は無視する。重なりが無ければ None。
    """
    last: int | None = None
    for i, (s, e) in enumerate(offsets):
        if e <= s:
            continue  # 幅 0 (special token)
        if s < char_end and char_start < e:
            last = i
    return last


# ---------------------------------------------------------------------------
# 指標
# ---------------------------------------------------------------------------


def kl_from_logits(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    """KL(softmax(p) ‖ softmax(q)) を logits から計算する (1 位置分)."""
    logp = torch.log_softmax(p_logits.float(), dim=-1)
    logq = torch.log_softmax(q_logits.float(), dim=-1)
    return float((logp.exp() * (logp - logq)).sum().item())


# ---------------------------------------------------------------------------
# flip ペア選定・シャード (CLI 用)
# ---------------------------------------------------------------------------


def select_flip_pairs(pairs: Sequence, n: int | None, seed: int = 42) -> list:
    """flip ペア (clean 正解 ∧ 摂動誤答) を決定論的に選ぶ.

    アーカイブ analysis の pattern="correct→incorrect" と同値の判定を
    PairRecord (is_correct_clean / extra["is_correct_typo"]) に対して行い、
    sample_id ソート → seed シャッフル → 先頭 n 件を返す (n=None は全件)。
    """
    import random

    flips = [
        p
        for p in pairs
        if p.is_correct_clean and not p.extra.get("is_correct_typo", False)
    ]
    flips.sort(key=lambda p: p.sample_id)
    random.Random(seed).shuffle(flips)
    return flips if n is None else flips[:n]


def shard_slice(items: Sequence, shard_index: int, num_shards: int) -> list:
    """決定論的なシャード分割 items[shard_index::num_shards]."""
    if num_shards <= 0:
        raise ValueError(f"num_shards は正の整数: {num_shards}")
    if not 0 <= shard_index < num_shards:
        raise ValueError(f"shard_index {shard_index} が範囲 [0, {num_shards}) の外")
    return list(items[shard_index::num_shards])


# ---------------------------------------------------------------------------
# 冪等実行 (CLI 用)
# ---------------------------------------------------------------------------


def result_is_current(path: str | Path, config_hash: str) -> bool:
    """既存の結果 JSON が現在の設定 (config_hash) で書かれたものなら True."""
    path = Path(path)
    if not path.exists():
        return False
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    return payload.get("config_hash") == config_hash
