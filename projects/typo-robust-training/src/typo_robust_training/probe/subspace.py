"""Deterministic semantic subspaces derived from frozen linear probes.

The subspace is defined by the row space of the *class-centred* word identity
classifier at the independently selected transition layer.  The implementation
never materialises a dense ``hidden_size x hidden_size`` projector: projection
is always the rank-sized operation ``(x @ Q.T) @ Q``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from typo_robust_training.probe.artifacts import ProbeTransitionArtifact


SEMANTIC_SUBSPACE_RANK = 16


def _finite_matrix(value: object, *, field: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or not array.size or not np.isfinite(array).all():
        raise ValueError(f"{field} must be one non-empty finite matrix")
    return np.ascontiguousarray(array)


def _finite_vector(value: object, *, field: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or not array.size or not np.isfinite(array).all():
        raise ValueError(f"{field} must be one non-empty finite vector")
    return np.ascontiguousarray(array)


def canonicalize_basis_signs(basis: object) -> np.ndarray:
    """Make each orthonormal row's largest-magnitude coordinate positive."""

    rows = _finite_matrix(basis, field="semantic basis")
    result = rows.copy()
    for row in result:
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] == 0.0:
            raise ValueError("semantic basis contains a zero row")
        if row[pivot] < 0.0:
            row *= -1.0
    return np.ascontiguousarray(result)


def validate_orthonormal_rows(basis: object, *, tolerance: float = 1e-10) -> np.ndarray:
    """Return one float64 row basis or fail closed on numerical drift."""

    rows = _finite_matrix(basis, field="semantic basis")
    if (
        not math.isfinite(float(tolerance))
        or float(tolerance) <= 0.0
        or rows.shape[0] > rows.shape[1]
    ):
        raise ValueError("semantic orthogonality validation settings differ")
    gram = rows @ rows.T
    if not np.allclose(gram, np.eye(rows.shape[0]), atol=tolerance, rtol=tolerance):
        raise ValueError("semantic basis rows are not orthonormal")
    return rows


@dataclass(frozen=True, slots=True)
class SemanticProbeSubspace:
    """Rank-limited classifier row space used by patching and training."""

    rank: int
    hidden_size: int
    class_count: int
    basis: np.ndarray
    projected_class_weights: np.ndarray
    classifier_bias: np.ndarray
    singular_values: np.ndarray

    def __post_init__(self) -> None:
        if (
            isinstance(self.rank, bool)
            or not isinstance(self.rank, int)
            or self.rank <= 0
            or isinstance(self.hidden_size, bool)
            or not isinstance(self.hidden_size, int)
            or self.hidden_size < self.rank
            or isinstance(self.class_count, bool)
            or not isinstance(self.class_count, int)
            or self.class_count - 1 < self.rank
        ):
            raise ValueError("semantic subspace dimensions differ")
        raw_basis = _finite_matrix(self.basis, field="semantic basis")
        if raw_basis.shape != (self.rank, self.hidden_size):
            raise ValueError("semantic subspace tensor shapes differ")
        basis = validate_orthonormal_rows(raw_basis)
        projected = _finite_matrix(
            self.projected_class_weights,
            field="projected classifier weights",
        )
        bias = _finite_vector(self.classifier_bias, field="classifier bias")
        singular = _finite_vector(self.singular_values, field="semantic singular values")
        if (
            projected.shape != (self.class_count, self.rank)
            or bias.shape != (self.class_count,)
            or singular.shape[0] < self.rank
            or np.any(singular[: self.rank] <= 0.0)
        ):
            raise ValueError("semantic subspace tensor shapes differ")
        object.__setattr__(self, "basis", np.ascontiguousarray(basis))
        object.__setattr__(self, "projected_class_weights", np.ascontiguousarray(projected))
        object.__setattr__(self, "classifier_bias", np.ascontiguousarray(bias))
        object.__setattr__(self, "singular_values", np.ascontiguousarray(singular))

    def project(self, values: object) -> np.ndarray:
        """Apply ``Q.T Q`` without allocating the dense projector."""

        array = np.asarray(values, dtype=np.float64)
        if (
            array.ndim < 1
            or array.shape[-1] != self.hidden_size
            or not np.isfinite(array).all()
        ):
            raise ValueError("semantic projection input differs")
        return np.ascontiguousarray((array @ self.basis.T) @ self.basis)

    def logits(self, values: object) -> np.ndarray:
        """Evaluate the frozen centred classifier through the rank basis."""

        array = np.asarray(values, dtype=np.float64)
        if (
            array.ndim < 1
            or array.shape[-1] != self.hidden_size
            or not np.isfinite(array).all()
        ):
            raise ValueError("semantic classifier input differs")
        coordinates = array @ self.basis.T
        return np.ascontiguousarray(
            coordinates @ self.projected_class_weights.T + self.classifier_bias
        )


def derive_semantic_probe_subspace(
    class_weights: object,
    classifier_bias: object,
    *,
    rank: int = SEMANTIC_SUBSPACE_RANK,
) -> SemanticProbeSubspace:
    """Derive the frozen rank-``rank`` semantic row basis in float64."""

    weights = _finite_matrix(class_weights, field="probe class weights")
    bias = _finite_vector(classifier_bias, field="probe classifier bias")
    if (
        isinstance(rank, bool)
        or not isinstance(rank, int)
        or rank <= 0
        or weights.shape[0] != bias.shape[0]
        or weights.shape[0] - 1 < rank
        or weights.shape[1] < rank
    ):
        raise ValueError("probe classifier cannot support the requested semantic rank")
    centred = weights - weights.mean(axis=0, keepdims=True)
    _left, singular_values, right = np.linalg.svd(centred, full_matrices=False)
    if not np.isfinite(singular_values).all() or singular_values.size < rank:
        raise ValueError("semantic SVD did not return the requested finite rank")
    threshold = (
        np.finfo(np.float64).eps
        * max(centred.shape)
        * float(singular_values[0])
    )
    numerical_rank = int(np.count_nonzero(singular_values > threshold))
    if numerical_rank < rank:
        raise ValueError("probe classifier numerical rank is below the requested rank")
    basis = canonicalize_basis_signs(right[:rank])
    validate_orthonormal_rows(basis)
    return SemanticProbeSubspace(
        rank=rank,
        hidden_size=weights.shape[1],
        class_count=weights.shape[0],
        basis=basis,
        projected_class_weights=centred @ basis.T,
        classifier_bias=bias,
        singular_values=singular_values,
    )


def derive_pca_basis(activations: object, *, rank: int = SEMANTIC_SUBSPACE_RANK) -> np.ndarray:
    """Derive a canonical clean-activation PCA control basis in float64."""

    values = _finite_matrix(activations, field="clean PCA activations")
    if (
        isinstance(rank, bool)
        or not isinstance(rank, int)
        or rank <= 0
        or min(values.shape[0] - 1, values.shape[1]) < rank
    ):
        raise ValueError("clean PCA activations cannot support the requested rank")
    centred = values - values.mean(axis=0, keepdims=True)
    _left, singular, right = np.linalg.svd(centred, full_matrices=False)
    threshold = np.finfo(np.float64).eps * max(centred.shape) * float(singular[0])
    if int(np.count_nonzero(singular > threshold)) < rank:
        raise ValueError("clean PCA activations have insufficient numerical rank")
    basis = canonicalize_basis_signs(right[:rank])
    return validate_orthonormal_rows(basis)


def deterministic_haar_basis(
    hidden_size: int,
    *,
    rank: int = SEMANTIC_SUBSPACE_RANK,
    seed: int,
) -> np.ndarray:
    """Generate a deterministic rank-matched Haar control row basis."""

    if (
        isinstance(hidden_size, bool)
        or not isinstance(hidden_size, int)
        or isinstance(rank, bool)
        or not isinstance(rank, int)
        or not 0 < rank <= hidden_size
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
    ):
        raise ValueError("Haar basis dimensions or seed differ")
    matrix = np.random.default_rng(seed).standard_normal((hidden_size, rank))
    columns, _upper = np.linalg.qr(matrix, mode="reduced")
    return validate_orthonormal_rows(canonicalize_basis_signs(columns.T))


def deterministic_complement_basis(
    semantic_basis: object,
    *,
    seed: int,
) -> np.ndarray:
    """Generate a deterministic rank-matched basis orthogonal to semantics."""

    semantic = validate_orthonormal_rows(semantic_basis)
    rank, hidden_size = semantic.shape
    if hidden_size - rank < rank:
        raise ValueError("semantic complement cannot support a rank-matched control")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("semantic complement seed differs")
    matrix = np.random.default_rng(seed).standard_normal((hidden_size, rank))
    matrix = matrix - semantic.T @ (semantic @ matrix)
    columns, upper = np.linalg.qr(matrix, mode="reduced")
    if np.any(np.abs(np.diag(upper)) <= np.finfo(np.float64).eps * hidden_size):
        raise ValueError("semantic complement random draw is rank deficient")
    basis = validate_orthonormal_rows(canonicalize_basis_signs(columns.T))
    if not np.allclose(basis @ semantic.T, 0.0, atol=1e-10, rtol=1e-10):
        raise ValueError("semantic complement is not orthogonal to semantics")
    return basis


def load_probe_layer_classifier(
    artifact: ProbeTransitionArtifact,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Read the validated transition-layer classifier for one registered seed."""

    if not isinstance(artifact, ProbeTransitionArtifact):
        raise TypeError("probe artifact must be validated before classifier loading")
    if seed not in artifact.probe_seeds:
        raise ValueError("semantic classifier seed is outside the parent artifact")
    path = Path(artifact.probe_weights_by_seed[seed])
    if path.is_symlink() or not path.is_file():
        raise ValueError("semantic classifier weights are not one regular file")
    from safetensors import safe_open

    layer = artifact.selected_transition_layer
    with safe_open(path, framework="np") as handle:
        weight = handle.get_tensor(f"decoder_layer.{layer}.weight")
        bias = handle.get_tensor(f"decoder_layer.{layer}.bias")
    if (
        weight.shape != (artifact.class_count, artifact.hidden_size)
        or bias.shape != (artifact.class_count,)
        or weight.dtype != np.float32
        or bias.dtype != np.float32
        or not np.isfinite(weight).all()
        or not np.isfinite(bias).all()
    ):
        raise ValueError("semantic classifier tensor shape, dtype, or values differ")
    return (
        np.ascontiguousarray(weight, dtype=np.float64),
        np.ascontiguousarray(bias, dtype=np.float64),
    )


def derive_artifact_semantic_subspace(
    artifact: ProbeTransitionArtifact,
    *,
    seed: int,
    rank: int = SEMANTIC_SUBSPACE_RANK,
) -> SemanticProbeSubspace:
    """Re-derive, rather than trust, a semantic basis from parent probe weights."""

    weight, bias = load_probe_layer_classifier(artifact, seed=seed)
    return derive_semantic_probe_subspace(weight, bias, rank=rank)


__all__ = [
    "SEMANTIC_SUBSPACE_RANK",
    "SemanticProbeSubspace",
    "canonicalize_basis_signs",
    "derive_pca_basis",
    "derive_artifact_semantic_subspace",
    "derive_semantic_probe_subspace",
    "deterministic_complement_basis",
    "deterministic_haar_basis",
    "load_probe_layer_classifier",
    "validate_orthonormal_rows",
]
