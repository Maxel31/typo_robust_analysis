"""few-shot キャッシュ機構のテスト。"""

from quant_typo_neuron.benchmarks.base import BenchmarkExample
from quant_typo_neuron.few_shot import FewShotCache


def _make_pool(n=20):
    return [
        BenchmarkExample(id=f"ex{i}", question=f"Q{i}?", choices=["A", "B"], answer=0)
        for i in range(n)
    ]


def test_get_or_create_returns_correct_count(tmp_path):
    cache = FewShotCache(cache_dir=tmp_path)
    pool = _make_pool()
    shots = cache.get_or_create("test_bench", num_shots=5, seed=42, pool=pool)
    assert len(shots) == 5
    assert all(isinstance(s, BenchmarkExample) for s in shots)


def test_get_or_create_deterministic(tmp_path):
    cache = FewShotCache(cache_dir=tmp_path)
    pool = _make_pool()
    shots1 = cache.get_or_create("test_bench", num_shots=5, seed=42, pool=pool)
    shots2 = cache.get_or_create("test_bench", num_shots=5, seed=42, pool=pool)
    assert [s.id for s in shots1] == [s.id for s in shots2]


def test_cache_file_created(tmp_path):
    cache = FewShotCache(cache_dir=tmp_path)
    cache.get_or_create("mybench", num_shots=3, seed=0, pool=_make_pool())
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    assert "mybench" in files[0].name


def test_cache_persists_across_instances(tmp_path):
    pool = _make_pool()
    cache1 = FewShotCache(cache_dir=tmp_path)
    shots1 = cache1.get_or_create("bench", num_shots=3, seed=0, pool=pool)
    cache2 = FewShotCache(cache_dir=tmp_path)
    shots2 = cache2.get_or_create("bench", num_shots=3, seed=0, pool=pool)
    assert [s.id for s in shots1] == [s.id for s in shots2]


def test_zero_shots_returns_empty(tmp_path):
    cache = FewShotCache(cache_dir=tmp_path)
    shots = cache.get_or_create("bench", num_shots=0, seed=0, pool=_make_pool())
    assert shots == []


def test_pool_larger_than_requested(tmp_path):
    cache = FewShotCache(cache_dir=tmp_path)
    pool = _make_pool(3)
    shots = cache.get_or_create("bench", num_shots=10, seed=0, pool=pool)
    assert len(shots) == 3
