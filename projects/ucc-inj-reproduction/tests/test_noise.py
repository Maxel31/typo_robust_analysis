from ucc_inj_reproduction.noise import inject_variation_selector_noise, variation_selector_for_byte


def test_noise_level_zero_is_identity() -> None:
    assert inject_variation_selector_noise("abc", noise_level=0, seed=42) == "abc"


def test_noise_is_deterministic_and_adds_exact_selectors() -> None:
    first = inject_variation_selector_noise("aB", noise_level=3, seed=9)
    assert first == inject_variation_selector_noise("aB", noise_level=3, seed=9)
    assert len(first) == 2 + 2 * 3


def test_variation_selector_byte_boundaries() -> None:
    assert ord(variation_selector_for_byte(0)) == 0xFE00
    assert ord(variation_selector_for_byte(15)) == 0xFE0F
    assert ord(variation_selector_for_byte(16)) == 0xE0100
    assert ord(variation_selector_for_byte(255)) == 0xE01EF
