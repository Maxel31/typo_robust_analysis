from quant_typo_neuron.contracts import ItemResult, QuantVariant


def test_item_result_roundtrip():
    r = ItemResult(
        model="llama-3.2-1b-instruct",
        method="rtn",
        bit=4,
        typo_type="sub_keyboard",
        eps=1,
        dataset="gsm8k",
        seed=0,
        item_id="gsm8k-0001",
        correct_clean=1,
        correct_typo=0,
        conf=0.8,
    )
    assert ItemResult.from_dict(r.to_dict()) == r


def test_quant_variant_defaults():
    v = QuantVariant(name="fp16", method="fp16")
    assert v.bits is None
    assert v.group_size is None
