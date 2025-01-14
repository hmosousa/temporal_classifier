import pytest

from src.model import load_model


@pytest.mark.skip(reason="One time tests. Skipping to avoid long running times.")
def test_load_model_custom_classifier():
    model = load_model("classifier", "hugosousa/smol-135-tq")

    text = "This is an <start_source>example</start_source> with <end_target>two</end_target> entities."
    input_ids = model.tokenizer.encode(text)
    assert model.tokenizer.encode("<start_source>")[0] in input_ids
    assert model.tokenizer.encode("</start_source>")[0] in input_ids
    assert model.tokenizer.encode("<end_target>")[0] in input_ids
    assert model.tokenizer.encode("</end_target>")[0] in input_ids

    pred = model(text)
    assert pred is not None


@pytest.mark.skip(reason="One time tests. Skipping to avoid long running times.")
def test_load_model_hf_classifier():
    model = load_model("hf_classifier", "hugosousa/smol-135-tq-closure-augment")

    text = "This is an <start_source>example</start_source> with <end_target>two</end_target> entities."
    input_ids = model.tokenizer.encode(text)
    assert model.tokenizer.encode("<start_source>")[0] in input_ids
    assert model.tokenizer.encode("</start_source>")[0] in input_ids
    assert model.tokenizer.encode("<end_target>")[0] in input_ids
    assert model.tokenizer.encode("</end_target>")[0] in input_ids

    pred = model(text)
    assert pred is not None
