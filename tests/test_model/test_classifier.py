import torch
from src.base import ID2RELATIONS, RELATIONS2ID
from src.model.classifier import ContextClassifier
from transformers import LlamaConfig


# @pytest.mark.skip(reason="Skipping due to slow loading times")
class TestClassifier:
    def test_classifier_one_example(self):
        model_config = LlamaConfig.from_pretrained(
            "HuggingFaceTB/SmolLM2-135M",
            num_labels=len(RELATIONS2ID),
            finetuning_task="text-classification",
            label2id=RELATIONS2ID,
            id2label=ID2RELATIONS,
        )
        model_config.tokens_to_encode_ids = [1, 2, 3, 4]
        model_config.pad_token_id = model_config.eos_token_id
        model_config.n_hidden_layers = 1
        model = ContextClassifier.from_pretrained(
            "HuggingFaceTB/SmolLM2-135M",
            config=model_config,
        )

        inputs = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]]),
        }
        outputs = model(**inputs)
        assert outputs.logits.shape == (1, len(RELATIONS2ID))

    def test_n_hidden_layers(self):
        model_config = LlamaConfig.from_pretrained(
            "HuggingFaceTB/SmolLM2-135M",
            num_labels=len(RELATIONS2ID),
            finetuning_task="text-classification",
            label2id=RELATIONS2ID,
            id2label=ID2RELATIONS,
        )
        model_config.tokens_to_encode_ids = [1, 2, 3, 4]
        model_config.pad_token_id = model_config.eos_token_id
        model_config.n_hidden_layers = 2
        model = ContextClassifier.from_pretrained(
            "HuggingFaceTB/SmolLM2-135M",
            config=model_config,
        )
        assert len(model.score) == 3  # 2 hidden layers + 1 activation

    def test_classifier_batch_example(self):
        model_config = LlamaConfig.from_pretrained(
            "HuggingFaceTB/SmolLM2-135M",
            num_labels=len(RELATIONS2ID),
            finetuning_task="text-classification",
            label2id=RELATIONS2ID,
            id2label=ID2RELATIONS,
        )
        model_config.tokens_to_encode_ids = [1, 2, 3, 4]
        model_config.pad_token_id = model_config.eos_token_id
        model_config.n_hidden_layers = 1
        model = ContextClassifier.from_pretrained(
            "HuggingFaceTB/SmolLM2-135M",
            config=model_config,
        )

        inputs = {
            "input_ids": torch.tensor(
                [
                    [1, 2, 3, 4, 5, 0, 0],
                    [1, 2, 3, 4, 5, 6, 7],
                ]
            ),
            "attention_mask": torch.tensor(
                [
                    [1, 1, 1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1],
                ]
            ),
        }
        outputs = model(**inputs)
        assert outputs.logits.shape == (2, len(RELATIONS2ID))
