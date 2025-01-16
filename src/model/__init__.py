import torch
from transformers import AutoTokenizer, pipeline

from src.model.classifier import ContextClassifier
from src.model.gemini import GeminiAPI
from src.model.majority import MajorityClassifier
from src.model.random import RandomClassifier


def load_model(model_name: str, model_path: str, revision: str = "main"):
    match model_name:
        case "classifier":
            tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision)
            model = ContextClassifier.from_pretrained(model_path, revision=revision)
            return pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                revision=revision,
            )
        case "hf_classifier":
            return pipeline(
                "text-classification",
                model=model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                revision=revision,
            )
        case "gemini":
            return GeminiAPI()
        case _:
            raise ValueError(f"Model {model_name} not found")


__all__ = [
    "load_model",
    "ContextClassifier",
    "Gemini",
    "MajorityClassifier",
    "RandomClassifier",
]
