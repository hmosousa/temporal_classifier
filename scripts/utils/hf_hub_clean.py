"""
This script is intended to delete a list of models from the Hugging Face Hub.
"""

import huggingface_hub


api = huggingface_hub.HfApi()

models = api.list_models(author="hugosousa")
for model in models:
    if model.id.startswith("hugosousa/debug"):
        huggingface_hub.delete_repo(model.id)
