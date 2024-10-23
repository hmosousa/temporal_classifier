from fire import Fire
from omegaconf import OmegaConf

from src.constants import CONFIGS_DIR
from src.data import load_dataset
from src.model import load_model
from src.trainer import load_trainer


def main(config_name: str = "classifier"):
    """Train a model with a given configuration.

    Args:
        config_name: The name of the configuration file to use. Check `configs/` for available configurations.
    """
    config = OmegaConf.load(CONFIGS_DIR / f"{config_name}.yaml")

    model, tokenizer = load_model(
        model_name=config.model.type, config=config.model.params
    )

    trainer = load_trainer(
        trainer_name=config.trainer.type,
        model=model,
        tokenizer=tokenizer,
        config=config.trainer.params,
    )

    trainset, validset = load_dataset(
        dataset_name=config.dataset.type, config=config.dataset.params
    )

    trainer.train(trainset, validset)


if __name__ == "__main__":
    Fire(main)
