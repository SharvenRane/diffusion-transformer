"""
Training entry point.
"""
import hydra
from omegaconf import DictConfig
from src.model import build_model
from src.dataset import get_dataloader
from src.trainer import Trainer


@hydra.main(config_path="configs", config_name="default", version_base=None)
def main(config: DictConfig):
    train_loader = get_dataloader(config.data_root, "train", config)
    val_loader = get_dataloader(config.data_root, "val", config)
    model = build_model(config)
    trainer = Trainer(model, config, train_loader, val_loader)
    trainer.train()


if __name__ == "__main__":
    main()
