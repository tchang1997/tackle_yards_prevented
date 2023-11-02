from argparse import ArgumentParser
import os

import lightning.pytorch as pl
import lightning.pytorch.callbacks
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml

from datasets import PlayByPlayDataset, collate_padded_play_data
from dragonnet.losses import tarreg_loss
import dragonnet.models
from dragonnet.models import DragonNet, TransformerRepresentor
from utils import get_default_text_spinner_context

np.random.seed(42)
torch.manual_seed(42)
class BaseCounterfactualTackleTrainer(pl.LightningModule):
    def __init__(self, model, loss_hparams, optimizer_settings, scheduler_settings=None):
        super().__init__()
        self.model = model
        self.loss_hparams = loss_hparams
        self.optimizer_settings = optimizer_settings
        self.scheduler_settings = scheduler_settings

    def training_step(self, batch, batch_idx):
        t_true = batch["treatment"]
        y_true = batch["target"]
        y0_pred, y1_pred, t_pred, eps = self.model(batch["time_series_features"].float())
        loss_record = tarreg_loss(
            y_true,
            t_true,
            t_pred,
            y0_pred,
            y1_pred,
            eps,
            alpha=self.loss_hparams["alpha"],
            beta=self.loss_hparams["beta"]
        )
        for loss_key, loss_value in loss_record.items():
            self.log(f"train/{loss_key}", loss_value)
        self.log("global_step", self.global_step, on_step=False, on_epoch=True)
        if "loss_total_tarreg" in loss_record:
            return loss_record["loss_total_tarreg"]
        else:
            return loss_record["loss_total"]

    def validation_step(self, batch, batch_idx):   # this is not very DRY of me...but I don't want to mess up any pytorch-lightning stuff under the hood
        t_true = batch["treatment"]
        y_true = batch["target"]
        y0_pred, y1_pred, t_pred, eps = self.model(batch["time_series_features"].float())
        loss_record = tarreg_loss(
            y_true,
            t_true,
            t_pred,
            y0_pred,
            y1_pred,
            eps,
            alpha=self.loss_hparams["alpha"],
            beta=self.loss_hparams["beta"]
        )
        for loss_key, loss_value in loss_record.items():
            self.log(f"val/{loss_key}", loss_value)

        y0_resid = y_true - y0_pred
        y1_resid = y_true - y1_pred
        factual_y0_mae = torch.mean((1 - t_true) * torch.abs(y0_resid))
        factual_y1_mae = torch.mean(t_true * torch.abs(y1_resid))

        factual_y0_mse = torch.mean((1 - t_true) * torch.square(y0_resid))  # squared yards off on missed tackles
        factual_y1_mse = torch.mean(t_true * torch.square(y1_resid))  # squared yards off on successful tackles

        p_t = torch.mean(t_true)
        self.log("val/factual_y0_mae", factual_y0_mae)
        self.log("val/factual_y1_mae", factual_y1_mae)
        self.log("val/factual_mae", (1 - p_t) * factual_y0_mae + p_t * factual_y1_mae)

        # same thing as y0_loss, y1_loss but we take the mean instead of the sum
        self.log("val/factual_y0_mse", factual_y0_mse)
        self.log("val/factual_y1_mse", factual_y1_mse)
        self.log("val/factual_mse", (1 - p_t) * factual_y0_mse + p_t * factual_y1_mse)

        if "loss_total_tarreg" in loss_record:
            return loss_record["loss_total_tarreg"]
        else:
            return loss_record["loss_total"]

    def configure_optimizers(self):
        optimizer = getattr(optim, self.optimizer_settings["name"])(self.parameters(), **self.optimizer_settings["params"])
        if self.scheduler_settings is None:
            # default: Adam with lr 1e-4
            return optimizer
        else:
            lr_scheduler = getattr(optim.lr_scheduler, self.scheduler_settings["name"])(optimizer, **self.scheduler_settings["params"])
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    **self.scheduler_settings["lightning_params"],
                }
            }

def get_callbacks(cfg):
    return [
        getattr(lightning.pytorch.callbacks, callback_dict["name"])(**callback_dict["params"])
        for callback_dict in cfg["callbacks"]
    ]

DEFAULT_CONFIG_PATH = "./configs/defaults.yaml"
if __name__ == '__main__':
    psr = ArgumentParser()
    psr.add_argument("--config", type=str, help="Config YAML file with experimental settings.")
    args = psr.parse_args()

    with open(DEFAULT_CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    if args.config is not None:
        with open(args.config, "r") as f:
            run_cfg = yaml.safe_load(f)
        cfg.update(run_cfg)
        print("Using configuration file", args.config, f"(overriding {DEFAULT_CONFIG_PATH})")
    print(yaml.dump(cfg, allow_unicode=True, default_flow_style=False))

    with get_default_text_spinner_context("Initializing model...") as spinner:
        model_settings = cfg["model_settings"]
        dragonnet_model = DragonNet(
            input_dim=model_settings["input_size"],
            backbone_class=getattr(dragonnet.models, model_settings["representation_class"]),
        )
        spinner.ok("✅ ")

    with get_default_text_spinner_context("Setting up trainer...") as spinner:
        logger = TensorBoardLogger(**cfg["logger"])
        trainer = pl.Trainer(
            logger=logger,
            callbacks=get_callbacks(cfg),
            **cfg["trainer_args"]
        )
        wrapped_module = BaseCounterfactualTackleTrainer(
            dragonnet_model,
            cfg["loss_hparams"],
            cfg["optimizer_settings"],
            scheduler_settings=cfg.get("scheduler_settings", None)
        )
        spinner.ok("✅ ")

    dataloaders = {}
    for split, path in cfg["data"].items():
        with get_default_text_spinner_context(f"Loading {split} split from {path}...") as spinner:
            dataset = PlayByPlayDataset(path)
            dataloaders[split] = DataLoader(dataset, collate_fn=collate_padded_play_data, **cfg["dataloader_settings"])
            spinner.ok(f"✅ ({len(dataset)} examples)")

    try:
        trainer.fit(
            model=wrapped_module,
            train_dataloaders=dataloaders["train"],
            val_dataloaders=dataloaders["val"]
        )
    except Exception as e:
        print("Raised exception", e, "in training")
    finally:
        with get_default_text_spinner_context("Saving config...") as spinner:
            final_config_path = os.path.join(logger.log_dir, "config.yml")
            with open(final_config_path, "w") as f:
                yaml.dump(cfg, f, default_flow_style=False)
            spinner.ok(f"✅ (saved at {final_config_path})")
