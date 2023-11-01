from argparse import ArgumentParser

import lightning.pytorch as pl
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from yaspin import yaspin

from datasets import PlayByPlayDataset, collate_padded_play_data
from dragonnet.losses import tarreg_loss
import dragonnet.models
from dragonnet.models import DragonNet, TransformerRepresentor

class BaseCounterfactualTackleTrainer(pl.LightningModule):
    def __init__(self, model, loss_hparams):
        super().__init__()
        self.model = model
        self.loss_hparams = loss_hparams

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

        if "loss_total_tarreg" in loss_record:
            return loss_record["loss_total_tarreg"]
        else:
            return loss_record["loss_total"]

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

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

    dataloaders = {}
    for split, path in cfg["data"].items():
        with yaspin(text=f"Loading {split} split from {path}...", color="green", timer=True) as spinner:
            dataset = PlayByPlayDataset(path)
            dataloaders[split] = DataLoader(dataset, collate_fn=collate_padded_play_data, **cfg["dataloader_settings"])
            spinner.ok("✅ ")

    with yaspin(text="Initializing model...", color="green", timer=True) as spinner:
        model_settings = cfg["model_settings"]
        dragonnet_model = DragonNet(
            input_dim=model_settings["input_size"],
            backbone_class=getattr(dragonnet.models, model_settings["representation_class"]),
        )
        spinner.ok("✅ ")

    trainer = pl.Trainer(num_sanity_val_steps=2, overfit_batches=4, max_epochs=10)
    wrapped_module = BaseCounterfactualTackleTrainer(dragonnet_model, cfg["loss_hparams"])
    trainer.fit(model=wrapped_module, train_dataloaders=dataloaders["train"])
