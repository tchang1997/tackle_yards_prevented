import lightning.pytorch as pl
import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
from torchmetrics.aggregation import CatMetric, MeanMetric
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score

import dragonnet.losses

np.random.seed(42)
torch.manual_seed(42)

TRAIN_Y_MEAN = 3.4594572025052197
class BaseCounterfactualTackleTrainer(pl.LightningModule):
    def __init__(self, model, loss_hparams, optimizer_settings, scheduler_settings=None, loss_fn="tarreg_loss"):
        super().__init__()
        self.model = model
        self.loss_hparams = loss_hparams
        self.optimizer_settings = optimizer_settings
        self.scheduler_settings = scheduler_settings
        self.loss_fn = getattr(dragonnet.losses, loss_fn)

        self.train_auroc = torchmetrics.AUROC(task="binary")
        self.val_auroc = torchmetrics.AUROC(task="binary")

        self.test_outputs = []

        self.test_auroc = torchmetrics.AUROC(task="binary")
        self.t_true_agg = CatMetric()
        self.y_true_agg = CatMetric()
        self.y0_pred_agg = CatMetric()
        self.y1_pred_agg = CatMetric()
        self.t_pred_agg = CatMetric()
        self.eps_agg = CatMetric()
        self.cate_agg = CatMetric()

        self.loss_t_mean = MeanMetric()
        self.loss_y0_mean = MeanMetric()
        self.loss_y1_mean = MeanMetric()
        self.loss_y_overall_mean = MeanMetric()
        self.loss_total_mean = MeanMetric()
        self.loss_total_tarreg_mean = MeanMetric()
        self.tarreg_mean = MeanMetric()

        regression_stack = torchmetrics.MetricCollection({
            "mae": MeanAbsoluteError(),
            "mse": MeanSquaredError(),
            "r2": R2Score(),
        })
        self.factual_y0_metrics = regression_stack.clone(prefix="factual_y0_")
        self.factual_y1_metrics = regression_stack.clone(prefix="factual_y1_")
        self.factual_metrics = regression_stack.clone(prefix="factual_")
        self.zero_metrics = regression_stack.clone(prefix="zero_baseline_")
        self.const_metrics = regression_stack.clone(prefix="const_baseline_")

        self.test_results = None

    def training_step(self, batch, batch_idx):
        t_true = batch["treatment"]
        y_true = batch["target"]

        y0_pred, y1_pred, t_pred, eps = self.model(batch["time_series_features"])
        loss_record = self.loss_fn(
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
        self.train_auroc.update(t_pred, t_true)
        if "loss_total_tarreg" in loss_record:
            return loss_record["loss_total_tarreg"]
        else:
            return loss_record["loss_total"]

    def validation_step(self, batch, batch_idx):   # this is not very DRY of me...but I don't want to mess up any pytorch-lightning stuff under the hood
        t_true = batch["treatment"]
        y_true = batch["target"]
        y0_pred, y1_pred, t_pred, eps = self.model(batch["time_series_features"])
        loss_record = self.loss_fn(
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
        self.val_auroc.update(t_pred, t_true)

        y0_resid = y_true - y0_pred
        y1_resid = y_true - y1_pred
        factual_y0_mae = torch.mean(torch.abs(y0_resid[t_true == 0]))
        factual_y1_mae = torch.mean(torch.abs(y1_resid[t_true == 1]))

        factual_y0_mse = torch.mean(torch.square(y0_resid[t_true == 0]))  # squared yards off on missed tackles
        factual_y1_mse = torch.mean(torch.square(y1_resid[t_true == 1]))  # squared yards off on successful tackles

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

    def test_step(self, batch, batch_idx):   # this is not very DRY of me...but I don't want to mess up any pytorch-lightning stuff under the hood
        t_true = batch["treatment"]
        y_true = batch["target"]
        y0_pred, y1_pred, t_pred, eps = self.model(batch["time_series_features"])
        loss_record = self.loss_fn(
            y_true,
            t_true,
            t_pred,
            y0_pred,
            y1_pred,
            eps,
            alpha=self.loss_hparams["alpha"],
            beta=self.loss_hparams["beta"]
        )
        self.test_auroc.update(t_pred, t_true)

        y0_resid = y_true - y0_pred
        y1_resid = y_true - y1_pred
        factual_y0_mae = torch.mean((1 - t_true) * torch.abs(y0_resid))
        factual_y1_mae = torch.mean(t_true * torch.abs(y1_resid))

        factual_y0_mse = torch.mean((1 - t_true) * torch.square(y0_resid))  # squared yards off on missed tackles
        factual_y1_mse = torch.mean(t_true * torch.square(y1_resid))  # squared yards off on successful tackles

        p_t = torch.mean(t_true)

        self.t_true_agg.update(t_true)
        self.y_true_agg.update(y_true)
        self.y0_pred_agg.update(y0_pred)
        self.y1_pred_agg.update(y1_pred)
        self.t_pred_agg.update(t_pred)
        self.eps_agg.update(eps)
        self.cate_agg.update(y1_pred - y0_pred)

        self.loss_t_mean.update(loss_record["loss_t"])
        self.loss_y0_mean.update(loss_record["loss_y0"])
        self.loss_y1_mean.update(loss_record["loss_y1"])
        self.loss_y_overall_mean.update(loss_record["loss_y_overall"])
        self.loss_total_mean.update(loss_record["loss_total"])
        if "tarreg" in loss_record:
            self.loss_total_tarreg_mean.update(loss_record["loss_total_tarreg"])
            self.tarreg_mean.update(loss_record["tarreg"])

        self.factual_y0_metrics.update(y0_pred[t_true == 0], y_true[t_true == 0])
        self.factual_y1_metrics.update(y1_pred[t_true == 1], y_true[t_true == 1])
        self.factual_metrics.update((1 - t_true) * y0_pred + t_true * y1_pred, y_true)
        self.zero_metrics.update(torch.zeros_like(y_true), y_true)
        self.const_metrics.update(torch.ones_like(y_true) * TRAIN_Y_MEAN, y_true)

    def on_train_epoch_end(self):
        self.log("train/t_auroc", self.train_auroc.compute())
        self.train_auroc.reset()

    def on_validation_epoch_end(self):
        self.log("val/t_auroc", self.val_auroc.compute())
        self.val_auroc.reset()

    def on_test_epoch_end(self):
        results_dict = {
            "t_true": self.t_true_agg.compute(),
            "y_true": self.y_true_agg.compute(),
            "y0_pred": self.y0_pred_agg.compute(),
            "y1_pred": self.y1_pred_agg.compute(),
            "t_pred": self.t_pred_agg.compute(),
            "eps": self.eps_agg.compute(),
            "cate": self.cate_agg.compute(),
            "t_loss": self.loss_t_mean.compute(),
            "y0_loss": self.loss_y0_mean.compute(),
            "y1_loss": self.loss_y1_mean.compute(),
            "y_loss": self.loss_y_overall_mean.compute(),
            "loss_total": self.loss_total_mean.compute(),
            "loss_total_tarreg": self.loss_total_tarreg_mean.compute(),
            "tarreg": self.tarreg_mean.compute(),
            "t_auroc": self.test_auroc.compute(),
        }
        results_dict.update(
            **self.factual_y0_metrics.compute(),
            **self.factual_y1_metrics.compute(),
            **self.factual_metrics.compute(),
            **self.zero_metrics.compute(),
            **self.const_metrics.compute(),
        )
        for k, v in results_dict.items():
            results_dict[k] = v.cpu()
        self.test_results = results_dict

    def configure_optimizers(self):
        requires_grad_params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = getattr(optim, self.optimizer_settings["name"])(requires_grad_params, **self.optimizer_settings["params"])
        if self.scheduler_settings is None:
            return optimizer
        else:
            lr_scheduler = getattr(optim.lr_scheduler, self.scheduler_settings["name"])(optimizer, **self.scheduler_settings["params"])
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    **self.scheduler_settings["params"],
                }
            }
