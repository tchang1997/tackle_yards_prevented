from lightning.pytorch.callbacks import BaseFinetuning

class TwoStageTraining(BaseFinetuning):
    def __init__(self, unfreeze_at_epoch=5, finetune_factor=100., freeze_backbone=True, freeze_propensity_model=False, verbose=False):
        super().__init__()
        self._unfreeze_at_epoch = unfreeze_at_epoch
        self.freeze_backbone = freeze_backbone
        self.freeze_propensity_model = freeze_propensity_model
        self.finetune_factor = finetune_factor
        self.verbose = verbose

    def freeze_before_training(self, pl_module):
        self.freeze(pl_module.model.y0_model)
        self.freeze(pl_module.model.y1_model)

    def finetune_function(self, pl_module, current_epoch, optimizer):
        if current_epoch == self._unfreeze_at_epoch:
            if self.verbose:
                print("Current epoch is", current_epoch, "-- unfreezing outcome models")
            self.unfreeze_and_add_param_group(
                modules=pl_module.model.y0_model,
                optimizer=optimizer,
                initial_denom_lr=self.finetune_factor,
            )
            self.unfreeze_and_add_param_group(
                modules=pl_module.model.y1_model,
                optimizer=optimizer,
                initial_denom_lr=self.finetune_factor,
            )
            if self.freeze_backbone:
                self.freeze(pl_module.model.backbone)
                if self.verbose:
                    print("Current epoch is", current_epoch, "-- backbone frozen")
            if self.freeze_propensity_model:
                self.freeze(pl_module.model.propensity_model)
                if self.verbose:
                    print("Current epoch is", current_epoch, "-- propensity model frozen")
