from argparse import ArgumentParser
import glob
import os
import pickle
import yaml

import lightning.pytorch as pl
from torch.utils.data import DataLoader

from datasets import PlayByPlayDataset, COLLATE_FN_DICT
from main import get_data_split, get_torch_model
from trainer import BaseCounterfactualTackleTrainer
from utils import get_default_text_spinner_context

def load_checkpoint(ckpt_path, rank, cfg):
    ckpts = sorted(glob.glob(os.path.join(ckpt_path, "checkpoints/**/*.ckpt")))
    print("Found checkpoints:")
    print(" - ", end="")
    print(*ckpts, sep="\n - ")
    print()

    ckpts_final = sorted(ckpts, key=lambda p: float(os.path.splitext(os.path.basename(p))[0].split("=")[-1]))
    final_ckpt = ckpts_final[rank - 1]
    print(f"Loading #{rank}-ranked checkpoint:", os.path.basename(final_ckpt))
    model = BaseCounterfactualTackleTrainer.load_from_checkpoint(
        final_ckpt,
        model=get_torch_model(cfg["model_settings"]),
        loss_hparams=cfg["loss_hparams"],
        optimizer_settings=cfg["optimizer_settings"],
        scheduler_settings=cfg.get("scheduler_settings", None)
    )
    model.eval()
    return model

def report_on_results(test_results):
    for k, v in test_results.items():
        if len(v.size()) == 0:
            if "loss" not in k:
                metric_str = k + ": " + str(v.item())
                if k.endswith("mae"):
                    metric_str += " yds"
                elif k.endswith("mse"):
                    metric_str += " yds^2"
                print(metric_str)

if __name__ == '__main__':
    psr = ArgumentParser()
    psr.add_argument("--ckpt-path", type=str, required=True)
    psr.add_argument("--split", type=str, default="val")
    psr.add_argument("--data-path", type=str, default="./data/nfl-big-data-bowl-2024/play_by_play_{}.pkl")
    psr.add_argument("--rank", type=int, default=1)
    args = psr.parse_args()

    with open(os.path.join(args.ckpt_path, "config.yml"), "r") as f:
        cfg = yaml.safe_load(f)
    model = load_checkpoint(args.ckpt_path, args.rank, cfg)

    collate_fn = COLLATE_FN_DICT[cfg["model_settings"]["representation_class"]]
    dataloader = get_data_split(args.split, args.data_path.format(args.split), collate_fn, cfg["dataloader_settings"])
    trainer = pl.Trainer(logger=None)
    trainer.test(model, dataloaders=dataloader)  # val and test is the same procedure for us, but with different returns

    test_results = trainer.model.test_results
    report_on_results(test_results)

    save_path = os.path.join(args.ckpt_path, "results.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(test_results, f)
        print("Saved evaluation results to", save_path)
