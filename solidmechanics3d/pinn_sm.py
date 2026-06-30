import os
import sys
import argparse
from contextlib import contextmanager

import yaml
import h5py
import numpy as np
import torch
import torch.multiprocessing as mp
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from utils_sm_unet import FEMPhysicsModule, SimplifiedOutputCallback


torch.set_float32_matmul_precision("medium")


# ----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train NewPINNs on 3D linear elasticity (YAML-configured)"
    )
    parser.add_argument("--config", type=str, default="config_sm.yaml",
                        help="Path to YAML config file")
    return parser.parse_args()


# ----------------------------------------------------------------------
class ElasticityDataset(torch.utils.data.Dataset):
    """
    Returns
    -------
    For training:
        img1 : tensor (2, nz, ny, nx)   channels: [ν_norm, mask]
    For validation:
        img1, Ueq, Veq, Weq            displacement ground truth shapes (nz, ny, nx)
    """

    def __init__(self, file_path):
        self.file_path = file_path
        with h5py.File(file_path, "r") as f:
            self.num_samples = f["img1"].shape[0]
            self.mask = torch.from_numpy(f["mask"][...]).float()
            self.has_gt = "Ueq" in f

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        with h5py.File(self.file_path, "r") as f:
            nu_norm = torch.from_numpy(f["img1"][idx]).float()  # (nz, ny, nx)
            # Stack ν_norm and mask as channels.
            img1 = torch.stack([nu_norm, self.mask], dim=0)     # (2, nz, ny, nx)
            if self.has_gt:
                Ueq = torch.from_numpy(f["Ueq"][idx]).float()
                Veq = torch.from_numpy(f["Veq"][idx]).float()
                Weq = torch.from_numpy(f["Weq"][idx]).float()
                return img1, Ueq, Veq, Weq
            return img1


# ----------------------------------------------------------------------
def run_training(config_path):
    mp.set_start_method("spawn", force=True)

    num_threads = os.cpu_count()
    print(f"You have access to {num_threads} CPU threads.")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    print("\n=== YAML Configuration File Contents ===")
    with open(config_path, "r") as file:
        print(file.read())
    print("======================================\n")

    # ---- config extraction ----
    train_num_samples = config["data"]["train_num_samples"]
    valid_num_samples = config["data"]["valid_num_samples"]
    model_config = config.get("model", {}).get("unet_config", None)

    train_file_path = f"./data/train_data_{train_num_samples}.h5"
    valid_file_path = f"./data/valid_data_{valid_num_samples}.h5"
    print(f"train_file_path = {train_file_path}")
    print(f"valid_file_path = {valid_file_path}")

    model_type = config["model"]["type"]
    learning_rate = config["model"]["learning_rate"]

    save_dir = config["training"]["save_dir"]
    read_ckpt = config["training"]["read_ckpt"]
    if read_ckpt == "None":
        read_ckpt = None

    fem_iterations = config["training"]["fem_iterations"]
    Tmax = config["training"]["Tmax"]
    lambda_u = config["training"]["lambda_u"]
    lambda_v = config["training"]["lambda_v"]
    lambda_w = config["training"]["lambda_w"]
    num_epochs = config["training"]["num_epochs"]
    batch_size = config["training"]["batch_size"]

    enable_progress_bar  = config["training"].get("enable_progress_bar", False)
    enable_model_summary = config["training"].get("enable_model_summary", False)
    enable_validation    = config["training"].get("enable_validation", True)
    num_gpus             = config["training"].get("num_gpus", 1)
    precision            = config["training"].get("precision", "16-mixed")

    # ---- datasets / loaders ----
    train_dataset = ElasticityDataset(train_file_path)
    valid_dataset = ElasticityDataset(valid_file_path)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=config["dataloader"].get("num_workers", 1),
        pin_memory=config["dataloader"].get("pin_memory", True),
        persistent_workers=config["dataloader"].get("persistent_workers", True),
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False,
        num_workers=config["dataloader"].get("num_workers", 1),
        pin_memory=config["dataloader"].get("pin_memory", True),
        persistent_workers=config["dataloader"].get("persistent_workers", True),
    )

    # ---- model ----
    model = FEMPhysicsModule(
        model_type=model_type,
        learning_rate=learning_rate,
        fem_iterations=fem_iterations,
        Tmax=Tmax,
        lambda_u=lambda_u,
        lambda_v=lambda_v,
        lambda_w=lambda_w,
        model_config=model_config,
    )

    if read_ckpt is not None:
        print(f"Loading weights from checkpoint: {read_ckpt}")
        #checkpoint = torch.load(read_ckpt, map_location=lambda s, l: s)
        #model.load_state_dict(checkpoint["state_dict"])
        print("Checkpoint loaded successfully!")

    # ---- callbacks ----
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        filename=config["checkpoint"].get("filename", "model_{epoch}"),
        save_top_k=config["checkpoint"].get("save_top_k", -1),
        monitor=config["checkpoint"].get("monitor", "train_loss"),
        mode=config["checkpoint"].get("mode", "min"),
        save_last=config["checkpoint"].get("save_last", True),
        every_n_epochs=config["checkpoint"].get("every_n_epochs", 10),
    )
    output_callback = SimplifiedOutputCallback(num_epochs)

    logger = TensorBoardLogger(save_dir=os.path.join(save_dir, "logs"))

    # ---- trainer ----
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        callbacks=[checkpoint_callback, output_callback],
        logger=logger,
        log_every_n_steps=config["training"].get("log_every_n_steps", 1),
        accelerator=config["training"].get("accelerator", "auto"),
        devices=num_gpus,
        strategy=config["training"].get("strategy", "ddp"),
        precision=precision,
        enable_progress_bar=enable_progress_bar,
        enable_model_summary=enable_model_summary,
        accumulate_grad_batches=config["training"].get("accumulate_grad_batches", 64),
    )

    if enable_validation:
        #trainer.fit(model, train_loader, valid_loader)
        trainer.fit(model, train_loader, valid_loader, ckpt_path=read_ckpt)
    else:
        #trainer.fit(model, train_loader, None)
        trainer.fit(model, train_loader, None, ckpt_path=read_ckpt)

    print("Done!")


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


if __name__ == "__main__":
    args = parse_args()
    run_training(args.config)