from typing import Callable, Dict, List, Tuple
from torchvision import datasets, models
from torchvision.transforms import v2 as T
from torchvision.tv_tensors import BoundingBoxes, Image
from torch import nn
from torch.utils.data import DataLoader, BatchSampler, SequentialSampler, RandomSampler
import json, os
import torch
import numpy as np
import lightning as L
import lightning.pytorch.loggers as loggers
import lightning.pytorch.callbacks as callbacks
from lightning.pytorch.tuner import Tuner
from PIL import Image, ImageOps, ImageDraw
import random
from pathlib import Path

# Parameters (modify this part as needed)

DATA_DIR = Path("../data/xishen/A_classification/")
TENSORBOARD_LOGS_DIR = "../training/classif"

TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "test"

# Constants

# norm_vals = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # ImageNet normalization
norm_vals = ([ 0.75,0.70,0.65],[ 0.14,0.15,0.16]) # Custom watermark dataset normalization

def unnormalize(x):
    return x*torch.tensor(norm_vals[1])[:, None, None].to(x.device) + torch.tensor(norm_vals[0])[:, None, None].to(x.device)

tforms_common = T.Compose([
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(mean=norm_vals[0], std=norm_vals[1]),
])

sz_resize = 224

tforms_train = T.Compose([
    T.ToImage(),
    T.RandomRotation(7),
    T.RandomResizedCrop(sz_resize),
    T.RandomPhotometricDistort(p=1, saturation=(0., 1.4), brightness=(0.8, 1.2), contrast=(0.6, 1.4)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    tforms_common
])
tforms_val = T.Compose([
    T.ToImage(),
    T.Resize(256),
    T.CenterCrop(sz_resize),
    tforms_common
])

class CosineClassifierHead(nn.Linear):
    def __init__(self, in_features, out_features, dropout_ratio=0.7, scale_factor=5.0):
        super().__init__(in_features, out_features, bias=False)
        self.cbias = nn.Parameter(torch.tensor(0.), requires_grad=True)
        self.scale_factor = scale_factor
        self.dropout = nn.Dropout(dropout_ratio)
    
    def forward(self, input):
        input = self.dropout(input)
        n_input = torch.nn.functional.normalize(input, dim=-1)
        n_weight = torch.nn.functional.normalize(self.weight, dim=-1)
        return self.scale_factor * (
            torch.nn.functional.linear(n_input, n_weight, self.cbias))

train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=tforms_train)
val_dataset = datasets.ImageFolder(root=VAL_DIR, transform=tforms_val)
device = torch.device('cuda')

def log_images(mod, x, prefix="train", model=None):
    N = 4
    for k in range(N):
        im = unnormalize(x[k])
        mod.logger.experiment.add_image(f"{prefix}_img/{k}/", im, mod.current_epoch)

class ViTClassif(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
    
    def forward(self, x):
        return self.model(x)

class SimpleClassifier(L.LightningModule):
    TOT_EPOCHS = 300
    BASE_LR = 1e-3
    BATCH_SIZE = 64

    def __init__(self):
        super().__init__()
        
        self.init_model()
    
        self.criterion = nn.CrossEntropyLoss()
        self.validation_step_outputs = []
        self.lr = self.BASE_LR

    def init_model(self):
        self.model = models.resnet18()
        self.model.fc = CosineClassifierHead(512, len(train_dataset.classes), scale_factor=30.)

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx, log_label="train"):
        x, classes = batch
        x, classes = x.to(device), classes.to(device)
        y = self(x)
        loss = self.criterion(y, classes)
        accuracy = (y.argmax(dim=1) == classes).float().mean()

        self.log(f"{log_label}/loss", loss.mean())
        self.log(f"{log_label}/accuracy", accuracy)
        if batch_idx == 0 and log_label == "train":
            log_images(self, x, "train")
            self.log("learning_rate", self.trainer.optimizers[0].param_groups[0]["lr"])

        return {"loss": loss}

    def train_dataloader(self):
        return DataLoader(train_dataset, self.BATCH_SIZE, True, num_workers=7)
    
    def val_dataloader(self):
        return DataLoader(val_dataset, self.BATCH_SIZE*4, False, num_workers=7)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=self.lr, total_steps=self.TOT_EPOCHS, 
                final_div_factor=10, pct_start=0., anneal_strategy="cos")
        }
    
    def validation_step(self, batch, batch_idx):
        x, classes = batch
        x, classes = x.to(device), classes.to(device)
        y = self(x)
        loss = self.criterion(y, classes).mean()
        accuracy = (y.argmax(dim=1) == classes).float().sum()
        totl = x.size(0)
        self.validation_step_outputs.append((float(loss), float(accuracy), totl))
    
    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        avg_loss = torch.tensor([k[0] for k in outputs]).mean()
        avg_accuracy = torch.tensor([k[1] for k in outputs]).sum() / torch.tensor([k[2] for k in outputs]).sum()
        self.log("val/avg_loss", avg_loss)
        self.log("val/avg_accuracy", avg_accuracy)
        self.log("acc_val", avg_accuracy)
        self.validation_step_outputs = []

if __name__ == "__main__":

    model = SimpleClassifier()
    model.to(device)

    logger = loggers.TensorBoardLogger(TENSORBOARD_LOGS_DIR, name="resnet18+cc")

    checkpointer = callbacks.ModelCheckpoint(
        monitor="acc_val",
        dirpath=os.path.join(logger.log_dir, "checkpoints"),
        filename="model-{epoch:02d}-{acc_val:.2f}",
        save_top_k=3,
        mode="max",
    )

    trainer = L.Trainer(max_epochs=model.TOT_EPOCHS, logger=logger, log_every_n_steps=10, callbacks=[checkpointer])
    trainer.fit(model)
 