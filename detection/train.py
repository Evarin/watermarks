from torchvision import datasets
from torchvision.transforms import v2 as transforms
from torchvision.tv_tensors import BoundingBoxes, Image
import json, os
import torch
import lightning as L
import lightning.pytorch.loggers as loggers
from pl_bolts.models.detection import FasterRCNN
from PIL import Image, ImageOps, ImageDraw
from pathlib import Path

# Parameters (modify this part as needed)
DATASET_ANNOTATIONS_DIR = "../data/datasets/detection/"
DATASET_IMAGES_DIR = "../data/"
TENSORBOARD_LOGS_DIR = "../training/rcnn"
device = torch.device('cuda')

# Constants

DATASET_TRAIN_JSON = Path(DATASET_ANNOTATIONS_DIR) / "train.json"
DATASET_VAL_JSON = Path(DATASET_ANNOTATIONS_DIR) / "val.json"

TOT_EPOCHS = 100
BASE_LR = 1e-4

# Helper functions

def draw_bboxes(img, bboxes, color, thickness=3):
    """
    Helper function to draw bounding boxes on an image.
    """
    draw = ImageDraw.Draw(img)
    for box in bboxes:
        draw.rectangle([int(t) for t in box], outline=color, width=thickness)

@torch.no_grad()
def log_images(mod, batch, prefix="train", model=None):
    x, bboxes = batch
    N = 4
    if model is not None:
        preds = model(x[:N])

    for k in range(N):
        img = transforms.ToPILImage()(x[k]).convert("RGB")
        draw_bboxes(img, bboxes[k]["boxes"], (255, 0, 0))
        if model is not None:
            draw_bboxes(img, preds[k]["boxes"], (0, 0, 255))
        im = transforms.ToTensor()(img)
        mod.logger.experiment.add_image(f"{prefix}_img/{k}/", im, mod.current_epoch)

# Define transformations

tforms_common = transforms.Compose([
    transforms.ToDtype(torch.float32, scale=True),
    #transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Already integrated to FasterRCNN
])
tforms_train = transforms.Compose([
    transforms.ToImage(),
    transforms.RandomPhotometricDistort(p=1, saturation=(0., 1.5), brightness=(0.6, 1.2)),
    transforms.RandomZoomOut(fill={Image: (123, 117, 104), "others": 0}, side_range=(1., 2.)),
    transforms.RandomIoUCrop(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomResizedCrop(640),
    transforms.SanitizeBoundingBoxes(),#labels_getter=lambda k: torch.zeros(1)),
    tforms_common
])
tforms_val = transforms.Compose([
    transforms.ToImage(),
    transforms.Resize(900),
    transforms.CenterCrop(900),
    tforms_common
])

# A dataset compatible with our annotations

class WatermarkDetectionDataset(datasets.VisionDataset):
    def __init__(self, root: str, src_json: str, transform=None):
        super().__init__(root, transform=transform)
        self.samples = []
        with open(src_json) as f:
            self.samples = json.load(f)

    def __getitem__(self, idx):
        data = self.samples[idx]
        img = Image.open(os.path.join(self.root, data["image"]))
        img = ImageOps.exif_transpose(img)
        w, h = img.size
        bbox = [data["x0"]*w, data["y0"]*h, data["x1"]*w, data["y1"]*h]
        bbox = BoundingBoxes([bbox], format="xyxy", canvas_size=(h, w))
        bbox = {"boxes": bbox, "labels": torch.ones(1, dtype=torch.int64)}
        img, bbox = self.transform(img, bbox)
        return img, bbox
    
    def __len__(self) -> int:
        return len(self.samples)

train_dataset = WatermarkDetectionDataset(root=DATASET_IMAGES_DIR, src_json=DATASET_TRAIN_JSON, transform=tforms_train)
val_dataset = WatermarkDetectionDataset(root=DATASET_IMAGES_DIR, src_json=DATASET_VAL_JSON, transform=tforms_val)

# Main model

class SimpleDetector(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = FasterRCNN(num_classes=2, backbone="resnet18", pretrained_backbone=True, trainable_backbone_layers=5)
        self.validation_step_outputs = []
        self.lr = BASE_LR

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        data = self.model.training_step(batch, batch_idx)
        self.log("train_loss", data["loss"])
        if batch_idx == 0:
            self.model.eval()
            log_images(self, batch, "train", self.model)
            self.model.train()
        self.log("learning_rate", self.trainer.optimizers[0].param_groups[0]["lr"])
        return data
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(train_dataset, batch_size=12, num_workers=8, shuffle=True, collate_fn=lambda batch: tuple(zip(*batch)))
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(val_dataset, batch_size=24, num_workers=8, shuffle=False, collate_fn=lambda batch: tuple(zip(*batch)))
    
    def configure_optimizers(self):
        #return self.model.configure_optimizers()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.0005)
        return {
            "optimizer": optimizer,
            "lr_scheduler": torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.lr, total_steps=TOT_EPOCHS, pct_start=0.1, anneal_strategy="cos")
        }
    
    def validation_step(self, batch, batch_idx):
        data = self.model.validation_step(batch, batch_idx)
        self.validation_step_outputs.append(float(data["val_iou"]))
        if batch_idx == 0:
            log_images(self, batch, "val", self.model)
        return data
    
    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        avg_iou = torch.tensor(outputs).mean()
        self.log("avg_val_iou", avg_iou)
        self.validation_step_outputs = []

if __name__ == "__main__":

    model = SimpleDetector()
    model.to(device)

    logger = loggers.TensorBoardLogger(TENSORBOARD_LOGS_DIR, name="bbox")

    trainer = L.Trainer(max_epochs=TOT_EPOCHS, logger=logger, log_every_n_steps=10)
    trainer.fit(model)
