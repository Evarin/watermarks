import torch
from pathlib import Path
import re
import random
from torchvision import transforms as T
from PIL import Image, ImageOps
from tqdm import tqdm
import shutil

# WATERMARK DETECTION MODEL
DETECTION_CHECKPOINT = "../detection.pth"

# DATA DIRECTORIES
DATA_DIR = Path("../data/")
SOURCE_DIR = DATA_DIR / "xishen" / "A_classification"
TARGET_DIR = DATA_DIR / "datasets" / "xishen-classif"

source_train_dir = SOURCE_DIR / "train"
source_val_dir = SOURCE_DIR / "test"
train_dir = TARGET_DIR / "train"
val_dir = TARGET_DIR / "val"
tmp_cropped_dir = TARGET_DIR / "tmp"


if TARGET_DIR.exists():
    shutil.rmtree(TARGET_DIR)

train_dir.mkdir(parents=True, exist_ok=True)
val_dir.mkdir(parents=True, exist_ok=True)

cats_trainval = set()

def process_images(model, src_dir, tgt_dir):
    """
    Process images in src_dir and save them to tgt_dir.
    """
    device = next(model.parameters()).device
    all_boxes = []
    no_box = []
    for f in src_dir.iterdir():
        if not f.is_file() or f.suffix.lower() not in [".jpg", ".jpeg", ".png"] or f.name.startswith("."):
            continue
        img = Image.open(f)
        img0 = ImageOps.exif_transpose(img)
        img = T.ToTensor()(img)
        img = img.to(device)

        with torch.no_grad():
            r = model([img])[0]
            boxes = r["boxes"]
            scores = r["scores"]

        # crop around best box
        if len(boxes) > 0 and scores[0] > 0.5:
            box = boxes[0]
            x0, y0, x1, y1 = box
            # rescale to original size
            sx, sy = img0.size[0] / img.shape[-1], img0.size[1] / img.shape[-2]
            x0, y0, x1, y1 = int(x0 * sx), int(y0 * sy), int(x1 * sx), int(y1 * sy)
            # convert to cx cy w h
            cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
            # add 15% padding and square
            sz = max(w, h) * 1.2
            all_boxes.append((cx, cy, sz))
            # crop
            x0, y0, x1, y1 = int(cx - sz / 2), int(cy - sz / 2), int(cx + sz / 2), int(cy + sz / 2)
            crop = img0.crop((x0, y0, x1, y1))
            crop.thumbnail((320, 320))
            crop.save(tgt_dir / f.name, quality=85)
        else:
            no_box.append((f, img0))

    # If detection failed, propagate average box for the whole subdirectory
    if len(no_box) > 0:
        av_box = torch.tensor(all_boxes).mean(dim=0)
        cx, cy, sz = av_box
        print(f"Using average box for {len(no_box)} images of {src_dir}: {av_box}")
        x0, y0, x1, y1 = int(cx - sz / 2), int(cy - sz / 2), int(cx + sz / 2), int(cy + sz / 2)
        for (f, img0) in no_box:
            crop = img0.crop((x0, y0, x1, y1))
            crop.thumbnail((320, 320))
            crop.save(tgt_dir / ("0_"+f.name), quality=85)

if __name__ == "__main__":
    device = torch.device("cuda:0")

    # Load detection model
    m = torch.load(DETECTION_CHECKPOINT).to(device).eval()

    for src_dir in tqdm(list(source_train_dir.iterdir())):
        cat = src_dir.name
        tgt_dir = train_dir / cat
        tgt_dir.mkdir(parents=True, exist_ok=True)
        process_images(m, src_dir, tgt_dir)

    for src_dir in tqdm(list(source_val_dir.iterdir())):
        cat = src_dir.name
        tgt_dir = val_dir / cat
        tgt_dir.mkdir(parents=True, exist_ok=True)
        process_images(m, src_dir, tgt_dir)
