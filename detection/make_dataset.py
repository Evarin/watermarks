from sqlalchemy import create_engine, text
from PIL import Image, ImageOps
import random, numpy as np, os
import shutil
from tqdm import tqdm
import json
from pathlib import Path

def fetch_annotations(db_file="./annotator/db.sqlite3"):
    engine = create_engine(f'sqlite:///{db_file}')

    with engine.connect() as connection:
        annotations = connection.execute(text('SELECT * FROM ANNOTATIONS')).fetchall()

    return annotations

def process_dir(source_dir, target_dir, annotations, clear=False, ignore_prefixes=["xi/", "newton/"], one_for_all=False):
    """
    This function processes a directory of images and annotations, and creates a dataset split, stored in json files.

    source_dir: str, path to the source directory
    target_dir: str, path to the target directory
    annotations: list, list of annotations (image, x0, y0, x1, y1)
    clear: bool, whether to clear the target directory
    ignore_prefixes: list, list of path prefixes to ignore (relative to source_dir)
    one_for_all: bool, whether to use one annotation for all images in a directory
    """
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    if clear:
        shutil.rmtree(target_dir, ignore_errors=True)
    os.makedirs(target_dir, exist_ok=True)
    val_prop = 0.1

    train_file = "train.json"
    val_file = "val.json"

    data = {}
    for i, annotation in enumerate(tqdm(annotations)):
        if annotation.image.startswith(tuple(ignore_prefixes)):
            continue
        if not os.path.isfile(source_dir / annotation.image):
            print(f"File not found: {source_dir / annotation.image}")
            continue
        is_train = random.random() < val_prop

        if annotation.image in data:
            data[annotation.image].update({"x0": annotation.x0, "y0": annotation.y0, "x1": annotation.x1, "y1": annotation.y1})
            continue

        if one_for_all:
            idir = os.path.dirname(annotation.image)
            candidates = [os.path.join(idir, f) for f in os.listdir(source_dir + idir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        else:
            candidates = [annotation.image]

        for candidate in candidates:
            data[candidate] = {
                "split": "val" if is_train else "train",
                "image": candidate,
                "x0": annotation.x0, "y0": annotation.y0, "x1": annotation.x1, "y1": annotation.y1
            }

    with open(target_dir / train_file, "w") as f:
        json.dump([v for v in data.values() if v["split"] == "train"], f)

    with open(target_dir / val_file, "w") as f:
        json.dump([v for v in data.values() if v["split"] == "val"], f)

if __name__ == "__main__":
    annotations = fetch_annotations()
    process_dir("../data/", "../data/datasets/detection/", annotations, clear=True)
    print("Done!")