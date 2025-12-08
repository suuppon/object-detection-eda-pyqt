from pathlib import Path

import cv2
import pandas as pd
import yaml

from core.data.data_loader import UnifiedDataLoader


class YoloDataLoader:
    """Factory for loading YOLO datasets into UnifiedDataLoader."""

    def __new__(cls, yaml_path):
        loader = UnifiedDataLoader()

        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"YAML file not found: {yaml_path}")

        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)

        base_path = Path(config.get("path", yaml_path.parent))
        train_path_str = config.get("train", "")

        if train_path_str:
            if Path(train_path_str).is_absolute():
                train_path = Path(train_path_str)
            else:
                train_path = base_path / train_path_str
        else:
            train_path = base_path

        if (base_path / "images").exists() and (base_path / "labels").exists():
            img_dir = base_path / "images"
            label_dir = base_path / "labels"
        elif train_path.name == "images" or (train_path / "images").exists():
            if train_path.name == "images":
                img_dir = train_path
                labels_candidate = train_path.parent / "labels" / train_path.name
                if labels_candidate.exists():
                    label_dir = labels_candidate
                else:
                    label_dir = train_path.parent / "labels"
            else:
                img_dir = train_path / "images"
                label_dir = train_path / "labels"
        else:
            img_dir = train_path
            if (base_path / "labels").exists():
                label_dir = base_path / "labels"
            else:
                label_dir = (
                    train_path.parent / "labels"
                    if train_path.parent.exists()
                    else train_path
                )

        categories = {}
        names = config.get("names", {})
        if isinstance(names, dict):
            categories = {int(k): str(v) for k, v in names.items()}
        elif isinstance(names, list):
            categories = {i: str(name) for i, name in enumerate(names)}

        if not categories:
            for filename in ["classes.txt", "names.txt", "labels.txt"]:
                classes_path = base_path / filename
                if classes_path.exists():
                    with open(classes_path, "r") as f:
                        for class_id, line in enumerate(f):
                            class_name = line.strip()
                            if class_name:
                                categories[class_id] = class_name
                    break

        loader.categories = categories
        # Exclude category 0 (super category)
        if 0 in loader.categories:
            del loader.categories[0]
        loader.img_root = img_dir
        loader.base_path = base_path  # Store base path for reference

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        image_paths = []

        if img_dir.exists():
            for f in img_dir.iterdir():
                if f.suffix.lower() in image_extensions and f.is_file():
                    image_paths.append((f, label_dir))

        val_path_str = config.get("val", "")
        if val_path_str:
            if Path(val_path_str).is_absolute():
                val_path = Path(val_path_str)
            else:
                val_path = base_path / val_path_str

            if (val_path / "images").exists():
                val_img_dir = val_path / "images"
                val_label_dir = val_path / "labels"
            elif val_path.name == "images":
                val_img_dir = val_path
                val_label_dir = val_path.parent / "labels"
            elif (base_path / "images").exists():
                val_img_dir = base_path / "images"
                val_label_dir = base_path / "labels"
            else:
                val_img_dir = val_path
                val_label_dir = base_path / "labels"

            if val_img_dir.exists() and val_img_dir != img_dir:
                for f in val_img_dir.iterdir():
                    if f.suffix.lower() in image_extensions and f.is_file():
                        image_paths.append((f, val_label_dir))

        loader.images = {}
        annotations_list = []
        ann_id = 0

        for img_file, label_d in sorted(image_paths):
            img_path = (
                img_file if isinstance(img_file, Path) else (img_dir / img_file.name)
            )
            if not img_path.exists():
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            img_height, img_width = img.shape[:2]
            img_id = len(loader.images)

            try:
                rel_path = img_path.relative_to(base_path)
                file_name = str(rel_path)
            except ValueError:
                file_name = img_file.name

            loader.images[img_id] = {
                "id": img_id,
                "file_name": file_name,
                "width": img_width,
                "height": img_height,
                "abs_path": str(img_path.absolute()),
            }

            label_file = label_d / f"{img_file.stem}.txt"
            if not label_file.exists():
                continue

            with open(label_file, "r") as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 5:
                    continue

                try:
                    class_id = int(parts[0])
                    # Skip category 0 (super category)
                    if class_id == 0:
                        continue
                    center_x = float(parts[1])
                    center_y = float(parts[2])
                    width_norm = float(parts[3])
                    height_norm = float(parts[4])

                    x_abs = (center_x - width_norm / 2) * img_width
                    y_abs = (center_y - height_norm / 2) * img_height
                    w_abs = width_norm * img_width
                    h_abs = height_norm * img_height

                    bbox = [x_abs, y_abs, w_abs, h_abs]

                    annotations_list.append(
                        {
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": class_id,
                            "bbox": bbox,
                        }
                    )
                    ann_id += 1

                except (ValueError, IndexError):
                    continue

        loader.annotations = pd.DataFrame(annotations_list)

        if not loader.annotations.empty:
            loader.annotations["bbox_w"] = loader.annotations["bbox"].apply(
                lambda x: x[2]
            )
            loader.annotations["bbox_h"] = loader.annotations["bbox"].apply(
                lambda x: x[3]
            )
            loader.annotations["area"] = (
                loader.annotations["bbox_w"] * loader.annotations["bbox_h"]
            )
            loader.annotations["aspect_ratio"] = (
                loader.annotations["bbox_w"] / loader.annotations["bbox_h"]
            )
            if loader.categories:
                loader.annotations["category_name"] = loader.annotations[
                    "category_id"
                ].map(loader.categories)
                loader.annotations["category_name"] = loader.annotations[
                    "category_name"
                ].fillna(loader.annotations["category_id"].astype(str))
            else:
                loader.annotations["category_name"] = loader.annotations[
                    "category_id"
                ].astype(str)
        else:
            for col in ["bbox_w", "bbox_h", "area", "aspect_ratio", "category_name"]:
                loader.annotations[col] = []

        return loader
