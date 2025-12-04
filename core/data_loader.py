"""Data loader for COCO and YOLO format object detection datasets."""

import json
import shutil
from pathlib import Path

import cv2
import pandas as pd
import yaml


class UnifiedDataLoader:
    """Common Data Loader for Object Detection Datasets."""

    def __init__(self):
        self.images = {}
        self.categories = {}
        self.annotations = pd.DataFrame()
        self.img_root = None
        self.base_path = None  # For YOLO dataset base path

    def get_stats(self):
        """Get basic dataset statistics."""
        return {
            "Total Images": len(self.images),
            "Total Instances": len(self.annotations),
            "Total Classes": (
                len(self.categories)
                if self.categories
                else (
                    len(self.annotations["category_id"].unique())
                    if not self.annotations.empty
                    else 0
                )
            ),
        }

    def set_img_root(self, img_root):
        """Set image root directory and update absolute paths."""
        self.img_root = Path(img_root) if img_root else None
        if self.img_root:
            for img_id, img_info in self.images.items():
                if "file_name" in img_info:
                    # Check if file_name is already absolute
                    if Path(img_info["file_name"]).is_absolute():
                        img_info["abs_path"] = str(img_info["file_name"])
                    else:
                        img_info["abs_path"] = str(
                            self.img_root / img_info["file_name"]
                        )

    def merge(self, other):
        """Merge another DataLoader instance into this one."""
        if not other or not other.images:
            return

        # 1. Merge Categories (Name-based mapping)
        cat_map = {}

        if not self.categories and not self.annotations.empty:
            unique_ids = self.annotations["category_id"].unique()
            for uid in unique_ids:
                self.categories[uid] = str(uid)

        next_cat_id = int(max(self.categories.keys()) + 1) if self.categories else 0
        name_to_id = {v: k for k, v in self.categories.items()}

        other_cats = other.categories.copy()
        if not other_cats and not other.annotations.empty:
            unique_ids = other.annotations["category_id"].unique()
            for uid in unique_ids:
                other_cats[uid] = str(uid)

        for other_id, other_name in other_cats.items():
            if other_name in name_to_id:
                cat_map[other_id] = name_to_id[other_name]
            else:
                self.categories[next_cat_id] = other_name
                cat_map[other_id] = next_cat_id
                name_to_id[other_name] = next_cat_id
                next_cat_id += 1

        if not other.annotations.empty:
            missing_ids = set(other.annotations["category_id"].unique()) - set(
                cat_map.keys()
            )
            for mid in missing_ids:
                name = str(mid)
                if name in name_to_id:
                    cat_map[mid] = name_to_id[name]
                else:
                    self.categories[next_cat_id] = name
                    cat_map[mid] = next_cat_id
                    name_to_id[name] = next_cat_id
                    next_cat_id += 1

        # 2. Merge Images
        next_img_id = int(max(self.images.keys()) + 1) if self.images else 0
        img_map = {}

        for other_img_id, img_info in other.images.items():
            new_img_id = next_img_id
            img_map[other_img_id] = new_img_id

            new_img_info = img_info.copy()
            new_img_info["id"] = new_img_id
            self.images[new_img_id] = new_img_info

            next_img_id += 1

        # 3. Merge Annotations
        if not other.annotations.empty:
            new_anns = other.annotations.copy()
            new_anns["image_id"] = new_anns["image_id"].map(img_map)
            new_anns["category_id"] = new_anns["category_id"].map(cat_map)
            new_anns["category_name"] = new_anns["category_id"].map(self.categories)

            if not self.annotations.empty:
                next_ann_id = int(self.annotations["id"].max() + 1)
            else:
                next_ann_id = 0

            new_anns["id"] = range(next_ann_id, next_ann_id + len(new_anns))
            self.annotations = pd.concat(
                [self.annotations, new_anns], ignore_index=True
            )

    def rename_category(self, cat_id, new_name):
        """Rename a category."""
        if cat_id in self.categories:
            self.categories[cat_id] = new_name
            if not self.annotations.empty:
                # Update category_name column for consistency
                mask = self.annotations["category_id"] == cat_id
                self.annotations.loc[mask, "category_name"] = new_name

    def export_as_yolo(self, save_dir):
        """Export dataset in YOLO format."""
        save_dir = Path(save_dir)
        images_dir = save_dir / "images"
        labels_dir = save_dir / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        cat_id_to_idx = {
            cat_id: i for i, cat_id in enumerate(sorted(self.categories.keys()))
        }
        class_names = [
            self.categories[cat_id] for cat_id in sorted(self.categories.keys())
        ]

        if not self.annotations.empty:
            anns_by_img = self.annotations.groupby("image_id")
        else:
            anns_by_img = None

        for img_id, img_info in self.images.items():
            src_path = img_info.get("abs_path")
            if not src_path or not Path(src_path).exists():
                if self.img_root:
                    src_path = self.img_root / img_info["file_name"]
                elif Path(img_info["file_name"]).exists():
                    src_path = Path(img_info["file_name"])
                else:
                    print(
                        f"Warning: Image source not found for {img_info['file_name']}"
                    )
                    continue

            if not Path(src_path).exists():
                print(f"Warning: Image file does not exist: {src_path}")
                continue

            dst_img_path = images_dir / Path(img_info["file_name"]).name
            shutil.copy2(src_path, dst_img_path)

            label_lines = []
            if anns_by_img and img_id in anns_by_img.groups:
                anns = anns_by_img.get_group(img_id)
                img_w = img_info["width"]
                img_h = img_info["height"]

                for _, ann in anns.iterrows():
                    cat_id = ann["category_id"]
                    if cat_id not in cat_id_to_idx:
                        continue

                    cls_idx = cat_id_to_idx[cat_id]
                    x, y, w, h = ann["bbox"]

                    x_c = (x + w / 2) / img_w
                    y_c = (y + h / 2) / img_h
                    w_n = w / img_w
                    h_n = h / img_h

                    x_c = max(0, min(1, x_c))
                    y_c = max(0, min(1, y_c))
                    w_n = max(0, min(1, w_n))
                    h_n = max(0, min(1, h_n))

                    label_lines.append(
                        f"{cls_idx} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}"
                    )

            label_path = labels_dir / f"{Path(img_info['file_name']).stem}.txt"
            with open(label_path, "w") as f:
                f.write("\n".join(label_lines))

        yaml_content = {
            "path": str(save_dir.absolute()),
            "train": "images",
            "val": "images",
            "names": {i: name for i, name in enumerate(class_names)},
        }
        with open(save_dir / "data.yaml", "w") as f:
            yaml.dump(yaml_content, f, sort_keys=False)

    def export_as_coco(self, save_path):
        """Export dataset in COCO format (JSON)."""
        save_path = Path(save_path)

        coco_images = []
        for img in self.images.values():
            img_copy = img.copy()
            img_copy["file_name"] = str(
                Path(img_copy["file_name"]).name
            )  # COCO uses relative filename
            if "abs_path" in img_copy:
                del img_copy["abs_path"]
            coco_images.append(img_copy)

        coco_dict = {
            "images": coco_images,
            "annotations": (
                self.annotations.to_dict(orient="records")
                if not self.annotations.empty
                else []
            ),
            "categories": [{"id": k, "name": v} for k, v in self.categories.items()],
        }

        derived_cols = ["bbox_w", "bbox_h", "area", "aspect_ratio", "category_name"]
        for ann in coco_dict["annotations"]:
            for col in derived_cols:
                if col in ann:
                    del ann[col]

        with open(save_path, "w") as f:
            json.dump(coco_dict, f, indent=2)

        # Copy images to the same directory as json (common practice)
        save_dir = save_path.parent
        for img_info in self.images.values():
            src_path = img_info.get("abs_path")
            if not src_path or not Path(src_path).exists():
                if self.img_root:
                    src_path = self.img_root / img_info["file_name"]
                elif Path(img_info["file_name"]).exists():
                    src_path = Path(img_info["file_name"])
                else:
                    continue

            if Path(src_path).exists():
                dst_path = save_dir / Path(img_info["file_name"]).name
                if not dst_path.exists():
                    shutil.copy2(src_path, dst_path)


class CocoDataLoader:
    """Factory for loading COCO datasets into UnifiedDataLoader."""

    def __new__(cls, json_path, img_root=None):
        loader = UnifiedDataLoader()

        with open(json_path, "r") as f:
            data = json.load(f)

        loader.img_root = Path(img_root) if img_root else None
        loader.images = {img["id"]: img for img in data["images"]}
        loader.categories = {cat["id"]: cat["name"] for cat in data["categories"]}
        loader.annotations = pd.DataFrame(data["annotations"])

        if loader.img_root:
            for img_id, img_info in loader.images.items():
                if "file_name" in img_info:
                    img_info["abs_path"] = str(loader.img_root / img_info["file_name"])

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
            loader.annotations["category_name"] = loader.annotations["category_id"].map(
                loader.categories
            )
        else:
            for col in ["bbox_w", "bbox_h", "area", "aspect_ratio", "category_name"]:
                loader.annotations[col] = []

        return loader


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
