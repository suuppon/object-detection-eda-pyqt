import json
from pathlib import Path

import pandas as pd

from core.data.data_loader import UnifiedDataLoader


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
        
        # Exclude category 0 (super category)
        if 0 in loader.categories:
            del loader.categories[0]
        if not loader.annotations.empty:
            loader.annotations = loader.annotations[loader.annotations["category_id"] != 0]

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
