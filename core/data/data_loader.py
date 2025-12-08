"""Data loader for COCO and YOLO format object detection datasets."""

import json
import shutil
from pathlib import Path

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
        self.excluded_image_ids = set()  # Images marked for exclusion
        self.duplicate_groups = []  # List of sets containing duplicate image IDs
        self.source_tracking = {}  # {source_name: set(image_ids)} - Track which images came from which source

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
                # If abs_path already exists, keep it (support multiple sources)
                if "abs_path" in img_info and Path(img_info["abs_path"]).exists():
                    continue
                    
                if "file_name" in img_info:
                    # Check if file_name is already absolute
                    if Path(img_info["file_name"]).is_absolute():
                        img_info["abs_path"] = str(img_info["file_name"])
                    else:
                        img_info["abs_path"] = str(
                            self.img_root / img_info["file_name"]
                        )

    def mark_image_for_exclusion(self, image_id):
        """Mark an image for exclusion from export."""
        if image_id in self.images:
            self.excluded_image_ids.add(image_id)

    def unmark_image_for_exclusion(self, image_id):
        """Unmark an image for exclusion."""
        self.excluded_image_ids.discard(image_id)

    def get_excluded_images(self):
        """Get set of excluded image IDs."""
        return self.excluded_image_ids.copy()

    def get_exportable_images(self):
        """Get dict of images excluding marked ones."""
        return {
            img_id: img_info
            for img_id, img_info in self.images.items()
            if img_id not in self.excluded_image_ids
        }

    def remove_images(self, image_ids):
        """Remove specific images and their annotations from the dataset.
        
        Args:
            image_ids: Set or list of image IDs to remove.
        """
        image_ids = set(image_ids)
        
        # Remove images
        for img_id in image_ids:
            self.images.pop(img_id, None)
            self.excluded_image_ids.discard(img_id)  # Also remove from excluded if present
        
        # Remove annotations for these images
        if not self.annotations.empty:
            self.annotations = self.annotations[
                ~self.annotations["image_id"].isin(image_ids)
            ]
        
        # Update duplicate groups
        if self.duplicate_groups:
            self.duplicate_groups = [
                group - image_ids for group in self.duplicate_groups
            ]
            # Remove empty groups
            self.duplicate_groups = [g for g in self.duplicate_groups if len(g) > 1]
        
        # Update source tracking
        for source_name in list(self.source_tracking.keys()):
            self.source_tracking[source_name] -= image_ids
            if not self.source_tracking[source_name]:
                del self.source_tracking[source_name]
    
    def set_source(self, source_name, image_ids=None):
        """Set source name for images.
        
        Args:
            source_name: Name of the source dataset.
            image_ids: Optional set of image IDs. If None, uses all current images.
        """
        if image_ids is None:
            image_ids = set(self.images.keys())
        else:
            image_ids = set(image_ids)
        
        if source_name not in self.source_tracking:
            self.source_tracking[source_name] = set()
        self.source_tracking[source_name].update(image_ids)
        
        # Also add source metadata to each image
        for img_id in image_ids:
            if img_id in self.images:
                self.images[img_id]["source"] = source_name
    
    def get_sources(self):
        """Get list of source names."""
        return list(self.source_tracking.keys())
    
    def get_source_image_ids(self, source_name):
        """Get image IDs for a specific source."""
        return self.source_tracking.get(source_name, set()).copy()

    def set_duplicate_groups(self, groups):
        """Set duplicate image groups.

        Args:
            groups: List of sets, each containing duplicate image IDs.
        """
        self.duplicate_groups = groups

    def get_image_duplicate_group(self, image_id):
        """Get the duplicate group containing the given image ID.

        Returns:
            Set of image IDs in the same group, or None if not in any group.
        """
        for group in self.duplicate_groups:
            if image_id in group:
                return group
        return None

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
            # Preserve source information if present
            if "source" in img_info:
                new_img_info["source"] = img_info["source"]
            self.images[new_img_id] = new_img_info

            # If the other loader had this image excluded, exclude it in merged loader too
            if other_img_id in other.excluded_image_ids:
                self.excluded_image_ids.add(new_img_id)

            next_img_id += 1
        
        # 2.5. Merge source tracking information
        for source_name, img_ids in other.source_tracking.items():
            # Map old image IDs to new image IDs
            new_img_ids = {img_map[old_id] for old_id in img_ids if old_id in img_map}
            if source_name in self.source_tracking:
                self.source_tracking[source_name].update(new_img_ids)
            else:
                self.source_tracking[source_name] = new_img_ids

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

    def normalize_category_ids(self):
        """Normalize category IDs to start from 0 sequentially.

        Categories with the same name will be merged into a single ID.
        This modifies the internal state and should typically be called before export.

        Returns:
            Dictionary mapping old category_id to new category_id.
        """
        if not self.categories:
            return {}

        # Group categories by name (merge duplicates)
        name_to_ids = {}
        for cat_id, cat_name in self.categories.items():
            if cat_name not in name_to_ids:
                name_to_ids[cat_name] = []
            name_to_ids[cat_name].append(cat_id)

        # Create new categories with sequential IDs
        new_categories = {}
        old_to_new_map = {}

        for new_id, (cat_name, old_ids) in enumerate(sorted(name_to_ids.items())):
            new_categories[new_id] = cat_name
            for old_id in old_ids:
                old_to_new_map[old_id] = new_id

        # Ensure category 0 is always "vehicle"
        if 0 in new_categories:
            # Category 0 exists, rename it to "vehicle"
            new_categories[0] = "vehicle"
        else:
            # Category 0 doesn't exist, insert "vehicle" as category 0 and shift others
            shifted_categories = {0: "vehicle"}
            for old_id, cat_name in new_categories.items():
                shifted_categories[old_id + 1] = cat_name
            new_categories = shifted_categories
            # Update the mapping to account for the shift (all new IDs are incremented by 1)
            for old_id in old_to_new_map:
                old_to_new_map[old_id] = old_to_new_map[old_id] + 1

        # Update categories
        self.categories = new_categories

        # Update annotations
        if not self.annotations.empty:
            self.annotations["category_id"] = self.annotations["category_id"].map(
                old_to_new_map
            )
            self.annotations["category_name"] = self.annotations["category_id"].map(
                self.categories
            )

        return old_to_new_map

    def rename_category(self, cat_id, new_name):
        """Rename a category."""
        if cat_id in self.categories:
            self.categories[cat_id] = new_name
            if not self.annotations.empty:
                # Update category_name column for consistency
                mask = self.annotations["category_id"] == cat_id
                self.annotations.loc[mask, "category_name"] = new_name

    def export_as_yolo(self, save_dir, split_info=None, exclude_marked=True):
        """Export dataset in YOLO format with optional split and exclusion.

        Args:
            save_dir: Directory to save YOLO dataset.
            split_info: Optional dict with 'train', 'val', 'test' keys containing image IDs.
            exclude_marked: Whether to exclude images marked for deletion.
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Normalize categories first
        self.normalize_category_ids()

        # Get exportable images
        if exclude_marked:
            exportable_imgs = self.get_exportable_images()
        else:
            exportable_imgs = self.images

        # Prepare category mapping
        class_names = [
            self.categories[cat_id] for cat_id in sorted(self.categories.keys())
        ]

        # Prepare annotations grouped by image
        # Filter annotations to only include exportable images
        if not self.annotations.empty:
            exportable_anns = self.annotations[
                self.annotations["image_id"].isin(exportable_imgs.keys())
            ]
            anns_by_img = exportable_anns.groupby("image_id")
        else:
            anns_by_img = None

        # Determine splits
        if split_info and any(split_info.values()):
            # With splits
            splits = {}
            for split_name in ["train", "val", "test"]:
                if split_name in split_info and split_info[split_name]:
                    splits[split_name] = set(split_info[split_name]) & set(
                        exportable_imgs.keys()
                    )

            # Create split directories
            for split_name in splits.keys():
                (save_dir / "images" / split_name).mkdir(parents=True, exist_ok=True)
                (save_dir / "labels" / split_name).mkdir(parents=True, exist_ok=True)

            # Export by split
            for split_name, img_ids in splits.items():
                for img_id in img_ids:
                    self._export_yolo_image(
                        img_id,
                        exportable_imgs[img_id],
                        save_dir / "images" / split_name,
                        save_dir / "labels" / split_name,
                        anns_by_img,
                    )

            # Create data.yaml with splits
            yaml_content = {
                "path": str(save_dir.absolute()),
                "train": str(Path("images") / "train") if "train" in splits else "",
                "val": str(Path("images") / "val") if "val" in splits else "",
                "test": str(Path("images") / "test") if "test" in splits else "",
                "names": {i: name for i, name in enumerate(class_names)},
            }
        else:
            # Without splits
            images_dir = save_dir / "images"
            labels_dir = save_dir / "labels"
            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)

            for img_id, img_info in exportable_imgs.items():
                self._export_yolo_image(
                    img_id, img_info, images_dir, labels_dir, anns_by_img
                )

            # Create data.yaml without splits
            yaml_content = {
                "path": str(save_dir.absolute()),
                "train": "images",
                "val": "images",
                "names": {i: name for i, name in enumerate(class_names)},
            }

        # Write data.yaml
        with open(save_dir / "data.yaml", "w") as f:
            yaml.dump(yaml_content, f, sort_keys=False)

    def _export_yolo_image(self, img_id, img_info, images_dir, labels_dir, anns_by_img):
        """Helper to export a single image and its label in YOLO format."""
        # Find source image
        src_path = img_info.get("abs_path")
        if not src_path or not Path(src_path).exists():
            if self.img_root:
                src_path = self.img_root / img_info["file_name"]
            elif Path(img_info["file_name"]).exists():
                src_path = Path(img_info["file_name"])
            else:
                print(f"Warning: Image source not found for {img_info['file_name']}")
                return

        if not Path(src_path).exists():
            print(f"Warning: Image file does not exist: {src_path}")
            return

        # Copy image
        dst_img_path = images_dir / Path(img_info["file_name"]).name
        shutil.copy2(src_path, dst_img_path)

        # Generate label
        label_lines = []
        if anns_by_img and img_id in anns_by_img.groups:
            anns = anns_by_img.get_group(img_id)
            img_w = img_info["width"]
            img_h = img_info["height"]

            for _, ann in anns.iterrows():
                cat_id = int(ann["category_id"])
                cls_idx = cat_id  # After normalization, cat_id == index
                x, y, w, h = ann["bbox"]

                # Convert to YOLO format
                x_c = (x + w / 2) / img_w
                y_c = (y + h / 2) / img_h
                w_n = w / img_w
                h_n = h / img_h

                # Clamp values
                x_c = max(0, min(1, x_c))
                y_c = max(0, min(1, y_c))
                w_n = max(0, min(1, w_n))
                h_n = max(0, min(1, h_n))

                label_lines.append(f"{cls_idx} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}")

        # Write label file
        label_path = labels_dir / f"{Path(img_info['file_name']).stem}.txt"
        with open(label_path, "w") as f:
            f.write("\n".join(label_lines))

    def export_as_coco(self, save_path, split_info=None, exclude_marked=True):
        """Export dataset in COCO format with optional split and exclusion.

        Args:
            save_path: Path for JSON file. If splits are used, this is treated as a directory.
            split_info: Optional dict with 'train', 'val', 'test' keys containing image IDs.
            exclude_marked: Whether to exclude images marked for deletion.
        """
        save_path = Path(save_path)

        # Normalize categories first
        self.normalize_category_ids()

        # Get exportable images
        if exclude_marked:
            exportable_imgs = self.get_exportable_images()
        else:
            exportable_imgs = self.images

        # Prepare categories
        categories_list = [
            {"id": k, "name": v} for k, v in sorted(self.categories.items())
        ]

        # Derived columns to remove from annotations
        derived_cols = ["bbox_w", "bbox_h", "area", "aspect_ratio", "category_name"]

        if split_info and any(split_info.values()):
            # With splits - save_path is treated as directory
            save_dir = (
                save_path
                if save_path.is_dir() or not save_path.suffix
                else save_path.parent
            )
            save_dir.mkdir(parents=True, exist_ok=True)

            for split_name in ["train", "val", "test"]:
                if split_name not in split_info or not split_info[split_name]:
                    continue

                split_img_ids = set(split_info[split_name]) & set(
                    exportable_imgs.keys()
                )
                if not split_img_ids:
                    continue

                # Create images directory for this split
                split_images_dir = save_dir / "images" / split_name
                split_images_dir.mkdir(parents=True, exist_ok=True)

                # Filter images for this split
                split_images = []
                for img_id in split_img_ids:
                    img_info = exportable_imgs[img_id].copy()
                    img_info["file_name"] = Path(img_info["file_name"]).name
                    if "abs_path" in img_info:
                        del img_info["abs_path"]
                    split_images.append(img_info)

                    # Copy image file
                    self._copy_image_file(
                        img_id, exportable_imgs[img_id], split_images_dir
                    )

                # Filter annotations for this split
                if not self.annotations.empty:
                    split_anns = self.annotations[
                        self.annotations["image_id"].isin(split_img_ids)
                    ]
                    split_anns_list = split_anns.to_dict(orient="records")
                    for ann in split_anns_list:
                        for col in derived_cols:
                            ann.pop(col, None)
                else:
                    split_anns_list = []

                # Create COCO dict for this split
                coco_dict = {
                    "images": split_images,
                    "annotations": split_anns_list,
                    "categories": categories_list,
                }

                # Save JSON
                json_path = save_dir / f"{split_name}.json"
                with open(json_path, "w") as f:
                    json.dump(coco_dict, f, indent=2)
        else:
            # Without splits - single JSON file
            if save_path.suffix != ".json":
                save_path = save_path / "annotations.json"

            save_dir = save_path.parent
            save_dir.mkdir(parents=True, exist_ok=True)

            # Create images directory
            images_dir = save_dir / "images"
            images_dir.mkdir(parents=True, exist_ok=True)

            # Prepare images
            coco_images = []
            for img_id, img_info in exportable_imgs.items():
                img_copy = img_info.copy()
                img_copy["file_name"] = Path(img_copy["file_name"]).name
                if "abs_path" in img_copy:
                    del img_copy["abs_path"]
                coco_images.append(img_copy)

                # Copy image file
                self._copy_image_file(img_id, img_info, images_dir)

            # Prepare annotations (filter to only exportable images)
            if not self.annotations.empty:
                export_anns = self.annotations[
                    self.annotations["image_id"].isin(exportable_imgs.keys())
                ]
                anns_list = export_anns.to_dict(orient="records")
                for ann in anns_list:
                    for col in derived_cols:
                        ann.pop(col, None)
            else:
                anns_list = []

            # Create COCO dict
            coco_dict = {
                "images": coco_images,
                "annotations": anns_list,
                "categories": categories_list,
            }

            # Save JSON
            with open(save_path, "w") as f:
                json.dump(coco_dict, f, indent=2)

    def _copy_image_file(self, img_id, img_info, dest_dir):
        """Helper to copy image file to destination directory."""
        src_path = img_info.get("abs_path")
        if not src_path or not Path(src_path).exists():
            if self.img_root:
                src_path = self.img_root / img_info["file_name"]
            elif Path(img_info["file_name"]).exists():
                src_path = Path(img_info["file_name"])
            else:
                print(f"Warning: Image source not found for {img_info['file_name']}")
                return

        if not Path(src_path).exists():
            print(f"Warning: Image file does not exist: {src_path}")
            return

        dst_path = dest_dir / Path(img_info["file_name"]).name
        if not dst_path.exists():
            shutil.copy2(src_path, dst_path)
