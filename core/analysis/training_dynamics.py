import os
import shutil

import numpy as np
import pandas as pd
import yaml
from PySide6.QtCore import QThread, Signal
from ultralytics import YOLO


class TrainingDynamicsAnalyzerThread(QThread):
    """
    Background worker for Training Dynamics Analysis using YOLOv8.
    Handles COCO -> YOLO conversion and Training Dynamics Analysis.
    """

    progress_updated = Signal(int, str)  # value, message
    analysis_finished = Signal(pd.DataFrame)
    error_occurred = Signal(str)

    def __init__(
        self, loader=None, img_root=None, epochs=5, batch_size=16, dataset_yaml=None
    ):
        super().__init__()
        self.loader = loader
        self.img_root = img_root
        self.dataset_yaml = dataset_yaml  # Optional: use existing yaml if provided
        self.epochs = epochs
        self.batch_size = batch_size
        self.is_running = True
        self.temp_dir = None
        self.project_dir = None

    def run(self):
        try:
            self.progress_updated.emit(0, "Initializing Training Dynamics Analysis...")

            # Setup workspace
            base_cache_dir = os.path.join(os.getcwd(), "yolo_cartography_cache")
            if not os.path.exists(base_cache_dir):
                os.makedirs(base_cache_dir)

            self.temp_dir = os.path.join(base_cache_dir, "current_dataset")
            self.project_dir = os.path.join(base_cache_dir, "training_runs")

            # Clean previous runs to avoid confusion
            if os.path.exists(self.project_dir):
                shutil.rmtree(self.project_dir)

            # 1. Check if we need to convert COCO to YOLO
            if not self.dataset_yaml:
                if not self.loader or not self.img_root:
                    raise ValueError(
                        "No dataset loaded (CocoDataLoader required if no YAML provided)."
                    )

                self.dataset_yaml = self._convert_coco_to_yolo()

            # 2. Parse Dataset YAML to get image paths (Validation)
            with open(self.dataset_yaml, "r") as f:
                data_config = yaml.safe_load(f)

            train_path = data_config.get("train")
            if not os.path.isabs(train_path):
                train_path = os.path.abspath(
                    os.path.join(os.path.dirname(self.dataset_yaml), train_path)
                )

            image_files = []
            if os.path.isdir(train_path):
                for root, dirs, files in os.walk(train_path):
                    for file in files:
                        if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                            image_files.append(os.path.join(root, file))
            elif os.path.isfile(train_path) and train_path.endswith(".txt"):
                with open(train_path, "r") as f:
                    lines = f.readlines()
                    base_dir = os.path.dirname(train_path)
                    for line in lines:
                        path = line.strip()
                        if not os.path.isabs(path):
                            path = os.path.join(base_dir, path)
                        image_files.append(path)

            image_files = sorted(image_files)
            if not image_files:
                raise ValueError("No training images found in the dataset.")

            self.dynamics = {img: [] for img in image_files}

            # 3. Custom Training Loop
            # Start with pretrained model
            current_weights = "yolov8n.pt"

            for epoch in range(self.epochs):
                if not self.is_running:
                    break

                msg = f"Training Epoch {epoch + 1}/{self.epochs}..."
                prog_start = 20 + int((epoch / self.epochs) * 70)
                self.progress_updated.emit(prog_start, msg)

                # Re-instantiate model for each epoch to avoid state issues
                # Use the weights from the previous epoch (or pretrained for the first)
                model = YOLO(current_weights)

                # Train for 1 epoch
                # We use a new project/name for each epoch run to keep them separate
                run_name = f"epoch_{epoch}"

                # Note: To continue training, we should ideally use 'resume=True' with the same run dir,
                # but here we are manually chaining 1-epoch runs.
                # Using 'pretrained=True' loads weights.

                model.train(
                    data=self.dataset_yaml,
                    epochs=1,
                    project=self.project_dir,
                    name=run_name,
                    batch=self.batch_size,
                    plots=False,
                    save=True,
                    pretrained=True,  # This loads weights from 'model' argument (current_weights)
                    verbose=False,
                    exist_ok=True,
                )

                # Update weights for next iteration
                # The model saves weights in project/name/weights/last.pt
                # We need to find where it saved.
                # ultralytics typically saves to: {project}/{name}/weights/last.pt
                saved_weights = os.path.join(
                    self.project_dir, run_name, "weights", "last.pt"
                )
                if os.path.exists(saved_weights):
                    current_weights = saved_weights
                else:
                    # Fallback or error handling
                    print(f"Warning: Could not find saved weights at {saved_weights}")

                self.progress_updated.emit(
                    prog_start + 5, f"Analyzing dynamics for Epoch {epoch + 1}..."
                )

                # Predict using the CURRENT model state (which is now trained for 1 more epoch)
                preds = model.predict(
                    source=image_files, stream=True, conf=0.01, iou=0.5, verbose=False
                )

                for result in preds:
                    path = result.path
                    if result.boxes is not None and len(result.boxes) > 0:
                        conf = result.boxes.conf.cpu().numpy().mean()
                    else:
                        conf = 0.0

                    if path in self.dynamics:
                        self.dynamics[path].append(float(conf))

            # 5. Calculate Metrics
            self.progress_updated.emit(95, "Calculating Training Dynamics Metrics...")

            data_list = []
            for img_path, confs in self.dynamics.items():
                if not confs:
                    continue

                confs_arr = np.array(confs)
                confidence = np.mean(confs_arr)
                variability = np.std(confs_arr)

                region = "Ambiguous"
                if confidence >= 0.7 and variability <= 0.2:
                    region = "Easy-to-Learn"
                elif confidence <= 0.4 and variability <= 0.2:
                    region = "Hard-to-Learn"
                elif variability > 0.2:
                    region = "Ambiguous"
                else:
                    region = "Ambiguous"

                data_list.append(
                    {
                        "image_path": img_path,
                        "confidence": confidence,
                        "variability": variability,
                        "region": region,
                        "history": confs,
                    }
                )

            df = pd.DataFrame(data_list)
            self.analysis_finished.emit(df)

        except Exception as e:
            import traceback

            traceback.print_exc()
            self.error_occurred.emit(str(e))

    def _convert_coco_to_yolo(self):
        """
        Convert loaded COCO data to YOLO format structure in self.temp_dir.
        Returns path to the generated data.yaml.
        """
        self.progress_updated.emit(5, "Converting COCO dataset to YOLO format...")

        images_dir = os.path.join(self.temp_dir, "images", "train")
        labels_dir = os.path.join(self.temp_dir, "labels", "train")

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        os.makedirs(images_dir)
        os.makedirs(labels_dir)

        # Get Categories: self.loader.categories is {id: name}
        cat_id_to_idx = {
            cat_id: i for i, cat_id in enumerate(self.loader.categories.keys())
        }
        class_names = [
            self.loader.categories[cat_id] for cat_id in self.loader.categories.keys()
        ]

        img_ids = list(self.loader.images.keys())
        total_imgs = len(img_ids)

        # Pre-filter annotations by image_id for faster access
        anns_by_img = self.loader.annotations.groupby("image_id")

        for i, img_id in enumerate(img_ids):
            if not self.is_running:
                raise InterruptedError("Conversion stopped by user")

            if i % 100 == 0:
                self.progress_updated.emit(
                    5 + int((i / total_imgs) * 15),
                    f"Converting images {i}/{total_imgs}...",
                )

            img_info = self.loader.images[img_id]
            file_name = img_info["file_name"]
            src_path = os.path.join(self.img_root, file_name)

            if not os.path.exists(src_path):
                continue

            dst_img_path = os.path.join(images_dir, file_name)
            shutil.copy2(src_path, dst_img_path)

            # Get annotations for this image
            if img_id in anns_by_img.groups:
                anns = anns_by_img.get_group(img_id)
            else:
                anns = pd.DataFrame()

            label_lines = []
            img_w = img_info["width"]
            img_h = img_info["height"]

            for _, ann in anns.iterrows():
                bbox = ann["bbox"]
                cat_id = ann["category_id"]

                if cat_id not in cat_id_to_idx:
                    continue

                cls_idx = cat_id_to_idx[cat_id]
                x, y, w, h = bbox

                x_c = (x + w / 2) / img_w
                y_c = (y + h / 2) / img_h
                w_n = w / img_w
                h_n = h / img_h

                x_c = max(0, min(1, x_c))
                y_c = max(0, min(1, y_c))
                w_n = max(0, min(1, w_n))
                h_n = max(0, min(1, h_n))

                label_lines.append(f"{cls_idx} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}")

            label_name = os.path.splitext(file_name)[0] + ".txt"
            dst_label_path = os.path.join(labels_dir, label_name)
            with open(dst_label_path, "w") as f:
                f.write("\n".join(label_lines))

        yaml_path = os.path.join(self.temp_dir, "data.yaml")
        yaml_content = {
            "path": self.temp_dir,
            "train": os.path.join("images", "train"),
            "val": os.path.join("images", "train"),
            "names": {i: name for i, name in enumerate(class_names)},
        }

        with open(yaml_path, "w") as f:
            yaml.dump(yaml_content, f, sort_keys=False)

        return yaml_path

    def stop(self):
        self.is_running = False
