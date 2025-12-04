"""Deep signal analysis module for texture, camouflage, FFT, and PCA."""

import os

import cv2
import numpy as np
import pandas as pd
from PySide6.QtCore import QThread, Signal
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy
from sklearn.decomposition import PCA


class SignalAnalysisThread(QThread):
    """Thread for deep signal analysis including texture, camouflage, FFT, and PCA."""

    progress = Signal(int, str)
    finished_analysis = Signal(
        object, object, object
    )  # df_objects, df_images, pca_results
    error_occurred = Signal(str)

    def __init__(self, loader, root_path, max_pca_samples=1000):
        """Initialize signal analysis thread.

        Args:
            loader: CocoDataLoader instance with loaded dataset.
            root_path: Root directory path for image files.
            max_pca_samples: Maximum number of samples for PCA (default: 1000).
        """
        super().__init__()
        self.loader = loader
        self.root_path = root_path
        self.max_pca_samples = max_pca_samples

    def run(self):
        """Execute signal analysis including texture, camouflage, FFT, and PCA."""
        try:
            # 1. Setup
            annotations = self.loader.annotations
            images_meta = self.loader.images

            obj_results = []
            img_results = []
            pca_crops = []

            total_ops = len(annotations) + len(images_meta)
            current_op = 0

            # --- Stratified Sampling for PCA ---
            # Avoid pure random sampling, use stratified sampling by category
            # This ensures all classes are represented in PCA analysis if possible
            pca_indices = []
            if not annotations.empty:
                # Group by category
                grouped = annotations.groupby("category_name")
                n_classes = len(grouped)
                if n_classes > 0:
                    samples_per_class = max(1, self.max_pca_samples // n_classes)

                    for _, group in grouped:
                        n = min(len(group), samples_per_class)
                        if n > 0:
                            sampled = group.sample(n=n, random_state=42)
                            pca_indices.extend(sampled.index.tolist())

                    # If we still have room, fill with random samples from remaining
                    remaining_count = self.max_pca_samples - len(pca_indices)
                    if remaining_count > 0:
                        remaining_indices = list(
                            set(annotations.index) - set(pca_indices)
                        )
                        if remaining_indices:
                            additional = np.random.choice(
                                remaining_indices,
                                min(len(remaining_indices), remaining_count),
                                replace=False,
                            )
                            pca_indices.extend(additional)

            pca_indices_set = set(pca_indices)

            # --- Phase 1: Object-Level Analysis (Texture, Camouflage) ---
            for idx, row in annotations.iterrows():
                if self.isInterruptionRequested():
                    return

                img_id = row["image_id"]
                if img_id not in images_meta:
                    continue

                file_name = images_meta[img_id]["file_name"]
                full_path = os.path.join(self.root_path, file_name)

                # BBox coordinates (x, y, w, h)
                x, y, w, h = map(int, row["bbox"])

                # Metric Defaults
                metrics = {
                    "ann_id": row.get("id", idx),
                    "category": row["category_name"],
                    "entropy": 0,
                    "texture_contrast": 0,
                    "fg_bg_separability": 0,  # Low = Hard to see (Camouflage)
                    "is_valid": False,
                }

                if os.path.exists(full_path) and w > 0 and h > 0:
                    img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        h_img, w_img = img.shape

                        # 1. Extract Foreground (Object)
                        x1, y1 = max(0, x), max(0, y)
                        x2, y2 = min(w_img, x + w), min(h_img, y + h)
                        crop = img[y1:y2, x1:x2]

                        if crop.size > 0:
                            metrics["is_valid"] = True

                            # [Metric 1] Texture & Entropy
                            metrics["entropy"] = shannon_entropy(crop)

                            # GLCM (Texture Contrast) - Fast config
                            # Optimize: Only compute for small crops or downsample large ones
                            if (
                                crop.size > 10000
                            ):  # If too large, resize for texture analysis speed
                                texture_crop = cv2.resize(
                                    crop,
                                    (100, int(100 * crop.shape[0] / crop.shape[1])),
                                )
                            else:
                                texture_crop = crop

                            if (
                                texture_crop.shape[0] >= 2
                                and texture_crop.shape[1] >= 2
                            ):
                                glcm = graycomatrix(
                                    texture_crop,
                                    distances=[1],
                                    angles=[0],
                                    levels=256,
                                    symmetric=True,
                                    normed=True,
                                )
                                metrics["texture_contrast"] = graycoprops(
                                    glcm, "contrast"
                                )[0, 0]

                            # [Metric 2] Foreground vs Background Separability (Camouflage)
                            # Optimize: Background margin calculation
                            bg_margin = max(10, int(min(w, h) * 0.2))
                            bx1, by1 = max(0, x1 - bg_margin), max(0, y1 - bg_margin)
                            bx2, by2 = min(w_img, x2 + bg_margin), min(
                                h_img, y2 + bg_margin
                            )

                            bg_crop = img[by1:by2, bx1:bx2]

                            # Calculate Histograms
                            # Mask for FG in the BG crop coordinate system
                            mask_fg = np.zeros_like(bg_crop, dtype=np.uint8)
                            mask_fg[y1 - by1 : y2 - by1, x1 - bx1 : x2 - bx1] = 255
                            mask_bg = cv2.bitwise_not(mask_fg)

                            hist_fg = cv2.calcHist(
                                [bg_crop], [0], mask_fg, [32], [0, 256]
                            )
                            hist_bg = cv2.calcHist(
                                [bg_crop], [0], mask_bg, [32], [0, 256]
                            )

                            cv2.normalize(
                                hist_fg, hist_fg, alpha=1, norm_type=cv2.NORM_L1
                            )
                            cv2.normalize(
                                hist_bg, hist_bg, alpha=1, norm_type=cv2.NORM_L1
                            )

                            # Chi-Square Distance (0 = Identical/Camouflaged, High = Distinct)
                            metrics["fg_bg_separability"] = cv2.compareHist(
                                hist_fg, hist_bg, cv2.HISTCMP_CHISQR
                            )

                            # Collect for PCA (Resize to 64x64)
                            if idx in pca_indices_set:
                                pca_crop = cv2.resize(crop, (64, 64)).flatten()
                                pca_crops.append(pca_crop)

                obj_results.append(metrics)
                current_op += 1
                if current_op % 50 == 0:
                    self.progress.emit(
                        int((current_op / total_ops) * 100),
                        f"Object Analysis: {file_name}",
                    )

            # --- Phase 2: PCA Analysis (Eigen-Objects) ---
            pca_res = {}
            if pca_crops:
                pca_data = np.array(pca_crops)
                # Fit PCA
                # Optimize: Use RandomizedPCA for speed if many samples
                n_components = min(5, len(pca_data))
                pca = PCA(n_components=n_components, svd_solver="randomized")
                pca.fit(pca_data)

                pca_res["mean_image"] = pca.mean_.reshape(64, 64)
                pca_res["eigen_vectors"] = [
                    comp.reshape(64, 64) for comp in pca.components_
                ]
                pca_res["explained_variance"] = pca.explained_variance_ratio_

            # --- Phase 3: Image-Level Analysis (FFT) ---
            # Optimize: Only analyze sample images if too many
            max_img_samples = 500
            if len(images_meta) > max_img_samples:
                target_img_ids = list(images_meta.keys())[:max_img_samples]
            else:
                target_img_ids = images_meta.keys()

            for img_id in target_img_ids:
                if self.isInterruptionRequested():
                    return

                info = images_meta[img_id]
                file_name = info["file_name"]
                full_path = os.path.join(self.root_path, file_name)

                img_metric = {"file_name": file_name, "log_fft_mean": 0, "valid": False}

                if os.path.exists(full_path):
                    # Read small for FFT speed
                    img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.resize(
                            img, (256, 256)
                        )  # Standardize size for comparison
                        f = np.fft.fft2(img)
                        fshift = np.fft.fftshift(f)
                        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
                        img_metric["log_fft_mean"] = np.mean(magnitude_spectrum)
                        img_metric["valid"] = True

                img_results.append(img_metric)
                current_op += 1
                if current_op % 10 == 0:
                    self.progress.emit(
                        int((current_op / total_ops) * 100),
                        f"FFT Analysis: {file_name}",
                    )

            self.finished_analysis.emit(
                pd.DataFrame(obj_results), pd.DataFrame(img_results), pca_res
            )

        except Exception as e:
            import traceback

            traceback.print_exc()
            self.error_occurred.emit(str(e))
