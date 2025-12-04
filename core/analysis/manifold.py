"""HOG/t-SNE manifold and MSCN statistics analysis module."""

import os

import cv2
import numpy as np
import pandas as pd
from PySide6.QtCore import QThread, Signal
from skimage.feature import hog
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


class ManifoldAnalyzerThread(QThread):
    """Thread for HOG features, t-SNE manifold, and MSCN statistics analysis."""

    progress = Signal(int, str)
    finished_analysis = Signal(object, object)  # df_manifold, df_mscn
    error_occurred = Signal(str)

    def __init__(self, loader, root_path, max_samples=2000):
        """Initialize manifold analysis thread.

        Args:
            loader: CocoDataLoader instance with loaded dataset.
            root_path: Root directory path for image files.
            max_samples: Maximum number of samples for t-SNE (default: 2000).
        """
        super().__init__()
        self.loader = loader
        self.root_path = root_path
        self.max_samples = max_samples  # t-SNE는 느리므로 샘플링 필수

    def calculate_mscn_coefficients(self, img):
        """
        MSCN (Mean Subtracted Contrast Normalization)
        자연 이미지 통계(Natural Scene Statistics)의 핵심
        """
        img = img.astype(np.float32) + 1.0  # 0방지
        mu = cv2.GaussianBlur(img, (7, 7), 0)
        mu_sq = mu * mu
        sigma = np.sqrt(np.abs(cv2.GaussianBlur(img * img, (7, 7), 0) - mu_sq))
        structdis = (img - mu) / (sigma + 1)
        return structdis

    def run(self):
        try:
            annotations = self.loader.annotations
            images_meta = self.loader.images

            # --- 1. Data Manifold (HOG + t-SNE) ---
            # Sampling for t-SNE (너무 많으면 오래 걸림)
            # Optimize: Stratified Sampling
            if len(annotations) > self.max_samples:
                # Stratified sampling by category
                grouped = annotations.groupby("category_name")
                n_classes = len(grouped)
                samples_per_class = max(1, self.max_samples // n_classes)

                sampled_dfs = []
                for _, group in grouped:
                    n = min(len(group), samples_per_class)
                    sampled_dfs.append(group.sample(n=n, random_state=42))

                sampled_df = pd.concat(sampled_dfs)
                # If still more than max, sample again
                if len(sampled_df) > self.max_samples:
                    sampled_df = sampled_df.sample(n=self.max_samples, random_state=42)
            else:
                sampled_df = annotations

            hog_features = []
            valid_indices = []
            labels = []

            total_ops = len(sampled_df)
            current = 0

            for idx, row in sampled_df.iterrows():
                if self.isInterruptionRequested():
                    return

                img_id = row["image_id"]
                if img_id not in images_meta:
                    continue

                info = images_meta[img_id]
                full_path = os.path.join(self.root_path, info["file_name"])
                x, y, w, h = map(int, row["bbox"])

                if (
                    os.path.exists(full_path) and w > 16 and h > 16
                ):  # HOG needs min size
                    img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        # Crop & Resize (HOG는 고정 크기 필요)
                        x1, y1 = max(0, x), max(0, y)
                        x2, y2 = min(info["width"], x + w), min(info["height"], y + h)
                        crop = img[y1:y2, x1:x2]

                        if crop.size > 0:
                            crop_resized = cv2.resize(
                                crop, (64, 128)
                            )  # Standard HOG size
                            # Compute HOG
                            fd = hog(
                                crop_resized,
                                orientations=8,
                                pixels_per_cell=(16, 16),
                                cells_per_block=(1, 1),
                                visualize=False,
                            )

                            hog_features.append(fd)
                            valid_indices.append(idx)
                            labels.append(row["category_name"])

                current += 1
                if current % 50 == 0:
                    self.progress.emit(
                        int((current / total_ops) * 50),
                        f"Extracting HOG Features: {current}/{total_ops}",
                    )

            # Run t-SNE
            manifold_df = pd.DataFrame()
            if hog_features:
                self.progress.emit(55, "Running t-SNE (Dimensionality Reduction)...")
                X = np.array(hog_features)
                # Normalize features
                X = StandardScaler().fit_transform(X)

                # t-SNE (Perplexity는 보통 30)
                # Optimize: Use PCA init for faster convergence
                tsne = TSNE(
                    n_components=2,
                    perplexity=min(30, len(X) - 1),
                    random_state=42,
                    init="pca",
                    learning_rate="auto",
                    n_jobs=-1,  # Use all cores
                )
                X_embedded = tsne.fit_transform(X)

                manifold_df = pd.DataFrame(X_embedded, columns=["x", "y"])
                manifold_df["category"] = labels
                manifold_df["ann_id"] = valid_indices

            # --- 2. Natural Scene Statistics (MSCN) ---
            mscn_results = []
            # 전체 이미지를 다 하면 느리니 이미지 단위로 100장만 샘플링 (Demo용)
            img_ids = list(images_meta.keys())
            np.random.shuffle(img_ids)
            target_imgs = img_ids[:100]

            total_mscn = len(target_imgs)
            for i, img_id in enumerate(target_imgs):
                if self.isInterruptionRequested():
                    return

                info = images_meta[img_id]
                full_path = os.path.join(self.root_path, info["file_name"])

                if os.path.exists(full_path):
                    img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.resize(img, (512, 512))  # Normalize size

                        # Calculate MSCN
                        mscn = self.calculate_mscn_coefficients(img)

                        # MSCN 통계량 (이게 Gaussian을 따라야 함)
                        # 자연스러운 이미지일수록 Kurtosis(첨도)가 일정 범위 내에 있음
                        mscn_var = np.var(mscn)

                        mscn_results.append(
                            {"file_name": info["file_name"], "mscn_var": mscn_var}
                        )

                self.progress.emit(
                    80 + int((i / total_mscn) * 20),
                    f"Calculating Natural Stats: {i}/{total_mscn}",
                )

            self.finished_analysis.emit(manifold_df, pd.DataFrame(mscn_results))

        except Exception as e:
            import traceback

            traceback.print_exc()
            self.error_occurred.emit(str(e))
