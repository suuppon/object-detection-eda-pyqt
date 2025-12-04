"""Image quality analysis module for brightness, contrast, and blur detection."""

import os

import cv2
import numpy as np
import pandas as pd
from PySide6.QtCore import QThread, Signal


class QualityAnalysisThread(QThread):
    """Thread for analyzing image quality metrics (brightness, contrast, blur)."""

    progress = Signal(int, int)
    finished_analysis = Signal(object)
    error_occurred = Signal(str)

    def __init__(self, image_dict, root_path):
        """Initialize quality analysis thread.

        Args:
            image_dict: Dictionary mapping image_id to image metadata.
            root_path: Root directory path for image files.
        """
        super().__init__()
        self.image_dict = image_dict
        self.root_path = root_path

    def run(self):
        """Execute quality analysis for all images."""
        results = []
        total_images = len(self.image_dict)

        if total_images == 0:
            self.finished_analysis.emit(pd.DataFrame())
            return

        count = 0
        for img_id, img_info in self.image_dict.items():
            if self.isInterruptionRequested():
                break

            file_name = img_info["file_name"]
            full_path = os.path.join(self.root_path, file_name)

            metrics = {
                "image_id": img_id,
                "brightness": 0,
                "contrast": 0,
                "blur_score": 0,
                "width": img_info["width"],
                "height": img_info["height"],
                "file_exists": False,
            }

            if os.path.exists(full_path):
                try:
                    img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        metrics["file_exists"] = True
                        metrics["brightness"] = np.mean(img)
                        metrics["contrast"] = np.std(img)
                        metrics["blur_score"] = cv2.Laplacian(img, cv2.CV_64F).var()

                except Exception as e:
                    print(f"Error processing {file_name}: {e}")

            results.append(metrics)
            count += 1

            if count % 10 == 0 or count == total_images:
                self.progress.emit(int((count / total_images) * 100), total_images)

        df = pd.DataFrame(results)
        self.finished_analysis.emit(df)
