"""Data loader for COCO format object detection datasets."""

import json

import pandas as pd


class CocoDataLoader:
    """Load and preprocess COCO format JSON annotation files."""

    def __init__(self, json_path):
        """Initialize COCO data loader.

        Args:
            json_path: Path to COCO format JSON file.
        """
        with open(json_path, "r") as f:
            self.data = json.load(f)

        # 이미지 정보와 어노테이션 정보를 DataFrame으로 변환하여 매핑 속도 향상
        self.images = {img["id"]: img for img in self.data["images"]}
        self.categories = {cat["id"]: cat["name"] for cat in self.data["categories"]}
        self.annotations = pd.DataFrame(self.data["annotations"])

        # 분석을 위한 파생 변수 추가 (넓이, 비율)
        # COCO bbox format: [x, y, width, height]
        if not self.annotations.empty:
            self.annotations["bbox_w"] = self.annotations["bbox"].apply(lambda x: x[2])
            self.annotations["bbox_h"] = self.annotations["bbox"].apply(lambda x: x[3])
            self.annotations["area"] = (
                self.annotations["bbox_w"] * self.annotations["bbox_h"]
            )
            self.annotations["aspect_ratio"] = (
                self.annotations["bbox_w"] / self.annotations["bbox_h"]
            )
            self.annotations["category_name"] = self.annotations["category_id"].map(
                self.categories
            )
        else:
            # 빈 데이터프레임 처리 (컬럼 생성)
            for col in ["bbox_w", "bbox_h", "area", "aspect_ratio", "category_name"]:
                self.annotations[col] = []

    def get_stats(self):
        """Get basic dataset statistics.

        Returns:
            Dictionary with 'Total Images', 'Total Instances', 'Total Classes'.
        """
        return {
            "Total Images": len(self.images),
            "Total Instances": len(self.annotations),
            "Total Classes": len(self.categories),
        }
