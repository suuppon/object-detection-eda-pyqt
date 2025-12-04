"""Analysis utilities for object detection dataset."""

from sklearn.cluster import KMeans


class StatisticsAnalyzer:
    """Static analysis methods for dataset statistics and health checks."""

    @staticmethod
    def get_kmeans_anchors(df, k=9):
        """Compute K-Means anchor boxes from bounding box dimensions.

        Args:
            df: DataFrame with 'bbox_w' and 'bbox_h' columns.
            k: Number of clusters (default: 9).

        Returns:
            Tuple of (cluster_centers, labels) or (None, None) if df is empty.
        """
        if df.empty:
            return None, None
        X = df[["bbox_w", "bbox_h"]].values
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        return kmeans.cluster_centers_, kmeans.labels_

    @staticmethod
    def get_size_distribution(df):
        """Calculate Small/Medium/Large object distribution (COCO metric).

        Args:
            df: DataFrame with 'area' column.

        Returns:
            List of [small_count, medium_count, large_count].
        """
        if df.empty:
            return [0, 0, 0]
        area = df["area"]
        small = (area < 32**2).sum()
        medium = ((area >= 32**2) & (area < 96**2)).sum()
        large = (area >= 96**2).sum()
        return [small, medium, large]

    @staticmethod
    def check_health(df, images_dict):
        """Check dataset health and detect annotation errors.

        Args:
            df: DataFrame with annotations.
            images_dict: Dictionary mapping image_id to image metadata.

        Returns:
            List of error dictionaries with 'type', 'img_id', 'detail', 'bbox' keys.
        """
        errors = []
        if df.empty:
            return errors

        # 1. Tiny Boxes
        tiny_mask = (df["bbox_w"] < 1) | (df["bbox_h"] < 1)
        tiny_boxes = df[tiny_mask]
        for _, row in tiny_boxes.iterrows():
            errors.append(
                {
                    "type": "Tiny Box",
                    "img_id": row["image_id"],
                    "ann_id": row["id"],
                    "detail": f"w={row['bbox_w']:.1f}, h={row['bbox_h']:.1f}",
                    "bbox": row["bbox"],
                }
            )

        # 2. Out of Bounds & 3. Giant Boxes
        for _, row in df.iterrows():
            img_id = row["image_id"]
            if img_id not in images_dict:
                continue

            img_w = images_dict[img_id]["width"]
            img_h = images_dict[img_id]["height"]
            x, y, w, h = row["bbox"]

            # OOB
            if x < 0 or y < 0 or (x + w) > img_w or (y + h) > img_h:
                errors.append(
                    {
                        "type": "Out of Bounds",
                        "img_id": img_id,
                        "ann_id": row["id"],
                        "detail": f"Box[{x},{y},{w},{h}] vs Img[{img_w}x{img_h}]",
                        "bbox": row["bbox"],
                    }
                )

            # Giant Box
            img_area = img_w * img_h
            if img_area > 0 and (row["area"] / img_area) > 0.95:
                errors.append(
                    {
                        "type": "Giant Box",
                        "img_id": img_id,
                        "ann_id": row["id"],
                        "detail": f"Area Ratio: {row['area'] / img_area:.2f}",
                        "bbox": row["bbox"],
                    }
                )

        return errors
