"""Spatial analysis widget for object location and density analysis."""

import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import QApplication, QHBoxLayout, QPushButton, QVBoxLayout, QWidget


class SpatialWidget(QWidget):
    """Widget for spatial analysis including center heatmaps, density, and class-wise distribution."""

    def __init__(self, data_loader=None):
        """Initialize the spatial widget.

        Args:
            data_loader: CocoDataLoader instance (optional).
        """
        super().__init__()
        self.loader = data_loader
        self.main_layout = None
        self.initUI()

    def initUI(self):
        """Initialize the UI components."""
        self.main_layout = QVBoxLayout(self)

        # Canvas
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        self.main_layout.addWidget(self.canvas)

        # Guide Button
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.btn_guide = QPushButton("View Guide")
        # Note: Signal connection is handled in main_window or locally if needed
        # self.btn_guide.clicked.connect(self._navigate_to_guide)
        btn_layout.addWidget(self.btn_guide)
        self.main_layout.addLayout(btn_layout)

    def update_data(self, data_loader):
        """Update the data loader and refresh plots.

        Args:
            data_loader: CocoDataLoader instance.
        """
        self.loader = data_loader
        self.plot_charts()

    def plot_charts(self):
        """Plot all spatial analysis charts."""
        if not self.loader or self.loader.annotations.empty:
            return

        self.figure.clear()
        df = self.loader.annotations

        # 2x2 그리드 사용 (1: Center Heatmap, 2: Objects per Image, 3: IoU Dist, 4: Class Spatial - Placeholder)

        # 1. BBox Center Heatmap (Vectorized)
        ax1 = self.figure.add_subplot(221)

        # Vectorized approach: merge with image info
        img_df = pd.DataFrame.from_dict(self.loader.images, orient="index")
        if not img_df.empty and "width" in img_df.columns and "height" in img_df.columns:
            df_merged = df.merge(img_df[["width", "height"]], left_on="image_id", right_index=True, how="inner")
            if not df_merged.empty:
                # Calculate normalized centers using vectorized operations
                import numpy as np
                bbox_array = np.array(df_merged["bbox"].tolist())
                x, y, w, h = bbox_array[:, 0], bbox_array[:, 1], bbox_array[:, 2], bbox_array[:, 3]
                cx_norm = (x + w / 2) / df_merged["width"].values
                cy_norm = (y + h / 2) / df_merged["height"].values
                
                h = ax1.hist2d(
                    cx_norm,
                    cy_norm,
                    bins=50,
                    range=[[0, 1], [0, 1]],
                    cmap="hot_r",
                )
                self.figure.colorbar(h[3], ax=ax1)
                ax1.set_title("Normalized Object Center Distribution")
                ax1.invert_yaxis()
            else:
                ax1.text(0.5, 0.5, "Image size info missing", ha="center")
        else:
            ax1.text(0.5, 0.5, "Image size info missing", ha="center")
        
        QApplication.processEvents()

        # 2. Objects per Image Histogram
        ax2 = self.figure.add_subplot(222)
        counts = df["image_id"].value_counts()

        all_img_ids = set(self.loader.images.keys())
        obj_img_ids = set(counts.index)
        zero_count = len(all_img_ids - obj_img_ids)
        data_counts = counts.tolist() + [0] * zero_count

        sns.histplot(data_counts, bins=30, kde=False, ax=ax2)
        ax2.set_title("Objects per Image Distribution")
        ax2.set_yscale("log")
        ax2.set_xlabel("Objects Count")
        
        QApplication.processEvents()

        # 3. IoU Distribution (Overlap Analysis) - Simplified (Same Image Objects)
        # Warning: Calculating full IoU for all pairs is slow.
        # Here we approximate by checking overlapping areas or skip full N^2 check for performance.
        # Instead, let's show "BBox Area / Image Area Ratio" distribution which is faster and useful.
        ax3 = self.figure.add_subplot(223)

        # Vectorized approach
        if not img_df.empty and "width" in img_df.columns and "height" in img_df.columns:
            df_merged = df.merge(img_df[["width", "height"]], left_on="image_id", right_index=True, how="inner")
            if not df_merged.empty:
                img_area = df_merged["width"] * df_merged["height"]
                ratios = df_merged["area"] / img_area
                ratios = ratios[ratios > 0]  # Filter out zero/negative ratios
                
                if len(ratios) > 0:
                    sns.histplot(ratios, bins=50, ax=ax3, kde=True)
                    ax3.set_title("BBox Area / Image Area Ratio")
                    ax3.set_xlabel("Ratio (0~1)")
                    ax3.set_yscale("log")
        
        QApplication.processEvents()

        # 4. Class-wise Spatial Distribution (Top 5 Classes)
        ax4 = self.figure.add_subplot(224)
        top_classes = df["category_name"].value_counts().head(5).index

        plotted_labels = []
        for cls in top_classes:
            cls_df = df[df["category_name"] == cls]
            if cls_df.empty:
                continue
                
            # Vectorized approach
            if not img_df.empty and "width" in img_df.columns and "height" in img_df.columns:
                cls_merged = cls_df.merge(img_df[["width", "height"]], left_on="image_id", right_index=True, how="inner")
                if not cls_merged.empty:
                    import numpy as np
                    bbox_array = np.array(cls_merged["bbox"].tolist())
                    x, y, w, h = bbox_array[:, 0], bbox_array[:, 1], bbox_array[:, 2], bbox_array[:, 3]
                    cls_cx = (x + w / 2) / cls_merged["width"].values
                    cls_cy = (y + h / 2) / cls_merged["height"].values
                    
                    if len(cls_cx) > 0:
                        sns.kdeplot(x=cls_cx, y=cls_cy, ax=ax4, label=cls, alpha=0.5)
                        plotted_labels.append(cls)
            
            QApplication.processEvents()

        ax4.set_title("Spatial Dist. of Top 5 Classes (KDE)")
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.invert_yaxis()
        # Only show legend if we actually plotted something with labels
        if plotted_labels:
            # Check if there are any labeled artists before calling legend
            handles, labels = ax4.get_legend_handles_labels()
            if handles and labels:
                ax4.legend(fontsize="small")

        self.figure.tight_layout()
        self.canvas.draw()
        QApplication.processEvents()

    def _navigate_to_guide(self):
        """Navigate to the guide tab and scroll to spatial section."""
        # This method is kept for compatibility if called internally,
        # but mainly the signal is connected in MainWindow.
        pass
