"""Spatial analysis widget for object location and density analysis."""

import seaborn as sns
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QVBoxLayout, QWidget


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
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        self.main_layout.addWidget(self.canvas)

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

        # 1. BBox Center Heatmap
        ax1 = self.figure.add_subplot(221)

        cx_norm_list = []
        cy_norm_list = []

        for _, row in df.iterrows():
            img_id = row["image_id"]
            if img_id in self.loader.images:
                img_w = self.loader.images[img_id]["width"]
                img_h = self.loader.images[img_id]["height"]
                x, y, w, h = row["bbox"]
                cx_norm_list.append((x + w / 2) / img_w)
                cy_norm_list.append((y + h / 2) / img_h)

        if cx_norm_list:
            h = ax1.hist2d(
                cx_norm_list,
                cy_norm_list,
                bins=50,
                range=[[0, 1], [0, 1]],
                cmap="hot_r",
            )
            self.figure.colorbar(h[3], ax=ax1)
            ax1.set_title("Normalized Object Center Distribution")
            ax1.invert_yaxis()
        else:
            ax1.text(0.5, 0.5, "Image size info missing", ha="center")

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

        # 3. IoU Distribution (Overlap Analysis) - Simplified (Same Image Objects)
        # Warning: Calculating full IoU for all pairs is slow.
        # Here we approximate by checking overlapping areas or skip full N^2 check for performance.
        # Instead, let's show "BBox Area / Image Area Ratio" distribution which is faster and useful.
        ax3 = self.figure.add_subplot(223)

        ratios = []
        for _, row in df.iterrows():
            img_id = row["image_id"]
            if img_id in self.loader.images:
                img_area = (
                    self.loader.images[img_id]["width"]
                    * self.loader.images[img_id]["height"]
                )
                if img_area > 0:
                    ratios.append(row["area"] / img_area)

        if ratios:
            sns.histplot(ratios, bins=50, ax=ax3, kde=True)
            ax3.set_title("BBox Area / Image Area Ratio")
            ax3.set_xlabel("Ratio (0~1)")
            ax3.set_yscale("log")

        # 4. Class-wise Spatial Distribution (Top 5 Classes)
        ax4 = self.figure.add_subplot(224)
        top_classes = df["category_name"].value_counts().head(5).index

        for cls in top_classes:
            cls_df = df[df["category_name"] == cls]
            # Calculate simplified center spread (std dev of x and y)
            # Or just scatter plot of centers
            cls_cx = []
            cls_cy = []
            for _, row in cls_df.iterrows():
                img_id = row["image_id"]
                if img_id in self.loader.images:
                    img_w = self.loader.images[img_id]["width"]
                    img_h = self.loader.images[img_id]["height"]
                    cls_cx.append((row["bbox"][0] + row["bbox"][2] / 2) / img_w)
                    cls_cy.append((row["bbox"][1] + row["bbox"][3] / 2) / img_h)

            if cls_cx:
                sns.kdeplot(x=cls_cx, y=cls_cy, ax=ax4, label=cls, alpha=0.5)

        ax4.set_title("Spatial Dist. of Top 5 Classes (KDE)")
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.invert_yaxis()
        ax4.legend(fontsize="small")

        self.figure.tight_layout()
        self.canvas.draw()

        # Guide button at bottom (only add once)
        if not hasattr(self, "btn_guide"):
            btn_layout = QHBoxLayout()
            btn_layout.addStretch()
            self.btn_guide = QPushButton("📖 View Guide: Spatial Analysis")
            self.btn_guide.clicked.connect(lambda: self._navigate_to_guide())
            btn_layout.addWidget(self.btn_guide)
            self.main_layout.addLayout(btn_layout)

    def _navigate_to_guide(self):
        """Navigate to the guide tab and scroll to spatial section."""
        main_window = self.window()
        if hasattr(main_window, "navigate_to_guide"):
            main_window.navigate_to_guide("spatial")
