"""Geometry analysis widget for anchor box and size distribution analysis."""

import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.analyzer import Analyzer


class GeometryWidget(QWidget):
    """Widget for geometry analysis including anchor boxes, aspect ratios, and size distribution."""

    def __init__(self, data_loader=None):
        """Initialize the geometry widget.

        Args:
            data_loader: CocoDataLoader instance (optional).
        """
        super().__init__()
        self.loader = data_loader
        self.initUI()

    def initUI(self):
        """Initialize the UI components."""
        layout = QVBoxLayout(self)

        # 컨트롤 패널
        control_panel = QHBoxLayout()

        self.k_spin = QSpinBox()
        self.k_spin.setRange(1, 20)
        self.k_spin.setValue(9)
        self.k_spin.setPrefix("K = ")

        self.btn_kmeans = QPushButton("Run K-Means (Anchor Analysis)")
        self.btn_kmeans.clicked.connect(self.run_kmeans)

        control_panel.addWidget(QLabel("Anchor Box Analysis:"))
        control_panel.addWidget(self.k_spin)
        control_panel.addWidget(self.btn_kmeans)
        control_panel.addStretch()

        layout.addLayout(control_panel)

        # 그래프 영역
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Guide button at bottom
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.btn_guide = QPushButton("📖 View Guide: Geometry Analysis")
        self.btn_guide.clicked.connect(lambda: self._navigate_to_guide())
        btn_layout.addWidget(self.btn_guide)
        layout.addLayout(btn_layout)

    def _navigate_to_guide(self):
        """Navigate to the guide tab and scroll to geometry section."""
        main_window = self.window()
        if hasattr(main_window, "navigate_to_guide"):
            main_window.navigate_to_guide("geometry")

    def update_data(self, data_loader):
        """Update the data loader and refresh plots.

        Args:
            data_loader: CocoDataLoader instance.
        """
        self.loader = data_loader
        self.plot_charts()

    def plot_charts(self):
        """Plot all geometry analysis charts."""
        if not self.loader or self.loader.annotations.empty:
            return

        self.figure.clear()
        df = self.loader.annotations

        # 2x3 그리드 사용

        # 1. Width vs Height Scatter Plot
        ax1 = self.figure.add_subplot(231)
        ax1.scatter(df["bbox_w"], df["bbox_h"], alpha=0.1, s=2, c="blue")
        ax1.set_title("BBox Width vs Height Distribution")
        ax1.set_xlabel("Width")
        ax1.set_ylabel("Height")
        ax1.grid(True, alpha=0.3)

        # 2. Aspect Ratio Histogram
        ax2 = self.figure.add_subplot(232)
        sns.histplot(df["aspect_ratio"], bins=50, ax=ax2, kde=True)
        ax2.set_title("Aspect Ratio Distribution (w/h)")
        ax2.set_xlabel("Ratio")

        # 3. Small/Medium/Large Ratio (Pie Chart)
        ax3 = self.figure.add_subplot(233)
        sizes = Analyzer.get_size_distribution(df)
        labels = ["Small (<32²)", "Medium (32²~96²)", "Large (>96²)"]
        colors = ["#ff9999", "#66b3ff", "#99ff99"]

        ax3.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90, colors=colors)
        ax3.set_title("Object Size Distribution (COCO Metric)")

        # 4. Image Resolution Analysis (New)
        ax4 = self.figure.add_subplot(234)
        img_df = pd.DataFrame.from_dict(self.loader.images, orient="index")
        if (
            not img_df.empty
            and "width" in img_df.columns
            and "height" in img_df.columns
        ):
            ax4.scatter(img_df["width"], img_df["height"], alpha=0.3, s=10, c="purple")
            ax4.set_title("Image Resolution Distribution")
            ax4.set_xlabel("Image Width")
            ax4.set_ylabel("Image Height")
            ax4.grid(True)
        else:
            ax4.text(0.5, 0.5, "No Image Info", ha="center")

        # 5. K-Means Result (Placeholder)
        self.ax_kmeans = self.figure.add_subplot(235)
        self.ax_kmeans.text(
            0.5, 0.5, "Click 'Run K-Means' to analyze anchors", ha="center", va="center"
        )
        self.ax_kmeans.axis("off")

        self.figure.tight_layout()
        self.canvas.draw()

    def run_kmeans(self):
        """Run K-Means clustering on bounding box dimensions for anchor analysis."""
        if not self.loader or self.loader.annotations.empty:
            return

        k = self.k_spin.value()
        centers, labels = Analyzer.get_kmeans_anchors(self.loader.annotations, k)

        self.ax_kmeans.clear()
        self.ax_kmeans.axis("on")

        X = self.loader.annotations[["bbox_w", "bbox_h"]].values
        self.ax_kmeans.scatter(
            X[:, 0], X[:, 1], c=labels, cmap="viridis", alpha=0.1, s=2
        )
        self.ax_kmeans.scatter(
            centers[:, 0],
            centers[:, 1],
            c="red",
            marker="x",
            s=100,
            linewidths=2,
            label="Anchors",
        )

        for cx, cy in centers:
            self.ax_kmeans.text(
                cx,
                cy,
                f"({int(cx)},{int(cy)})",
                color="red",
                fontsize=8,
                fontweight="bold",
            )

        self.ax_kmeans.set_title(f"K-Means Clustering (K={k})")
        self.ax_kmeans.set_xlabel("Width")
        self.ax_kmeans.set_ylabel("Height")
        self.ax_kmeans.legend()
        self.ax_kmeans.grid(True, alpha=0.3)

        self.canvas.draw()
