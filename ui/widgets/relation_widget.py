"""Class relation analysis widget for co-occurrence and imbalance analysis."""

import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import QApplication, QHBoxLayout, QPushButton, QVBoxLayout, QWidget


class RelationWidget(QWidget):
    """Widget for class relation analysis including co-occurrence matrix and class imbalance."""

    def __init__(self, data_loader=None):
        """Initialize the relation widget.

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
        self.btn_guide = QPushButton("View Guide: Class Relation")
        # Note: Signal connection is handled in main_window
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
        """Plot all class relation analysis charts."""
        if not self.loader or self.loader.annotations.empty:
            return

        self.figure.clear()
        df = self.loader.annotations

        # 1. Class Imbalance
        ax1 = self.figure.add_subplot(221)
        counts = df["category_name"].value_counts()
        sns.barplot(x=counts.index, y=counts.values, ax=ax1, hue=counts.index, palette="viridis", legend=False)
        ax1.set_title("Class Distribution (Log Scale)")
        ax1.set_yscale("log")
        ax1.set_xticks(ax1.get_xticks())
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha="right")
        QApplication.processEvents()

        # 2. Co-occurrence Matrix
        ax2 = self.figure.add_subplot(222)
        img_cat_matrix = pd.crosstab(df["image_id"], df["category_name"])
        img_cat_matrix = (img_cat_matrix > 0).astype(int)
        co_matrix = img_cat_matrix.T.dot(img_cat_matrix)

        sns.heatmap(co_matrix, annot=True, fmt="d", cmap="Blues", ax=ax2)
        ax2.set_title("Class Co-occurrence Matrix")
        QApplication.processEvents()

        # 3. Class-wise Average Area (New)
        ax3 = self.figure.add_subplot(223)
        avg_areas = (
            df.groupby("category_name")["area"].mean().sort_values(ascending=False)
        )
        sns.barplot(x=avg_areas.index, y=avg_areas.values, ax=ax3, hue=avg_areas.index, palette="magma", legend=False)
        ax3.set_title("Average BBox Area per Class")
        ax3.set_xticks(ax3.get_xticks())
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha="right")
        ax3.set_yscale("log")
        QApplication.processEvents()

        # 4. Class-wise Aspect Ratio Boxplot (New)
        ax4 = self.figure.add_subplot(224)
        sns.boxplot(
            x="category_name", y="aspect_ratio", data=df, ax=ax4, hue="category_name", palette="Set2", legend=False
        )
        ax4.set_title("Aspect Ratio Distribution per Class")
        ax4.set_xticks(ax4.get_xticks())
        ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha="right")
        ax4.set_ylim(0, 5)  # 극단적인 값 제외하고 보기 위해 제한
        QApplication.processEvents()

        self.figure.tight_layout()
        self.canvas.draw()
        QApplication.processEvents()

    def _navigate_to_guide(self):
        """Navigate to the guide tab and scroll to relation section."""
        pass
