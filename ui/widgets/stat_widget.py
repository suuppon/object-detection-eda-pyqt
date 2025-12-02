"""Statistics widget for displaying class distribution."""

import seaborn as sns
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QVBoxLayout, QWidget


class StatWidget(QWidget):
    """Widget for displaying class distribution statistics."""

    def __init__(self):
        """Initialize the statistics widget."""
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)

        # Guide button at bottom
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.btn_guide = QPushButton("📖 View Guide: Dashboard")
        self.btn_guide.clicked.connect(lambda: self._navigate_to_guide("Dashboard"))
        btn_layout.addWidget(self.btn_guide)
        self.layout.addLayout(btn_layout)

    def _navigate_to_guide(self, section):
        """Navigate to the guide tab and scroll to dashboard section."""
        main_window = self.window()
        if hasattr(main_window, "navigate_to_guide"):
            main_window.navigate_to_guide("dashboard")

    def plot_class_distribution(self, df):
        """Plot class distribution bar chart.

        Args:
            df: DataFrame with 'category_name' column.
        """
        self.figure.clear()
        if df.empty:
            return

        ax = self.figure.add_subplot(111)
        # 클래스별 개수 세기 및 시각화
        sns.countplot(
            y="category_name",
            data=df,
            order=df["category_name"].value_counts().index,
            ax=ax,
        )
        ax.set_title("Class Distribution")
        ax.set_xlabel("Count")
        self.figure.tight_layout()
        self.canvas.draw()
