import seaborn as sns
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.analysis.manifold import ManifoldAnalyzerThread


class AdvancedWidget(QWidget):
    def __init__(self, data_loader=None):
        super().__init__()
        self.loader = data_loader
        self.df_manifold = None
        self.df_mscn = None
        self.img_root_path = ""
        self.initUI()

    def initUI(self):
        main_layout = QVBoxLayout(self)

        # Controls
        control_layout = QHBoxLayout()
        self.btn_run = QPushButton("Run Top-tier Analysis (HOG/t-SNE + NSS)")
        self.btn_run.clicked.connect(self.run_analysis)
        control_layout.addWidget(self.btn_run)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)  # 텍스트 보이게
        control_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready")
        control_layout.addWidget(self.status_label)
        main_layout.addLayout(control_layout)

        # Plot
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        main_layout.addWidget(self.canvas)

        # Guide button at bottom
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.btn_guide = QPushButton("📖 View Guide: Advanced Analysis")
        self.btn_guide.clicked.connect(lambda: self._navigate_to_guide())
        btn_layout.addWidget(self.btn_guide)
        main_layout.addLayout(btn_layout)

    def _navigate_to_guide(self):
        main_window = self.window()
        if hasattr(main_window, "navigate_to_guide"):
            main_window.navigate_to_guide("advanced")

    def run_analysis(self):
        if not self.loader:
            QMessageBox.warning(self, "Warning", "No dataset loaded.")
            return
        if not self.img_root_path:
            self.img_root_path = QFileDialog.getExistingDirectory(
                self, "Select Image Root Directory"
            )
            if not self.img_root_path:
                return

        self.btn_run.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.worker = ManifoldAnalyzerThread(self.loader, self.img_root_path)
        self.worker.progress.connect(
            lambda v, m: (self.progress_bar.setValue(v), self.status_label.setText(m))
        )
        self.worker.finished_analysis.connect(self.on_finished)
        self.worker.error_occurred.connect(self.on_error)
        self.worker.start()

    def on_error(self, msg):
        self.btn_run.setEnabled(True)
        self.status_label.setText("Error")
        QMessageBox.critical(self, "Error", msg)

    def update_data(self, data_loader):
        """Update data loader when dataset is loaded"""
        self.loader = data_loader
        self.df_manifold = None
        self.df_mscn = None
        self.figure.clear()
        self.canvas.draw()
        self.status_label.setText("Data Loaded. Ready to analyze (Requires Images).")
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)

    def set_img_root(self, path):
        """Set image root path from main window"""
        self.img_root_path = path

    def on_finished(self, df_manifold, df_mscn):
        self.df_manifold = df_manifold
        self.df_mscn = df_mscn
        self.btn_run.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Done.")
        self.plot_charts()

    def plot_charts(self):
        self.figure.clear()

        # 1. t-SNE Manifold
        ax1 = self.figure.add_subplot(121)
        if not self.df_manifold.empty:
            sns.scatterplot(
                data=self.df_manifold,
                x="x",
                y="y",
                hue="category",
                ax=ax1,
                palette="tab10",
                s=20,
                alpha=0.7,
            )
            ax1.set_title("Dataset Manifold (HOG + t-SNE)")
            ax1.set_xlabel("Latent Dim 1")
            ax1.set_ylabel("Latent Dim 2")
            # Insight: 같은 색깔끼리 뭉쳐 있어야 정상.
            # 엉뚱한 색깔이 섞여 있으면 Label Noise.

        # 2. Natural Scene Statistics (MSCN Variance)
        ax2 = self.figure.add_subplot(122)
        if not self.df_mscn.empty:
            sns.histplot(self.df_mscn["mscn_var"], kde=True, ax=ax2, color="green")
            ax2.set_title("Naturalness Score (MSCN Variance)")
            ax2.set_xlabel("Variance (Deviations from Gaussian)")
            # Insight: 특정 값 주변에 정규분포를 그려야 함.
            # 꼬리가 길거나 봉우리가 두 개면 데이터 품질 이슈.

        self.figure.tight_layout()
        self.canvas.draw()
