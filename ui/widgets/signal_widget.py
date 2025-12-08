import seaborn as sns
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QProgressDialog,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.analysis.texture_analysis import TextureAnalyzerThread


class SignalWidget(QWidget):
    def __init__(self, data_loader=None):
        super().__init__()
        self.loader = data_loader
        self.df_objects = None
        self.df_images = None
        self.pca_data = None
        self.img_root_path = ""
        self.initUI()

    def initUI(self):
        main_layout = QVBoxLayout(self)

        # Controls
        control_layout = QHBoxLayout()
        self.btn_run = QPushButton("Run Deep Signal Analysis (Slow)")
        self.btn_run.clicked.connect(self.run_analysis)
        control_layout.addWidget(self.btn_run)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)  # 텍스트 보이게
        control_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready (Requires Images)")
        control_layout.addWidget(self.status_label)
        main_layout.addLayout(control_layout)

        # Visualization (2x2 Grid)
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        main_layout.addWidget(self.canvas)

        # Guide button at bottom
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.btn_guide = QPushButton("📖 View Guide: Signal Analysis")
        self.btn_guide.clicked.connect(lambda: self._navigate_to_guide())
        btn_layout.addWidget(self.btn_guide)
        main_layout.addLayout(btn_layout)

    def _navigate_to_guide(self):
        main_window = self.window()
        if hasattr(main_window, "navigate_to_guide"):
            main_window.navigate_to_guide("signal")

    def update_data(self, data_loader):
        """Update data loader when dataset is loaded"""
        self.loader = data_loader
        self.df_objects = None
        self.df_images = None
        self.pca_data = None
        self.figure.clear()
        self.canvas.draw()
        self.status_label.setText("Data Loaded. Ready to analyze (Requires Images).")
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)

    def set_img_root(self, path):
        """Set image root path from main window"""
        self.img_root_path = path

    def run_analysis(self):
        if not self.loader:
            QMessageBox.warning(self, "Warning", "No dataset loaded.")
            return

        # Ask for image path if not set
        if not self.img_root_path:
            self.img_root_path = QFileDialog.getExistingDirectory(
                self, "Select Image Root Directory"
            )
            if not self.img_root_path:
                return

        # Show modal loading dialog
        self.loading_dialog = QProgressDialog(
            "Running texture analysis (Texture, Camouflage, FFT, PCA)...\n\n"
            "Please wait while the analysis is being performed.\n"
            "This may take a while for large datasets.",
            None, 0, 0, self
        )
        self.loading_dialog.setWindowTitle("Analyzing")
        self.loading_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.loading_dialog.setCancelButton(None)
        self.loading_dialog.setMinimumDuration(0)
        self.loading_dialog.setRange(0, 0)
        self.loading_dialog.show()
        QApplication.processEvents()

        self.btn_run.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        self.worker = TextureAnalyzerThread(self.loader, self.img_root_path)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished_analysis.connect(self.on_finished)
        self.worker.error_occurred.connect(self.on_error)
        self.worker.start()

    def update_progress(self, val, msg):
        self.progress_bar.setValue(val)
        self.status_label.setText(msg)

    def on_error(self, msg):
        if hasattr(self, 'loading_dialog'):
            self.loading_dialog.close()
        self.btn_run.setEnabled(True)
        self.status_label.setText("Error")
        QMessageBox.critical(self, "Error", msg)

    def on_finished(self, df_obj, df_img, pca_res):
        if hasattr(self, 'loading_dialog'):
            self.loading_dialog.close()
        self.df_objects = df_obj
        self.df_images = df_img
        self.pca_data = pca_res

        self.btn_run.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Deep Analysis Complete.")
        self.plot_results()

    def plot_results(self):
        self.figure.clear()

        # Filter valid data
        df_obj = self.df_objects[self.df_objects["is_valid"]]

        # 1. Texture Analysis (Entropy vs Contrast)
        ax1 = self.figure.add_subplot(221)
        if not df_obj.empty:
            sns.scatterplot(
                data=df_obj,
                x="entropy",
                y="texture_contrast",
                hue="category",
                alpha=0.6,
                s=15,
                ax=ax1,
                legend=False,
            )
            ax1.set_title("Texture Analysis: Entropy vs Contrast")
            ax1.set_xlabel("Shannon Entropy (Information Content)")
            ax1.set_ylabel("GLCM Contrast (Sharpness)")
            # Insight: Left-Bottom = Featureless objects (Hard to learn)

        # 2. Camouflage Analysis (Fg/Bg Separability)
        ax2 = self.figure.add_subplot(222)
        if not df_obj.empty:
            sns.histplot(df_obj["fg_bg_separability"], kde=True, ax=ax2, color="salmon")
            ax2.set_title("Camouflage Score (Chi-Square Dist)")
            ax2.set_xlabel("Distance (Left=Camouflaged, Right=Distinct)")
            # Insight: Low scores mean objects look just like the background

        # 3. FFT Analysis (Image Quality in Frequency Domain)
        ax3 = self.figure.add_subplot(223)
        if self.df_images is not None and not self.df_images.empty:
            sns.histplot(self.df_images["log_fft_mean"], kde=True, ax=ax3, color="teal")
            ax3.set_title("Image Frequency Spectrum Energy")
            ax3.set_xlabel("Mean Log Magnitude")
            # Insight: Very low values = Blurry/Low-res images

        # 4. Eigen-Objects (Mean Image)
        ax4 = self.figure.add_subplot(224)
        if self.pca_data and "mean_image" in self.pca_data:
            ax4.imshow(self.pca_data["mean_image"], cmap="gray")
            ax4.set_title("The 'Average' Object (PCA Mean)")
            ax4.axis("off")
        else:
            ax4.text(0.5, 0.5, "Insufficient data for PCA", ha="center")

        self.figure.tight_layout()
        self.canvas.draw()
