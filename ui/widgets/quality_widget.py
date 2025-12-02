import numpy as np
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

from core.quality_analysis import QualityAnalysisThread


class ImageQualityWidget(QWidget):
    def __init__(self, data_loader=None):
        super().__init__()
        self.loader = data_loader
        self.analysis_df = None  # 분석 결과 캐싱
        self.img_root_path = ""  # 이미지가 있는 폴더 경로
        self.initUI()

    def initUI(self):
        main_layout = QVBoxLayout(self)

        # 컨트롤 패널 (분석 시작 버튼 등)
        control_layout = QHBoxLayout()

        self.btn_load_path = QPushButton("Analyze Image Quality")
        self.btn_load_path.clicked.connect(self.start_analysis)
        control_layout.addWidget(self.btn_load_path)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setVisible(False)  # 초기에는 숨김
        control_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready")
        control_layout.addWidget(self.status_label)

        main_layout.addLayout(control_layout)

        # 차트 영역
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        main_layout.addWidget(self.canvas)

        # Guide button at bottom
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.btn_guide = QPushButton("📖 View Guide: Image Quality")
        self.btn_guide.clicked.connect(lambda: self._navigate_to_guide())
        btn_layout.addWidget(self.btn_guide)
        main_layout.addLayout(btn_layout)

    def _navigate_to_guide(self):
        main_window = self.window()
        if hasattr(main_window, "navigate_to_guide"):
            main_window.navigate_to_guide("quality")

    def update_data(self, data_loader):
        # Loader가 바뀌면 초기화 (단, 자동 분석은 하지 않음 - 무거우니까)
        self.loader = data_loader
        self.analysis_df = None
        self.figure.clear()
        self.canvas.draw()
        self.status_label.setText("Data Loaded. Click 'Analyze' to scan images.")
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)

    def start_analysis(self):
        if not self.loader or not self.loader.images:
            QMessageBox.warning(self, "Warning", "No dataset loaded.")
            return

        # 이미지 폴더가 설정되지 않았다면 선택 요청
        if not self.img_root_path:
            dir_path = QFileDialog.getExistingDirectory(
                self, "Select Image Root Directory"
            )
            if not dir_path:
                return
            self.img_root_path = dir_path

        self.btn_load_path.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Analyzing images... This may take a while.")

        # 워커 스레드 시작
        self.worker = QualityAnalysisThread(self.loader.images, self.img_root_path)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished_analysis.connect(self.on_analysis_finished)
        self.worker.error_occurred.connect(self.on_error)
        self.worker.start()

    def update_progress(self, percentage, total):
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(percentage)
        self.status_label.setText(f"Processing: {percentage}% ({total} images)")

    def on_analysis_finished(self, df):
        self.btn_load_path.setEnabled(True)
        self.progress_bar.setValue(100)
        self.status_label.setText("Analysis Complete.")
        self.analysis_df = df
        self.plot_charts()

    def on_error(self, error_msg):
        self.btn_load_path.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"Error: {error_msg}")
        QMessageBox.critical(self, "Error", f"Analysis failed: {error_msg}")

    def set_img_root(self, path):
        """Main window에서 이미지 루트 경로를 설정할 수 있도록 함"""
        self.img_root_path = path

    def plot_charts(self):
        if self.analysis_df is None or self.analysis_df.empty:
            return

        self.figure.clear()

        # 데이터 필터링 (파일 있는 것만)
        df = self.analysis_df[self.analysis_df["file_exists"]]

        if df.empty:
            ax = self.figure.add_subplot(111)
            ax.text(
                0.5,
                0.5,
                "No valid images found in the selected directory.",
                ha="center",
            )
            self.canvas.draw()
            return

        # 1. Brightness Distribution
        ax1 = self.figure.add_subplot(221)
        sns.histplot(df["brightness"], bins=30, ax=ax1, color="orange", kde=True)
        ax1.set_title("Brightness Distribution (Mean Pixel)")
        ax1.set_xlabel("Brightness (0=Black, 255=White)")
        # 팁: 너무 어둡거나(<50) 너무 밝은(>200) 데이터 비율 표시해주면 좋음

        # 2. Blur Score (Laplacian Variance) - Log Scale
        ax2 = self.figure.add_subplot(222)
        # 0인 값이 있을 수 있으므로 log 처리를 위해 작은 값 더함
        log_blur = np.log1p(df["blur_score"])
        sns.histplot(log_blur, bins=30, ax=ax2, color="purple", kde=True)
        ax2.set_title("Blur Score Distribution (Log Scale)")
        ax2.set_xlabel("Log(Laplacian Variance)")
        # Insight: 왼쪽 꼬리(Low value)에 있는 이미지들이 'Blurry' 후보군

        # 3. Brightness vs Contrast (Scatter)
        ax3 = self.figure.add_subplot(223)
        sns.scatterplot(data=df, x="brightness", y="contrast", ax=ax3, alpha=0.5, s=15)
        ax3.set_title("Brightness vs Contrast")
        ax3.set_xlabel("Brightness")
        ax3.set_ylabel("Contrast")
        # Insight: 왼쪽 아래(어둡고 대비 낮음), 오른쪽 아래(밝고 대비 낮음) = Low Quality

        # 4. Image Size vs Blur Score
        ax4 = self.figure.add_subplot(224)
        # 크기를 Area로 단순화
        df["img_area"] = df["width"] * df["height"]
        sns.scatterplot(data=df, x="img_area", y="blur_score", ax=ax4, alpha=0.5, s=15)
        ax4.set_title("Image Area vs Blur Score")
        ax4.set_xlabel("Image Area (px)")
        ax4.set_ylabel("Blur Score (Sharpness)")
        ax4.set_yscale("log")
        # Insight: 해상도가 높은데 Blur Score가 낮다면 -> 초점이 나간 '진짜 Blurry' 이미지

        self.figure.tight_layout()
        self.canvas.draw()
