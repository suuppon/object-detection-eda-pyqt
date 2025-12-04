import functools

import numpy as np
import seaborn as sns
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.analysis.quality import QualityAnalysisThread


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

        self.lbl_marked = QLabel("Marked for deletion: 0")
        self.lbl_marked.setStyleSheet("color: red; font-weight: bold;")
        control_layout.addWidget(self.lbl_marked)

        main_layout.addLayout(control_layout)

        # Use QSplitter to allow resizing between graph and table
        splitter = QSplitter(Qt.Vertical)

        # 차트 영역
        self.figure = Figure(figsize=(10, 5))
        self.canvas = FigureCanvas(self.figure)
        splitter.addWidget(self.canvas)

        # Low Quality Images Section
        quality_group = QGroupBox("Low Quality Images (Check to mark for deletion)")
        quality_layout = QVBoxLayout()

        self.quality_table = QTableWidget()
        self.quality_table.setColumnCount(8)
        self.quality_table.setHorizontalHeaderLabels(
            [
                "Marked for deletion",
                "ID",
                "File Name",
                "Brightness",
                "Contrast",
                "Blur Score",
                "Issue",
                "View",
            ]
        )
        self.quality_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.Stretch
        )
        self.quality_table.setMinimumHeight(200)

        quality_layout.addWidget(self.quality_table)
        quality_group.setLayout(quality_layout)
        splitter.addWidget(quality_group)

        # Set splitter proportions (graph:table = 60:40)
        splitter.setSizes([600, 400])
        main_layout.addWidget(splitter)

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
        self.populate_quality_table()
        self.update_marked_count()

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

    def populate_quality_table(self):
        """Populate table with low quality images."""
        if self.analysis_df is None or self.analysis_df.empty:
            return

        df = self.analysis_df[self.analysis_df["file_exists"]].copy()
        if df.empty:
            return

        # Define thresholds for low quality
        # Too dark or too bright
        brightness_issues = (df["brightness"] < 50) | (df["brightness"] > 200)
        # Low blur score (blurry)
        blur_threshold = df["blur_score"].quantile(0.1)  # Bottom 10%
        blur_issues = df["blur_score"] < blur_threshold
        # Low contrast
        contrast_threshold = df["contrast"].quantile(0.1)
        contrast_issues = df["contrast"] < contrast_threshold

        # Combine issues
        df["has_issue"] = brightness_issues | blur_issues | contrast_issues
        df["issue_type"] = ""
        df.loc[
            brightness_issues & (df["brightness"] < 50), "issue_type"
        ] += "Too Dark; "
        df.loc[
            brightness_issues & (df["brightness"] > 200), "issue_type"
        ] += "Too Bright; "
        df.loc[blur_issues, "issue_type"] += "Blurry; "
        df.loc[contrast_issues, "issue_type"] += "Low Contrast; "

        # Filter problematic images
        problem_df = df[df["has_issue"]].copy()

        # Populate table
        self.quality_table.setRowCount(len(problem_df))
        self.quality_table.blockSignals(True)

        for row, (_, img_data) in enumerate(problem_df.iterrows()):
            img_id = img_data["image_id"]
            is_excluded = (
                img_id in self.loader.excluded_image_ids if self.loader else False
            )

            # Checkbox - use QProperty to store img_id
            chk = QCheckBox()
            chk.setProperty("img_id", img_id)  # Store img_id in widget
            # Block signals while setting initial state to prevent unwanted triggers
            chk.blockSignals(True)
            chk.setChecked(is_excluded)
            chk.blockSignals(False)
            chk.stateChanged.connect(self.on_checkbox_state_changed)
            self.quality_table.setCellWidget(row, 0, chk)

            # ID
            self.quality_table.setItem(row, 1, QTableWidgetItem(str(img_id)))

            # File Name
            file_name = (
                self.loader.images[img_id]["file_name"]
                if self.loader and img_id in self.loader.images
                else "N/A"
            )
            self.quality_table.setItem(row, 2, QTableWidgetItem(file_name))

            # Brightness
            self.quality_table.setItem(
                row, 3, QTableWidgetItem(f"{img_data['brightness']:.1f}")
            )

            # Contrast
            self.quality_table.setItem(
                row, 4, QTableWidgetItem(f"{img_data['contrast']:.1f}")
            )

            # Blur Score
            self.quality_table.setItem(
                row, 5, QTableWidgetItem(f"{img_data['blur_score']:.2f}")
            )

            # Issue
            issue_item = QTableWidgetItem(img_data["issue_type"].strip("; "))
            self.quality_table.setItem(row, 6, issue_item)

            # View button
            btn_view = QPushButton("👁 View")
            btn_view.clicked.connect(functools.partial(self.view_image, img_id))
            self.quality_table.setCellWidget(row, 7, btn_view)

            # Highlight if excluded
            if is_excluded:
                self.update_row_highlighting(row, True)

        self.quality_table.blockSignals(False)

    def on_checkbox_state_changed(self, state):
        """Handle any checkbox state change - simplified version."""
        sender = self.sender()
        if not isinstance(sender, QCheckBox):
            return

        if not self.loader:
            return

        # Safely get img_id
        try:
            raw_id = sender.property("img_id")
            if raw_id is None:
                return
            img_id = int(raw_id)
        except (ValueError, TypeError):
            return

        # Mark/unmark image
        # Qt.Checked is 2, Qt.Unchecked is 0.
        # Using explicit integer comparison for robustness.
        if state == 2:
            self.loader.mark_image_for_exclusion(img_id)
        else:
            self.loader.unmark_image_for_exclusion(img_id)

        # Update UI
        self.update_marked_count()
        self.update_row_highlighting_by_img_id(img_id)

    def refresh_marked_status(self):
        """Refresh marked status in table."""
        if not self.loader or not hasattr(self, "quality_table"):
            return

        # Update all rows' highlighting based on current excluded status
        for row in range(self.quality_table.rowCount()):
            item = self.quality_table.item(row, 1)  # ID column
            if item:
                try:
                    img_id = int(item.text())
                    is_excluded = img_id in self.loader.excluded_image_ids
                    # Update checkbox
                    widget = self.quality_table.cellWidget(row, 0)
                    if isinstance(widget, QCheckBox):
                        widget.blockSignals(True)
                        widget.setChecked(is_excluded)
                        widget.blockSignals(False)
                    # Update highlighting
                    self.update_row_highlighting(row, is_excluded)
                except ValueError:
                    continue

        self.update_marked_count()

    def update_row_highlighting_by_img_id(self, img_id):
        """Update highlighting for a specific image ID."""
        is_excluded = img_id in self.loader.excluded_image_ids if self.loader else False

        # Find row with this img_id
        for row in range(self.quality_table.rowCount()):
            item = self.quality_table.item(row, 1)  # ID column
            if item and int(item.text()) == img_id:
                self.update_row_highlighting(row, is_excluded)
                break

    def update_row_highlighting(self, row, is_excluded):
        """Update highlighting for a specific row."""
        for col in range(
            1, self.quality_table.columnCount() - 1
        ):  # Exclude View column (button)
            item = self.quality_table.item(row, col)
            if item:
                if is_excluded:
                    item.setBackground(Qt.red)
                    item.setForeground(Qt.white)
                else:
                    item.setBackground(Qt.transparent)
                    item.setForeground(Qt.black)

    def view_image(self, img_id):
        """Open image in viewer tab."""
        main_window = self.window()
        if hasattr(main_window, "open_image_in_viewer"):
            main_window.open_image_in_viewer(img_id)
        else:
            QMessageBox.warning(self, "Warning", "Viewer not available.")

    def update_marked_count(self):
        """Update the marked images counter."""
        if self.loader:
            count = len(self.loader.excluded_image_ids)
            self.lbl_marked.setText(f"Marked for deletion: {count}")
