from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressDialog,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class DifficultyWidget(QWidget):
    def __init__(self, data_loader=None):
        super().__init__()
        self.loader = data_loader
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)

        # 상단 패널
        top_panel = QGroupBox("Hard Sample Discovery")
        top_layout = QHBoxLayout()

        self.btn_analyze = QPushButton("🔍 Analyze Difficulty")
        self.btn_analyze.setFixedWidth(150)
        self.btn_analyze.clicked.connect(self.run_analysis)

        self.lbl_summary = QLabel("Ready to analyze.")

        top_layout.addWidget(self.btn_analyze)
        top_layout.addWidget(self.lbl_summary)
        top_panel.setLayout(top_layout)

        layout.addWidget(top_panel)

        # 결과 테이블
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(
            ["Image ID", "Difficulty Score", "Small Objects", "Total Objects", "Action"]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSortingEnabled(True)  # 점수 정렬 가능하도록

        layout.addWidget(self.table)

        # Guide button at bottom
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.btn_guide = QPushButton("📖 View Guide: Difficulty Analysis")
        self.btn_guide.clicked.connect(lambda: self._navigate_to_guide())
        btn_layout.addWidget(self.btn_guide)
        layout.addLayout(btn_layout)

    def _navigate_to_guide(self):
        main_window = self.window()
        if hasattr(main_window, "navigate_to_guide"):
            main_window.navigate_to_guide("difficulty")

    def update_data(self, data_loader):
        self.loader = data_loader
        self.table.setRowCount(0)
        self.lbl_summary.setText("Data loaded. Click Analyze.")

    def run_analysis(self):
        if not self.loader:
            return

        # Show modal loading dialog
        self.loading_dialog = QProgressDialog(
            "Analyzing image difficulty...\n\n"
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

        self.table.setRowCount(0)
        df = self.loader.annotations
        if df.empty:
            self.loading_dialog.close()
            return

        # 이미지별 난이도 분석
        # 1. Small Object Count
        # 2. Total Object Count (Density)

        # 이미지별 집계
        img_stats = (
            df.groupby("image_id")
            .agg(
                {
                    "bbox": "count",  # Total objects
                    "area": lambda x: (x < 32**2).sum(),  # Small objects
                }
            )
            .rename(columns={"bbox": "total_count", "area": "small_count"})
        )

        # 점수 계산 (단순 예시: Small 비중 * 0.5 + Total count * 0.1)
        # 정규화가 필요하지만 여기선 단순 가중치
        img_stats["score"] = (
            img_stats["small_count"] / img_stats["total_count"].replace(0, 1)
        ) * 50 + (
            img_stats["total_count"] / 50.0
        ) * 50  # 50개 이상이면 고밀도 점수 높음

        img_stats = img_stats.sort_values("score", ascending=False).head(
            100
        )  # 상위 100개만 표시

        # Close loading dialog
        self.loading_dialog.close()

        self.table.setRowCount(len(img_stats))
        for i, (img_id, row) in enumerate(img_stats.iterrows()):
            self.table.setItem(i, 0, QTableWidgetItem(str(img_id)))

            # 숫자 정렬을 위해 DataRole 설정 필요하나 간편하게 str로
            score_item = QTableWidgetItem(f"{row['score']:.1f}")
            self.table.setItem(i, 1, score_item)

            self.table.setItem(i, 2, QTableWidgetItem(str(int(row["small_count"]))))
            self.table.setItem(i, 3, QTableWidgetItem(str(int(row["total_count"]))))

            btn_view = QPushButton("View")
            btn_view.clicked.connect(lambda checked, eid=img_id: self.open_viewer(eid))
            self.table.setCellWidget(i, 4, btn_view)

        self.lbl_summary.setText(
            f"Analysis Complete. Showing top {len(img_stats)} hard samples."
        )

    def open_viewer(self, img_id):
        main_window = self.window()
        if hasattr(main_window, "open_image_in_viewer"):
            main_window.open_image_in_viewer(img_id)
