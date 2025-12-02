from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.analyzer import Analyzer


class HealthCheckWidget(QWidget):
    def __init__(self, data_loader=None):
        super().__init__()
        self.loader = data_loader
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)

        top_panel = QGroupBox("Data Hygiene Check")
        top_layout = QHBoxLayout()

        self.btn_scan = QPushButton("🔍 Scan for Errors")
        self.btn_scan.setFixedWidth(150)
        self.btn_scan.clicked.connect(self.run_scan)

        self.lbl_summary = QLabel("Ready to scan.")

        top_layout.addWidget(self.btn_scan)
        top_layout.addWidget(self.lbl_summary)
        top_panel.setLayout(top_layout)

        layout.addWidget(top_panel)

        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(
            ["Error Type", "Image ID", "BBox / Details", "Action"]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        layout.addWidget(self.table)

        # Guide button at bottom
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.btn_guide = QPushButton("📖 View Guide: Data Health Check")
        self.btn_guide.clicked.connect(lambda: self._navigate_to_guide())
        btn_layout.addWidget(self.btn_guide)
        layout.addLayout(btn_layout)

    def _navigate_to_guide(self):
        main_window = self.window()
        if hasattr(main_window, "navigate_to_guide"):
            main_window.navigate_to_guide("health")

    def update_data(self, data_loader):
        self.loader = data_loader
        self.table.setRowCount(0)
        self.lbl_summary.setText("Data loaded. Click Scan to check health.")

    def run_scan(self):
        if not self.loader:
            return

        self.table.setRowCount(0)
        errors = Analyzer.check_health(self.loader.annotations, self.loader.images)

        self.table.setRowCount(len(errors))
        for i, err in enumerate(errors):
            self.table.setItem(i, 0, QTableWidgetItem(err["type"]))
            self.table.setItem(i, 1, QTableWidgetItem(str(err["img_id"])))
            self.table.setItem(i, 2, QTableWidgetItem(err["detail"]))

            btn_view = QPushButton("View")
            btn_view.clicked.connect(
                lambda checked, eid=err["img_id"]: self.open_viewer(eid)
            )
            self.table.setCellWidget(i, 3, btn_view)

        self.lbl_summary.setText(f"Scan Complete. Found {len(errors)} issues.")

    def open_viewer(self, img_id):
        main_window = self.window()
        if hasattr(main_window, "open_image_in_viewer"):
            main_window.open_image_in_viewer(img_id)
