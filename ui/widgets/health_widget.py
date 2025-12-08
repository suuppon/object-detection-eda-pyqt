from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
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

from core.analysis.statistics import StatisticsAnalyzer


class HealthWidget(QWidget):
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
        self.lbl_marked = QLabel("Marked for deletion: 0")
        self.lbl_marked.setStyleSheet("color: red; font-weight: bold;")

        top_layout.addWidget(self.btn_scan)
        top_layout.addWidget(self.lbl_summary)
        top_layout.addWidget(self.lbl_marked)
        top_panel.setLayout(top_layout)

        layout.addWidget(top_panel)

        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(
            [
                "Marked for deletion",
                "Error Type",
                "Image ID",
                "BBox / Details",
                "Action",
            ]
        )
        self.table.setColumnWidth(0, 80)
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

        # Show modal loading dialog
        self.loading_dialog = QProgressDialog(
            "Scanning for data health issues...\n\n"
            "Please wait while the dataset is being analyzed.\n"
            "This may take a while for large datasets.",
            None, 0, 0, self
        )
        self.loading_dialog.setWindowTitle("Scanning")
        self.loading_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.loading_dialog.setCancelButton(None)
        self.loading_dialog.setMinimumDuration(0)
        self.loading_dialog.setRange(0, 0)
        self.loading_dialog.show()
        QApplication.processEvents()

        self.table.setRowCount(0)
        errors = StatisticsAnalyzer.check_health(
            self.loader.annotations, self.loader.images
        )
        
        # Close loading dialog
        self.loading_dialog.close()

        self.table.setRowCount(len(errors))

        for i, err in enumerate(errors):
            img_id = err["img_id"]
            is_excluded = img_id in self.loader.excluded_image_ids

            # Checkbox for marking IMAGE (not bbox)
            chk = QCheckBox()
            chk.setProperty("img_id", img_id)
            chk.setChecked(is_excluded)
            chk.stateChanged.connect(self.on_checkbox_state_changed)
            self.table.setCellWidget(i, 0, chk)

            # Error type
            self.table.setItem(i, 1, QTableWidgetItem(err["type"]))

            # Image ID
            self.table.setItem(i, 2, QTableWidgetItem(str(img_id)))

            # Details
            self.table.setItem(i, 3, QTableWidgetItem(err["detail"]))

            # View button
            btn_view = QPushButton("View")
            btn_view.clicked.connect(lambda checked, eid=img_id: self.open_viewer(eid))
            self.table.setCellWidget(i, 4, btn_view)

            # Highlight if excluded
            if is_excluded:
                self.update_row_highlighting(i, True)

        self.lbl_summary.setText(f"Scan Complete. Found {len(errors)} issues.")
        self.update_marked_count()

    def on_checkbox_state_changed(self, state):
        """Handle checkbox state change for marking images."""
        sender = self.sender()
        if not isinstance(sender, QCheckBox) or not self.loader:
            return

        try:
            img_id = int(sender.property("img_id"))
        except (ValueError, TypeError):
            return

        # Mark/unmark image
        if state == 2:
            self.loader.mark_image_for_exclusion(img_id)
        else:
            self.loader.unmark_image_for_exclusion(img_id)

        # Update UI
        self.update_marked_count()

        # Update row highlighting for this img_id
        for row in range(self.table.rowCount()):
            widget = self.table.cellWidget(row, 0)
            if widget and widget.property("img_id") == img_id:
                self.update_row_highlighting(row, state == 2)

    def update_row_highlighting(self, row, is_excluded):
        """Update highlighting for a specific row."""
        for col in range(1, 4):  # Exclude checkbox and view button columns
            item = self.table.item(row, col)
            if item:
                if is_excluded:
                    item.setBackground(Qt.red)
                    item.setForeground(Qt.white)
                else:
                    item.setBackground(Qt.transparent)
                    item.setForeground(Qt.black)

    def update_marked_count(self):
        """Update the marked images counter."""
        if self.loader:
            count = len(self.loader.excluded_image_ids)
            self.lbl_marked.setText(f"Marked images: {count}")

    def open_viewer(self, img_id):
        main_window = self.window()
        if hasattr(main_window, "open_image_in_viewer"):
            main_window.open_image_in_viewer(img_id)
