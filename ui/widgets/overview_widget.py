"""Overview widget for dataset summary and management."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class OverviewWidget(QWidget):
    """Widget for displaying dataset overview and managing classes."""

    def __init__(self):
        super().__init__()
        self.loader = None
        self.initUI()

    def initUI(self):
        """Initialize the UI."""
        layout = QVBoxLayout(self)

        # 1. Dataset Info
        info_group = QGroupBox("Dataset Summary")
        info_layout = QHBoxLayout()

        self.lbl_images = QLabel("Images: 0")
        self.lbl_instances = QLabel("Instances: 0")
        self.lbl_classes = QLabel("Classes: 0")

        info_layout.addWidget(self.lbl_images)
        info_layout.addWidget(self.lbl_instances)
        info_layout.addWidget(self.lbl_classes)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        # 2. Class Management
        class_group = QGroupBox("Class Management (Double-click name to rename)")
        class_layout = QVBoxLayout()

        self.class_table = QTableWidget()
        self.class_table.setColumnCount(3)
        self.class_table.setHorizontalHeaderLabels(["ID", "Name", "Count"])
        self.class_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.class_table.cellChanged.connect(self.on_class_rename)

        class_layout.addWidget(self.class_table)
        class_group.setLayout(class_layout)
        layout.addWidget(class_group)

        # 3. Export Actions
        export_group = QGroupBox("Export Dataset")
        export_layout = QHBoxLayout()

        self.btn_export_yolo = QPushButton("Export as YOLO")
        self.btn_export_yolo.clicked.connect(self.export_yolo)

        self.btn_export_coco = QPushButton("Export as COCO")
        self.btn_export_coco.clicked.connect(self.export_coco)

        export_layout.addWidget(self.btn_export_yolo)
        export_layout.addWidget(self.btn_export_coco)
        export_group.setLayout(export_layout)
        layout.addWidget(export_group)

        # Guide button
        guide_layout = QHBoxLayout()
        guide_layout.addStretch()
        self.btn_guide = QPushButton("📖 View Guide: Overview")
        guide_layout.addWidget(self.btn_guide)
        layout.addLayout(guide_layout)

    def update_data(self, loader):
        """Update widget with new data loader."""
        self.loader = loader
        if not self.loader:
            return

        stats = self.loader.get_stats()
        self.lbl_images.setText(f"Images: {stats['Total Images']}")
        self.lbl_instances.setText(f"Instances: {stats['Total Instances']}")
        self.lbl_classes.setText(f"Classes: {stats['Total Classes']}")

        # Update Class Table
        self.class_table.blockSignals(True)
        self.class_table.setRowCount(0)

        if self.loader.categories:
            self.class_table.setRowCount(len(self.loader.categories))

            # Count instances per category
            if not self.loader.annotations.empty:
                counts = self.loader.annotations["category_id"].value_counts()
            else:
                counts = {}

            for row, (cat_id, cat_name) in enumerate(
                sorted(self.loader.categories.items())
            ):
                # ID (Read-only)
                id_item = QTableWidgetItem(str(cat_id))
                id_item.setFlags(
                    id_item.flags() & ~Qt.ItemIsEditable
                )  # Disable editing
                self.class_table.setItem(row, 0, id_item)

                # Name (Editable)
                name_item = QTableWidgetItem(str(cat_name))
                self.class_table.setItem(row, 1, name_item)

                # Count (Read-only)
                count = counts.get(cat_id, 0)
                count_item = QTableWidgetItem(str(count))
                count_item.setFlags(count_item.flags() & ~Qt.ItemIsEditable)
                self.class_table.setItem(row, 2, count_item)

        self.class_table.blockSignals(False)

    def on_class_rename(self, row, column):
        """Handle class name change."""
        if column != 1 or not self.loader:
            return

        new_name = self.class_table.item(row, column).text()
        try:
            cat_id = int(self.class_table.item(row, 0).text())
            self.loader.rename_category(cat_id, new_name)
        except ValueError:
            pass

    def export_yolo(self):
        """Export dataset as YOLO."""
        if not self.loader:
            return

        save_dir = QFileDialog.getExistingDirectory(
            self, "Select Directory to Save YOLO Dataset"
        )
        if not save_dir:
            return

        try:
            self.loader.export_as_yolo(save_dir)
            QMessageBox.information(self, "Success", f"Dataset exported to {save_dir}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export: {e}")

    def export_coco(self):
        """Export dataset as COCO."""
        if not self.loader:
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save COCO JSON", "", "JSON Files (*.json)"
        )
        if not save_path:
            return

        try:
            self.loader.export_as_coco(save_path)
            QMessageBox.information(self, "Success", f"Dataset exported to {save_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export: {e}")
