"""Overview widget for dataset summary and management."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.dataset_splitter import split_dataset
from ui.widgets.export_dialog import ExportDialog


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
        self.lbl_excluded = QLabel("Excluded: 0")
        self.lbl_excluded.setStyleSheet("color: red; font-weight: bold;")

        info_layout.addWidget(self.lbl_images)
        info_layout.addWidget(self.lbl_instances)
        info_layout.addWidget(self.lbl_classes)
        info_layout.addWidget(self.lbl_excluded)
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
        self.update_excluded_count()

        self.lbl_images.setText(f"Images: {stats['Total Images']}")
        self.lbl_instances.setText(f"Instances: {stats['Total Instances']}")
        self.lbl_classes.setText(f"Classes: {stats['Total Classes']}")

    def update_excluded_count(self):
        """Update excluded count from loader."""
        if not self.loader:
            return
        excluded_count = len(self.loader.excluded_image_ids)
        self.lbl_excluded.setText(f"Excluded: {excluded_count} images")

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
        """Export dataset as YOLO with options dialog."""
        if not self.loader:
            QMessageBox.warning(self, "Warning", "No dataset loaded.")
            return

        # Show export dialog
        dialog = ExportDialog(self.loader, export_format="YOLO", parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        config = dialog.get_export_config()

        # Show excluded images warning if any
        if config["exclude_marked"]:
            excluded_ids = self.loader.get_excluded_images()
            excluded_count = len(excluded_ids)
            if excluded_count > 0:
                # Build detailed message
                msg = f"⚠️ {excluded_count} image(s) are marked for exclusion.\n\n"
                msg += "These images (and their labels) will NOT be included in the export.\n\n"

                # Show first few image IDs/names as examples
                sample_ids = list(excluded_ids)[:5]
                sample_info = []
                for img_id in sample_ids:
                    if img_id in self.loader.images:
                        file_name = self.loader.images[img_id].get(
                            "file_name", str(img_id)
                        )
                        sample_info.append(f"  • ID {img_id}: {file_name}")

                if sample_info:
                    msg += "Sample excluded images:\n"
                    msg += "\n".join(sample_info)
                    if excluded_count > 5:
                        msg += f"\n  ... and {excluded_count - 5} more\n"
                    msg += "\n"

                msg += "\nDo you want to continue?"

                reply = QMessageBox.question(
                    self,
                    "Excluded Images",
                    msg,
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes,
                )

                if reply != QMessageBox.Yes:
                    return

        # Select directory
        save_dir = QFileDialog.getExistingDirectory(
            self, "Select Directory to Save YOLO Dataset"
        )
        if not save_dir:
            return

        # Prepare split if enabled
        split_info = None
        if config["enable_split"]:
            try:
                split_info = split_dataset(
                    self.loader,
                    train_ratio=config["train_ratio"],
                    val_ratio=config["val_ratio"],
                    test_ratio=config["test_ratio"],
                    random_seed=config["random_seed"],
                    exclude_marked=config["exclude_marked"],
                )
            except Exception as e:
                QMessageBox.critical(
                    self, "Split Error", f"Failed to split dataset: {e}"
                )
                return

        # Export with progress
        progress = QProgressDialog("Exporting dataset...", "Cancel", 0, 0, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()

        try:
            self.loader.export_as_yolo(
                save_dir, split_info=split_info, exclude_marked=config["exclude_marked"]
            )
            progress.close()

            # Show success message with stats
            msg = f"Dataset exported successfully to:\n{save_dir}\n\n"
            if split_info:
                msg += f"Train: {len(split_info['train'])} images\n"
                msg += f"Val: {len(split_info['val'])} images\n"
                msg += f"Test: {len(split_info['test'])} images\n"

            # Show excluded images info
            if config["exclude_marked"]:
                excluded_count = len(self.loader.excluded_image_ids)
                if excluded_count > 0:
                    msg += f"\n⚠️ {excluded_count} image(s) were excluded from export."

            QMessageBox.information(self, "Export Complete", msg)
        except Exception as e:
            progress.close()
            QMessageBox.critical(
                self,
                "Export Error",
                f"Failed to export: {e}\n\nCheck console for details.",
            )

    def export_coco(self):
        """Export dataset as COCO with options dialog."""
        if not self.loader:
            QMessageBox.warning(self, "Warning", "No dataset loaded.")
            return

        # Show export dialog
        dialog = ExportDialog(self.loader, export_format="COCO", parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        config = dialog.get_export_config()

        # Show excluded images warning if any
        if config["exclude_marked"]:
            excluded_ids = self.loader.get_excluded_images()
            excluded_count = len(excluded_ids)
            if excluded_count > 0:
                # Build detailed message
                msg = f"⚠️ {excluded_count} image(s) are marked for exclusion.\n\n"
                msg += "These images will NOT be included in the export.\n\n"

                # Show first few image IDs/names as examples
                sample_ids = list(excluded_ids)[:5]
                sample_info = []
                for img_id in sample_ids:
                    if img_id in self.loader.images:
                        file_name = self.loader.images[img_id].get(
                            "file_name", str(img_id)
                        )
                        sample_info.append(f"  • ID {img_id}: {file_name}")

                if sample_info:
                    msg += "Sample excluded images:\n"
                    msg += "\n".join(sample_info)
                    if excluded_count > 5:
                        msg += f"\n  ... and {excluded_count - 5} more\n"
                    msg += "\n"

                msg += "\nDo you want to continue?"

                reply = QMessageBox.question(
                    self,
                    "Excluded Images",
                    msg,
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes,
                )

                if reply != QMessageBox.Yes:
                    return

        # Prepare split if enabled
        split_info = None
        if config["enable_split"]:
            # For COCO with split, ask for directory instead of file
            save_path = QFileDialog.getExistingDirectory(
                self, "Select Directory to Save COCO Dataset (with splits)"
            )
            if not save_path:
                return

            try:
                split_info = split_dataset(
                    self.loader,
                    train_ratio=config["train_ratio"],
                    val_ratio=config["val_ratio"],
                    test_ratio=config["test_ratio"],
                    random_seed=config["random_seed"],
                    exclude_marked=config["exclude_marked"],
                )
            except Exception as e:
                QMessageBox.critical(
                    self, "Split Error", f"Failed to split dataset: {e}"
                )
                return
        else:
            # Without split, ask for JSON file path
            save_path, _ = QFileDialog.getSaveFileName(
                self, "Save COCO JSON", "", "JSON Files (*.json)"
            )
            if not save_path:
                return

        # Export with progress
        progress = QProgressDialog("Exporting dataset...", "Cancel", 0, 0, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()

        try:
            self.loader.export_as_coco(
                save_path,
                split_info=split_info,
                exclude_marked=config["exclude_marked"],
            )
            progress.close()

            # Show success message with stats
            msg = f"Dataset exported successfully to:\n{save_path}\n\n"
            if split_info:
                msg += f"Train: {len(split_info['train'])} images\n"
                msg += f"Val: {len(split_info['val'])} images\n"
                msg += f"Test: {len(split_info['test'])} images\n"

            # Show excluded images info
            if config["exclude_marked"]:
                excluded_count = len(self.loader.excluded_image_ids)
                if excluded_count > 0:
                    msg += f"\n⚠️ {excluded_count} image(s) were excluded from export."

            QMessageBox.information(self, "Export Complete", msg)
        except Exception as e:
            progress.close()
            QMessageBox.critical(
                self,
                "Export Error",
                f"Failed to export: {e}\n\nCheck console for details.",
            )
