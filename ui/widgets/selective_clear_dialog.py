"""Dialog for selectively clearing data from the dataset."""

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class SelectiveClearDialog(QDialog):
    """Dialog to select criteria for clearing specific data."""

    def __init__(self, loader, parent=None):
        super().__init__(parent)
        self.loader = loader
        self.setWindowTitle("Selective Data Removal")
        self.resize(500, 400)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)

        # Info label
        info_label = QLabel(
            "Select criteria to remove specific images and their annotations from the dataset."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: gray; padding: 10px;")
        layout.addWidget(info_label)

        # Selection method
        method_group = QGroupBox("Removal Method")
        method_layout = QVBoxLayout()

        self.radio_filename = QRadioButton("By Filename Pattern")
        self.radio_id_range = QRadioButton("By Image ID Range")
        self.radio_category = QRadioButton("By Category")
        self.radio_source = QRadioButton("By Source Dataset")
        self.radio_source.setChecked(True)  # Make source the default

        method_layout.addWidget(self.radio_filename)
        method_layout.addWidget(self.radio_id_range)
        method_layout.addWidget(self.radio_category)
        method_layout.addWidget(self.radio_source)
        method_group.setLayout(method_layout)
        layout.addWidget(method_group)

        # Options
        options_group = QGroupBox("Options")
        options_layout = QFormLayout()

        # Filename pattern
        self.filename_pattern = QLineEdit()
        self.filename_pattern.setPlaceholderText("e.g., 'dataset1', 'train/', '*.jpg'")
        options_layout.addRow("Filename contains:", self.filename_pattern)

        # ID range
        id_range_layout = QWidget()
        id_range_hbox = QHBoxLayout(id_range_layout)
        id_range_hbox.setContentsMargins(0, 0, 0, 0)
        self.id_from = QSpinBox()
        self.id_from.setMinimum(0)
        self.id_from.setMaximum(999999)
        self.id_to = QSpinBox()
        self.id_to.setMinimum(0)
        self.id_to.setMaximum(999999)
        id_range_hbox.addWidget(QLabel("From:"))
        id_range_hbox.addWidget(self.id_from)
        id_range_hbox.addWidget(QLabel("To:"))
        id_range_hbox.addWidget(self.id_to)
        id_range_hbox.addStretch()
        options_layout.addRow("Image ID Range:", id_range_layout)

        # Category
        self.category_combo = QComboBox()
        if self.loader and self.loader.categories:
            for cat_id, cat_name in sorted(self.loader.categories.items()):
                self.category_combo.addItem(f"{cat_name} (ID: {cat_id})", cat_id)
        options_layout.addRow("Category:", self.category_combo)

        # Source
        self.source_combo = QComboBox()
        if self.loader and hasattr(self.loader, 'get_sources'):
            sources = self.loader.get_sources()
            for source_name in sorted(sources):
                img_count = len(self.loader.get_source_image_ids(source_name))
                self.source_combo.addItem(f"{source_name} ({img_count} images)", source_name)
        options_layout.addRow("Source Dataset:", self.source_combo)

        options_group.setLayout(options_layout)
        layout.addWidget(options_group)

        # Preview
        self.preview_label = QLabel("Preview: Click 'Preview' to see how many items will be removed")
        self.preview_label.setWordWrap(True)
        self.preview_label.setStyleSheet("padding: 10px; background-color: #f0f0f0;")
        layout.addWidget(self.preview_label)

        # Buttons
        btn_layout = QHBoxLayout()
        self.btn_preview = QDialogButtonBox(QDialogButtonBox.NoButton)
        preview_btn = self.btn_preview.addButton("Preview", QDialogButtonBox.ActionRole)
        preview_btn.clicked.connect(self.preview_removal)
        btn_layout.addWidget(self.btn_preview)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.validate_and_accept)
        self.button_box.rejected.connect(self.reject)
        btn_layout.addWidget(self.button_box)

        layout.addLayout(btn_layout)

        # Connect radio buttons to enable/disable fields
        self.radio_filename.toggled.connect(self.on_method_changed)
        self.radio_id_range.toggled.connect(self.on_method_changed)
        self.radio_category.toggled.connect(self.on_method_changed)
        self.radio_source.toggled.connect(self.on_method_changed)
        self.on_method_changed()

    def on_method_changed(self):
        """Enable/disable form fields based on selected method."""
        filename_enabled = self.radio_filename.isChecked()
        id_range_enabled = self.radio_id_range.isChecked()
        category_enabled = self.radio_category.isChecked()
        source_enabled = self.radio_source.isChecked()

        self.filename_pattern.setEnabled(filename_enabled)
        self.id_from.setEnabled(id_range_enabled)
        self.id_to.setEnabled(id_range_enabled)
        self.category_combo.setEnabled(category_enabled)
        self.source_combo.setEnabled(source_enabled)

    def preview_removal(self):
        """Preview how many items will be removed."""
        if not self.loader:
            QMessageBox.warning(self, "No Data", "No data loaded.")
            return

        img_ids_to_remove = self._get_image_ids_to_remove()
        
        if not img_ids_to_remove:
            self.preview_label.setText("Preview: No items match the criteria.")
            return

        # Count annotations
        if not self.loader.annotations.empty:
            ann_count = len(self.loader.annotations[self.loader.annotations["image_id"].isin(img_ids_to_remove)])
        else:
            ann_count = 0

        self.preview_label.setText(
            f"Preview: {len(img_ids_to_remove)} image(s) and {ann_count} annotation(s) will be removed."
        )

    def _get_image_ids_to_remove(self):
        """Get set of image IDs that match the removal criteria."""
        if not self.loader:
            return set()

        img_ids = set()

        if self.radio_filename.isChecked():
            pattern = self.filename_pattern.text().strip()
            if not pattern:
                return set()
            
            for img_id, img_info in self.loader.images.items():
                file_name = img_info.get("file_name", "")
                if pattern in file_name:
                    img_ids.add(img_id)

        elif self.radio_id_range.isChecked():
            from_id = self.id_from.value()
            to_id = self.id_to.value()
            
            if from_id > to_id:
                return set()
            
            for img_id in self.loader.images.keys():
                if from_id <= img_id <= to_id:
                    img_ids.add(img_id)

        elif self.radio_category.isChecked():
            if self.category_combo.count() == 0:
                return set()
            
            cat_id = self.category_combo.currentData()
            if cat_id is None:
                return set()
            
            # Get all images that have annotations with this category
            if not self.loader.annotations.empty:
                matching_anns = self.loader.annotations[
                    self.loader.annotations["category_id"] == cat_id
                ]
                img_ids = set(matching_anns["image_id"].unique())
            else:
                return set()

        elif self.radio_source.isChecked():
            if self.source_combo.count() == 0:
                return set()
            
            source_name = self.source_combo.currentData()
            if source_name is None:
                return set()
            
            # Get all images from this source
            if hasattr(self.loader, 'get_source_image_ids'):
                img_ids = self.loader.get_source_image_ids(source_name)
            else:
                return set()

        return img_ids

    def validate_and_accept(self):
        """Validate input and accept dialog."""
        img_ids = self._get_image_ids_to_remove()
        
        if not img_ids:
            QMessageBox.warning(
                self, "No Match", "No items match the selected criteria."
            )
            return

        # Confirm
        reply = QMessageBox.question(
            self,
            "Confirm Removal",
            f"Are you sure you want to remove {len(img_ids)} image(s)?\n\n"
            "This action cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            self.accept()

    def get_image_ids_to_remove(self):
        """Get the image IDs to remove."""
        return self._get_image_ids_to_remove()

