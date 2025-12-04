"""Export dialog for dataset export with split options."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QSpinBox,
    QVBoxLayout,
)


class ExportDialog(QDialog):
    """Dialog for configuring dataset export options."""

    def __init__(self, loader, export_format="YOLO", parent=None):
        """Initialize export dialog.

        Args:
            loader: UnifiedDataLoader instance.
            export_format: "YOLO" or "COCO".
            parent: Parent widget.
        """
        super().__init__(parent)
        self.loader = loader
        self.export_format = export_format
        self.setWindowTitle(f"Export as {export_format}")
        self.setMinimumWidth(450)
        self.initUI()

    def initUI(self):
        """Initialize the UI."""
        layout = QVBoxLayout(self)

        # Dataset Info
        info_group = QGroupBox("Dataset Information")
        info_layout = QFormLayout()

        stats = self.loader.get_stats()
        total_images = stats["Total Images"]
        excluded_count = len(self.loader.excluded_image_ids)
        exportable_count = total_images - excluded_count

        info_layout.addRow("Total Images:", QLabel(str(total_images)))
        info_layout.addRow(
            "Excluded Images:", QLabel(f"<b style='color:red;'>{excluded_count}</b>")
        )
        info_layout.addRow("Exportable Images:", QLabel(f"<b>{exportable_count}</b>"))
        info_layout.addRow("Total Categories:", QLabel(str(stats["Total Classes"])))

        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        # Split Options
        split_group = QGroupBox("Train/Val/Test Split")
        split_layout = QVBoxLayout()

        self.chk_enable_split = QCheckBox("Enable dataset split")
        self.chk_enable_split.setChecked(True)
        self.chk_enable_split.stateChanged.connect(self.on_split_toggled)
        split_layout.addWidget(self.chk_enable_split)

        # Ratios
        ratio_form = QFormLayout()

        self.spin_train = QDoubleSpinBox()
        self.spin_train.setRange(0.0, 1.0)
        self.spin_train.setSingleStep(0.05)
        self.spin_train.setValue(0.7)
        self.spin_train.setDecimals(2)
        self.spin_train.valueChanged.connect(self.on_ratio_changed)
        ratio_form.addRow("Train Ratio:", self.spin_train)

        self.spin_val = QDoubleSpinBox()
        self.spin_val.setRange(0.0, 1.0)
        self.spin_val.setSingleStep(0.05)
        self.spin_val.setValue(0.2)
        self.spin_val.setDecimals(2)
        self.spin_val.valueChanged.connect(self.on_ratio_changed)
        ratio_form.addRow("Val Ratio:", self.spin_val)

        self.spin_test = QDoubleSpinBox()
        self.spin_test.setRange(0.0, 1.0)
        self.spin_test.setSingleStep(0.05)
        self.spin_test.setValue(0.1)
        self.spin_test.setDecimals(2)
        self.spin_test.valueChanged.connect(self.on_ratio_changed)
        ratio_form.addRow("Test Ratio:", self.spin_test)

        self.lbl_ratio_sum = QLabel("Total: 1.00")
        ratio_form.addRow("", self.lbl_ratio_sum)

        split_layout.addLayout(ratio_form)

        # Random Seed
        seed_form = QFormLayout()
        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 999999)
        self.spin_seed.setValue(42)
        seed_form.addRow("Random Seed:", self.spin_seed)
        split_layout.addLayout(seed_form)

        split_group.setLayout(split_layout)
        layout.addWidget(split_group)

        # Duplicate Groups Info
        if self.loader.duplicate_groups:
            dup_label = QLabel(
                f"ℹ️ <i>{len(self.loader.duplicate_groups)} duplicate groups detected. "
                f"Images in the same group will stay together in the same split.</i>"
            )
            dup_label.setWordWrap(True)
            dup_label.setStyleSheet("color: blue;")
            layout.addWidget(dup_label)

        # Buttons
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

        # Initial state
        self.on_ratio_changed()

    def on_split_toggled(self, state):
        """Handle split checkbox toggle."""
        enabled = state == Qt.Checked
        self.spin_train.setEnabled(enabled)
        self.spin_val.setEnabled(enabled)
        self.spin_test.setEnabled(enabled)
        self.spin_seed.setEnabled(enabled)

    def on_ratio_changed(self):
        """Update ratio sum display."""
        total = self.spin_train.value() + self.spin_val.value() + self.spin_test.value()
        self.lbl_ratio_sum.setText(f"Total: {total:.2f}")

        # Validate
        if abs(total - 1.0) > 0.01:
            self.lbl_ratio_sum.setStyleSheet("color: red; font-weight: bold;")
            self.button_box.button(QDialogButtonBox.Ok).setEnabled(False)
        else:
            self.lbl_ratio_sum.setStyleSheet("color: green; font-weight: bold;")
            self.button_box.button(QDialogButtonBox.Ok).setEnabled(True)

    def get_export_config(self):
        """Get export configuration.

        Returns:
            Dictionary with export settings.
        """
        return {
            "format": self.export_format,
            "enable_split": self.chk_enable_split.isChecked(),
            "train_ratio": self.spin_train.value(),
            "val_ratio": self.spin_val.value(),
            "test_ratio": self.spin_test.value(),
            "random_seed": self.spin_seed.value(),
            "exclude_marked": True,  # Always exclude marked images
        }
