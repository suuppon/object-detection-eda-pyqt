import os

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.path import Path as MplPath
from matplotlib.widgets import LassoSelector
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from core.cartography import CartographyWorker


class CartographyWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.loader = None
        self.img_root = None
        self.worker = None
        self.df = None
        self.selected_indices = []

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        # 1. Top Controls
        control_layout = QHBoxLayout()

        self.btn_run = QPushButton("Start Cartography Analysis")
        self.btn_run.clicked.connect(self.run_analysis)
        self.btn_run.setEnabled(False)  # Disabled until data loaded

        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(3, 20)
        self.spin_epochs.setValue(5)
        self.spin_epochs.setPrefix("Epochs: ")

        self.combo_batch = QComboBox()
        self.combo_batch.addItems(["8", "16", "32"])
        self.combo_batch.setCurrentText("16")

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)

        control_layout.addWidget(self.btn_run)
        control_layout.addWidget(self.spin_epochs)
        control_layout.addWidget(QLabel("Batch:"))
        control_layout.addWidget(self.combo_batch)
        control_layout.addWidget(self.progress_bar)
        control_layout.addStretch()

        self.btn_guide = QPushButton("View Guide")
        control_layout.addWidget(self.btn_guide)

        main_layout.addLayout(control_layout)

        # 2. Main Content (Splitter)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: Plot Area
        plot_container = QWidget()
        plot_layout = QVBoxLayout(plot_container)

        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        plot_layout.addWidget(self.canvas)
        plot_layout.addWidget(QLabel("Tip: Drag mouse to select points (Lasso)"))

        splitter.addWidget(plot_container)

        # Right: Inspection Panel
        inspect_container = QWidget()
        inspect_layout = QVBoxLayout(inspect_container)

        inspect_layout.addWidget(QLabel("Selected Images (Hard-to-Learn / Ambiguous)"))
        self.list_widget = QListWidget()
        self.list_widget.itemClicked.connect(self.on_item_clicked)
        inspect_layout.addWidget(self.list_widget)

        self.preview_label = QLabel("Preview")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumHeight(200)
        self.preview_label.setStyleSheet("border: 1px solid #ccc; background: #f0f0f0;")
        inspect_layout.addWidget(self.preview_label)

        splitter.addWidget(inspect_container)

        # Set initial sizes (60% plot, 40% list)
        splitter.setSizes([600, 400])

        main_layout.addWidget(splitter)

        # Initialize empty plot
        self.ax.set_title("Data Cartography Map")
        self.ax.set_xlabel("Variability (Standard Deviation)")
        self.ax.set_ylabel("Confidence (Mean Probability)")
        self.ax.grid(True, linestyle="--", alpha=0.6)
        self.canvas.draw()

    def update_data(self, loader):
        """Receive data loader from main window."""
        self.loader = loader
        self.check_ready()

    def set_img_root(self, img_root):
        """Receive image root from main window."""
        self.img_root = img_root
        self.check_ready()

    def check_ready(self):
        if self.loader and self.img_root:
            self.btn_run.setEnabled(True)
            self.btn_run.setText(f"Start Analysis ({len(self.loader.images)} imgs)")
        else:
            self.btn_run.setEnabled(False)

    def run_analysis(self):
        if not self.loader or not self.img_root:
            QMessageBox.warning(self, "Error", "Dataset not loaded properly.")
            return

        # Confirmation for Conversion
        reply = QMessageBox.question(
            self,
            "Dataset Conversion Required",
            "To perform Data Cartography, the current COCO dataset needs to be converted to YOLO format.\n"
            "This will create a copy of images and labels in a 'yolo_cartography_cache' folder.\n\n"
            "Do you want to proceed?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.No:
            return

        epochs = self.spin_epochs.value()
        batch = int(self.combo_batch.currentText())

        self.btn_run.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.list_widget.clear()

        # Pass loader and img_root for auto-conversion
        self.worker = CartographyWorker(
            loader=self.loader, img_root=self.img_root, epochs=epochs, batch_size=batch
        )
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.analysis_finished.connect(self.on_analysis_finished)
        self.worker.error_occurred.connect(self.on_error)
        self.worker.start()

    @Slot(int, str)
    def update_progress(self, val, msg):
        self.progress_bar.setValue(val)
        self.progress_bar.setFormat(f"{msg} (%p%)")

    @Slot(pd.DataFrame)
    def on_analysis_finished(self, df):
        self.df = df
        self.progress_bar.setVisible(False)
        self.btn_run.setEnabled(True)
        self.plot_data()
        QMessageBox.information(
            self,
            "Analysis Complete",
            f"Analyzed {len(df)} images.\n" "Drag on the plot to select images.",
        )

    @Slot(str)
    def on_error(self, err):
        self.progress_bar.setVisible(False)
        self.btn_run.setEnabled(True)
        QMessageBox.critical(self, "Error", f"An error occurred:\n{err}")

    def plot_data(self):
        self.ax.clear()

        if self.df is None or self.df.empty:
            return

        # Color mapping
        colors = {
            "Easy-to-Learn": "blue",
            "Ambiguous": "orange",
            "Hard-to-Learn": "red",
        }
        c_map = self.df["region"].map(colors)

        # Scatter plot
        self.collection = self.ax.scatter(
            self.df["variability"],
            self.df["confidence"],
            c=c_map,
            alpha=0.6,
            edgecolors="w",
            s=40,
            picker=True,
        )

        # Regions annotation (Guidelines)
        self.ax.axhline(y=0.7, color="gray", linestyle=":", alpha=0.5)
        self.ax.axhline(y=0.4, color="gray", linestyle=":", alpha=0.5)
        self.ax.axvline(x=0.15, color="gray", linestyle=":", alpha=0.5)

        self.ax.set_title("Data Cartography Map")
        self.ax.set_xlabel("Variability (Standard Deviation)")
        self.ax.set_ylabel("Confidence (Mean Probability)")
        self.ax.set_xlim(left=0)
        self.ax.set_ylim(0, 1.05)
        self.ax.grid(True, linestyle="--", alpha=0.6)

        # Setup Lasso Selector
        self.lasso = LassoSelector(self.ax, self.on_select)

        self.canvas.draw()

    def on_select(self, verts):
        if self.df is None:
            return

        path = MplPath(verts)

        # Get points data
        x_data = self.df["variability"].values
        y_data = self.df["confidence"].values
        points = np.column_stack((x_data, y_data))

        # Check which points are inside the lasso path
        self.selected_indices = np.nonzero(path.contains_points(points))[0]

        self.update_list_widget()

    def update_list_widget(self):
        self.list_widget.clear()

        if len(self.selected_indices) == 0:
            return

        selected_df = self.df.iloc[self.selected_indices]

        for idx, row in selected_df.iterrows():
            item_text = f"[{row['region']}] {os.path.basename(row['image_path'])}"
            item = QListWidgetItem(item_text)
            # Store original image path
            item.setData(Qt.ItemDataRole.UserRole, row["image_path"])

            # Color code text based on region
            if row["region"] == "Hard-to-Learn":
                item.setForeground(Qt.GlobalColor.red)
            elif row["region"] == "Ambiguous":
                item.setForeground(Qt.GlobalColor.darkYellow)

            self.list_widget.addItem(item)

        self.list_widget.addItem(f"--- {len(selected_df)} images selected ---")

    def on_item_clicked(self, item):
        path = item.data(Qt.ItemDataRole.UserRole)
        if path and os.path.exists(path):
            pixmap = QPixmap(path)
            if not pixmap.isNull():
                scaled = pixmap.scaled(
                    self.preview_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                self.preview_label.setPixmap(scaled)
            else:
                self.preview_label.setText("Invalid Image")

    def _navigate_to_guide(self):
        # This signal will be connected in main_window
        pass
