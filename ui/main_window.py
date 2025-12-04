"""Main window for the Object Detection EDA Tool."""

from PySide6.QtWidgets import (
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core.data.coco import CocoDataLoader
from core.data.yolo import YoloDataLoader
from ui.widgets import (
    AdvancedStatsWidget,
    CartographyWidget,
    DifficultyWidget,
    DuplicateWidget,
    GeometryWidget,
    GuideWidget,
    HealthCheckWidget,
    ImageQualityWidget,
    ImageViewer,
    OverviewWidget,
    RelationWidget,
    SignalAnalysisWidget,
    SpatialWidget,
    StatWidget,
    StrategyWidget,
)


class MainWindow(QMainWindow):
    """Main window for the Object Detection EDA Tool."""

    def __init__(self):
        """Initialize the main window."""
        super().__init__()
        self.initUI()

    def initUI(self):
        """Initialize the UI."""
        self.setWindowTitle("Object Detection EDA Tool")
        self.resize(1400, 900)

        # Top Layout
        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)

        # Controls
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)

        self.btn_load_coco = QPushButton("Load COCO JSON")
        self.btn_load_coco.clicked.connect(self.load_data_coco)
        self.btn_load_yolo = QPushButton("Load YOLO Format")
        self.btn_load_yolo.clicked.connect(self.load_data_yolo)
        self.lbl_status = QLabel("Data not loaded")

        control_layout.addWidget(self.btn_load_coco)
        control_layout.addWidget(self.btn_load_yolo)
        control_layout.addWidget(self.lbl_status)

        top_layout.addWidget(control_panel)

        # Tab Widget
        self.tabs = QTabWidget()

        # Initialize tabs
        self.overview_tab = OverviewWidget()
        self.guide_tab = GuideWidget()
        self.stat_tab = StatWidget()
        self.geo_tab = GeometryWidget()
        self.spatial_tab = SpatialWidget()
        self.rel_tab = RelationWidget()
        self.diff_tab = DifficultyWidget()
        self.dup_tab = DuplicateWidget()
        self.health_tab = HealthCheckWidget()
        self.quality_tab = ImageQualityWidget()
        self.strat_tab = StrategyWidget()
        self.signal_tab = SignalAnalysisWidget()
        self.carto_tab = CartographyWidget()  # New Tab
        self.advanced_tab = AdvancedStatsWidget()
        self.viewer_tab = QWidget()

        # Connect Guide Signals
        self.overview_tab.btn_guide.clicked.connect(
            lambda: self.navigate_to_guide("overview")
        )
        self.stat_tab.btn_guide.clicked.connect(
            lambda: self.navigate_to_guide("dashboard")
        )
        self.geo_tab.btn_guide.clicked.connect(
            lambda: self.navigate_to_guide("geometry")
        )
        self.spatial_tab.btn_guide.clicked.connect(
            lambda: self.navigate_to_guide("spatial")
        )
        self.rel_tab.btn_guide.clicked.connect(
            lambda: self.navigate_to_guide("relation")
        )
        self.diff_tab.btn_guide.clicked.connect(
            lambda: self.navigate_to_guide("difficulty")
        )
        self.dup_tab.btn_guide.clicked.connect(
            lambda: self.navigate_to_guide("duplicates")
        )
        self.health_tab.btn_guide.clicked.connect(
            lambda: self.navigate_to_guide("health")
        )
        self.quality_tab.btn_guide.clicked.connect(
            lambda: self.navigate_to_guide("quality")
        )
        self.strat_tab.btn_guide.clicked.connect(
            lambda: self.navigate_to_guide("strategy")
        )
        self.signal_tab.btn_guide.clicked.connect(
            lambda: self.navigate_to_guide("signal")
        )
        self.carto_tab.btn_guide.clicked.connect(
            lambda: self.navigate_to_guide("cartography")
        )  # To be added in Guide
        self.advanced_tab.btn_guide.clicked.connect(
            lambda: self.navigate_to_guide("advanced")
        )

        # Add tabs in logical order
        self.tabs.addTab(self.overview_tab, "📋 Overview")
        self.tabs.addTab(self.guide_tab, "📖 Guide")
        self.tabs.addTab(self.stat_tab, "📊 Dashboard")
        self.tabs.addTab(self.geo_tab, "📏 Geometry")
        self.tabs.addTab(self.spatial_tab, "🗺️ Spatial")
        self.tabs.addTab(self.rel_tab, "🤝 Relation")
        self.tabs.addTab(self.diff_tab, "🎯 Difficulty")
        self.tabs.addTab(self.dup_tab, "🔍 Duplicates")
        self.tabs.addTab(self.health_tab, "🧹 Health")
        self.tabs.addTab(self.quality_tab, "🎨 Quality")
        self.tabs.addTab(self.strat_tab, "🚀 Strategy")
        self.tabs.addTab(self.signal_tab, "🔍 Signal")
        self.tabs.addTab(self.carto_tab, "🗺️ Cartography")  # New Tab
        self.tabs.addTab(self.advanced_tab, "🔍 Advanced")
        self.tabs.addTab(self.viewer_tab, "📸 Viewer")

        self.loader = None
        self.img_root = ""

        # Layout Setup
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)
        layout.addWidget(top_widget)
        layout.addWidget(self.tabs)

    def _handle_new_loader(self, new_loader, new_img_root):
        """Handle new data loader (merge or replace)."""
        if self.loader is not None:
            reply = QMessageBox.question(
                self,
                "Merge Datasets",
                "A dataset is already loaded. Do you want to merge the new data into the existing dataset?\n"
                "Click 'Yes' to merge, 'No' to replace the existing dataset.",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                QMessageBox.Yes,
            )

            if reply == QMessageBox.Yes:
                try:
                    self.loader.merge(new_loader)
                    # If merging, we might need to handle img_root.
                    # For simplicity in this version, we keep the original img_root
                    # if it works, or we might need a more complex image path handling
                    # if datasets are in different locations.
                    # The Viewer currently uses self.img_root.
                    # To support multiple roots, DataLoader should handle absolute paths.
                    # But for now, let's assume user might want to keep using the primary root or we just update stats.
                    self.lbl_status.setText("Datasets merged successfully.")
                except Exception as e:
                    QMessageBox.critical(
                        self, "Merge Error", f"Failed to merge datasets: {e}"
                    )
                    return
            elif reply == QMessageBox.No:
                self.loader = new_loader
                self.img_root = new_img_root
            else:
                return
        else:
            self.loader = new_loader
            self.img_root = new_img_root

        self._update_ui_with_loader()

    def load_data_coco(self):
        """Load the data from COCO JSON file."""
        json_path, _ = QFileDialog.getOpenFileName(
            self, "Open COCO JSON", "", "JSON Files (*.json)"
        )
        if not json_path:
            return

        img_root = QFileDialog.getExistingDirectory(self, "Select Image Root Directory")
        if not img_root:
            return

        try:
            new_loader = CocoDataLoader(json_path)
            self._handle_new_loader(new_loader, img_root)

        except Exception as e:
            import traceback

            traceback.print_exc()
            self.lbl_status.setText(f"Error: {e}")

    def load_data_yolo(self):
        """Load the data from YOLO format YAML file."""
        # Select YAML file
        yaml_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select YOLO data.yaml File",
            "",
            "YAML Files (*.yaml *.yml);;All Files (*)",
        )
        if not yaml_path:
            return

        try:
            new_loader = YoloDataLoader(yaml_path)
            # Use base_path if available, otherwise use img_root
            if hasattr(new_loader, "base_path") and new_loader.base_path:
                img_root_path = str(new_loader.base_path)
            else:
                img_root_path = str(new_loader.img_root) if new_loader.img_root else ""
            self._handle_new_loader(new_loader, img_root_path)

        except Exception as e:
            import traceback

            traceback.print_exc()
            self.lbl_status.setText(f"Error: {e}")

    def _update_ui_with_loader(self):
        """Update UI with the loaded data loader."""
        stats = self.loader.get_stats()
        self.lbl_status.setText(
            f"Loaded: {stats['Total Images']} images, {stats['Total Instances']} objects"
        )

        self.overview_tab.update_data(self.loader)
        self.stat_tab.plot_class_distribution(self.loader.annotations)
        self.geo_tab.update_data(self.loader)
        self.spatial_tab.update_data(self.loader)
        self.rel_tab.update_data(self.loader)
        self.diff_tab.update_data(self.loader)
        self.health_tab.update_data(self.loader)
        self.dup_tab.update_data(self.loader)
        self.dup_tab.set_img_root(self.img_root)
        self.strat_tab.update_data(self.loader)
        self.quality_tab.update_data(self.loader)
        self.quality_tab.set_img_root(self.img_root)
        self.signal_tab.update_data(self.loader)
        self.signal_tab.set_img_root(self.img_root)
        self.advanced_tab.update_data(self.loader)
        self.advanced_tab.set_img_root(self.img_root)

        # Data Cartography Setup
        self.carto_tab.update_data(self.loader)
        self.carto_tab.set_img_root(self.img_root)

        # Re-initialize Viewer
        viewer_idx = self.tabs.indexOf(self.viewer_tab)
        self.tabs.removeTab(viewer_idx)
        self.viewer_tab = ImageViewer(self.loader, self.img_root)
        self.tabs.insertTab(viewer_idx, self.viewer_tab, "📸 Viewer")

        self.tabs.setCurrentIndex(0)

    def open_image_in_viewer(self, img_id):
        """Open the image in the viewer."""
        if isinstance(self.viewer_tab, ImageViewer):
            idx = self.tabs.indexOf(self.viewer_tab)
            if idx != -1:
                self.tabs.setCurrentIndex(idx)
                self.viewer_tab.select_image_by_id(img_id)
        else:
            self.lbl_status.setText("Viewer not initialized.")

    def navigate_to_guide(self, section_name):
        """Navigate to Guide tab and scroll to specific section."""
        guide_idx = self.tabs.indexOf(self.guide_tab)
        if guide_idx != -1:
            self.tabs.setCurrentIndex(guide_idx)
            self.guide_tab.scroll_to_section(section_name)
