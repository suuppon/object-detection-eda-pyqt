"""Main window for the Object Detection EDA Tool."""

from PySide6.QtWidgets import (
    QFileDialog,
    QLabel,
    QMainWindow,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core.data_loader import CocoDataLoader
from ui.widgets import (
    AdvancedStatsWidget,
    DifficultyWidget,
    DuplicateWidget,
    GeometryWidget,
    GuideWidget,
    HealthCheckWidget,
    ImageQualityWidget,
    ImageViewer,
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
        self.resize(1400, 900)  # 조금 더 넓게

        # 파일 로드 버튼
        self.btn_load = QPushButton("Load COCO JSON")
        self.btn_load.clicked.connect(self.load_data)
        self.lbl_status = QLabel("Data not loaded")

        # 탭 위젯 구성
        self.tabs = QTabWidget()

        # Initialize tabs
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
        self.advanced_tab = AdvancedStatsWidget()
        self.viewer_tab = QWidget()

        # Add tabs in logical order
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
        self.tabs.addTab(self.advanced_tab, "🔍 Advanced")
        self.tabs.addTab(self.viewer_tab, "📸 Viewer")

        self.loader = None
        self.img_root = ""

        # 레이아웃 설정
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)
        layout.addWidget(self.btn_load)
        layout.addWidget(self.lbl_status)
        layout.addWidget(self.tabs)

    def load_data(self):
        """Load the data from the file."""
        json_path, _ = QFileDialog.getOpenFileName(
            self, "Open COCO JSON", "", "JSON Files (*.json)"
        )
        if not json_path:
            return

        img_root = QFileDialog.getExistingDirectory(self, "Select Image Root Directory")
        if not img_root:
            return
        self.img_root = img_root

        try:
            self.loader = CocoDataLoader(json_path)
            stats = self.loader.get_stats()
            self.lbl_status.setText(
                f"Loaded: {stats['Total Images']} images, {stats['Total Instances']} objects"
            )

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

            viewer_idx = self.tabs.indexOf(self.viewer_tab)
            self.tabs.removeTab(viewer_idx)

            self.viewer_tab = ImageViewer(self.loader, self.img_root)
            self.tabs.insertTab(viewer_idx, self.viewer_tab, "Viewer")
            self.tabs.setCurrentIndex(0)

        except Exception as e:
            import traceback

            traceback.print_exc()
            self.lbl_status.setText(f"Error: {e}")

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
            # Scroll to section
            self.guide_tab.scroll_to_section(section_name)
