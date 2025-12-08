"""Main window for the Object Detection EDA Tool."""

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core.data.coco import CocoDataLoader
from core.data.yolo import YoloDataLoader
from ui.widgets.multi_load_dialog import MultiDatasetLoadDialog
from ui.widgets import (
    AdvancedWidget,
    CartographyWidget,
    DifficultyWidget,
    DuplicateWidget,
    GeometryWidget,
    GuideWidget,
    HealthWidget,
    OverviewWidget,
    QualityWidget,
    RelationWidget,
    SignalWidget,
    SpatialWidget,
    StatWidget,
    StrategyWidget,
    ViewerWidget,
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
        self.btn_load_multi = QPushButton("Load Multiple COCO (Train/Val/Test)")
        self.btn_load_multi.clicked.connect(self.load_data_multi_coco)
        self.btn_load_yolo = QPushButton("Load YOLO Format")
        self.btn_load_yolo.clicked.connect(self.load_data_yolo)
        self.lbl_status = QLabel("Data not loaded")

        control_layout.addWidget(self.btn_load_coco)
        control_layout.addWidget(self.btn_load_multi)
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
        self.health_tab = HealthWidget()
        self.quality_tab = QualityWidget()
        self.strat_tab = StrategyWidget()
        self.signal_tab = SignalWidget()
        self.carto_tab = CartographyWidget()  # New Tab
        self.advanced_tab = AdvancedWidget()
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
        self.widgets_initialized = {}  # Track which widgets have been initialized
        self.pending_updates = []  # Queue for deferred widget updates

        # Connect tab change signal for lazy loading
        self.tabs.currentChanged.connect(self.on_tab_changed)

        # Layout Setup
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)
        layout.addWidget(top_widget)
        layout.addWidget(self.tabs)

    def clear_data(self):
        """Clear all loaded data."""
        self.loader = None
        self.img_root = ""
        self.widgets_initialized = {}
        self.pending_updates = []
        
        # Reset all widgets
        self.overview_tab.update_data(None)
        self.stat_tab.figure.clear()
        self.stat_tab.canvas.draw()
        self.geo_tab.update_data(None)
        self.spatial_tab.update_data(None)
        self.rel_tab.update_data(None)
        self.diff_tab.update_data(None)
        self.health_tab.update_data(None)
        self.dup_tab.update_data(None)
        self.strat_tab.update_data(None)
        self.quality_tab.update_data(None)
        self.signal_tab.update_data(None)
        self.advanced_tab.update_data(None)
        self.carto_tab.update_data(None)
        
        # Reset viewer
        viewer_idx = self.tabs.indexOf(self.viewer_tab)
        self.tabs.removeTab(viewer_idx)
        self.viewer_tab = QWidget()
        self.tabs.insertTab(viewer_idx, self.viewer_tab, "📸 Viewer")
        
        self.lbl_status.setText("Data not loaded")

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

        # Show loading dialog immediately after file selection
        progress = QProgressDialog(
            "Loading COCO dataset...\n\nPlease wait while the data is being loaded.\n"
            "This may take a moment for large datasets.",
            None, 0, 0, self
        )
        progress.setWindowTitle("Loading Data")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setCancelButton(None)  # Disable cancel button
        progress.setMinimumDuration(0)  # Show immediately
        progress.setRange(0, 0)  # Indeterminate progress
        progress.show()
        progress.setValue(0)
        
        # Process events to show the dialog
        from PySide6.QtWidgets import QApplication
        QApplication.processEvents()

        try:
            # Generate source name from JSON path
            from pathlib import Path
            source_name = Path(json_path).stem  # Use filename without extension as source name
            new_loader = CocoDataLoader(json_path, source_name=source_name)
            progress.close()
            self._handle_new_loader(new_loader, img_root)

        except Exception as e:
            import traceback

            traceback.print_exc()
            progress.close()
            self.lbl_status.setText(f"Error: {e}")
            QMessageBox.critical(self, "Loading Error", f"Failed to load COCO dataset:\n{e}")

    def load_data_multi_coco(self):
        """Load multiple COCO datasets from root directories."""
        dialog = MultiDatasetLoadDialog(self)
        if dialog.exec() != QDialog.Accepted:
            return
        
        config = dialog.get_config()
        root_dirs = config["dirs"]
        splits = config["splits"]
        
        if not root_dirs:
            return
            
        # Show loading dialog immediately after dialog closes
        progress = QProgressDialog(
            "Scanning directories and loading datasets...\n\n"
            "Please wait while multiple datasets are being loaded.\n"
            "This may take a moment for large datasets.",
            None, 0, 0, self
        )
        progress.setWindowTitle("Loading Multiple Datasets")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setCancelButton(None)
        progress.setMinimumDuration(0)
        progress.setRange(0, 0)
        progress.show()
        
        from PySide6.QtWidgets import QApplication
        QApplication.processEvents()
        
        combined_loader = None
        from pathlib import Path
        
        try:
            # Collect all tasks
            tasks = []
            for root in root_dirs:
                root_path = Path(root)
                for split in splits:
                    split_path = root_path / split
                    
                    # Fallback for 'valid' -> 'val' or 'val' -> 'valid' if needed?
                    # User specified "valid", but standard is often "val"
                    if not split_path.exists() and split == "valid":
                        if (root_path / "val").exists():
                            split_path = root_path / "val"
                    
                    if split_path.exists() and split_path.is_dir():
                        # Find json
                        json_files = list(split_path.glob("*.json"))
                        if json_files:
                            # Use the first found json
                            # Store (json_path, img_root, root_name) for source tracking
                            tasks.append((json_files[0], split_path, root_path.name))
            
            if not tasks:
                progress.close()
                QMessageBox.warning(self, "No Data Found", "No JSON files found in the selected splits.")
                return
            
            progress.setLabelText(f"Loading {len(tasks)} datasets...")
            QApplication.processEvents()
            
            first_img_root = None  # Store first split's img_root for widget initialization
            for i, (json_path, img_root, root_name) in enumerate(tasks):
                progress.setLabelText(f"Loading {i+1}/{len(tasks)}: {img_root.name}...")
                QApplication.processEvents()
                
                # Store first img_root for widget initialization
                if first_img_root is None:
                    first_img_root = str(img_root)
                
                # Generate source name: root_dir_name/split_name
                img_root_path = Path(img_root)
                split_name = img_root_path.name
                source_name = f"{root_name}/{split_name}"
                
                loader = CocoDataLoader(str(json_path), str(img_root), source_name=source_name)
                
                if combined_loader is None:
                    combined_loader = loader
                else:
                    combined_loader.merge(loader)
            
            progress.close()
            
            if combined_loader:
                # Use the first split's img_root path for widget initialization
                # Since CocoDataLoader sets abs_path for all images, widgets can use abs_path
                # But we still need to set img_root for widgets that check it
                if first_img_root:
                    self._handle_new_loader(combined_loader, first_img_root)
                else:
                    # Fallback to first root dir if no tasks were processed
                    self._handle_new_loader(combined_loader, root_dirs[0])
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            progress.close()
            self.lbl_status.setText(f"Error: {e}")
            QMessageBox.critical(self, "Loading Error", f"Failed to load datasets:\n{e}")

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

        # Show loading dialog immediately after file selection
        progress = QProgressDialog(
            "Loading YOLO dataset...\n\nPlease wait while the data is being loaded.\n"
            "This may take a moment for large datasets.",
            None, 0, 0, self
        )
        progress.setWindowTitle("Loading Data")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setCancelButton(None)  # Disable cancel button
        progress.setMinimumDuration(0)  # Show immediately
        progress.setRange(0, 0)  # Indeterminate progress
        progress.show()
        progress.setValue(0)
        
        # Process events to show the dialog
        from PySide6.QtWidgets import QApplication
        QApplication.processEvents()

        try:
            # Generate source name from YAML path
            from pathlib import Path
            source_name = Path(yaml_path).stem  # Use filename without extension as source name
            new_loader = YoloDataLoader(yaml_path)
            new_loader.set_source(source_name)  # Set source after loading
            # Use base_path if available, otherwise use img_root
            if hasattr(new_loader, "base_path") and new_loader.base_path:
                img_root_path = str(new_loader.base_path)
            else:
                img_root_path = str(new_loader.img_root) if new_loader.img_root else ""
            progress.close()
            self._handle_new_loader(new_loader, img_root_path)

        except Exception as e:
            import traceback

            traceback.print_exc()
            progress.close()
            self.lbl_status.setText(f"Error: {e}")
            QMessageBox.critical(self, "Loading Error", f"Failed to load YOLO dataset:\n{e}")

    def _update_ui_with_loader(self):
        """Update UI with the loaded data loader (optimized for large datasets)."""
        from PySide6.QtWidgets import QApplication
        
        stats = self.loader.get_stats()
        
        # Show loading dialog for UI updates (especially for large datasets)
        is_large_dataset = stats['Total Images'] > 10000 or stats['Total Instances'] > 100000
        if is_large_dataset:
            progress = QProgressDialog("Updating UI...", None, 0, 0, self)
            progress.setWindowTitle("Processing Data")
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setCancelButton(None)  # Disable cancel button
            progress.setMinimumDuration(0)  # Show immediately
            progress.setRange(0, 0)  # Indeterminate progress
            progress.show()
            progress.setValue(0)
            QApplication.processEvents()
        else:
            progress = None
        
        self.lbl_status.setText(
            f"Loaded: {stats['Total Images']} images, {stats['Total Instances']} objects"
        )
        QApplication.processEvents()

        # Reset initialization tracking
        self.widgets_initialized = {}
        self.pending_updates = []

        # Always update overview tab immediately (lightweight)
        self.overview_tab.update_data(self.loader)
        self.widgets_initialized['overview'] = True
        QApplication.processEvents()

        # For large datasets, defer heavy widget updates until tab is accessed
        # Only update lightweight widgets immediately
        if is_large_dataset:
            # Defer heavy operations - only update when tab is accessed
            self.widgets_initialized['stat'] = False
            self.widgets_initialized['geo'] = False
            self.widgets_initialized['spatial'] = False
            self.widgets_initialized['relation'] = False
            self.widgets_initialized['viewer'] = False
            
            # Update lightweight widgets in batches with processEvents
            # Batch 1
            self.diff_tab.update_data(self.loader)
            QApplication.processEvents()
            
            self.health_tab.update_data(self.loader)
            QApplication.processEvents()
            
            self.dup_tab.update_data(self.loader)
            self.dup_tab.set_img_root(self.img_root)
            QApplication.processEvents()
            
            # Batch 2
            self.strat_tab.update_data(self.loader)
            QApplication.processEvents()
            
            self.quality_tab.update_data(self.loader)
            self.quality_tab.set_img_root(self.img_root)
            QApplication.processEvents()
            
            # Batch 3
            self.signal_tab.update_data(self.loader)
            self.signal_tab.set_img_root(self.img_root)
            QApplication.processEvents()
            
            self.advanced_tab.update_data(self.loader)
            self.advanced_tab.set_img_root(self.img_root)
            QApplication.processEvents()
            
            self.carto_tab.update_data(self.loader)
            self.carto_tab.set_img_root(self.img_root)
            QApplication.processEvents()
            
            # Create placeholder viewer (will be initialized on first access)
            viewer_idx = self.tabs.indexOf(self.viewer_tab)
            self.tabs.removeTab(viewer_idx)
            self.viewer_tab = QWidget()  # Placeholder
            self.tabs.insertTab(viewer_idx, self.viewer_tab, "📸 Viewer")
            QApplication.processEvents()
        else:
            # Small dataset - update everything in batches
            self.stat_tab.plot_class_distribution(self.loader.annotations)
            QApplication.processEvents()
            
            self.geo_tab.update_data(self.loader)
            QApplication.processEvents()
            
            self.spatial_tab.update_data(self.loader)
            QApplication.processEvents()
            
            self.rel_tab.update_data(self.loader)
            QApplication.processEvents()
            
            self.diff_tab.update_data(self.loader)
            QApplication.processEvents()
            
            self.health_tab.update_data(self.loader)
            QApplication.processEvents()
            
            self.dup_tab.update_data(self.loader)
            self.dup_tab.set_img_root(self.img_root)
            QApplication.processEvents()
            
            self.strat_tab.update_data(self.loader)
            QApplication.processEvents()
            
            self.quality_tab.update_data(self.loader)
            self.quality_tab.set_img_root(self.img_root)
            QApplication.processEvents()
            
            self.signal_tab.update_data(self.loader)
            self.signal_tab.set_img_root(self.img_root)
            QApplication.processEvents()
            
            self.advanced_tab.update_data(self.loader)
            self.advanced_tab.set_img_root(self.img_root)
            QApplication.processEvents()
            
            self.carto_tab.update_data(self.loader)
            self.carto_tab.set_img_root(self.img_root)
            QApplication.processEvents()
            
            # Initialize Viewer immediately for small datasets
            viewer_idx = self.tabs.indexOf(self.viewer_tab)
            self.tabs.removeTab(viewer_idx)
            self.viewer_tab = ViewerWidget(self.loader, self.img_root)
            self.tabs.insertTab(viewer_idx, self.viewer_tab, "📸 Viewer")
            self.widgets_initialized['viewer'] = True
            QApplication.processEvents()

        self.tabs.setCurrentIndex(0)
        QApplication.processEvents()
        
        # Close loading dialog if shown
        if progress:
            progress.close()
            QApplication.processEvents()
        
        # Show friendly loading completion message
        QMessageBox.information(
            self,
            "Data Loaded Successfully! 🎉",
            f"Dataset loaded successfully!\n\n"
            f"📊 Statistics:\n"
            f"  • Total Images: {stats['Total Images']}\n"
            f"  • Total Objects: {stats['Total Instances']}\n"
            f"  • Total Classes: {stats['Total Classes']}\n\n"
            f"You can now explore the data using the various analysis tabs."
        )

    def on_tab_changed(self, index):
        """Handle tab change for lazy loading of widgets."""
        if not self.loader:
            return

        tab_name = self.tabs.tabText(index)
        tab_widget = self.tabs.widget(index)

        # Initialize widgets on first access for large datasets
        if tab_widget == self.stat_tab and not self.widgets_initialized.get('stat', False):
            self._initialize_stat_tab()
        elif tab_widget == self.geo_tab and not self.widgets_initialized.get('geo', False):
            self._initialize_geo_tab()
        elif tab_widget == self.spatial_tab and not self.widgets_initialized.get('spatial', False):
            self._initialize_spatial_tab()
        elif tab_widget == self.rel_tab and not self.widgets_initialized.get('relation', False):
            self._initialize_relation_tab()
        elif tab_widget == self.viewer_tab and not self.widgets_initialized.get('viewer', False):
            self._initialize_viewer_tab()

    def _initialize_stat_tab(self):
        """Initialize stat tab."""
        df = self.loader.annotations
        if not df.empty:
            self.stat_tab.plot_class_distribution(df)
        self.widgets_initialized['stat'] = True

    def _initialize_geo_tab(self):
        """Initialize geometry tab."""
        self.geo_tab.update_data(self.loader)
        self.widgets_initialized['geo'] = True

    def _initialize_spatial_tab(self):
        """Initialize spatial tab."""
        self.spatial_tab.update_data(self.loader)
        self.widgets_initialized['spatial'] = True

    def _initialize_relation_tab(self):
        """Initialize relation tab (sampling handled in widget)."""
        self.rel_tab.update_data(self.loader)
        self.widgets_initialized['relation'] = True

    def _initialize_viewer_tab(self):
        """Initialize viewer tab (lazy loading for large datasets)."""
        viewer_idx = self.tabs.indexOf(self.viewer_tab)
        self.tabs.removeTab(viewer_idx)
        self.viewer_tab = ViewerWidget(self.loader, self.img_root)
        self.tabs.insertTab(viewer_idx, self.viewer_tab, "📸 Viewer")
        self.widgets_initialized['viewer'] = True

    def open_image_in_viewer(self, img_id):
        """Open the image in the viewer."""
        if isinstance(self.viewer_tab, ViewerWidget):
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
