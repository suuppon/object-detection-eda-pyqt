"""Dialog for loading multiple COCO datasets."""

from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QListWidget,
    QListView,
    QTreeView,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QLabel,
)
import os


class FileDropListWidget(QListWidget):
    """ListWidget that accepts dropped directories."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
            # Get parent dialog to access selected_dirs logic
            # This assumes the parent or grandparent is MultiDatasetLoadDialog
            # But better to emit a signal or just handle items directly if possible.
            # We can't easily access 'selected_dirs' list from here without coupling.
            # So we will emit a signal that the parent connects to? Or just add items.
            # Wait, the parent maintains `selected_dirs` list. We should update it.
            
            # Alternative: The parent implements drop logic? No, drag events go to the widget.
            # Let's just add items here and let parent sync, or let parent handle dropping.
            # Easier: Parent assigns a callback or we emit a signal.
            pass  # Implemented in parent's event filter or simply here if we pass callback.
            
    # Actually, it is cleaner to implement drag events in the QListWidget subclass 
    # and expose a signal 'filesDropped'
    
from PySide6.QtCore import Signal

class FileDropListWidget(QListWidget):
    """ListWidget that accepts dropped directories."""
    
    filesDropped = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
            paths = []
            for url in event.mimeData().urls():
                path = url.toLocalFile()
                if os.path.isdir(path):
                    paths.append(path)
            if paths:
                self.filesDropped.emit(paths)


class MultiDatasetLoadDialog(QDialog):
    """Dialog to select multiple root directories and splits."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Load Multiple COCO Datasets")
        self.resize(600, 450)
        self.selected_dirs = []
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)

        # 1. Directory Selection
        dir_group = QGroupBox("Selected Root Directories")
        dir_layout = QVBoxLayout()

        self.list_dirs = FileDropListWidget()
        self.list_dirs.filesDropped.connect(self.add_dropped_dirs)
        dir_layout.addWidget(self.list_dirs)
        
        lbl_hint = QLabel("💡 Tip: Drag and drop folders here or use 'Add Directories'")
        lbl_hint.setStyleSheet("color: gray; font-size: 11px;")
        dir_layout.addWidget(lbl_hint)

        btn_layout = QHBoxLayout()
        self.btn_add = QPushButton("Add Directories")
        self.btn_add.clicked.connect(self.add_directory)
        self.btn_remove = QPushButton("Remove Selected")
        self.btn_remove.clicked.connect(self.remove_directory)
        
        btn_layout.addWidget(self.btn_add)
        btn_layout.addWidget(self.btn_remove)
        dir_layout.addLayout(btn_layout)
        
        dir_group.setLayout(dir_layout)
        layout.addWidget(dir_group)

        # 2. Split Selection
        split_group = QGroupBox("Select Splits to Load")
        split_layout = QHBoxLayout()

        self.chk_train = QCheckBox("train")
        self.chk_train.setChecked(True)
        self.chk_val = QCheckBox("valid")
        self.chk_val.setChecked(True)
        self.chk_test = QCheckBox("test")
        self.chk_test.setChecked(True)

        split_layout.addWidget(self.chk_train)
        split_layout.addWidget(self.chk_val)
        split_layout.addWidget(self.chk_test)
        split_group.setLayout(split_layout)
        layout.addWidget(split_group)
        
        # Description
        lbl_desc = QLabel(
            "Note: Each root directory should contain subdirectories matching the selected splits\n"
            "(e.g. 'train', 'valid', 'test'), each containing a JSON file and images."
        )
        lbl_desc.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(lbl_desc)

        # 3. Dialog Buttons
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    def add_dropped_dirs(self, paths):
        """Handle directories dropped onto the list."""
        for path in paths:
            if path not in self.selected_dirs:
                self.selected_dirs.append(path)
                self.list_dirs.addItem(path)

    def add_directory(self):
        """Open file dialog to select one or multiple directories."""
        # Use QFileDialog instance to enable multi-selection of directories
        dialog = QFileDialog(self, "Select Root Directories")
        dialog.setFileMode(QFileDialog.Directory)
        dialog.setOption(QFileDialog.ShowDirsOnly, True)
        
        # Try to enable multiple selection
        # Note: This often requires using the non-native dialog
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        
        # Find the view to set selection mode
        view = dialog.findChild(QListView, "listView")
        if view:
            view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        
        tree = dialog.findChild(QTreeView)
        if tree:
            tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
            
        if dialog.exec():
            paths = dialog.selectedFiles()
            for path in paths:
                if path not in self.selected_dirs:
                    self.selected_dirs.append(path)
                    self.list_dirs.addItem(path)

    def remove_directory(self):
        """Remove selected directories from list."""
        # Support removing multiple selected items
        selected_items = self.list_dirs.selectedItems()
        if not selected_items:
            return
            
        for item in selected_items:
            row = self.list_dirs.row(item)
            path = item.text()
            if path in self.selected_dirs:
                self.selected_dirs.remove(path)
            self.list_dirs.takeItem(row)

    def get_config(self):
        """Get the dialog configuration."""
        splits = []
        if self.chk_train.isChecked():
            splits.append("train")
        if self.chk_val.isChecked():
            splits.append("valid")
        if self.chk_test.isChecked():
            splits.append("test")
        
        return {
            "dirs": self.selected_dirs,
            "splits": splits
        }

