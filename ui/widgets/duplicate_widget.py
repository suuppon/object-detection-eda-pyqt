import os

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.duplicate_finder import DuplicateFinderThread


class DuplicateWidget(QWidget):
    def __init__(self, data_loader=None, img_root=""):
        super().__init__()
        self.loader = data_loader
        self.img_root = img_root
        self.finder_thread = None
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)

        # 컨트롤 패널
        control_panel = QGroupBox("Duplicate Image Detection")
        control_layout = QHBoxLayout()

        self.btn_scan = QPushButton("🔍 Scan for Duplicates (PHash)")
        self.btn_scan.clicked.connect(self.run_scan)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)

        self.lbl_status = QLabel("Ready to scan.")

        control_layout.addWidget(self.btn_scan)
        control_layout.addWidget(self.progress_bar)
        control_layout.addWidget(self.lbl_status)
        control_panel.setLayout(control_layout)

        layout.addWidget(control_panel)

        # 메인 영역 (트리 + 프리뷰)
        content_layout = QHBoxLayout()

        # 결과 트리
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Hash Group / Image ID", "File Name"])
        self.tree.setColumnWidth(0, 200)
        self.tree.itemClicked.connect(self.on_item_clicked)
        content_layout.addWidget(self.tree, 1)

        # 프리뷰 영역 (스크롤 가능)
        preview_container = QWidget()
        self.preview_layout = QHBoxLayout(preview_container)
        self.preview_layout.setAlignment(Qt.AlignTop)

        scroll = QScrollArea()
        scroll.setWidget(preview_container)
        scroll.setWidgetResizable(True)
        content_layout.addWidget(scroll, 2)

        layout.addLayout(content_layout)

        # Guide button at bottom
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.btn_guide = QPushButton("📖 View Guide: Duplicate Detection")
        self.btn_guide.clicked.connect(lambda: self._navigate_to_guide())
        btn_layout.addWidget(self.btn_guide)
        layout.addLayout(btn_layout)

    def _navigate_to_guide(self):
        main_window = self.window()
        if hasattr(main_window, "navigate_to_guide"):
            main_window.navigate_to_guide("duplicates")

    def update_data(self, data_loader):
        self.loader = data_loader
        # img_root는 main_window에서 직접 설정해주거나 loader에 포함시켜야 함
        # 여기서는 일단 빈 상태, main_window에서 set_img_root 호출 필요
        self.tree.clear()
        self.lbl_status.setText("Data loaded. Ready to scan.")

    def set_img_root(self, path):
        self.img_root = path

    def run_scan(self):
        if not self.loader or not self.img_root:
            self.lbl_status.setText("Error: Data or Image Root not loaded.")
            return

        self.btn_scan.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.tree.clear()
        self.clear_preview()

        self.finder_thread = DuplicateFinderThread(self.loader.images, self.img_root)
        self.finder_thread.progress.connect(self.update_progress)
        self.finder_thread.finished.connect(self.on_scan_finished)
        self.finder_thread.error.connect(self.on_scan_error)
        self.finder_thread.start()

    def update_progress(self, current, total):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.lbl_status.setText(f"Scanning... {current}/{total}")

    def on_scan_finished(self, duplicates):
        self.btn_scan.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.lbl_status.setText(
            f"Scan Complete. Found {len(duplicates)} duplicate groups."
        )

        self.tree.clear()
        for hash_val, img_ids in duplicates.items():
            group_item = QTreeWidgetItem(self.tree)
            group_item.setText(0, f"Group: {hash_val[:8]}... ({len(img_ids)} files)")
            group_item.setData(0, Qt.UserRole, img_ids)  # 그룹 데이터 저장

            for img_id in img_ids:
                img_info = self.loader.images[img_id]
                child_item = QTreeWidgetItem(group_item)
                child_item.setText(0, str(img_id))
                child_item.setText(1, img_info["file_name"])
                child_item.setData(0, Qt.UserRole, img_id)  # 개별 ID 저장

        self.tree.expandAll()

    def on_scan_error(self, msg):
        self.btn_scan.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.lbl_status.setText(f"Error: {msg}")

    def on_item_clicked(self, item, column):
        data = item.data(0, Qt.UserRole)

        if isinstance(data, list):  # 그룹 선택 시
            self.show_preview_images(data)
        elif isinstance(data, int):  # 개별 이미지 선택 시
            # 부모 그룹의 모든 이미지를 보여주되, 선택된 것을 강조?
            # 일단 그룹 전체를 보여주는 게 비교에 좋음
            parent = item.parent()
            if parent:
                group_ids = parent.data(0, Qt.UserRole)
                self.show_preview_images(group_ids)

    def clear_preview(self):
        while self.preview_layout.count():
            child = self.preview_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def show_preview_images(self, img_ids):
        self.clear_preview()

        for img_id in img_ids:
            if img_id not in self.loader.images:
                continue

            img_info = self.loader.images[img_id]
            file_path = os.path.join(self.img_root, img_info["file_name"])

            # 이미지 컨테이너 (이미지 + 라벨)
            container = QWidget()
            v_layout = QVBoxLayout(container)

            lbl_img = QLabel()
            pixmap = QPixmap(file_path)
            if not pixmap.isNull():
                lbl_img.setPixmap(pixmap.scaled(300, 300, Qt.KeepAspectRatio))
            else:
                lbl_img.setText("Load Failed")

            lbl_info = QLabel(f"ID: {img_id}\n{img_info['file_name']}")
            lbl_info.setAlignment(Qt.AlignCenter)

            v_layout.addWidget(lbl_img)
            v_layout.addWidget(lbl_info)

            self.preview_layout.addWidget(container)
