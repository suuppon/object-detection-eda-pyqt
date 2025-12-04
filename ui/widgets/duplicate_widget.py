import os

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
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

from core.analysis.duplicate import DuplicateFinderThread


class DuplicateWidget(QWidget):
    def __init__(self, data_loader=None, img_root=""):
        super().__init__()
        self.loader = data_loader
        self.img_root = img_root
        self.finder_thread = None
        self.duplicates_dict = {}  # Store duplicate results
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
        self.lbl_marked = QLabel("Marked for deletion: 0")
        self.lbl_marked.setStyleSheet("color: red; font-weight: bold;")

        control_layout.addWidget(self.btn_scan)
        control_layout.addWidget(self.progress_bar)
        control_layout.addWidget(self.lbl_status)
        control_layout.addWidget(self.lbl_marked)
        control_panel.setLayout(control_layout)

        layout.addWidget(control_panel)

        # 메인 영역 (트리 + 프리뷰)
        content_layout = QHBoxLayout()

        # 결과 트리
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(
            ["Marked for deletion", "Hash Group / Image ID", "File Name"]
        )
        self.tree.setColumnWidth(0, 50)
        self.tree.setColumnWidth(1, 200)
        self.tree.itemClicked.connect(self.on_item_clicked)
        self.tree.itemChanged.connect(self.on_tree_item_changed)
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

        self.duplicates_dict = duplicates

        # Store duplicate groups in loader
        if self.loader:
            duplicate_groups = DuplicateFinderThread.convert_to_groups(duplicates)
            self.loader.set_duplicate_groups(duplicate_groups)

        # Temporarily disconnect itemChanged to avoid triggering during tree population
        self.tree.itemChanged.disconnect(self.on_tree_item_changed)
        self.tree.clear()

        for hash_val, img_ids in duplicates.items():
            group_item = QTreeWidgetItem(self.tree)
            group_item.setText(1, f"Group: {hash_val[:8]}... ({len(img_ids)} files)")
            group_item.setData(1, Qt.UserRole, img_ids)  # 그룹 데이터 저장
            group_item.setFlags(
                group_item.flags() & ~Qt.ItemIsUserCheckable
            )  # Group items are not checkable

            for img_id in img_ids:
                img_info = self.loader.images[img_id]
                child_item = QTreeWidgetItem(group_item)

                # Add checkbox in column 0
                is_excluded = img_id in self.loader.excluded_image_ids
                child_item.setFlags(child_item.flags() | Qt.ItemIsUserCheckable)
                child_item.setCheckState(0, Qt.Checked if is_excluded else Qt.Unchecked)

                child_item.setText(1, str(img_id))
                child_item.setText(2, img_info["file_name"])
                child_item.setData(1, Qt.UserRole, img_id)  # 개별 ID 저장

                # Color if excluded
                if is_excluded:
                    child_item.setForeground(1, Qt.red)
                    child_item.setForeground(2, Qt.red)

        self.tree.expandAll()
        # Reconnect itemChanged signal
        self.tree.itemChanged.connect(self.on_tree_item_changed)
        self.update_marked_count()

    def on_scan_error(self, msg):
        self.btn_scan.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.lbl_status.setText(f"Error: {msg}")

    def on_item_clicked(self, item, column):
        # Don't show preview when clicking checkbox column
        if column == 0:
            return

        data = item.data(1, Qt.UserRole)

        if isinstance(data, list):  # 그룹 선택 시
            self.show_preview_images(data)
        elif isinstance(data, int):  # 개별 이미지 선택 시
            # 부모 그룹의 모든 이미지를 보여주되, 선택된 것을 강조?
            # 일단 그룹 전체를 보여주는 게 비교에 좋음
            parent = item.parent()
            if parent:
                group_ids = parent.data(1, Qt.UserRole)
                self.show_preview_images(group_ids)

    def on_tree_item_changed(self, item, column):
        """Handle checkbox state change in tree."""
        if column != 0:
            return

        img_id = item.data(1, Qt.UserRole)
        if not isinstance(img_id, int):
            return

        state = item.checkState(0)

        if state == 2:  # Qt.Checked
            self.loader.mark_image_for_exclusion(img_id)
        else:
            self.loader.unmark_image_for_exclusion(img_id)

        # Update colors
        is_excluded = img_id in self.loader.excluded_image_ids
        color = Qt.red if is_excluded else Qt.black
        item.setForeground(1, color)
        item.setForeground(2, color)

        self.update_marked_count()

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

            # 이미지 컨테이너 (이미지 + 체크박스 + 라벨)
            container = QWidget()
            v_layout = QVBoxLayout(container)

            # Check if image is marked for exclusion
            is_excluded = img_id in self.loader.excluded_image_ids

            lbl_img = QLabel()
            pixmap = QPixmap(file_path)
            if not pixmap.isNull():
                lbl_img.setPixmap(pixmap.scaled(300, 300, Qt.KeepAspectRatio))
            else:
                lbl_img.setText("Load Failed")

            # Apply visual style if excluded
            if is_excluded:
                container.setStyleSheet(
                    "QWidget { border: 3px solid red; background-color: rgba(255, 0, 0, 30); }"
                )

            lbl_info = QLabel(f"ID: {img_id}\n{img_info['file_name']}")
            lbl_info.setAlignment(Qt.AlignCenter)

            # Checkbox for marking deletion
            chk_delete = QCheckBox("Mark for deletion")
            chk_delete.setChecked(is_excluded)
            chk_delete.stateChanged.connect(
                lambda state, img_id=img_id: self.on_mark_changed(img_id, state)
            )

            v_layout.addWidget(lbl_img)
            v_layout.addWidget(lbl_info)
            v_layout.addWidget(chk_delete)

            self.preview_layout.addWidget(container)

    def on_mark_changed(self, img_id, state):
        """Handle marking/unmarking images for deletion."""
        if state == 2:
            self.loader.mark_image_for_exclusion(img_id)
        else:
            self.loader.unmark_image_for_exclusion(img_id)

        self.update_marked_count()

        # Refresh preview to update visual styling
        if self.tree.currentItem():
            item = self.tree.currentItem()
            data = item.data(1, Qt.UserRole)
            if isinstance(data, list):
                self.show_preview_images(data)
            elif isinstance(data, int):
                parent = item.parent()
                if parent:
                    group_ids = parent.data(1, Qt.UserRole)
                    self.show_preview_images(group_ids)

    def refresh_marked_status(self):
        """Refresh marked status in tree."""
        if not self.loader:
            return

        # Update tree item checkboxes and colors
        for i in range(self.tree.topLevelItemCount()):
            group_item = self.tree.topLevelItem(i)
            for j in range(group_item.childCount()):
                child_item = group_item.child(j)
                img_id = child_item.data(1, Qt.UserRole)
                if isinstance(img_id, int):
                    is_excluded = img_id in self.loader.excluded_image_ids
                    child_item.setCheckState(
                        0, Qt.Checked if is_excluded else Qt.Unchecked
                    )
                    color = Qt.red if is_excluded else Qt.black
                    child_item.setForeground(1, color)
                    child_item.setForeground(2, color)

        self.update_marked_count()

    def update_marked_count(self):
        """Update the marked images counter."""
        if self.loader:
            count = len(self.loader.excluded_image_ids)
            self.lbl_marked.setText(f"Marked for deletion: {count}")
