from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class ViewerWidget(QWidget):
    def __init__(self, data_loader, img_root_path):
        super().__init__()
        self.loader = data_loader
        self.img_root = img_root_path
        self.all_image_ids = list(self.loader.images.keys()) if self.loader else []
        self.filtered_image_ids = self.all_image_ids.copy()

        main_layout = QVBoxLayout(self)

        # Search and navigation controls
        controls_layout = QHBoxLayout()
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search by Image ID or filename...")
        self.search_box.textChanged.connect(self.filter_images)
        
        self.page_spin = QSpinBox()
        self.page_spin.setMinimum(1)
        self.page_spin.setMaximum(1)
        self.page_spin.setValue(1)
        self.page_spin.valueChanged.connect(self.on_page_changed)
        self.page_size = 1000  # Show 1000 items per page
        
        controls_layout.addWidget(QLabel("Search:"))
        controls_layout.addWidget(self.search_box, 2)
        controls_layout.addWidget(QLabel("Page:"))
        controls_layout.addWidget(self.page_spin)
        main_layout.addLayout(controls_layout)

        content_layout = QHBoxLayout()

        self.img_list = QListWidget()
        self.update_image_list()
        self.img_list.currentRowChanged.connect(self.display_image)

        self.image_label = QLabel("Select an Image")
        self.image_label.setAlignment(Qt.AlignCenter)

        content_layout.addWidget(self.img_list, 1)
        content_layout.addWidget(self.image_label, 4)
        main_layout.addLayout(content_layout)

        # Guide button at bottom
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.btn_guide = QPushButton("📖 View Guide: Visual Explorer")
        self.btn_guide.clicked.connect(lambda: self._navigate_to_guide())
        btn_layout.addWidget(self.btn_guide)
        main_layout.addLayout(btn_layout)

    def _navigate_to_guide(self):
        main_window = self.window()
        if hasattr(main_window, "navigate_to_guide"):
            main_window.navigate_to_guide("viewer")

    def display_image(self, row_index):
        if row_index < 0:
            return

        img_id = int(self.img_list.item(row_index).text())
        img_info = self.loader.images[img_id]
        
        # Use absolute path if available, otherwise construct from img_root
        if "abs_path" in img_info and img_info["abs_path"]:
            file_path = img_info["abs_path"]
        else:
            file_path = f"{self.img_root}/{img_info['file_name']}"

        pixmap = QPixmap(file_path)
        if pixmap.isNull():
            self.image_label.setText(f"Failed to load: {file_path}")
            return

        # 원본을 유지해야 줌인/줌아웃 등에서 깨지지 않음
        # 여기서는 단순히 복사본을 사용
        display_pixmap = pixmap.copy()

        painter = QPainter(display_pixmap)
        pen = QPen(QColor(255, 0, 0), 2)
        painter.setPen(pen)

        # 폰트 설정 (이모지 지원을 위해 기본 폰트 사용하지만, 시스템에 따라 다를 수 있음)
        # font = painter.font()
        # font.setPointSize(10)
        # painter.setFont(font)

        anns = self.loader.annotations[self.loader.annotations["image_id"] == img_id]
        for _, row in anns.iterrows():
            x, y, w, h = row["bbox"]
            painter.drawRect(int(x), int(y), int(w), int(h))

            # 텍스트 배경 (가독성 향상)
            text = row["category_name"]
            # 이모지 문제: PyQt의 QPainter.drawText는 컬러 이모지를 제대로 렌더링하지 못할 수 있음.
            # 해결책: 단순 텍스트로 표시하거나, QLabel 오버레이 사용.
            # 여기서는 텍스트만 그림
            painter.drawText(int(x), int(y) - 5, text)

        painter.end()

        self.image_label.setPixmap(
            display_pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio)
        )

    def filter_images(self, text):
        """Filter images based on search text."""
        if not text:
            self.filtered_image_ids = self.all_image_ids.copy()
        else:
            text_lower = text.lower()
            self.filtered_image_ids = []
            for img_id in self.all_image_ids:
                img_info = self.loader.images[img_id]
                # Search by ID or filename
                if (text_lower in str(img_id).lower() or 
                    text_lower in img_info.get("file_name", "").lower()):
                    self.filtered_image_ids.append(img_id)
        
        self.page_spin.setMaximum(max(1, (len(self.filtered_image_ids) + self.page_size - 1) // self.page_size))
        self.page_spin.setValue(1)
        self.update_image_list()

    def update_image_list(self):
        """Update the image list with current page."""
        self.img_list.clear()
        
        if not self.filtered_image_ids:
            return
        
        # Calculate page range
        page = self.page_spin.value() - 1
        start_idx = page * self.page_size
        end_idx = min(start_idx + self.page_size, len(self.filtered_image_ids))
        
        # Add items for current page
        page_ids = self.filtered_image_ids[start_idx:end_idx]
        self.img_list.addItems([str(img_id) for img_id in page_ids])
        
        # Update page info
        total_pages = (len(self.filtered_image_ids) + self.page_size - 1) // self.page_size
        if total_pages > 0:
            self.page_spin.setMaximum(total_pages)
            self.page_spin.setSuffix(f" / {total_pages}")

    def on_page_changed(self, value):
        """Handle page change."""
        self.update_image_list()

    def select_image_by_id(self, img_id):
        """Select image by ID, navigating to the correct page if needed."""
        if img_id not in self.filtered_image_ids:
            # If not in filtered list, clear filter and add to filtered list
            self.search_box.clear()
            self.filtered_image_ids = self.all_image_ids.copy()
            self.update_image_list()
        
        if img_id in self.filtered_image_ids:
            # Find which page contains this image
            idx = self.filtered_image_ids.index(img_id)
            page = (idx // self.page_size) + 1
            self.page_spin.setValue(page)
            
            # Find item in current list
            items = self.img_list.findItems(str(img_id), Qt.MatchExactly)
            if items:
                row = self.img_list.row(items[0])
                self.img_list.setCurrentRow(row)
