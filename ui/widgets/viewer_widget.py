from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class ImageViewer(QWidget):
    def __init__(self, data_loader, img_root_path):
        super().__init__()
        self.loader = data_loader
        self.img_root = img_root_path

        main_layout = QVBoxLayout(self)

        content_layout = QHBoxLayout()

        self.img_list = QListWidget()
        if self.loader:
            self.img_list.addItems(
                [str(img_id) for img_id in self.loader.images.keys()]
            )
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

    def select_image_by_id(self, img_id):
        items = self.img_list.findItems(str(img_id), Qt.MatchExactly)
        if items:
            row = self.img_list.row(items[0])
            self.img_list.setCurrentRow(row)
