from PySide6.QtWidgets import QHBoxLayout, QPushButton, QTextEdit, QVBoxLayout, QWidget

from core.analyzer import Analyzer


class StrategyWidget(QWidget):
    def __init__(self, data_loader=None):
        super().__init__()
        self.loader = data_loader
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)

        self.btn_gen = QPushButton("✨ Generate Training Strategy")
        self.btn_gen.clicked.connect(self.generate_strategy)
        layout.addWidget(self.btn_gen)

        self.text_report = QTextEdit()
        self.text_report.setReadOnly(True)
        self.text_report.setStyleSheet("font-size: 14px; line-height: 1.4;")
        layout.addWidget(self.text_report)

        # Guide button at bottom
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.btn_guide = QPushButton("📖 View Guide: Training Strategy")
        self.btn_guide.clicked.connect(lambda: self._navigate_to_guide())
        btn_layout.addWidget(self.btn_guide)
        layout.addLayout(btn_layout)

    def _navigate_to_guide(self):
        main_window = self.window()
        if hasattr(main_window, "navigate_to_guide"):
            main_window.navigate_to_guide("strategy")

    def update_data(self, data_loader):
        self.loader = data_loader
        self.text_report.clear()
        self.text_report.setText("Click the button to generate strategy.")

    def generate_strategy(self):
        if not self.loader:
            return

        df = self.loader.annotations
        if df.empty:
            return

        report = []
        report.append("# 🚀 Training Strategy Recommendation\n")

        # 1. Model Architecture
        sizes = Analyzer.get_size_distribution(df)  # S, M, L
        total = sum(sizes)
        small_ratio = sizes[0] / total if total > 0 else 0

        report.append("## 1. Model Architecture")
        if small_ratio > 0.4:
            report.append(
                "- **Recommendation**: Use models with strong feature pyramids (e.g., RetinaNet, YOLOv8-P2)."
            )
            report.append(
                f"- **Reason**: Small objects account for {small_ratio * 100:.1f}% of the dataset."
            )
        else:
            report.append(
                "- **Recommendation**: Standard detectors (YOLOv8, Faster R-CNN) should work fine."
            )
        report.append("")

        # 2. Augmentation
        report.append("## 2. Augmentation Strategy")
        report.append("- **Basic**: Horizontal Flip, Color Jitter.")
        if small_ratio > 0.3:
            report.append(
                "- **Advanced**: Copy-Paste, Mosaic (to improve small object detection)."
            )
            report.append("- **Resolution**: Consider multi-scale training.")

        # Density check
        img_counts = df["image_id"].value_counts()
        avg_obj = img_counts.mean()
        if avg_obj > 20:
            report.append(
                f"- **Density**: High density ({avg_obj:.1f} objs/img). MixUp augmentation recommended."
            )
        report.append("")

        # 3. Hyperparameters
        report.append("## 3. Hyperparameters")
        # Anchor check (K-Means wrapper needed, but let's use aspect ratio stats)
        ar_mean = df["aspect_ratio"].mean()
        ar_std = df["aspect_ratio"].std()

        report.append(
            "- **Input Resolution**: Check the 'Geometry Analysis' tab for image sizes. Generally 640px or higher."
        )
        report.append(
            f"- **Anchor Ratios**: Aspect Ratio Mean {ar_mean:.2f} (std {ar_std:.2f})."
        )
        if ar_std > 1.0:
            report.append(
                "  - High variance in aspect ratios. Consider using K-Means anchors or anchor-free models."
            )

        # Imbalance
        cls_counts = df["category_name"].value_counts()
        min_cls = cls_counts.min()
        max_cls = cls_counts.max()
        if max_cls / min_cls > 10:
            report.append(
                f"- **Loss Function**: Class imbalance detected (Max/Min = {max_cls / min_cls:.1f}). Use **Focal Loss**."
            )

        self.text_report.setMarkdown("\n".join(report))
