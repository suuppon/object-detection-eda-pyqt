"""Main entry point for the Object Detection EDA Tool."""

import sys

from PySide6.QtWidgets import QApplication

from ui.main_window import MainWindow


def main():
    """Run the main entry point for the Object Detection EDA Tool."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
