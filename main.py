"""Main entry point for the Object Detection EDA Tool."""

import os
import sys

from PySide6.QtWidgets import QApplication

from ui.main_window import MainWindow


def main():
    """Run the main entry point for the Object Detection EDA Tool."""
    # Set OpenBLAS thread limit to avoid memory allocation errors
    # This must be set before importing any libraries that use OpenBLAS (numpy, scipy, etc.)
    if "OPENBLAS_NUM_THREADS" not in os.environ:
        # Set to a safe default (4 threads) to avoid OpenBLAS errors
        # Users can override by setting the environment variable before running
        os.environ["OPENBLAS_NUM_THREADS"] = "4"
    
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
