# Object Detection EDA Tool

A PySide6-based desktop application for Exploratory Data Analysis (EDA) of object detection datasets in COCO/YOLO format.

## Quick Start

```bash
# Install dependencies
uv sync

# Run the application
uv run main.py
```

## Purpose

This tool provides comprehensive analysis for object detection datasets, including:
- Geometry analysis (anchor boxes, size distribution)
- Spatial analysis (location bias, density)
- Class relation analysis (co-occurrence, imbalance)
- Data quality checks (health, duplicates, image quality)
- Advanced signal analysis (texture, camouflage, FFT, PCA)
- Training strategy recommendations
- Dataset overview and summary
- Export functionality with train/val/test split options
- Support for excluding selected images during export
- Data cartography analysis
