from PySide6.QtCore import Qt
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import QScrollArea, QTextEdit, QVBoxLayout, QWidget


class GuideWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.text_edit = None
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)

        # Scrollable content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)

        # Guide content
        guide_text = self.generate_guide_text()
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setMarkdown(guide_text)
        self.text_edit.setStyleSheet(
            """
            QTextEdit {
                font-size: 12px;
                line-height: 1.6;
                padding: 10px;
            }
        """
        )

        content_layout.addWidget(self.text_edit)
        scroll.setWidget(content_widget)
        layout.addWidget(scroll)

    def scroll_to_section(self, section_name):
        """Scroll to a specific section in the guide"""
        if not self.text_edit:
            return

        # Map section names to search terms
        section_map = {
            "overview": "Overview Tab",
            "dashboard": "Dashboard Tab",
            "geometry": "Geometry Analysis Tab",
            "spatial": "Spatial Analysis Tab",
            "relation": "Class Relation Analysis Tab",
            "difficulty": "Difficulty Analysis Tab",
            "duplicates": "Duplicate Detection Tab",
            "health": "Data Health Check Tab",
            "quality": "Image Quality Tab",
            "strategy": "Training Strategy Tab",
            "signal": "Signal Analysis Tab",
            "viewer": "Visual Explorer Tab",
        }

        search_text = section_map.get(section_name.lower(), section_name)

        # Find the section
        cursor = self.text_edit.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.Start)
        self.text_edit.setTextCursor(cursor)

        # Search for the section header
        found = self.text_edit.find(search_text)
        if found:
            # Move cursor to beginning of line
            cursor = self.text_edit.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.StartOfLine)
            self.text_edit.setTextCursor(cursor)
            # Scroll to make it visible
            self.text_edit.ensureCursorVisible()

    def generate_guide_text(self):
        return """
# 📖 Object Detection EDA Tool - User Guide

This guide explains the meaning of metrics and visualizations across all analysis tabs.

---

## 📋 Overview Tab {#overview}

### Dataset Summary
- **Total Images**: Number of images in the dataset
- **Total Instances**: Number of object annotations
- **Total Classes**: Number of unique object categories
- **Excluded Images**: Number of images marked for deletion (from Duplicates/Quality tabs)

### Class Management
- **View and Edit**: See all classes with their IDs and instance counts
- **Rename Classes**: Double-click a class name to rename it
- **Merge Classes**: Renaming multiple classes to the same name will merge them on export

### Export Dataset
Export your dataset in YOLO or COCO format with powerful options:

#### Export Features:
1. **Format Selection**: Choose YOLO or COCO format
2. **Dataset Split**: Optionally split data into train/val/test sets
   - Configure ratios (e.g., 70/20/10)
   - Set random seed for reproducibility
   - Duplicate groups stay together in the same split
3. **Automatic Exclusion**: Images marked in Duplicates/Quality tabs are automatically excluded
4. **Category Normalization**: Categories are automatically renormalized (0, 1, 2...) with duplicate names merged

#### Export Process:
1. Click "Export as YOLO" or "Export as COCO"
2. Configure split options in the dialog
3. Select destination folder
4. Dataset is exported with images and labels/annotations

#### Directory Structure (YOLO with split):
```
output_folder/
├── data.yaml
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

#### Directory Structure (COCO with split):
```
output_folder/
├── train.json
├── val.json
├── test.json
└── images/
    ├── train/
    ├── val/
    └── test/
```

**Tip**: Always check the "Excluded Images" count before exporting to ensure you're not accidentally excluding important data.

---

## 📊 Dashboard Tab {#dashboard}

### Class Distribution
- **What it shows**: Number of instances per class in your dataset
- **Why it matters**: Identifies class imbalance. Severe imbalance may require:
  - Focal Loss instead of standard cross-entropy
  - Class weighting or oversampling strategies
  - Stratified train/val/test splits

---

## 📏 Geometry Analysis Tab {#geometry}

### Width/Height Scatter Plot
- **What it shows**: Distribution of bounding box dimensions across all objects
- **Why it matters**:
  - Helps design anchor boxes for anchor-based detectors (YOLO, RetinaNet)
  - Identifies dominant object sizes in your dataset
  - Use K-Means clustering to find optimal anchor box sizes

### Aspect Ratio Distribution
- **What it shows**: Ratio of width to height (w/h) for all bounding boxes
- **Why it matters**:
  - Extreme ratios (e.g., 1:5 or 5:1) indicate elongated objects (poles, signs)
  - Anchor box aspect ratios should match your data distribution
  - Consider anchor-free models if aspect ratios are highly diverse

### Small/Medium/Large Ratio
- **What it shows**: Proportion of objects by size category (COCO metric):
  - **Small**: Area < 32² pixels
  - **Medium**: 32² ≤ Area < 96² pixels
  - **Large**: Area ≥ 96² pixels
- **Why it matters**:
  - High small object ratio (>40%) → Use FPN (Feature Pyramid Network) or higher input resolution
  - Consider tiling/slicing strategies for very small objects
  - Multi-scale training may be beneficial

### Image Resolution Distribution
- **What it shows**: Width × Height distribution of all images
- **Why it matters**:
  - Determines optimal input resolution for training
  - Inconsistent resolutions may require resizing or padding strategies
  - Very high resolutions may need downsampling for efficiency

### K-Means Anchor Analysis
- **What it shows**: Clustered anchor box centers based on your data
- **Why it matters**:
  - Provides data-driven anchor box configuration
  - Reduces manual anchor design effort
  - Improves detection performance by matching data distribution

---

## 🗺️ Spatial Analysis Tab {#spatial}

### Normalized Object Center Distribution (Heatmap)
- **What it shows**: Where objects tend to appear in images (normalized 0-1 coordinates)
- **Why it matters**:
  - **Central bias**: Objects clustered in center → Use Random Crop augmentation
  - **Edge bias**: Objects near borders → May need padding or larger input size
  - **Uniform distribution**: Good for generalization

### Class-wise Spatial Distribution
- **What it shows**: Location patterns for top 5 classes (KDE visualization)
- **Why it matters**:
  - Identifies class-specific location biases
  - Example: "People always in center" → Strengthen Random Crop augmentation
  - Helps design class-specific augmentation strategies

### Objects per Image Distribution
- **What it shows**: Histogram of object counts per image
- **Why it matters**:
  - **High density** (>20 objects/image):
    - Lower NMS IoU threshold
    - Increase `max_det` parameter
    - Consider Dense Detection models
  - **Low density** (<5 objects/image):
    - Standard NMS settings should work
    - May benefit from Mosaic augmentation

### BBox Area / Image Area Ratio
- **What it shows**: How much of each image is covered by bounding boxes
- **Why it matters**:
  - High ratio → Objects are large relative to image
  - Low ratio → Objects are small, may need higher resolution
  - Helps understand scale relationships

---

## 🤝 Class Relation Analysis Tab {#relation}

### Class Distribution (Log Scale)
- **What it shows**: Instance counts per class on logarithmic scale
- **Why it matters**:
  - Identifies rare classes (long tail distribution)
  - Severe imbalance (max/min > 10) → Use Focal Loss or class weighting
  - Oversampling rare classes may improve performance

### Class Co-occurrence Matrix
- **What it shows**: How often classes appear together in the same image
- **Why it matters**:
  - **High co-occurrence**: Model may learn context dependencies (e.g., "person" + "bat")
  - May cause false positives when context is similar but class is different
  - Consider context-aware post-processing

### Average BBox Area per Class
- **What it shows**: Mean bounding box size for each class
- **Why it matters**:
  - Small classes may need FPN's lower pyramid levels (P3, P4)
  - Large classes can be detected at higher levels (P5, P6)
  - Informs feature pyramid design

### Aspect Ratio Distribution per Class
- **What it shows**: Box plot of width/height ratios for each class
- **Why it matters**:
  - Class-specific anchor ratios can be designed
  - Extreme ratios indicate specialized object shapes
  - Helps understand class-specific detection challenges

---

## 🎯 Difficulty Analysis Tab {#difficulty}

### Difficulty Score
- **What it shows**: Per-image difficulty metric based on:
  - Small object ratio
  - Total object count (density)
- **Why it matters**:
  - Identifies hard samples for curriculum learning
  - Focus annotation quality checks on high-difficulty images
  - May need additional augmentation for hard samples

### Hard Sample Categories
- **Extremely Small Objects**: Area < 16² pixels
- **High Occlusion**: Objects with IoU > 0.5 with others
- **Edge Cases**: Objects near image boundaries
- **Complex Scenes**: Multiple classes in single image

---

## 🔍 Duplicate Detection Tab {#duplicates}

### Perceptual Hash (PHash)
- **What it shows**: Groups of visually similar or identical images
- **Why it matters**:
  - **Data Leakage Prevention**: Remove duplicates between train/val/test splits
  - **Dataset Cleaning**: Identify and remove redundant images
  - **Quality Control**: Find mislabeled duplicates

### How it works:
- Uses perceptual hashing to detect similar images even with minor variations
- Groups images with identical or very similar hashes
- Review groups to decide which images to keep/remove

---

## 🧹 Data Health Check Tab {#health}

### Error Types

#### Tiny Boxes
- **Definition**: Width or height < 1 pixel
- **Impact**: Noise in annotations, can cause training instability
- **Action**: Remove or correct these annotations

#### Giant Boxes
- **Definition**: Bounding box covers >95% of image area
- **Impact**: Likely labeling error (e.g., entire image labeled as one object)
- **Action**: Review and correct annotations

#### Out of Bounds
- **Definition**: Bounding box coordinates exceed image dimensions
- **Impact**: Invalid annotations, can crash training
- **Action**: Clamp coordinates or remove invalid boxes

#### Duplicate Labels
- **Definition**: Multiple identical bounding boxes for same object
- **Impact**: Double-counting in loss calculation
- **Action**: Remove duplicate annotations

#### Missing Files
- **Definition**: Images referenced in JSON but not found in directory
- **Impact**: Training errors when loading images
- **Action**: Add missing files or remove references

---

## 🎨 Image Quality Tab {#quality}

### Brightness Distribution
- **What it shows**: Mean pixel intensity across all images
- **Why it matters**:
  - **Too dark** (<50): May need brightness augmentation or preprocessing
  - **Too bright** (>200): May indicate overexposure
  - **Normal range** (50-200): Good for training

### Blur Score (Laplacian Variance)
- **What it shows**: Image sharpness metric (higher = sharper)
- **Why it matters**:
  - **Low values** (<100): Blurry images may hurt model performance
  - **High values** (>1000): Sharp, good quality images
  - Consider removing very blurry images or applying deblurring

### Contrast
- **What it shows**: Standard deviation of pixel intensities
- **Why it matters**:
  - **Low contrast**: Objects may be hard to distinguish from background
  - **High contrast**: Good for detection
  - Low contrast images may need contrast enhancement

### Brightness vs Contrast Scatter
- **What it shows**: Relationship between brightness and contrast
- **Why it matters**:
  - **Bottom-left** (dark + low contrast): Worst quality
  - **Top-right** (bright + high contrast): Best quality
  - Identify problematic image regions

### Image Area vs Blur Score
- **What it shows**: Relationship between image size and sharpness
- **Why it matters**:
  - Large images with low blur score → Genuinely blurry (remove or fix)
  - Small images with low blur score → May be due to downsampling (acceptable)

---

## 🚀 Training Strategy Tab {#strategy}

### Automatic Recommendations
This tab analyzes your dataset and provides:
- **Model Architecture Suggestions**: Based on object size distribution
- **Augmentation Strategies**: Based on spatial biases
- **Hyperparameter Recommendations**: Based on data characteristics
- **Loss Function Suggestions**: Based on class imbalance

### How to Use
1. Load your dataset
2. Review other analysis tabs
3. Click "Generate Training Strategy"
4. Review recommendations and adjust based on your specific needs

---

## 🔍 Signal Analysis Tab {#signal}

### Texture Analysis (Entropy vs Contrast)
- **What it shows**:
  - **Entropy**: Shannon entropy measures information content/texture complexity
  - **Contrast**: GLCM (Gray-Level Co-occurrence Matrix) contrast measures sharpness
- **Why it matters**:
  - **Left-Bottom region**: Featureless objects (low entropy + low contrast) are hard to learn
  - **High entropy + High contrast**: Rich texture, easier to detect
  - Helps identify which objects may need more training data or augmentation

### Camouflage Score (Fg/Bg Separability)
- **What it shows**: Chi-square distance between foreground and background histograms
- **Why it matters**:
  - **Low scores**: Objects blend with background (camouflaged) → Hard to detect
  - **High scores**: Objects stand out from background → Easier to detect
  - Identifies objects that may need special attention during training

### Image Frequency Spectrum (FFT)
- **What it shows**: Mean log magnitude of FFT spectrum
- **Why it matters**:
  - **Very low values**: Blurry or low-resolution images
  - **High values**: Sharp, high-frequency content
  - Helps identify image quality issues beyond simple blur detection

### Eigen-Objects (PCA Mean)
- **What it shows**: The "average" object appearance across your dataset
- **Why it matters**:
  - Reveals dominant visual patterns in your data
  - Helps understand what the "typical" object looks like
  - Can inform data augmentation strategies

### Use Cases
- Identify hard-to-detect objects (low separability, low texture)
- Find images with quality issues (low FFT energy)
- Understand dataset visual characteristics
- Plan targeted augmentation strategies

**Note**: This analysis is computationally intensive and may take a while for large datasets.

---

## 📸 Visual Explorer Tab {#viewer}

### Features
- **Image List**: Browse all images in dataset
- **BBox Visualization**: See annotations overlaid on images
- **Class Colors**: Different colors for different classes
- **Navigation**: Jump to specific images from other tabs

### Use Cases
- Verify annotation quality
- Inspect problematic images from Health Check
- Review hard samples from Difficulty Analysis
- Compare duplicate images

---

## 💡 General Tips

1. **Start with Health Check**: Clean your data first
2. **Check Class Balance**: Address imbalance before training
3. **Review Spatial Biases**: Plan augmentation accordingly
4. **Analyze Object Sizes**: Choose appropriate model architecture
5. **Inspect Hard Samples**: Focus improvement efforts
6. **Remove Duplicates**: Prevent data leakage

---

## 📚 References

- **COCO Metrics**: https://cocodataset.org/#detection-eval
- **Anchor Box Design**: YOLO, RetinaNet papers
- **Data Augmentation**: Mosaic, MixUp, Random Crop strategies
- **Class Imbalance**: Focal Loss paper

---

*For questions or issues, refer to the project documentation or create an issue on the repository.*
"""
