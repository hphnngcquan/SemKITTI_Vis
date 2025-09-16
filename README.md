# Visualization Tools for SemanticKITTI Dataset Format
-   Visualize in Spherical Meshes
-   Box Compose
-   Visualize with Indices
-   Save Graphics

## Installation
```
conda create --n vistool python=3.9.20
pip install -r requirements.txt
```

## Data Preparation

```text
semantickitti
├── sequences
│   ├── 00
│   |   ├── labels
│   |   ├── velodyne
│   ├── 01
│   ├── ...
│   ├── 21
```

## Color Palette Prepare
Generate color palette to support 4D Panoptic Visualization
```
python tools/color_gen.py
```
