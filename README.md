# SpineVisualizer: 3D Spine Modeling & Analysis Toolkit

SpineVisualizer is a toolkit for extracting, labeling, analyzing, and visualizing human spine geometry from 3D body scans and multi-camera video. It supports Blender scripting for mesh processing, PyVista for 3D visualization, and OpenCV/Mediapipe for pose estimation.

---

## Features

- **Blender Mesh Processing**
  - Extract spine curve from 3D mesh (`extract_spine.py`)
  - Label vertebrae points (`label_vertebrae.py`)
  - Scale mesh to real-world height (`scale_model.py`)
  - Bake vertex colors (`modeling/bake.py`)

- **Spine Geometry Analysis**
  - Measure curvature and angles (`measure_curvature.py`, `vert_inpy.py`)
  - Visualize spline and deviations (`vert_inpy.py`, `stage2/render_spine_model.py`)

- **Multi-Camera Video Processing**
  - Calibrate cameras (`stage2/calibration_utils.py`)
  - Triangulate 3D landmarks (`stage2/triangulation_utils.py`)
  - Compute metrics: shoulder width, spine angle, lateral deviation (`stage2/main.py`)

- **Depth Estimation & Pose**
  - Integrate MiDaS depth model (`old/viswithdepthtest.py`)
  - Use Mediapipe for pose detection (`old/vissy.py`, `stage2/main.py`)

---

## Workflow

1. **Mesh Preparation (Blender)**
   - Import your body mesh (e.g., `BODY.obj`)
   - Run `scale_model.py` to scale to real height
   - Run `extract_spine.py` to extract spine curve
   - Run `label_vertebrae.py` to label vertebrae
   - Optionally bake vertex colors (`modeling/bake.py`)

2. **Spine Analysis (Python)**
   - Use `vert_inpy.py` for PyVista visualization and curvature analysis
   - Use `measure_curvature.py` for angle measurement

3. **Multi-Camera Capture**
   - Calibrate cameras (`stage2/calibration_utils.py`)
   - Run `stage2/main.py` for live metric extraction and visualization

4. **Depth & Pose (Optional)**
   - Run `old/viswithdepthtest.py` for depth + pose overlay

---

## Requirements

- **Blender** (for mesh scripts)
- **Python 3.8+**
- [PyVista](https://docs.pyvista.org/)
- [trimesh](https://trimsh.org/)
- [OpenCV](https://opencv.org/)
- [Mediapipe](https://google.github.io/mediapipe/)
- [torch](https://pytorch.org/) (for MiDaS depth)
- [open3d](http://www.open3d.org/) (for advanced 3D visualization)

---

## File Overview

- `extract_spine.py` – Extracts spine curve from mesh in Blender
- `label_vertebrae.py` – Labels vertebrae points in Blender
- `scale_model.py` – Scales mesh to real-world height in Blender
- `measure_curvature.py` – Measures and visualizes spine angles in Blender
- `vert_inpy.py` – PyVista-based 3D visualization and curvature analysis
- `stage2/main.py` – Multi-camera metric extraction and visualization
- `stage2/render_spine_model.py` – Advanced 3D rendering with Open3D
- `modeling/bake.py` – Vertex color baking in Blender

---

## Example Usage

**Extract and Label Spine in Blender:**
```sh
blender --python extract_spine.py
blender --python label_vertebrae.py
```

**Scale Mesh:**
```sh
blender --python scale_model.py
```

**Analyze Spine Curvature:**
```sh
python vert_inpy.py
```

**Run Multi-Camera Metric Extraction:**
```sh
python stage2/main.py
```

---

## License

MIT License. See LICENSE for details.

---

## Authors

Connor V-
