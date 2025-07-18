import pyvista as pv
import numpy as np
import trimesh

# === Load and Orient Mesh ===
tm_mesh = trimesh.load("/Users/connorv-e/Desktop/spinevis/modeling/BODY.obj", process=True)

# Optional: Apply transform if model is rotated incorrectly
tm_mesh.apply_transform(trimesh.transformations.rotation_matrix(
    np.radians(-90), [1, 0, 0]
))

verts = tm_mesh.vertices
faces = tm_mesh.faces

# === Convert to PyVista Format ===
faces_pv = np.column_stack((np.full(len(faces), 3), faces)).ravel()
pv_mesh = pv.PolyData(verts, faces_pv)

# === Constants ===
body_height_cm = 185.42
spine_length_cm = 72
vertebrae_counts = {'C': 7, 'T': 12, 'L': 5}
region_ratios = {'C': 0.25, 'T': 0.45, 'L': 0.30}
vertebrae_labels = (
    [f'C{i+1}' for i in range(7)] +
    [f'T{i+1}' for i in range(12)] +
    [f'L{i+1}' for i in range(5)]
)

# === Surface-Based Vertebrae Placement with Constraints ===
verts = pv_mesh.points
top_index = np.argmax(verts[:, 1])  # highest Y = skull top
skull_top = verts[top_index]
bottom_y = np.min(verts[:, 1])
mesh_height = skull_top[1] - bottom_y
scale_ratio = body_height_cm / mesh_height

# Vertebra placement: sample down Y axis
num_vertebrae = sum(vertebrae_counts.values())
y_scan_values = np.linspace(skull_top[1], bottom_y, num_vertebrae)

# Define region of interest (ROI)
x_min, x_max = -0.2, 0.1
y_max = -1.05  # only use points below this Y

vertebrae_coords = []
for y in y_scan_values:
    # ROI-constrained slice
    mask = (
        (np.abs(verts[:, 1] - y) < 0.5) &      # thin horizontal slice
        (verts[:, 0] >= x_min) & (verts[:, 0] <= x_max) &  # narrow X band
        (verts[:, 1] <= y_max)                 # lower Y only
    )
    slice_verts = verts[mask]

    if len(slice_verts) == 0:
        continue

    # Select the backmost point (max Z)
    back_idx = np.argmax(slice_verts[:, 2])
    spine_point = slice_verts[back_idx]
    vertebrae_coords.append(spine_point)

# Trim to label count
vertebrae_coords = np.array(vertebrae_coords[:len(vertebrae_labels)])

# === Visualization ===
plotter = pv.Plotter()
plotter.add_mesh(pv_mesh, opacity=0.3, color='white', show_edges=False)

sphere_radius = 0.01 * mesh_height
for i, coord in enumerate(vertebrae_coords):
    plotter.add_mesh(pv.Sphere(radius=sphere_radius, center=coord), color='red')
    plotter.add_point_labels([coord], [vertebrae_labels[i]], font_size=10, text_color='blue')

plotter.add_axes()
plotter.show_bounds(grid='front', location='outer')
plotter.show(title="ROI-Constrained Vertebrae Overlay", window_size=[1200, 900])
