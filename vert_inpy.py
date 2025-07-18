import pyvista as pv
import numpy as np
import trimesh

# === Load Mesh ===
tm_mesh = trimesh.load("/Users/connorv-e/Desktop/spinevis/modeling/BODY.obj", process=True)
verts = tm_mesh.vertices
faces = tm_mesh.faces

# PyVista requires a "flattened" array: [3, i0, i1, i2, 3, j0, j1, j2, ...]
faces_pv = np.hstack(
    [np.insert(face, 0, 3) for face in faces]
)

# Create the PyVista mesh
pv_mesh = pv.PolyData(verts, faces_pv)

# Visualize
plotter = pv.Plotter()
plotter.add_mesh(pv_mesh, show_edges=True, color="white", opacity=0.6)
plotter.show()

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

# === Estimate Spine Positions ===
top_y = np.max(verts[:, 1])
bottom_y = np.min(verts[:, 1])
mesh_height = top_y - bottom_y
scale_ratio = body_height_cm / mesh_height
spine_start_y = top_y - 5.0 * (mesh_height / body_height_cm)

vertebrae_y = []
current_y = spine_start_y

for region, count in vertebrae_counts.items():
    segment_height = region_ratios[region] * (spine_length_cm / scale_ratio)
    spacing = segment_height / count
    for i in range(count):
        vertebrae_y.append(current_y - i * spacing)
    current_y -= segment_height

vertebrae_y = vertebrae_y[:len(vertebrae_labels)]
center_x = np.median(verts[:, 0])
center_z = np.median(verts[:, 2])
vertebrae_coords = np.array([[center_x, y, center_z] for y in vertebrae_y])

# === Build Scene ===
plotter = pv.Plotter()
plotter.add_mesh(pv_mesh, opacity=0.3, color='white', show_edges=False)

for i, coord in enumerate(vertebrae_coords):
    plotter.add_mesh(pv.Sphere(radius=0.01, center=coord), color='red')
    plotter.add_point_labels([coord], [vertebrae_labels[i]], font_size=10, text_color='blue')

plotter.add_axes()
plotter.show_bounds(grid='front', location='outer')
plotter.show(title="Vertebrae Marker Overlay", window_size=[1200, 900])
