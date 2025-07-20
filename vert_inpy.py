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

# === Constants (cm) ===
body_height_cm = 185.42
spine_length_cm = body_height_cm * 0.385
skull_height_cm = body_height_cm*0.13     # ~9 inches
c1_offset_cm = skull_height_cm / 1.5

vertebrae_counts = {'C': 7, 'T': 12, 'L': 5}
region_ratios = {'C': 0.25, 'T': 0.45, 'L': 0.30}
vertebrae_labels = (
    [f'C{i+1}' for i in range(7)] +
    [f'T{i+1}' for i in range(12)] +
    [f'L{i+1}' for i in range(5)]
)

# === Extract Vertex Data ===
verts = pv_mesh.points
y_vals = verts[:, 1]

# === Identify Skull Top (lowest Y since flipped) ===
top_index = np.argmin(y_vals)
skull_top = verts[top_index]
top_y = skull_top[1]

# === Scale C1 Offset into Model Units ===
mesh_height = np.max(y_vals) - np.min(y_vals)
scale_ratio = body_height_cm / mesh_height
c1_offset_model = c1_offset_cm / scale_ratio
c1_y = top_y + c1_offset_model  # move up in Y

# === Find C1 Point on Back of Skull ===
x_min, x_max = -0.2, 0.1
y_tolerance = 0.5


# === C1 Point is Directly Below Skull Top ===
c1_y = top_y + c1_offset_model  # Y increases downward
c1_point = np.array([skull_top[0], c1_y, skull_top[2]])

# === Build Vertebrae Spine (Starting from C1) ===
num_remaining = sum(vertebrae_counts.values()) - 1
y_scan_values = np.linspace(c1_point[1] + 0.5, np.max(y_vals), num_remaining)

vertebrae_coords = [c1_point]
for y in y_scan_values:
    mask = (
        (np.abs(verts[:, 1] - y) < 0.5) &
        (verts[:, 0] >= x_min) & (verts[:, 0] <= x_max)
    )
    slice_verts = verts[mask]
    if len(slice_verts) == 0:
        continue
    back_idx = np.argmax(slice_verts[:, 2])
    spine_point = slice_verts[back_idx]
    vertebrae_coords.append(spine_point)

# === Trim to Vertebrae Count ===
vertebrae_coords = np.array(vertebrae_coords[:len(vertebrae_labels)])

# === Visualization ===
plotter = pv.Plotter()
plotter.add_mesh(pv_mesh, opacity=0.3, color='white', show_edges=False)

sphere_radius = 0.01 * mesh_height
for i, coord in enumerate(vertebrae_coords):
    color = 'blue' if i == 0 else 'red'  # C1 is blue
    plotter.add_mesh(pv.Sphere(radius=sphere_radius, center=coord), color=color)
    plotter.add_point_labels([coord], [vertebrae_labels[i]], font_size=10, text_color='blue')

plotter.add_axes()
plotter.show_bounds(grid='front', location='outer')
plotter.show(title="Scientific Vertebrae Placement (CM)", window_size=[1200, 900])
