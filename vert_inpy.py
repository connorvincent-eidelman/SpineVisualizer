import pyvista as pv
import numpy as np
import trimesh

# === Load and Orient Mesh ===
tm_mesh = trimesh.load("/Users/connorv-e/Desktop/spinevis/modeling/BODY.obj")
tm_mesh.apply_transform(trimesh.transformations.rotation_matrix(
    np.radians(-90), [1, 0, 0]
))

verts = tm_mesh.vertices
faces = tm_mesh.faces

# === Convert to PyVista Format ===
faces_pv = np.column_stack((np.full(len(faces), 3), faces)).ravel()
pv_mesh = pv.PolyData(verts, faces_pv)

# === Compute Surface Normals ===
pv_mesh.compute_normals(cell_normals=False, point_normals=True, inplace=True)
normals = pv_mesh.point_normals

# === Constants (in cm) ===
body_height_cm = 185.42
skull_height_cm = body_height_cm * 0.13
c1_offset_cm = skull_height_cm / 1.5

# Vertebrae counts per region
vertebrae_counts = {'C': 7, 'T': 12, 'L': 5}
vertebrae_labels = (
    [f'C{i+1}' for i in range(7)] +
    [f'T{i+1}' for i in range(12)] +
    [f'L{i+1}' for i in range(5)]
)

# Average distances between vertebrae (in cm)
region_spacing_cm = {'C': 1.71, 'T': 2.25, 'L': 3.4}

# === Get Vertex Data ===
verts = pv_mesh.points
y_vals = verts[:, 1]
mesh_height = np.max(y_vals) - np.min(y_vals)
scale_ratio = body_height_cm / mesh_height

# === Skull Top is min Y (flipped model) ===
top_index = np.argmin(y_vals)
skull_top = verts[top_index]
top_y = skull_top[1]

# === Scale C1 offset into model units ===
c1_offset_model = c1_offset_cm / scale_ratio
c1_y = top_y + c1_offset_model

# === C1 Placement ===
x_min, x_max = -0.15, 0
y_tolerance = 0.01
z_min = 0.1
normal_z_thresh = 0.3

c1_mask = (
    (np.abs(verts[:, 1] - c1_y) < y_tolerance) &
    (verts[:, 0] >= x_min) & (verts[:, 0] <= x_max) &
    (verts[:, 2] >= z_min) &
    (normals[:, 2] > normal_z_thresh)
)
c1_candidates = verts[c1_mask]
if len(c1_candidates) == 0:
    raise ValueError("No valid C1 candidates found.")

c1_index = np.argmax(c1_candidates[:, 2])
c1_point = c1_candidates[c1_index]

# === Build Vertebrae Spine with Region-Specific Spacing ===
vertebrae_coords = [c1_point]
current_y = c1_point[1]

region_sequence = (
    ['C'] * (vertebrae_counts['C'] - 1) +
    ['T'] * vertebrae_counts['T'] +
    ['L'] * vertebrae_counts['L']
)

for region in region_sequence:
    y_step_model = region_spacing_cm[region] / scale_ratio
    current_y += y_step_model

    mask = (
        (np.abs(verts[:, 1] - current_y) < y_tolerance) &
        (verts[:, 0] >= x_min) & (verts[:, 0] <= x_max) &
        (verts[:, 2] >= z_min) &
        (normals[:, 2] > normal_z_thresh)
    )
    candidates = verts[mask]
    if len(candidates) == 0:
        continue

    z_max_idx = np.argmax(candidates[:, 2])
    spine_point = candidates[z_max_idx]
    vertebrae_coords.append(spine_point)

# === Trim to 24 Vertebrae ===
vertebrae_coords = np.array(vertebrae_coords[:len(vertebrae_labels)])

# === Visualization ===
plotter = pv.Plotter()
plotter.add_mesh(pv_mesh, opacity=0.3, color='white', show_edges=False)

sphere_radius = 0.005 * mesh_height
for i, coord in enumerate(vertebrae_coords):
    color = 'blue' if i == 0 else 'red'
    plotter.add_mesh(pv.Sphere(radius=sphere_radius, center=coord), color=color)
    plotter.add_point_labels([coord], [vertebrae_labels[i]], font_size=15, text_color='blue')

plotter.add_axes()
plotter.show_bounds(grid='front', location='outer')
plotter.show(title="Realistic Vertebrae Spacing by Region", window_size=[1200, 900])
