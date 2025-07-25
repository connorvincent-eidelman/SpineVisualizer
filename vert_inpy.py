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
vertebrae_counts = {'C': 7, 'T': 12, 'L': 5, 'S': 5, 'Co': 4}
region_spacing_cm = {'C': 1.71, 'T': 2.25, 'L': 3.4, 'S': 1.5, 'Co': 0.8}

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

# === Build Vertebrae Spine ===
vertebrae_coords = [c1_point]
vertebrae_labels = ['C1']
used_coords = {tuple(np.round(c1_point, 5))}
current_y = c1_point[1]
min_distance = 0.002

region_sequence = (
    ['C'] * (vertebrae_counts['C'] - 1) +
    ['T'] * vertebrae_counts['T'] +
    ['L'] * vertebrae_counts['L'] +
    ['S'] * vertebrae_counts['S'] +
    ['Co'] * vertebrae_counts['Co']
)

label_sequence = (
    [f'C{i+2}' for i in range(vertebrae_counts['C'] - 1)] +
    [f'T{i+1}' for i in range(vertebrae_counts['T'])] +
    [f'L{i+1}' for i in range(vertebrae_counts['L'])] +
    [f'S{i+1}' for i in range(vertebrae_counts['S'])] +
    [f'Co{i+1}' for i in range(vertebrae_counts['Co'])]
)

max_angle_deg = 25
max_angle_rad = np.radians(max_angle_deg)

for region, label in zip(region_sequence, label_sequence):
    y_step_model = region_spacing_cm[region] / scale_ratio
    current_y += y_step_model

    if region == 'Co':
        y_tol = y_tolerance * 3
        z_thresh = 0.0
        normal_thresh = 0.05
    else:
        y_tol = y_tolerance
        z_thresh = z_min
        normal_thresh = normal_z_thresh

    mask = (
        (np.abs(verts[:, 1] - current_y) < y_tol) &
        (verts[:, 0] >= x_min) & (verts[:, 0] <= x_max) &
        (verts[:, 2] >= z_thresh) &
        (normals[:, 2] > normal_thresh)
    )
    candidates = verts[mask]
    candidates = [c for c in candidates if all(np.linalg.norm(c - np.array(p)) >= min_distance for p in used_coords)]

    if len(candidates) == 0:
        fallback_found = False
        y_diffs = np.abs(verts[:, 1] - current_y)
        fallback_indices = np.argsort(y_diffs)

        for idx in fallback_indices:
            point = verts[idx]
            if (x_min <= point[0] <= x_max) and (point[2] >= z_thresh):
                if all(np.linalg.norm(point - np.array(p)) >= min_distance for p in used_coords):
                    candidates = [point]
                    fallback_found = True
                    print(f"Fallback used for {label}: closest available unique point selected.")
                    break

        if not fallback_found:
            print(f"Warning: fallback failed for {label}, inserting synthetic offset.")
            last_point = vertebrae_coords[-1]
            spine_point = last_point + np.array([0.0, y_step_model, -0.001])
            vertebrae_coords.append(spine_point)
            vertebrae_labels.append(label)
            used_coords.add(tuple(np.round(spine_point, 5)))
            continue

    spine_point = candidates[np.argmax([p[2] for p in candidates])]

    prev_point = vertebrae_coords[-1]
    direction = spine_point - prev_point
    norm = np.linalg.norm(direction)
    if norm == 0:
        print(f"Skipped {label}: zero-length direction vector.")
        continue

    unit_dir = direction / norm
    angle = np.arccos(np.clip(np.dot(unit_dir, [0, 1, 0]), -1.0, 1.0))

    if angle > max_angle_rad:
        rotate_ratio = np.tan(max_angle_rad)
        dz = spine_point[2] - prev_point[2]
        dx = spine_point[0] - prev_point[0]
        dy = spine_point[1] - prev_point[1]
        dz = np.sign(dz) * abs(dy) * rotate_ratio
        dx = np.clip(dx, x_min, x_max)
        spine_point = prev_point + np.array([dx, dy, dz])
        spine_point[2] = max(spine_point[2], z_min)
        spine_point[0] = np.clip(spine_point[0], x_min, x_max)

    vertebrae_coords.append(spine_point)
    vertebrae_labels.append(label)
    used_coords.add(tuple(np.round(spine_point, 5)))

print("\n=== Vertebrae Placement Log ===")
for label, coord in zip(vertebrae_labels, vertebrae_coords):
    print(f"{label}: x={coord[0]:.4f}, y={coord[1]:.4f}, z={coord[2]:.4f}")

# === Visualization ===
plotter = pv.Plotter()
plotter.add_mesh(pv_mesh, opacity=0.3, color='white', show_edges=False)

colors = {'C': 'blue', 'T': 'green', 'L': 'orange', 'S': 'purple', 'Co': 'gray'}
region_shapes = {
    'C': {'radius': 0.007, 'height': 0.005},
    'T': {'radius': 0.01,  'height': 0.006},
    'L': {'radius': 0.002, 'height': 0.007},
    'S': {'radius': 0.004,  'height': 0.004},
    'Co': {'radius': 0.001, 'height': 0.007}
}

for i, coord in enumerate(vertebrae_coords):
    label = vertebrae_labels[i]
    region = 'Co' if label.startswith("Co") else label[0]
    color = colors.get(region, 'red')
    shape = region_shapes.get(region, {'radius': 0.01, 'height': 0.005})

    cylinder = pv.Cylinder(center=coord, direction=(1, 0, 0),
                           radius=shape['radius'] * mesh_height,
                           height=shape['height'] * mesh_height)
    plotter.add_mesh(cylinder, color=color)
    plotter.add_point_labels([coord], [label], font_size=30, text_color=color)

# Add line through vertebrae
for i in range(len(vertebrae_coords) - 1):
    p1, p2 = vertebrae_coords[i], vertebrae_coords[i + 1]
    label = vertebrae_labels[i]
    region = 'Co' if label.startswith("Co") else label[0]
    color = colors.get(region, 'red')
    line = pv.Line(p1, p2)
    plotter.add_mesh(line, color=color, line_width=4)

plotter.add_axes()
plotter.show_bounds(grid='front', location='outer')
plotter.show(title="Full Vertebrae Spine with Angle Constraints", window_size=[1200, 900])
