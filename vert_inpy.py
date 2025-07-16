import trimesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# === Load Mesh ===
mesh = trimesh.load("cleaned_scan.obj", process=True)

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

# === Mesh Analysis ===
verts = mesh.vertices
top_y = np.max(verts[:, 1])       # Head top
bottom_y = np.min(verts[:, 1])    # Foot bottom
mesh_height = top_y - bottom_y
scale_ratio = body_height_cm / mesh_height
spine_start_y = top_y - 5.0 * (mesh_height / body_height_cm)  # Estimate C1 5cm below head

# === Vertebrae Placement ===
vertebrae_y = []
current_y = spine_start_y

for region, count in vertebrae_counts.items():
    segment_height = region_ratios[region] * (spine_length_cm / scale_ratio)
    spacing = segment_height / count
    for i in range(count):
        vertebrae_y.append(current_y - i * spacing)
    current_y -= segment_height

# Truncate in case of float rounding
vertebrae_y = vertebrae_y[:len(vertebrae_labels)]

# === Center Line (XZ) ===
center_x = np.median(verts[:, 0])
center_z = np.median(verts[:, 2])

vertebrae_coords = np.array([[center_x, y, center_z] for y in vertebrae_y])

# === Visualization ===
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Mesh overlay
ax.scatter(verts[::10, 0], verts[::10, 1], verts[::10, 2], alpha=0.02, label="Mesh")

# Vertebrae markers
for i, coord in enumerate(vertebrae_coords):
    ax.scatter(*coord, color='red', s=20)
    ax.text(*coord, vertebrae_labels[i], fontsize=8, color='blue')

ax.set_title("Vertebrae Marker Estimation")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.view_init(elev=160, azim=-90)  # Side view

plt.show()
