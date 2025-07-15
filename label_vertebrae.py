import bpy

def label_vertebrae(spine_points, label_list=None):
    if label_list is None:
        label_list = ['C7'] + [f'T{i}' for i in range(1,13)] + [f'L{i}' for i in range(1,6)]

    step = len(spine_points) // len(label_list)
    used = []

    for i, label in enumerate(label_list):
        idx = min(i * step, len(spine_points)-1)
        co = spine_points[idx]
        used.append(co)

        # Add sphere
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.01, location=co)
        sphere = bpy.context.object
        sphere.name = f"VB_{label}"

        # Add label text
        bpy.ops.object.text_add(location=(co.x + 0.02, co.y, co.z))
        text = bpy.context.object
        text.data.body = label
        text.name = f"Label_{label}"

    return used

# You must have run extract_spine.py first to define spine_curve
try:
    labeled_points = label_vertebrae(spine_curve)
except NameError:
    print("Run extract_spine.py first to define spine_curve")
