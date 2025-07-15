import bpy

def extract_spine(obj_name, x_threshold=0.05, z_resolution=0.02):
    obj = bpy.data.objects[obj_name]
    mesh = obj.data
    spine_points = []

    for v in mesh.vertices:
        co = obj.matrix_world @ v.co
        if abs(co.x) < x_threshold:  # near midline
            spine_points.append((round(co.z / z_resolution), co.y, co))

    # Select lowest Y (deepest back) point per Z layer
    spine_dict = {}
    for z_bin, y, co in spine_points:
        if z_bin not in spine_dict or y < spine_dict[z_bin][1]:
            spine_dict[z_bin] = (co, y)

    sorted_spine = [v[0] for z, v in sorted(spine_dict.items())]
    return sorted_spine

# Save to global variable for use in other scripts
spine_curve = extract_spine("YourMeshName")
print(f"Extracted {len(spine_curve)} spine points.")