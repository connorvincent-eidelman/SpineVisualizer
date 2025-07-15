import bpy

def scale_to_real_height(obj_name, real_height_meters):
    obj = bpy.data.objects[obj_name]
    bbox = [obj.matrix_world @ v.co for v in obj.data.vertices]
    z_coords = [v.z for v in bbox]
    mesh_height = max(z_coords) - min(z_coords)

    scale_factor = real_height_meters / mesh_height
    obj.scale = (scale_factor, scale_factor, scale_factor)

# Change these values:
OBJECT_NAME = "baked_mesh"  # Name of your mesh object
REAL_HEIGHT = 1.8542  # in meters

scale_to_real_height(OBJECT_NAME, REAL_HEIGHT)
print("Model scaled.")
