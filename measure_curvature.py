import math
import bpy
from mathutils import Vector

def vector_angle(v1, v2):
    return math.degrees(v1.angle(v2))

def measure_spine_angles(points):
    angles = []
    for i in range(1, len(points) - 1):
        v1 = (points[i] - points[i-1]).normalized()
        v2 = (points[i+1] - points[i]).normalized()
        angle = vector_angle(v1, v2)
        angles.append((i, angle))
    return angles

# Visualize angle bends along spine
def draw_spine_angles(points):
    angles = measure_spine_angles(points)
    for idx, angle in angles:
        co = points[idx]
        bpy.ops.object.text_add(location=(co.x - 0.02, co.y, co.z))
        text = bpy.context.object
        text.data.body = f"{angle:.1f}°"
        text.name = f"Angle_{idx}"

# Must run extract_spine.py and label_vertebrae.py first
try:
    draw_spine_angles(spine_curve)
except NameError:
    print("Run extract_spine.py first to define spine_curve")
