import bpy
import mathutils
import math

vertebra_names = [
    "C1", "C2", "C3", "C4", "C5", "C6", "C7",
    "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11", "T12",
    "L1", "L2", "L3", "L4", "L5"
]

vertebrae = []

def add_marker(location, name):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.015, location=location)
    marker = bpy.context.active_object
    marker.name = name
    marker.show_name = True
    marker.display_type = 'SOLID'
    return marker

def fit_curve_through_points(points):
    curve_data = bpy.data.curves.new('SpineCurve', type='CURVE')
    curve_data.dimensions = '3D'
    polyline = curve_data.splines.new('NURBS')
    polyline.points.add(len(points) - 1)

    for i, pt in enumerate(points):
        polyline.points[i].co = (pt.x, pt.y, pt.z, 1)
    
    polyline.order_u = min(4, len(points))
    polyline.use_endpoint_u = True

    curve_obj = bpy.data.objects.new('SpineCurveObj', curve_data)
    bpy.context.collection.objects.link(curve_obj)
    return curve_obj

def compute_angles(points):
    angles = []
    for i in range(1, len(points) - 1):
        a = (points[i - 1] - points[i]).normalized()
        b = (points[i + 1] - points[i]).normalized()
        angle_rad = a.angle(b)
        angle_deg = math.degrees(angle_rad)
        angles.append((i, angle_deg))
    return angles

def display_angles(points, angles):
    for idx, ang in angles:
        loc = points[idx]
        bpy.ops.object.text_add(location=loc + mathutils.Vector((0.02, 0, 0)))
        txt = bpy.context.active_object
        txt.data.body = f"{round(ang, 1)}°"
        txt.data.size = 0.02
        txt.name = f"angle_{idx}"

# Blender UI Panel
class SPINE_PT_Tool(bpy.types.Panel):
    bl_label = "Spine Tracker"
    bl_idname = "SPINE_PT_tool"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Spine Tracker'

    def draw(self, context):
        layout = self.layout
        layout.operator("spine.add_marker", text="Add Vertebra Marker")
        layout.operator("spine.build_spine", text="Analyze Spine")
        layout.operator("spine.reset", text="Reset")

class SPINE_OT_AddMarker(bpy.types.Operator):
    bl_idname = "spine.add_marker"
    bl_label = "Add Vertebra Marker"

    def execute(self, context):
        idx = len(vertebrae)
        if idx >= len(vertebra_names):
            self.report({'WARNING'}, "Max vertebrae reached.")
            return {'CANCELLED'}
        loc = context.scene.cursor.location.copy()
        marker = add_marker(loc, vertebra_names[idx])
        vertebrae.append(loc)
        return {'FINISHED'}

class SPINE_OT_BuildSpine(bpy.types.Operator):
    bl_idname = "spine.build_spine"
    bl_label = "Build Spine Curve"

    def execute(self, context):
        if len(vertebrae) < 3:
            self.report({'WARNING'}, "Add at least 3 markers.")
            return {'CANCELLED'}
        curve = fit_curve_through_points(vertebrae)
        angles = compute_angles(vertebrae)
        display_angles(vertebrae, angles)
        return {'FINISHED'}

class SPINE_OT_Reset(bpy.types.Operator):
    bl_idname = "spine.reset"
    bl_label = "Reset Markers"

    def execute(self, context):
        global vertebrae
        vertebrae.clear()

        for obj in bpy.data.objects:
            if obj.name.startswith("C") or obj.name.startswith("T") or obj.name.startswith("L"):
                bpy.data.objects.remove(obj, do_unlink=True)
            if obj.name.startswith("angle_"):
                bpy.data.objects.remove(obj, do_unlink=True)
            if "SpineCurveObj" in obj.name:
                bpy.data.objects.remove(obj, do_unlink=True)
        return {'FINISHED'}

# Registration
classes = [
    SPINE_PT_Tool,
    SPINE_OT_AddMarker,
    SPINE_OT_BuildSpine,
    SPINE_OT_Reset,
]

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)

if __name__ == "__main__":
    register()
