import bpy

# Ensure you're in Object Mode
if bpy.context.object.mode != 'OBJECT':
    bpy.ops.object.mode_set(mode='OBJECT')

# Use Cycles render engine (required for baking)
bpy.context.scene.render.engine = 'CYCLES'

# Get the active object
obj = bpy.context.active_object

# Ensure it’s a mesh
if obj and obj.type == 'MESH':
    mesh = obj.data

    # Add a vertex color layer if it doesn't exist
    vc_layer_name = "bake_col"
    if vc_layer_name not in mesh.color_attributes:
        mesh.color_attributes.new(name=vc_layer_name, domain='CORNER', type='BYTE_COLOR')

    # Set this as the active color attribute for baking
    mesh.color_attributes.active_color = mesh.color_attributes[vc_layer_name]

    # Set up bake settings
    bpy.context.scene.cycles.bake_type = 'DIFFUSE'
    bpy.context.scene.render.bake.use_pass_direct = False
    bpy.context.scene.render.bake.use_pass_indirect = False
    bpy.context.scene.render.bake.use_pass_color = True

    # Set to bake to vertex colors
    bpy.context.scene.render.bake.target = 'VERTEX_COLORS'

    # Bake it!
    bpy.ops.object.bake(type='DIFFUSE')

    print(f"✅ Diffuse color baked to vertex color layer: '{vc_layer_name}'")
else:
    print("⚠️ Please select a mesh object.")
