import bpy
import os

# Clear all objects
try:
    bpy.ops.object.mode_set(mode='OBJECT')
except:
    pass
for obj in bpy.context.scene.objects:
    obj.select_set(True)
bpy.ops.object.delete()

# Create basic ico_sphere with 7 subdivisions
bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=7, location=(0.5,0.5,0.5), radius=0.5)
bpy.ops.object.mode_set(mode='SCULPT')
bpy.data.objects[0].select_set(True)
bpy.context.scene.tool_settings.sculpt.use_symmetry_x = False

filename = os.path.join("_PATH_", "_FILE_NAME_.py")
exec(compile(open(filename).read(), filename, 'exec'))