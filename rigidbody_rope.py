import bpy
import os
import json
import time
import sys
import bpy, bpy_extras
from math import *
from mathutils import *
import random
import numpy as np
import random
from random import sample
import bmesh

'''Usage: blender -P rigidbody-rope.py'''

def get_calibration_matrix_K_from_blender(camd):
    f_in_mm = camd.lens
    scene = bpy.context.scene
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100
    sensor_width_in_mm = camd.sensor_width
    sensor_height_in_mm = camd.sensor_height
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    if (camd.sensor_fit == 'VERTICAL'):
        # the sensor height is fixed (sensor fit is horizontal),
        # the sensor width is effectively changed with the pixel aspect ratio
        s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
    else: # 'HORIZONTAL' and 'AUTO'
        # the sensor width is fixed (sensor fit is horizontal),
        # the sensor height is effectively changed with the pixel aspect ratio
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm

    # Parameters of intrinsic calibration matrix K
    alpha_u = f_in_mm * s_u
    alpha_v = f_in_mm * s_v
    u_0 = resolution_x_in_px*scale / 2
    v_0 = resolution_y_in_px*scale / 2
    skew = 0 # only use rectangular pixels

    K = Matrix(
        ((alpha_u, skew,    u_0),
        (    0  ,  alpha_v, v_0),
        (    0  ,    0,      1 )))
    return K

def compute_world_to_camera_matrix(camera):
    if camera.type != 'CAMERA':
        raise Exception("Object {} is not a camera.".format(camera.name))
    # Get the two components to calculate M
    render = bpy.context.scene.render
    modelview_matrix = camera.matrix_world.inverted()
    projection_matrix = camera.calc_matrix_camera(
        bpy.data.scenes["Scene"].view_layers["View Layer"].depsgraph,
        x = render.resolution_x,
        y = render.resolution_y,
        scale_x = render.pixel_aspect_x,
        scale_y = render.pixel_aspect_y,
    )
    # print(projection_matrix * modelview_matrix)
    # Compute P??? = M * P
    transformation_matrix = projection_matrix @ modelview_matrix
    return transformation_matrix

def add_camera_light():
    bpy.ops.object.light_add(type='SUN', radius=1, location=(0,0,0))
    bpy.ops.object.camera_add(location=(2,0,28), rotation=(0,0,0))
    bpy.context.scene.camera = bpy.context.object
    camera = bpy.context.scene.camera
    # modelview_matrix = camera.matrix_world.inverted()
    # render = bpy.context.scene.render
    # depsgraph = bpy.context.evaluated_depsgraph_get()
    # camera_matrix = camera.calc_matrix_camera(depsgraph,
    #         x=render.resolution_x,
    #         y=render.resolution_y,
    #         scale_x=render.pixel_aspect_x,
    #         scale_y=render.pixel_aspect_y,
    #         )
    # camera_matrix = get_calibration_matrix_K_from_blender(camera.data)
    camera_matrix = compute_world_to_camera_matrix(camera)
    np.save('proj_matrix.npy', np.array(camera_matrix))

def clear_scene():
    '''Clear existing objects in scene'''
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)
    for block in bpy.data.textures:
        if block.users == 0:
            bpy.data.textures.remove(block)
    for block in bpy.data.images:
        if block.users == 0:
            bpy.data.images.remove(block)
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def make_rope(params):
    segment_radius = params["segment_radius"]
    num_segments = params["num_segments"]
    bpy.ops.mesh.primitive_cylinder_add(location=(segment_radius*num_segments,0,0))
    bpy.ops.transform.resize(value=(segment_radius, segment_radius, segment_radius))
    bpy.ops.object.editmode_toggle()
    bpy.ops.mesh.subdivide(number_cuts=5, quadcorner='INNERVERT')
    # bpy.ops.mesh.subdivide(number_cuts=1, quadcorner='INNERVERT')
    bpy.ops.object.editmode_toggle()
    cylinder = bpy.context.object
    cylinder.name = "Cylinder"
    cylinder.rotation_euler = (0, np.pi/2, 0)
    bpy.ops.rigidbody.object_add()
    cylinder.rigid_body.mass = params["segment_mass"]
    cylinder.rigid_body.friction = params["segment_friction"]
    cylinder.rigid_body.linear_damping = params["linear_damping"]
    cylinder.rigid_body.angular_damping = params["angular_damping"] # NOTE: this makes the rope a lot less wiggly
    bpy.context.scene.rigidbody_world.steps_per_second = 120
    bpy.context.scene.rigidbody_world.solver_iterations = 20
    for i in range(num_segments-1):
        bpy.ops.object.duplicate_move(TRANSFORM_OT_translate={"value":(-2*segment_radius, 0, 0)})
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.rigidbody.connect(con_type='POINT', connection_pattern='CHAIN_DISTANCE')
    bpy.ops.object.select_all(action='DESELECT')

    return [bpy.data.objects['Cylinder.%03d' % (i) if i>0 else "Cylinder"] for i in range(num_segments)]

def make_capsule_rope(params):
    radius = params["segment_radius"]
    #rope_length = radius * params["num_segments"] * 2 * 0.9 # HACKY -- shortening the rope artificially by 10% for now
    rope_length = radius * params["num_segments"]
    num_segments = int(rope_length / radius)
    separation = radius*1.1 # HACKY - artificially increase the separation to avoid link-to-link collision
    link_mass = params["segment_mass"] # TODO: this may need to be scaled
    link_friction = params["segment_friction"]
    twist_stiffness = 20
    twist_damping = 10
    bend_stiffness = 0
    bend_damping = 5
    num_joints = int(radius/separation)*2+1
    bpy.ops.import_mesh.stl(filepath="capsule.stl")
    loc0 = (radius*num_segments,0,0)
    link0 = bpy.context.object
    link0.location = loc0
    loc0 = loc0[0]
    link0.name = "Cylinder"
    bpy.ops.transform.resize(value=(radius, radius, radius))
    link0.rotation_euler = (0, pi/2, 0)
    bpy.ops.rigidbody.object_add()
    link0.rigid_body.mass = link_mass
    link0.rigid_body.friction = link_friction
    link0.rigid_body.linear_damping = params["linear_damping"]
    link0.rigid_body.angular_damping = params["angular_damping"] # NOTE: this makes the rope a lot less wiggly
    #link0.rigid_body.collision_shape = 'CAPSULE'
    bpy.context.scene.rigidbody_world.steps_per_second = 120
    bpy.context.scene.rigidbody_world.solver_iterations = 20
    for i in range(num_segments-1):
        bpy.ops.object.duplicate_move(TRANSFORM_OT_translate={"value":(-2*radius, 0, 0)})
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.rigidbody.connect(con_type='POINT', connection_pattern='CHAIN_DISTANCE')
    bpy.ops.object.select_all(action='DESELECT')
    links = [bpy.data.objects['Cylinder.%03d' % (i) if i>0 else "Cylinder"] for i in range(num_segments)]
    return links

def createNewBone(obj, new_bone_name, head, tail):
    bpy.ops.object.editmode_toggle()
    bpy.ops.armature.bone_primitive_add(name=new_bone_name)
    new_edit_bone = obj.data.edit_bones[new_bone_name]
    new_edit_bone.head = head
    new_edit_bone.tail = tail
    bpy.ops.object.editmode_toggle()
    bone = obj.pose.bones[-1]
    constraint = bone.constraints.new('COPY_TRANSFORMS')
    target_obj_name = "Cylinder" if new_bone_name == "Bone.000" else new_bone_name.replace("Bone", "Cylinder")
    constraint.target = bpy.data.objects[target_obj_name]

def make_braid_rig(params, bezier):
    n = params["num_segments"]
    radius = params["segment_radius"]
    bpy.ops.mesh.primitive_circle_add(location=(0,0,0))
    radius = 0.125
    bpy.ops.transform.resize(value=(radius, radius, radius))
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.transform.rotate(value= pi / 2, orient_axis='X')
    bpy.ops.transform.translate(value=(radius, 0, 0))
    bpy.ops.object.mode_set(mode='OBJECT')
    num_chords = 4
    # num_chords = 2
    for i in range(1, num_chords):
        bpy.ops.object.duplicate_move(OBJECT_OT_duplicate=None, TRANSFORM_OT_translate=None)
        ob = bpy.context.active_object
        ob.rotation_euler = (0, 0, i * (2*pi / num_chords))
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.join()
    bpy.ops.object.modifier_add(type='SCREW')
    rope = bpy.context.object
    rope.rotation_euler = (0,pi/2,0)
    rope.modifiers["Screw"].screw_offset = 12 # Arbitrary
    rope.modifiers["Screw"].iterations = 15 # Arbitrary
    bpy.ops.object.modifier_add(type='CURVE')
    rope.modifiers["Curve"].object = bezier
    rope.modifiers["Curve"].show_in_editmode = True
    rope.modifiers["Curve"].show_on_cage = True
    return rope

def make_cable_rig(params, bezier):
    bpy.ops.object.modifier_add(type='CURVE')
    bpy.ops.curve.primitive_bezier_circle_add(radius=0.02)
    #bpy.ops.curve.primitive_bezier_circle_add(radius=0.018)
    bezier.data.bevel_object = bpy.data.objects["BezierCircle"]
    bpy.context.view_layer.objects.active = bezier
    return bezier

def rig_rope(params, braid=0):
    bpy.ops.object.armature_add(enter_editmode=False, location=(0, 0, 0))
    arm = bpy.context.object
    n = params["num_segments"]
    radius = params["segment_radius"]
    for i in range(n):
        loc = 2*radius*((n-i) - n//2)
        createNewBone(arm, "Bone.%03d"%i, (loc,0,0), (loc,0,1))
    bpy.ops.curve.primitive_bezier_curve_add(location=(radius,0,0))
    bezier_scale = n*radius
    bpy.ops.transform.resize(value=(bezier_scale, bezier_scale, bezier_scale))
    bezier = bpy.context.active_object
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.curve.select_all(action='SELECT')
    bpy.ops.curve.handle_type_set(type='VECTOR')
    bpy.ops.curve.handle_type_set(type='AUTOMATIC')
    # NOTE: it segfaults for num_control_points > 20 for the braided rope!!
    num_control_points = 40 # Tune this
    bpy.ops.curve.subdivide(number_cuts=num_control_points-2)
    bpy.ops.object.mode_set(mode='OBJECT')
    bezier_points = bezier.data.splines[0].bezier_points
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.curve.select_all(action='DESELECT')
    for i in range(num_control_points):
        bpy.ops.curve.select_all(action='DESELECT')
        hook = bezier.modifiers.new(name = "Hook.%03d"%i, type = 'HOOK' )
        hook.object = arm
        hook.subtarget = "Bone.%03d"%(n-1-(i*n/num_control_points))
        pt = bpy.data.curves['BezierCurve'].splines[0].bezier_points[i]
        pt.select_control_point = True
        bpy.ops.object.hook_assign(modifier="Hook.%03d"%i)
        pt.select_control_point = False
    bpy.ops.object.mode_set(mode='OBJECT')
    for i in range(n):
        obj_name = "Cylinder.%03d"%i if i else "Cylinder"
        bpy.data.objects[obj_name].hide_set(True)
        bpy.data.objects[obj_name].hide_render = True
    bezier.select_set(False)
    if braid:
        rope = make_braid_rig(params, bezier)
    else:
        rope = make_cable_rig(params, bezier)
    return rope

def make_rope_v3(params):
    # This method relies on an STL file that contains a mesh for a
    # capsule.  The capsule cannot be non-unformly scaled without
    # distorting the end caps.  So instead we compute the rope_length
    # based on the param's radius and num_segments, and compute the
    # number of segments composed of the capsules that we need
    radius = params["segment_radius"]
    rope_length = radius * params["num_segments"] * 2 * 0.9 # HACKY -- shortening the rope artificially by 10% for now
    num_segments = int(rope_length / radius)
    separation = radius*1.1 # HACKY - artificially increase the separation to avoid link-to-link collision
    link_mass = params["segment_mass"] # TODO: this may need to be scaled
    link_friction = params["segment_friction"]

    # Parameters for how much the rope resists twisting
    twist_stiffness = 20
    twist_damping = 10

    # Parameters for how much the rope resists bending
    bend_stiffness = 0
    bend_damping = 5

    num_joints = int(radius/separation)*2+1
    loc0 = rope_length/2

    # Create the first link from the STL. In the filename: 12 = number
    # of radial subdivisions, 8 = number of length-wise subdivisions,
    # 1 = radius, 2 = height..
    bpy.ops.import_mesh.stl(filepath="capsule.stl")
    link0 = bpy.context.object
    #link0.name = "link_0"
    link0.name = "Cylinder"
    bpy.ops.transform.resize(value=(radius, radius, radius))
    # The link has Z-axis up, and the origin is in its center.
    link0.rotation_euler = (0, pi/2, 0)
    link0.location = (loc0, 0, 0)
    bpy.ops.rigidbody.object_add()
    link0.rigid_body.mass = link_mass
    link0.rigid_body.friction = link_friction
    # switch the collision_shape to CAPSULE--this is both faster and
    # more accurate (for an actual capsule) than the default.
    link0.rigid_body.collision_shape = 'CAPSULE'

    links = [link0]
    for i in range(1,num_segments):
        # copy link0 to create each additional link
        linki = link0.copy()
        linki.data = link0.data.copy()
        #linki.name = "link_" + str(i)
        #linki.name = "link_" + str(i)
        linki.location = (loc0 - i*separation, 0, 0)
        bpy.context.collection.objects.link(linki)

        links.append(linki)

        # Create a GENERIC_SPRING connecting this link to the previous.
        bpy.ops.object.empty_add(type='ARROWS', radius=radius*2, location=(loc0 - (i-0.5)*separation, 0, 0))
        bpy.ops.rigidbody.constraint_add(type='GENERIC_SPRING')
        joint = bpy.context.object
        joint.name = 'cc_' + str(i-1) + ':' + str(i)
        rbc = joint.rigid_body_constraint
        # connect the two links
        rbc.object1 = links[i-1]
        rbc.object2 = links[i]
        # disable translation from the joint.  Note: we can consider
        # making a "stretchy" rope by setting
        # limit_lin_x_{lower,upper} to a non-zero range.
        rbc.use_limit_lin_x = True
        rbc.use_limit_lin_y = True
        rbc.use_limit_lin_z = True
        rbc.limit_lin_x_lower = 0
        rbc.limit_lin_x_upper = 0
        rbc.limit_lin_y_lower = 0
        rbc.limit_lin_y_upper = 0
        rbc.limit_lin_z_lower = 0
        rbc.limit_lin_z_upper = 0
        if twist_stiffness > 0 or twist_damping > 0:
            rbc.use_spring_ang_x = True
            rbc.spring_stiffness_ang_x = twist_stiffness
            rbc.spring_damping_ang_x = twist_damping
        if bend_stiffness > 0 or bend_damping > 0:
            rbc.use_spring_ang_y = True
            rbc.use_spring_ang_z = True
            rbc.spring_stiffness_ang_y = bend_stiffness
            rbc.spring_stiffness_ang_z = bend_stiffness
            rbc.spring_damping_ang_y = bend_damping
            rbc.spring_damping_ang_z = bend_damping

    # After creating the rope, we connect every link to the link 1
    # separated by a joint that has no constraints.  This prevents
    # collision detection between the pairs of rope points.
    for i in range(2, num_segments):
        bpy.ops.object.empty_add(type='PLAIN_AXES', radius=radius*1.5, location=(loc0 - (i-1)*separation, 0, 0))
        bpy.ops.rigidbody.constraint_add(type='GENERIC_SPRING')
        joint = bpy.context.object
        joint.name = 'cc_' + str(i-2) + ':' + str(i)
        joint.rigid_body_constraint.object1 = links[i-2]
        joint.rigid_body_constraint.object2 = links[i]

    # the following parmaeters seem sufficient and fast for using this
    # rope.  steps_per_second can probably be lowered more to gain a
    # little speed.
    bpy.context.scene.rigidbody_world.steps_per_second = 1000
    bpy.context.scene.rigidbody_world.solver_iterations = 100

    return links



def make_table(params):
    bpy.ops.mesh.primitive_plane_add(size=params["table_size"], location=(0,0,-5))
    bpy.ops.rigidbody.object_add()
    table = bpy.context.object
    table.rigid_body.type = 'PASSIVE'
    #table.rigid_body.friction = 0.7
    table.rigid_body.friction = 0.8
    bpy.ops.object.select_all(action='DESELECT')

def tie_knot_with_fixture(end, fixture):
    end.rigid_body.kinematic = True
    end.keyframe_insert(data_path="location", frame=1)
    end.keyframe_insert(data_path="rotation_euler", frame=1)

    # Create a sequence of [ frame, angle, location ] for the end
    # we're moving to tie the knot.
    key_frames = [
        [ 20, -160, (-12, -1.5, -2) ],
        [ 80,  -160, (0, -1.5, -3.0) ],
        [ 110, -180, (4, -0.75, -3) ],
        [ 120, -270, (4, 0, -3) ],
        [ 130, -360, (4, 1, -3) ],
        [ 180, -450, (0, 1, -3) ],
        [ 200, -360, (0, -1, -3) ],
        [ 250, -360, (0, -3, -3) ],
        [ 330, -360, (0, -3, 14) ] ]

    # Apply the sequence as a set of keyframes to the end of the rope
    # we're pulling.
    for fno, rot, loc in key_frames:
        bpy.context.scene.frame_current = fno
        end.location = loc
        end.rotation_euler = (rot*pi/180, 90*pi/180, 0)
        end.keyframe_insert(data_path="location", frame=fno)
        end.keyframe_insert(data_path="rotation_euler", frame=fno)

    # move the fixture out of the way at frames 250 to 270
    bpy.context.scene.frame_current = 250
    fixture.keyframe_insert(data_path="location", frame=250)
    fixture.keyframe_insert(data_path="rotation_euler", frame=250)
    bpy.context.scene.frame_current = 270
    fixture.location = (0, 4, -1.5)
    fixture.rotation_euler = (radians(30), 0, 0)
    fixture.keyframe_insert(data_path="location", frame=270)
    fixture.keyframe_insert(data_path="rotation_euler", frame=270)

    # reset blender to show the first frame
    bpy.context.scene.frame_current = 1

if __name__ == '__main__':
    with open("rigidbody_params.json", "r") as f:
        params = json.load(f)
    clear_scene()
    links = make_rope_v3(params)
    make_table(params)

    # create a fixture for tying the knot
    bpy.ops.mesh.primitive_cube_add(location=(0,0,-1.5))
    bpy.ops.transform.resize(value=(1,2,0.25))
    bpy.ops.rigidbody.object_add()
    fixture = bpy.context.object
    fixture.rigid_body.friction = 0.5
    fixture.rigid_body.kinematic = True

    # Extend the number for frames in the animation and the rigid-body
    # simulation.
    frame_end = 500
    bpy.context.scene.rigidbody_world.point_cache.frame_end = frame_end
    bpy.context.scene.frame_end = frame_end

    # A little hacky... add extra weight to the last segment to have
    # it tighten the knot more when the opposite end is lifted
    links[0].rigid_body.mass *= 100
    tie_knot_with_fixture(links[-1], fixture)

    add_camera_light()
