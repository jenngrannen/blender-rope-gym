import bpy
import sys
import os
sys.path.append(os.getcwd())
sys.path.append('policies')
from untangle_utils import *
from knots import *
from rigidbody_rope import *
# Load policies
from oracle_policy import Oracle

def shorten_rope(params, start=0, end=35):
    if bpy.context.object.mode == 'EDIT':
        bpy.ops.object.mode_set(mode='OBJECT')

    for i in range(0, start):
        cyl_name = 'Cylinder.%03d'%i if i > 0 else 'Cylinder'
        bpy.ops.object.select_all(action='DESELECT')
        bpy.data.objects[cyl_name].select_set(True)
        bpy.ops.object.delete()

    for i in range(end, params["num_segments"]):
        bpy.ops.object.select_all(action='DESELECT')
        bpy.data.objects['Cylinder.%03d'%i].select_set(True)
        bpy.ops.object.delete()

    params["num_segments"] = end - start

def run_untangling_rollout(policy, params):
    set_animation_settings(15000)
    knot_end_frame = 0
    knot_end_frame = tie_pretzel_knot(params, render=False)

    knot_end_frame = random_perturb(knot_end_frame, params)
    render_offset = knot_end_frame

    shorten_rope(params, end=35)
    render_frame(knot_end_frame, render_offset=render_offset, step=1)

    # reid_end = policy.reidemeister(knot_end_frame, render=True, render_offset=render_offset)
    reid_end = knot_end_frame
    undo_end_frame = reid_end

    undo_end, pull_pix, hold_pix, pull_idx, hold_idx, action_vec = policy.undo(undo_end_frame, render=True, render_offset=render_offset)
    undone = policy.policy_undone_check(undo_end, pull_pix, hold_pix, pull_idx, hold_idx, action_vec, render_offset=render_offset)
    undo_end_frame = undo_end
    undo_end_frame = policy.reidemeister(undo_end_frame, render=True, render_offset=render_offset)
    print("undone", undone)
    return

if __name__ == '__main__':
    if not os.path.exists("./preds"):
        os.makedirs('./preds')
    else:
        os.system('rm -r ./preds')
        os.makedirs('./preds')

    with open("rigidbody_params.json", "r") as f:
        params = json.load(f)

    BASE_DIR = os.getcwd()
    policy = Oracle(params)

    clear_scene()
    make_capsule_rope(params)
    if not params["texture"] == "capsule":
        rig_rope(params, braid=params["texture"]=="braid")
    add_camera_light()
    set_render_settings(params["engine"],(params["render_width"],params["render_height"]))
    make_table(params)
    run_untangling_rollout(policy, params)
