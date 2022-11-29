import bpy
import numpy as np
import sys
from sklearn.neighbors import NearestNeighbors
from untangle_utils import *
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), '..'))

def find_knot(num_segments, chain=False, depth_thresh=0.4, idx_thresh=3, pull_offset=3,knot_idx=None, full_rope=True):

    piece = "Torus" if chain else "Cylinder"
    cache = {}
    num_segments = num_segments if full_rope else 34

    # Make a single pass, store the xy positions of the cylinders
    rope_range = range(num_segments) if full_rope else range(0,34)
    # for i in range(num_segments):
    for i in rope_range:
        cyl = get_piece(piece, i if i else -1)
        x,y,z = cyl.matrix_world.translation
        key = tuple((x,y))
        val = {"idx":i, "depth":z}
        cache[key] = val
    neigh = NearestNeighbors(2, 0)
    planar_coords = list(cache.keys())
    neigh.fit(planar_coords)
    # Now traverse and look for the under crossing
    idx_list = range(knot_idx[0], knot_idx[1]) if not knot_idx is None else range(num_segments)
    idx_list = idx_list[::-1]
    for i in idx_list:
        cyl = get_piece(piece, i if i else -1)
        x,y,z = cyl.matrix_world.translation
        match_idxs = neigh.kneighbors([(x,y)], 2, return_distance=False) # 1st neighbor is always identical, we want 2nd
        nearest = match_idxs.squeeze().tolist()[1:][0]
        x1,y1 = planar_coords[nearest]
        curr_cyl, match_cyl = cache[(x,y)], cache[(x1,y1)]
        depth_diff = match_cyl["depth"] - curr_cyl["depth"]
        idx_diff = abs(match_cyl["idx"] - curr_cyl["idx"])
        if depth_diff > depth_thresh and idx_diff > idx_thresh:
            pull_idx = i + pull_offset # Pick a point slightly past under crossing to do the pull
            dx = planar_coords[pull_idx][0] - x
            dy = planar_coords[pull_idx][1] - y
            hold_idx = match_cyl["idx"]
            SCALE_X = 1
            SCALE_Y = 1
            Z_OFF = 2
            action_vec = [SCALE_X*dx, SCALE_Y*dy, Z_OFF] # Pull in the direction of the rope (NOTE: 7 is an arbitrary scale for now, 6 is z offset)
            return pull_idx, hold_idx, action_vec # Found! Return the pull, hold, and action
    return 16, 25, [0,0,0] # Didn't find a pull/hold

def find_knot_cylinders(num_segments, chain=False, num_knots=1):
    piece = "Torus" if chain else "Cylinder"
    cache = {}
    curr_z = get_piece(piece, -1).matrix_world.translation[2]
    dz_thresh = 0.2
    dzs = []
    # for i in range(num_segments):
    for i in range(0,34):
        cyl = get_piece(piece, i if i else -1)
        # cyl = get_piece(piece, i)
        x,y,z = cyl.matrix_world.translation
        dz = abs(z - curr_z)
        dzs.append(dz)
        curr_z = z
    dzs = np.round(dzs, 2)
    if num_knots == 1:
        nonzero = np.where(dzs>0.2)
        # nonzero = np.where(dzs>0.3)
        start_idx, end_idx = np.amin(nonzero), np.amax(nonzero)
        result = [[start_idx, end_idx]]
    else:
        nonzero = np.where(dzs>0.2)[0]
        split_idx, x, dx = 0, nonzero[0], 0
        for i in range(len(nonzero)):
            dx_curr = nonzero[i] - x
            if dx_curr > dx:
                dx = dx_curr
                split_idx = i
                x = nonzero[i]
        s1, e1 = np.amin(nonzero[:split_idx]), np.amax(nonzero[:split_idx])
        s2, e2 = np.amin(nonzero[split_idx:]), np.amax(nonzero[split_idx:])
        result = [[s1,e1],[s2,e2]]
    return result

class Oracle(object):
    def __init__(self, params):
        self.action_count = 0
        self.max_actions = 7
        self.rope_length = params["num_segments"]
        self.num_knots = len(params["knots"])

    def bbox_untangle(self, start_frame, render_offset=0):
        print("BBOX UNTANGLE FIND KNOT")
        # if find_knot(self.rope_length)[-1]  == [0,0,0]:
        #     return None, None
        return find_knot(self.rope_length, full_rope=False)[-1]  == [0,0,0], None

    def policy_undone_check(self, start_frame, prev_pull, prev_hold, pull_idx, hold_idx, prev_action_vec, render_offset=0):
        if self.action_count > self.max_actions or find_knot(self.rope_length, full_rope=False)[-1]  == [0,0,0]:
            if not self.action_count > self.max_actions:
                print("POLICY UNDONE CHECK FIND KNOT")
            return True
        end2_idx = 35-1
        end1_idx = -1
        ret = undone_check(start_frame, prev_pull, prev_hold, pull_idx, hold_idx, prev_action_vec, end1_idx, end2_idx, render_offset=render_offset)
        if ret:
            self.num_knots -= 1
        return ret

    def undo(self, start_frame, render=False, render_offset=0):
        idx_lists = find_knot_cylinders(self.rope_length, num_knots=self.num_knots)
        if self.num_knots > 1:
            idx_list1, idx_list2 = idx_lists
            knot_idx_list = idx_list1 if min(idx_list1) < min(idx_list2) else idx_list2 # find the right most knot
        else:
            knot_idx_list = idx_lists[0]
        pull_idx, hold_idx, action_vec = find_knot(self.rope_length, knot_idx=knot_idx_list, full_rope=False)
        action_vec /= np.linalg.norm(action_vec)
        end_frame, action_vec = take_undo_action(start_frame, pull_idx, hold_idx, action_vec, render=render, render_offset=render_offset)
        pull_pixel, hold_pixel = cyl_to_pixels([pull_idx, hold_idx])
        self.action_count += 1
        return end_frame, pull_pixel[0], hold_pixel[0], pull_idx, hold_idx, action_vec

    def reidemeister(self, start_frame, render=False, render_offset=0):
        # middle_frame = reidemeister_right(start_frame, -1, self.rope_length-1, render=render, render_offset=render_offset)
        # end_frame = reidemeister_left(middle_frame, -1, self.rope_length-1, render=render, render_offset=render_offset)

        middle_frame = reidemeister_right(start_frame, 34, -1, render=render, render_offset=render_offset)
        end_frame = reidemeister_left(middle_frame, 34, -1, render=render, render_offset=render_offset)
        self.action_count += 2
        return end_frame
