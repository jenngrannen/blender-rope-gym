from gym import Env, spaces
import numpy as np
from untangle import shorten_rope
import sys
import os
sys.path.append(os.getcwd())
sys.path.append('policies')
from untangle_utils import *
from knots import *
from rigidbody_rope import *
from oracle_policy import find_knot_cylinders, find_knot


class UntangleEnv(Env):
    def __init__(self, seed=0):
        self.seed = seed
        self.curr_frame_num = 0
        self.render_offset = 0
        self.num_segments = 34
        self.num_knots = 1
        self.hold_idx = None
        self.pull_idx = None
        self.stable_hold = True
        # self.observation_shape = (64, 64, 3)
        self.observation_shape = self.num_segments*3

        self.observation_space = spaces.Box(low = np.zeros(self.observation_shape),
                                            high = np.ones(self.observation_shape),)
                                            # dtype = np.float16)
        self.action_scale = 2
        self.action_shape = 3 if self.stable_hold else 6
        self.action_space = spaces.Box(low = -1*np.ones(self.action_shape),
                                        high = np.ones(self.action_shape), )
                                        # shape = (self.action_shape))

    def get_obs(self):
        piece = "Cylinder"
        obs_space = []
        for i in range(0,self.num_segments):
            cyl = get_piece(piece, i if i else -1)
            x,y,z = cyl.matrix_world.translation
            obs_space.extend([x,y,z])
        return obs_space

    def reset(self):
        with open("rigidbody_params.json", "r") as f:
            self.params = json.load(f)

        clear_scene()
        make_capsule_rope(self.params)
        add_camera_light()
        set_render_settings(self.params["engine"],(self.params["render_width"],self.params["render_height"]))
        make_table(self.params)

        set_animation_settings(15000)
        self.curr_frame_num = 0
        self.curr_frame_num = tie_pretzel_knot(self.params, render=False)

        self.curr_frame_num = random_perturb(self.curr_frame_num, self.params)
        self.render_offset = self.curr_frame_num

        shorten_rope(self.params, end=35)
        # img = render_frame(self.curr_frame_num, render_offset=self.render_offset, step=1)
        obs = self.get_obs()

        # find the correct pull/hold locations
        idx_lists = find_knot_cylinders(self.num_segments, num_knots=self.num_knots)
        knot_idx_list = idx_lists[0]
        self.pull_idx, self.hold_idx, action_vec = find_knot(self.num_segments, knot_idx=knot_idx_list)

        return obs

    def step(self, action, render=True, time_steps=50):
        # hold hold_idx in place while we apply action to pull_idx
        action = np.array(action) * self.action_scale
        piece = "Cylinder"
        pull_cyl = get_piece(piece, self.pull_idx)
        hold_cyl = get_piece(piece, self.hold_idx)

        ## Undoing
        if self.stable_hold:
            take_action(hold_cyl, self.curr_frame_num + time_steps, (0,0,0))
        else:
            take_action(hold_cyl, self.curr_frame_num + time_steps, tuple(action[3:]))

        for step in range(self.curr_frame_num, self.curr_frame_num+10):
            bpy.context.scene.frame_set(step)
            if render:
                render_frame(step, render_offset=self.render_offset, step=1)

        take_action(pull_cyl, self.curr_frame_num + time_steps, tuple(action[:3]))

        ## Release both pull, hold
        toggle_animation(pull_cyl, self.curr_frame_num + time_steps, False)
        toggle_animation(hold_cyl, self.curr_frame_num + time_steps, False)

        settle_time = 10
        # Let the rope settle after the action, so we can know where the ends are afterwards
        # for step in range(self.curr_frame_num + 10, self.curr_frame_num + time_steps+ 90 + settle_time+1):
        for step in range(self.curr_frame_num + 10, self.curr_frame_num + time_steps + settle_time+1):
            bpy.context.scene.frame_set(step)
            if render:
                render_frame(step, render_offset=self.render_offset, step=1)
                # render_frame(step, render_offset=render_offset, step=4)
        self.curr_frame_num = self.curr_frame_num+time_steps+settle_time

        # get obs
        obs = self.get_obs()

        # reward, done?
        # reward can be distance from endpoint to hold cyl, plus bonus if done?
        end_cyl = get_piece(piece, self.num_segments-1)
        end_pos = np.array(end_cyl.matrix_world.translation)
        hold_pos = np.array(hold_cyl.matrix_world.translation)

        done = end_pos[0] > hold_pos[0] # if the end is to the right of the hold?

        if not done:
            reward = -1 * np.linalg.norm(end_pos-hold_pos)
        else:
            reward = np.linalg.norm(end_pos-hold_pos)

        print("reward", reward)
        print("done", done, "end", end_pos[0], "hold", hold_pos[0])
        return obs, reward, done, {}


    def render(self):
        img = render_frame(self.curr_frame_num, render_offset=self.render_offset, step=1)
        return img


if __name__ == '__main__':
    env = UntangleEnv()
    obs = env.reset()
    env.render()

    images = [obs]
    actions = [0]
    while True:
        # Take a random action
        action = env.action_space.sample()
        action = [0.5, 0.0, 0.5] # can complete in 4 actions
        print("action", action)
        print("pull, hold", env.pull_idx, env.hold_idx)
        obs, reward, done, info = env.step(action)
        print("action", action)
        print("step num", env.curr_frame_num)

        # if done == True:
        #     break

    # env.close()

    # print('done, showing states')
    # for action,img in zip(actions,images):
    #     cv2.putText(img, str(action), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1, cv2.LINE_AA)
    #     cv2.imshow('img', img)
    #     cv2.waitKey(0)
