import os
import gym
import gym.spaces
import numpy as np

import matplotlib.pyplot as plt


class CatchBall(gym.core.Env):
    metadata = {
        'render.modes': ['human'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        # parameters
        self.name = os.path.splitext(os.path.basename(__file__))[0]
        self.screen_n_rows = 8
        self.screen_n_cols = 8
        self.player_length = 3
        self.enable_actions = (0, 1, 2)
        self.frame_rate = 5

        # 行動空間。何もしない、左、右の3種
        self.action_space = gym.spaces.Discrete(3)
        # 観測空間(state)の次元 (2次元スクリーン) とその最大値
        self.observation_space = gym.spaces.Box(
            0.0, 1.0, (self.screen_n_rows, self.screen_n_cols))

        self.figure = None

        # 使うメンバ変数を初期化
        self.reward = 0
        self.screen = np.zeros((self.screen_n_rows, self.screen_n_cols))
        self.rgb_screen = np.zeros((self.screen_n_rows, self.screen_n_cols, 3))
        self.terminal = False
        self.player_row = self.screen_n_rows - 1
        self.player_col = np.random.randint(
            self.screen_n_cols - self.player_length)
        self.ball_row = 0
        self.ball_col = np.random.randint(self.screen_n_cols)

    # 各stepごとに呼ばれる
    # actionを受け取り、次のstateとreward、episodeが終了したかどうかを返すように実装
    def _step(self, action):
        """
        action:
            0: do nothing
            1: move left
            2: move right
        """
        self.update(action)
        self.draw()
        return self.screen, self.reward, self.terminal, {}

    def update(self, action):
        """
        action:
            0: do nothing
            1: move left
            2: move right
        """
        # update player position
        if action == self.enable_actions[1]:
            # move left
            self.player_col = max(0, self.player_col - 1)
        elif action == self.enable_actions[2]:
            # move right
            self.player_col = min(self.player_col + 1,
                                  self.screen_n_cols - self.player_length)
        else:
            # do nothing
            pass

        # update ball position
        self.ball_row += 1

        # collision detection
        self.reward = 0
        self.terminal = False
        if self.ball_row == self.screen_n_rows - 1:
            self.terminal = True
            if self.player_col <= self.ball_col < self.player_col + self.player_length:
                # catch
                self.reward = 1
            else:
                # drop
                self.reward = -1

    def draw(self):
        "update state"
        # reset screen
        self.screen = np.zeros((self.screen_n_rows, self.screen_n_cols))

        # draw player
        self.screen[self.player_row,
                    self.player_col:self.player_col + self.player_length] = 1

        # draw ball
        self.screen[self.ball_row, self.ball_col] = 1

    def _reset(self):
        # reset player position
        self.player_row = self.screen_n_rows - 1
        self.player_col = np.random.randint(
            self.screen_n_cols - self.player_length)

        # reset ball position
        self.ball_row = 0
        self.ball_col = np.random.randint(self.screen_n_cols)

        # reset other variables
        self.reward = 0
        self.terminal = False
        self.draw()
        return self.screen

    def _render(self, mode='human', close=False):
        if close:
            return
        if self.figure is None:
            # animate
            self.figure = plt.figure(
                figsize=(self.screen_n_rows / 2, self.screen_n_cols / 2))
            self.image = plt.imshow(
                self.screen, interpolation="none", cmap="gray")
        self.image.set_array(self.screen)
        plt.pause(.01)
