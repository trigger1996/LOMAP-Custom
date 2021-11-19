"""
Env 2D
@author: huiming zhou
"""


class Env:
    def __init__(self):
        self.x_range = 17  # size of background
        self.y_range = 12
        self.motions = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                        (1, 0), (1, -1), (0, -1), (-1, -1)]
        self.obs = self.obs_map()

    def update_obs(self, obs):
        self.obs = obs

    def obs_map(self):
        """
        Initialize obstacles' positions
        :return: map of obstacles
        """

        x = self.x_range
        y = self.y_range
        obs = set()

        for i in range(x):
            obs.add((i, 0))
        for i in range(x):
            obs.add((i, y - 1))

        for i in range(y):
            obs.add((0, i))
        for i in range(y):
            obs.add((x - 1, i))

        '''Obstacle 1'''
        for i in range(3, 6):
            for j in range(5, 8):
                obs.add((i, j))
        for i in range(6, 8):
            for j in range(4, 8):
                obs.add((i, j))
        for i in range(3, 5):
            for j in range(8, 10):
                obs.add((i, j))

        '''Obstacle 2'''
        for i in range(9, 11):
            for j in range(1, 4):
                obs.add((i, j))
        for i in range(10, 14):
            for j in range(3, 6):
                obs.add((i, j))


        '''Obstacle 3'''
        for i in range(10, 12):
            for j in range(9, 11):
                obs.add((i, j))




        return obs
