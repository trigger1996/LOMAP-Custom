"""
LPA_star 2D
@author: huiming zhou
"""

import os
import sys
import math
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import plotting, env
from lomap.algorithms.product import ts_times_ts
import networkx
from LPAstar import Ts_Grid

def main():
    x_start_1 = (1, 1)
    x_start_2 = (2, 10)
    x_goal = (15, 10)
    bot_1 = Ts_Grid("unicycle bot 1", x_start_1, x_goal)
    bot_2 = Ts_Grid("unicycle bot 2", x_start_2, x_goal)

    #ts_example = Ts_Grid.load('./user/IEEE2022/old/draft_modules/robot_1.yaml')

    ts_tuple = (bot_1, bot_2)
    product_ts = ts_times_ts(ts_tuple)

    ''' Find vertices in product_TS which makes singleton collisions'''
    #product_ts.g[('(1, 1)', '(1, 1)')]

    ''' Find vertices in product_TS which makes pairwise collisions'''

    ''' Find vertices that will cause  cycles'''

    print(233)



if __name__ == '__main__':
    main()
