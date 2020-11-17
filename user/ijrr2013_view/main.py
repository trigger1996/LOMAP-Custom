
'''
INFO __main__ - robot_1 run prefix: ['u1', '4', 'u1', '4', 'u1', '4', 'u1']
INFO __main__ - robot_1 control perfix: ['ufl', 'f', 'ufl', 'f', 'ufl', 'f']
INFO __main__ - robot_1 suffix cycle: ['u1', '4', '5', '6', '7', '8', '25', '26', 'g3', '26', '27', '3', '4', 'u1']
INFO __main__ - robot_1 control suffix cycle: ['ufl', 'rf', 'f', 'f', 'rf', 'rfr', 'lf', 'lf', 'ufl', 'f', 'fr', 'f', 'f']
INFO __main__ - robot_2 run prefix: ['u2', '10', 'u2', '10', 'u2', '10', 'u2']
INFO __main__ - robot_2 control perfix: ['ufl', 'f', 'ufl', 'f', 'ufl', 'f']
INFO __main__ - robot_2 suffix cycle: ['u2', '10', '11', '12', '1', '2', '21', '22', 'g1', '22', '23', '9', '10', 'u2']
INFO __main__ - robot_2 control suffix cycle: ['ufl', 'rf', 'f', 'f', 'rf', 'rfr', 'lf', 'lf', 'ufl', 'f', 'f', 'f', 'f']
INFO __main__ - robot_3 run prefix: ['1', '2', '21', '12', '1', '2', '21', '12', '1', '2']
INFO __main__ - robot_3 control perfix: ['rf', 'rfr', 'fr', 'f', 'rf', 'rfr', 'fr', 'f', 'rf']
INFO __main__ - robot_3 suffix cycle: ['2', '21', '12', '1', '2', '21', '12', '1', '2', '21', '12', '1', '2', '21', '12', '1', '2', '21', '12', '1', '2']
INFO __main__ - robot_3 control suffix cycle: ['rfr', 'fr', 'f', 'rf', 'rfr', 'fr', 'f', 'rf', 'rfr', 'fr', 'f', 'rf', 'rfr', 'fr', 'f', 'rf', 'rfr', 'fr', 'f', 'rf']
'''

# https://blog.csdn.net/u013180339/article/details/77002254


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from lomap import Ts
import view_animation

"""
animation example 2
author: Kiterun
"""

'''
fig, ax = plt.subplots()
x = np.linspace(0, 2*np.pi, 200)
y = np.sin(x)
l = ax.plot(x, y)



def init():
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(-1, 1)
    return l
'''

r1 = Ts.load('./robot_1.yaml')
r2 = Ts.load('./robot_2.yaml')
r3 = Ts.load('./robot_3.yaml')
run_1 = ['u1', '4', '5', '6', '7', '8', '25', '26', 'g3', '26', '27', '3', '4', 'u1']
run_2 = ['u2', '10', '11', '12', '1', '2', '21', '22', 'g1', '22', '23', '9', '10', 'u2']
run_3 = ['2', '21', '12', '1', '2', '21', '12', '1', '2', '21', '12', '1', '2', '21', '12', '1', '2', '21', '12', '1', '2']

# test_2
team_suffix = [(('27', '28', 1), 'u2', '2'), (('27', '28', 2), ('u2', '10', 1), '1'), ('28', '10', '12'), (('28', '21', 1), '11', '21'), (('28', '21', 2), '23', '2'), ('21', ('23', '24', 1), '1'), (('21', '22', 1), ('23', '24', 2), '12'), ('22', '24', '21'), ('g1', 'g2', '2'), (('g1', '22', 1), ('g2', '24', 1), '1'), ('22', '24', '12'), (('22', '23', 1), ('24', '25', 1), '21'), ('23', ('24', '25', 2), '2'), ('9', '25', '1'), ('10', ('25', '26', 1), '12'), ('u2', '26', '21'), (('u2', '10', 1), ('26', '27', 1), '2'), ('10', '27', '1'), ('11', '3', '12'), ('23', '4', '21'), (('23', '24', 1), 'u1', '2'), (('23', '24', 2), ('u1', '4', 1), '1'), ('24', '4', '12'), (('24', '25', 1), '5', '21'), (('24', '25', 2), '27', '2'), ('25', ('27', '28', 1), '1'), (('25', '26', 1), ('27', '28', 2), '12'), ('26', '28', '21'), ('g3', 'g4', '2'), (('g3', '26', 1), ('g4', '28', 1), '1'), ('26', '28', '12'), (('26', '27', 1), ('28', '21', 1), '21'), ('27', ('28', '21', 2), '2'), ('3', '21', '1'), ('4', ('21', '22', 1), '12'), ('u1', '22', '21'), (('u1', '4', 1), ('22', '23', 1), '2'), ('4', '23', '1'), ('5', '9', '12'), ('27', '10', '21'), (('27', '28', 1), 'u2', '2')]


#view_animation.visualize_animation(r1, run_1)

ts_tuple = tuple([r1, r2])
#run = [run_1, run_2]
#view_animation.visualize_two_animation(ts_tuple, run)

ts_tuple = tuple([r1, r2, r3])
run = [run_1, run_2, run_3]
#view_animation.visualize_three_animation(ts_tuple, run)
#view_animation.visualize_multi_animation(ts_tuple, run)

ts_tuple = tuple([r1, r2, r3])
view_animation.visualize_animation_w_team_run(ts_tuple, team_suffix)