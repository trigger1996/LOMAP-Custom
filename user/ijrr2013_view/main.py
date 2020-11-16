
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
dot, = ax.plot([], [], 'ro')


def init():
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(-1, 1)
    return l

def gen_dot():
    for i in np.linspace(0, 2*np.pi, 200):
        newdot = [i, np.sin(i)]
        yield newdot

def update_dot(newd):
    dot.set_data(newd[0], newd[1])
    return dot,
'''

r1 = Ts.load('./robot_1.yaml')
run = ['u1', '4', '5', '6', '7', '8', '25', '26', 'g3', '26', '27', '3', '4', 'u1']
view_animation.visualize_run(r1, run)


'''
ani = animation.FuncAnimation(fig, update_dot, frames = gen_dot, interval = 100, init_func=init)
ani.save('sin_dot.gif', writer='imagemagick', fps=30)

plt.show()

'''