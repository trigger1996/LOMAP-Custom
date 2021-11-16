#!/usr/bin/env python
# coding=utf-8

# Copyright (C) 2012-2015, Alphan Ulusoy (alphan@bu.edu)
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.


import lomap
import networkx
from lomap import Ts, Timer
import logging
from collections import namedtuple
from lomap.algorithms.product_ca import ts_times_ts

# custom packages
#import view
#import lomap.algorithms.multi_agent_optimal_run_ca as ca


def get_proximity_info(ts, ts_auto, run, run_auto, cost = 2):
    # in simulations, proximity info is obtained by team TS
    # as for gazebo or physical, proximity info should be obtained from distance sensor or UWB sensor
    #for state in run:

    run_w_cost = [[run[0], 0]]
    for i in range(1, run.__len__()):
        run_w_cost.append([run[i], run_w_cost[i - 1][1] + ts.g.edge[run[i - 1]][run[i]][0]['weight']])

    run_w_cost_auto = [[run_auto[0], 0]]
    for i in range(1, run_auto.__len__()):
        run_w_cost_auto.append([run_auto[i], run_w_cost_auto[i - 1][1] + ts_auto.g.edge[run_auto[i - 1]][run_auto[i]][0]['weight']])

    # construct team run
    j = 0
    mas_run = [[run[0], run_auto[0]]]
    for i in range(1, run_w_cost.__len__()):
        while run_w_cost[i][1] > run_w_cost_auto[j][1]:
            j = j + 1

        if run_w_cost[i][1] == run_w_cost_auto[j][1]:
            mas_run.append([run[i], run_auto[j]])
        elif run_w_cost[i][1] < run_w_cost_auto[j][1]:
            w      = run_w_cost[i][1]      - run_w_cost_auto[j - 1][1]
            mas_run.append([run[i], (run_auto[j - 1], run_auto[j], w)])

        if j >= run_w_cost_auto.__len__():
            #break
            j = run_w_cost_auto.__len__() - 1

    for i in range(0, mas_run.__len__()):
        if type(mas_run[i][1]) == tuple:
            cost_t = networkx.dijkstra_path_length(ts.g, mas_run[i][0], list(mas_run[i][1])[0]) - list(mas_run[i][1])[2]
        else:
            cost_t = networkx.dijkstra_path_length(ts.g, mas_run[i][0], mas_run[i][1])

        print(mas_run[i], '         distance: ', cost_t)
        if cost_t <= cost:
            return mas_run[i]

    return None

def main():
    r1 = Ts.load('./robot_1.yaml')
    r2 = Ts.load('./robot_1.yaml')
    run_r1 = ['23', '24', '25', '26', '27', '28', '21', '22', '23']
    run_r2 = ['9', '10', '11', '12', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    ret = get_proximity_info(r1, r2, run_r1, run_r2)
    if ret == None:
        print('NO Uncontrollable vehicle sighted!')
    else:
        print('Uncontrollable vehicle sighted!', ret)

    print('Finished!')


if __name__ == '__main__':
    main()
