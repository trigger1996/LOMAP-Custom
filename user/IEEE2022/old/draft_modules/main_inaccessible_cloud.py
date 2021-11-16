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

# custom packages
#import view
#import lomap.algorithms.multi_agent_optimal_run_ca as ca


def get_inaccessible_region_adjacency(ts, run):
    inaccessible_cloud = []

    # 现在这个只是根据临近点计算，最好是根据cost或者距离，引入traveling state
    for state in run:
        if state in ts.g.node:
            adjacency_list_t = networkx.neighbors(ts.g, state)
            adjacency_list_t.append(state)
            inaccessible_cloud.append(adjacency_list_t)
    return inaccessible_cloud

#def get_inaccessible_region_distance

#def get_inacc

def main():
    r1 = Ts.load('./robot_1.yaml')
    run_r1 = ['23', '24', '25', '26', '27', '28', '21', '22', '23']

    inaccessible_cloud = get_inaccessible_region_adjacency(r1, run_r1)
    print(inaccessible_cloud)

    print('Finished!')


if __name__ == '__main__':
    main()
