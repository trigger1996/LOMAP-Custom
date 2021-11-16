#!/usr/bin/env python
# coding=utf-8

import lomap
import networkx
from lomap import Ts
import copy


def get_possible_region(start_pos, map, turn = 5):
    next_pos = []
    current_pos = []
    possible_region = []

    current_pos.append(start_pos)
    for index in range(0, turn):
        next_pos = []
        #possible_region.append([])
        for u in current_pos:
            if type(u) == str:
                for v in map.succ[u]:
                    if map.succ[u][v][0]['weight'] == 1:
                        next_pos.append(v)
                    else:
                       next_pos.append([(u, v, 1), index])
            else:
                v_start    = list(u)[0][0]
                v_end      = list(u)[0][1]
                start_time = list(u)[1]
                curr_time  = list(u)[0][2]

                if index - start_time == map.succ[v_start][v_end][0]['weight'] - 1:
                    next_pos.append(v_end)
                else:
                    next_pos.append([(v_start, v_end, curr_time + 1), start_time])

        # get possible region according to next possible position
        possible_region.append(current_pos + next_pos)      # ere no need to extend, and this is the final possible region

        current_pos = copy.deepcopy(next_pos)

    return possible_region

def main():
    robot_2 = Ts.load('./robot_1.yaml')
    turn = 25

    possible_region = get_possible_region('28', robot_2.g, turn)

    for i in range(0, possible_region.__len__()):
        print(possible_region[i])

    print('\n')

    for i in range(0, possible_region.__len__()):
        state_to_remove = []
        for state in possible_region[i]:
            if type(state) != str:
                state_to_remove.append(state)
        for state in state_to_remove:
            possible_region[i].remove(state)
        possible_region[i] = list(set(possible_region[i]))  # remove identical elements

    for i in range(0, possible_region.__len__()):
        print(possible_region[i])

    print(233)


if __name__ == '__main__':
    main()
