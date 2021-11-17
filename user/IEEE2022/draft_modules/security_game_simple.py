"""
LPA_star 2D
@author: huiming zhou
"""
import copy
import os
import sys
import re
import math
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import plotting, env
from lomap import Ts
import itertools as it
from lomap.algorithms.product import ts_times_ts
import networkx
from LPAstar import Ts_Grid
from env import Env


def ts_times_ts_branching(ts_tuple, F_index, opt_path):
    '''TODO:
    add option to choose what to save on the automaton's
    add description
    add regression tests
    add option to create from current state
    '''
    # NOTE: We assume deterministic TS
    assert all((len(ts.init) == 1 for ts in ts_tuple))

    # Initial state label is the tuple of initial states' labels
    product_ts = Ts()

    init_state = tuple((next(iter(ts.init)) for ts in ts_tuple))
    product_ts.init[init_state] = 1

    # Props satisfied at init_state is the union of props
    # For each ts, get the prop of init state or empty set
    init_prop = set.union(*[ts.g.node[ts_init].get('prop', set())
                            for ts, ts_init in zip(ts_tuple, init_state)])

    # Finally, add the state
    product_ts.g.add_node(init_state, {'prop': init_prop,
                        'label': "{}\\n{}".format(init_state, list(init_prop))})

    # Start depth first search from the initial state
    stack=[]
    stack.append(init_state)
    while stack:
        cur_state = stack.pop()
        # Actual source states of traveling states
        source_state = tuple((q[0] if type(q) == tuple else q
                              for q in cur_state))
        # Time spent since actual source states
        time_spent = tuple((q[2] if type(q) == tuple else 0 for q in cur_state))

        # Iterate over all possible transitions
        for tran_tuple in it.product(*[t.next_states_of_wts(q)
                                       for t, q in zip(ts_tuple, cur_state)]):
            # tran_tuple is a tuple of m-tuples (m: size of ts_tuple)

            # First element of each tuple: next_state
            # Second element of each tuple: time_left
            next_state = tuple([t[0] for t in tran_tuple])
            time_left = tuple([t[1] for t in tran_tuple])
            control = tuple([t[2] for t in tran_tuple])

            if not next_state[F_index] in opt_path:
                continue

            # Min time until next transition
            w_min = min(time_left)

            # Next state label. Singleton if transition taken, tuple if
            # traveling state
            next_state = tuple(((ss, ns, w_min+ts) if w_min < tl else ns
                        for ss, ns, tl, ts in zip(
                            source_state, next_state, time_left, time_spent)))

            # Add node if new
            if next_state not in product_ts.g:
                # Props satisfied at next_state is the union of props
                # For each ts, get the prop of next state or empty set
                # Note: we use .get(ns, {}) as this might be a travelling state
                next_prop = set.union(*[ts.g.node.get(ns, {}).get('prop', set())
                                       for ts, ns in zip(ts_tuple, next_state)])

                # Add the new state
                product_ts.g.add_node(next_state, {'prop': next_prop,
                        'label': "{}\\n{}".format(next_state, list(next_prop))})

                # Add transition w/ weight
                product_ts.g.add_edge(cur_state, next_state,
                                attr_dict={'weight': w_min, 'control': control})
                # Continue dfs from ns
                stack.append(next_state)

            # Add tran w/ weight if new
            elif next_state not in product_ts.g[cur_state]:
                product_ts.g.add_edge(cur_state, next_state,
                                attr_dict={'weight': w_min, 'control': control})

    # Return ts_1 x ts_2 x ...
    return product_ts

def takeThird(elem):
    return elem[2]

def main():
    x_start_1 = (1, 1)      # default (1, 1)
    x_start_2 = (2, 8)      # default (2, 10)
    x_goal = (15, 10)
    expect_volume_CF = 1 # better be odd, default: 1, 3

    actual_path_F  = [x_start_1]
    actual_path_CF = [x_start_2]

    bot_1 = Ts_Grid("unicycle bot 1", x_start_1, x_goal)
    bot_2 = Ts_Grid("unicycle bot 2", x_start_2, x_goal)

    xy_curr_1 = x_start_1
    xy_curr_2 = x_start_2
    while abs(list(xy_curr_1)[0] - list(xy_curr_2)[0]) >= 1 or abs(list(xy_curr_1)[1] - list(xy_curr_2)[1]) >= 1:
        ''' Update bot 2 as obstacle '''
        Env_t = Env()
        Env_t.obs.add(xy_curr_2)
        for x_t in range(list(xy_curr_2)[0] - (expect_volume_CF - 1) / 2, list(xy_curr_2)[0] + (expect_volume_CF - 1) / 2 + 1):
            for y_t in range(list(xy_curr_2)[1] - (expect_volume_CF - 1) / 2, list(xy_curr_2)[1] + (expect_volume_CF - 1) / 2 + 1):
                if x_t > 0 and x_t < Env_t.x_range and y_t > 0 and y_t < Env_t.y_range:
                    Env_t.obs.add((x_t, y_t))


        ''' re-construct bot with their current position '''
        bot_1_t = Ts_Grid("unicycle bot 1 t", xy_curr_1, x_goal, enviro=Env_t)
        bot_2_t = Ts_Grid("unicycle bot 2 t", xy_curr_2, x_goal)

        opt_path_F = networkx.dijkstra_path(bot_1_t.g, str(xy_curr_1), str(x_goal))

        # build up product_TS and branching such that only states in optimal path of vehicle-F are listed
        F_index = 0
        ts_tuple = (bot_1_t, bot_2_t)
        product_ts = ts_times_ts_branching(ts_tuple, F_index, opt_path_F)

        ''' Find vertices in product_TS which makes singleton collisions'''
        # Find all available collision points
        stop_pos = []
        for pos_xy in product_ts.g.node.keys():
            for i in range(0, list(pos_xy).__len__()):
                if i != F_index and list(pos_xy)[i] == list(pos_xy)[F_index]:
                    stop_pos.append(pos_xy)

        # calculate costs to collision points
        cost_s = []
        for pos_xy in stop_pos:
            cost_t = []
            for i in range(0, list(pos_xy).__len__()):
                val = networkx.dijkstra_path_length(list(ts_tuple)[i].g, list(list(ts_tuple)[i].init)[0], pos_xy[i])
                cost_t.append(val)
            cost_s.append(cost_t)

        # find the collision point with minimum cost && vehicle-CF can arrive simultaneously or earlier
        cost_s_without_F = copy.deepcopy(cost_s)
        for i in range(0, cost_s_without_F.__len__()):
            cost_s_without_F[i].pop(F_index)

        # first for point which can arrive simultaneously
        min_val = 1e6
        min_index = 0
        tgt_pt = []
        for i in range(0, cost_s.__len__()):
            for j in range(0, list(cost_s[i]).__len__()):
                if j != F_index and min(cost_s_without_F[i]) == list(cost_s[i])[F_index] and min_val >= min(cost_s_without_F[i]):
                    # requirement 2 ensures CF will arrive simultaneously
                    # requirement 3 finds the minimum value
                    min_index = i
                    min_val = min(cost_s_without_F[i])
        # check whether the min_index is valid
        if min(cost_s_without_F[min_index]) == list(cost_s[min_index])[F_index]:
            tgt_pt.append([stop_pos[min_index], cost_s[min_index], min(cost_s_without_F[min_index])])

        # second for point which can arrive earlier
        min_val = 1e6
        min_index = 0
        for i in range(0, cost_s.__len__()):
            for j in range(0, list(cost_s[i]).__len__()):
                if j != F_index and  min(cost_s_without_F[i]) <= list(cost_s[i])[F_index] and min_val >= min(cost_s_without_F[i]):
                    # requirement 2 ensures CF will arrive earlier
                    # requirement 3 finds the minimum value
                    min_val = min(cost_s_without_F[i])
                    min_index = i
        if min(cost_s_without_F[min_index]) <= list(cost_s[min_index])[F_index]:
            tgt_pt.append([stop_pos[min_index], cost_s[min_index], min(cost_s_without_F[min_index])])

        # sort available collision point by cost
        tgt_pt.sort(key=takeThird, reverse=False)

        ''' Find vertices in product_TS which makes pairwise collisions'''

        ''' Find vertices that will cause cycles'''

        ''' Update route for F & CF '''
        path_CF = networkx.dijkstra_path(bot_2_t.g, list(bot_2_t.init)[0], list(tgt_pt[0][0])[1])

        pattern = re.compile(r'\d+')
        xy_curr_1 = (int(pattern.findall(opt_path_F[1])[0]), int(pattern.findall(opt_path_F[1])[1]))
        xy_curr_2 = (int(pattern.findall(path_CF[1])[0]),    int(pattern.findall(path_CF[1])[1]))
        actual_path_F.append(xy_curr_1)
        actual_path_CF.append(xy_curr_2)


        '''for debugging'''
        #print(opt_path_F)
        print(actual_path_F, actual_path_CF)


        '''Clear redundant variables'''
        del Env_t, bot_1_t, bot_2_t

    print(233)



if __name__ == '__main__':
    main()
