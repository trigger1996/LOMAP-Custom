"""
LPA_star 2D
@author: huiming zhou
"""
import copy
import os
import sys
import re
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import plotting
from lomap import Ts
import itertools as it
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

def in_region(pos, region):
    '''

    :param pos:
    :param region: [x_min, x_max, y_min, y_max]
    :return:
    '''
    if pos[0] >= region[0] and pos[0] <= region[1] and\
       pos[1] >= region[2] and pos[1] <= region[3]:
        return True
    else:
        return False


def node_str_to_list(val):
    pattern = re.compile(r'\d+')
    return [int(pattern.findall(val)[0]), int(pattern.findall(val)[1])]

def takeThird(elem):
    return elem[2]

def main():

    start_region = [1, 1, 1, 1]         # [x_min, x_max, y_min, y_max]
    goal_region  = [15, 15, 10, 10]

    x_start_1 = (1, 1)      # default (1, 1)
    x_start_2 = (2, 10)     # default (2, 10)
    x_goal = (15, 10)
    expect_volume_CF = 5    # better be odd, default: 1, alternative, 3, 5

    end_turn = 50
    turn = 0

    is_goal_arrived = False

    actual_path_F  = [x_start_1]
    actual_path_CF = [x_start_2]

    bot_1 = Ts_Grid("unicycle bot 1", x_start_1, x_goal)
    bot_2 = Ts_Grid("unicycle bot 2", x_start_2, x_goal)
    plt.close('all')

    try:
        xy_curr_1 = x_start_1
        xy_curr_2 = x_start_2
        x_goal_t = x_goal
        while (abs(list(xy_curr_1)[0] - list(xy_curr_2)[0]) >= 1 or abs(list(xy_curr_1)[1] - list(xy_curr_2)[1]) >= 1) and \
              not (is_goal_arrived and in_region(xy_curr_1, start_region)) and\
              turn < end_turn:

            ''' Update bot 2 as obstacle '''
            Env_t = Env()
            Env_t.obs.add(xy_curr_2)
            for x_t in range(list(xy_curr_2)[0] - (expect_volume_CF - 1) / 2, list(xy_curr_2)[0] + (expect_volume_CF - 1) / 2 + 1):
                for y_t in range(list(xy_curr_2)[1] - (expect_volume_CF - 1) / 2, list(xy_curr_2)[1] + (expect_volume_CF - 1) / 2 + 1):
                    if x_t > 0 and x_t < Env_t.x_range and y_t > 0 and y_t < Env_t.y_range:
                        if (x_t, y_t) != xy_curr_1:         # to solve the problem that F in the virtual volume
                            Env_t.obs.add((x_t, y_t))

            ''' If bot 1 arrived target region, change goal '''
            if not is_goal_arrived and in_region(xy_curr_1, goal_region):
                x_goal_t = x_start_1
                is_goal_arrived = True

            ''' re-construct bot with their current position '''
            bot_1_t = Ts_Grid("unicycle bot 1 t", xy_curr_1, x_goal_t, enviro=Env_t)
            bot_2_t = Ts_Grid("unicycle bot 2 t", xy_curr_2, x_goal_t)
            plt.close('all')

            try:
                opt_path_F = networkx.dijkstra_path(bot_1_t.g, str(xy_curr_1), str(x_goal_t))
            except networkx.exception.NetworkXNoPath as e:
                print("\033[1;35;40m" + "[WARNING]: " + str(e) + "\033[0m")
                print("\033[1;35;40m" + "[WARNING]: Perhaps route to goal region is blocked by virtual volume \033[0m")

                Env_t = Env()
                Env_t.obs.add(xy_curr_2)
                bot_1_t = Ts_Grid("unicycle bot 1 t", xy_curr_1, x_goal_t, enviro=Env_t)
                opt_path_F = networkx.dijkstra_path(bot_1_t.g, str(xy_curr_1), str(x_goal_t))

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
                        if in_region(node_str_to_list(list(pos_xy)[i]), start_region) or\
                           in_region(node_str_to_list(list(pos_xy)[i]), goal_region):  # forced, not allow CF to get to goal region
                            break
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

            ''' set end turn '''
            turn += 1

    except KeyError as e:
        # https://blog.csdn.net/ever_peng/article/details/91492491
        print("\033[1;36;40m" + "[ERROR]: " + str(e) + " F is in the expect volume of CF" + "\033[0m")
        pass

    ''' print results '''
    if is_goal_arrived and in_region(xy_curr_1, start_region):
        print("[Info] Vehicle_F wins!")
    elif abs(list(xy_curr_1)[0] - list(xy_curr_2)[0]) <= 1 or abs(list(xy_curr_1)[1] - list(xy_curr_2)[1]) <= 1:
        print("[Info] Vehicle_CF wins!")

    ''' plot path '''
    plt.close()
    plotting.plot_actual_path(actual_path_F, actual_path_CF, x_start_1, x_goal, bot_1.Env, expect_volume_CF)

if __name__ == '__main__':
    main()
