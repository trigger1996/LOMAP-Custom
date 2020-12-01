
import sys
import traceback
import logging
import copy
from lomap.algorithms.product_ca import ts_times_ts
from lomap.algorithms.product_ca import ts_times_ts_ca

def is_collision_within_team(prefix_on_team_ts, suffix_cycle_on_team_ts, is_modifible):
    ###
    ''' Check if collision '''
    prefix_length = prefix_on_team_ts.__len__()
    team_size = 0
    for i in range(0, len(is_modifible)):
        if is_modifible[i]:
            team_size += 1

    #
    is_singleton_collision = False
    is_pairwise_collision = False
    singleton_collision_list = [ [False] * team_size ] * (prefix_length + suffix_cycle_on_team_ts.__len__())
    pairwise_collision_list  = [ [False] * team_size ] * (prefix_length + suffix_cycle_on_team_ts.__len__())

    # singleton_collision
    for i in range(0, prefix_length):
        prefix = list(prefix_on_team_ts[i])
        for j in range(1, prefix.__len__()):
            if prefix[j - 1] == prefix[j] and is_modifible[j]:
                is_singleton_collision = True
                singleton_collision_list[i][j] = True
    for i in range(0, suffix_cycle_on_team_ts.__len__()):
        suffix = list(suffix_cycle_on_team_ts[i])
        for j in range(1, suffix.__len__()):
            if suffix[j - 1] == suffix[j] and is_modifible[j]:
                is_singleton_collision = True
                singleton_collision_list[prefix_length + i][j] = True

    # pairwise_collision
    for i in range(0, prefix_length - 1):
        curr_run = list(prefix_on_team_ts[i])
        next_run = list(prefix_on_team_ts[i + 1])
        for j in range(0, curr_run.__len__()):
            for k in range(0, next_run.__len__()):
                if j != k and curr_run[j] == next_run[k] and is_modifible[j] and is_modifible[k]:
                    is_pairwise_collision = True
                    pairwise_collision_list[i][j] = True
                    pairwise_collision_list[i + 1][k] = True
    for i in range(0, suffix_cycle_on_team_ts.__len__() - 1):
        curr_run = list(suffix_cycle_on_team_ts[i])
        next_run = list(suffix_cycle_on_team_ts[i + 1])
        for j in range(0, curr_run.__len__()):
            for k in range(0, next_run.__len__()):
                if j != k and curr_run[j] == next_run[k] and is_modifible[j] and is_modifible[k]:
                    is_pairwise_collision = True
                    pairwise_collision_list[prefix_length + i][j] = True
                    pairwise_collision_list[prefix_length + i + 1][k] = True

    return is_singleton_collision, is_pairwise_collision, singleton_collision_list, pairwise_collision_list



def search_agent_route(cur_ts, cur_state, weight_limit = 3, node_limit = 2, is_weight_based =  True):
    if is_weight_based:
        current_level_tgt = [cur_state]
        next_level_tgt = []
        target_node = []
        route = []
        last_route = []

        while True:
            for state in current_level_tgt:
                next_state_arr = cur_ts.next_states_of_wts(state, traveling_states=False)
                list_t = []
                for j in range(0, next_state_arr.__len__()):
                    node_t = [next_state_arr[j][0], cur_ts.g.edge[state][next_state_arr[j][0]][0]['weight'], state] # [node_label, edge_weight, father_node]
                    list_t.append(node_t)
                next_level_tgt = next_level_tgt + list_t
                target_node = target_node + next_level_tgt

            # generate route
            last_route = route
            route = []
            for state_to_add in next_level_tgt:
                father = state_to_add[2]
                father_path = None
                for temp in last_route:
                    if father == temp[0][0]:
                        father_path = []
                        father_path = copy.deepcopy(temp)
                        break
                # trace back and add a new route
                if not isinstance(father_path, list):
                    route_t = []
                    weight  = 0
                    state_t = state_to_add
                    while True:
                        father = state_t[2]
                        route_t.append(state_t[0])
                        weight = weight + state_t[1]
                        if father == cur_state:  # initial state to search
                            route_t.append(father)
                            break
                        for temp in target_node:
                            if father == temp[0]:
                                state_t = temp
                                break
                    route.append([route_t, weight])
                else:
                    father_path[0].insert(0, state_to_add[0])
                    father_path[1] = father_path[1] + state_to_add[1]
                    route_t = father_path
                    route.append(route_t)


            # check if all route weight is larger than weight limit
            is_continue = False
            for route_t in route:
                if route_t[1] < weight_limit:
                    is_continue = True
            if not is_continue:
                ''' remove point with EXCEEDED POINTS '''
                route_to_return = []
                for route_t in route:
                    exceeded_weight = 0
                    pop_index = -1
                    for i in range(1, route_t[0].__len__()):
                        if route_t[1] - exceeded_weight <= weight_limit:
                            break
                        curr_node = route_t[0][i - 1]
                        last_node = route_t[0][i]
                        exceeded_weight = exceeded_weight + cur_ts.g.edge[last_node][curr_node][0]['weight']
                        pop_index = i - 1
                    if pop_index >= 0:
                        for j in range(0, pop_index + 1):
                            route_t[0].pop(0)
                        route_t[1] = route_t[1] - exceeded_weight
                    route_to_return.append(route_t)
                break

            # extract next level target
            current_level_tgt = []
            for tgt in next_level_tgt:
                current_level_tgt.append(tgt[0])
            next_level_tgt = []

        # return
        target_node = []
        for route_t in route_to_return:
            for i in range(0, route_t[0].__len__()):
                is_found = False
                for j in range(0, target_node.__len__()):
                    if target_node[j] == route_t[0][i]:
                        is_found = True
                        break
                if not is_found:
                    target_node.append(route_t[0][i])

        return target_node

    elif not is_weight_based:

        current_level_tgt = [cur_state]
        next_level_tgt = []
        target_node = []

        for i in range(0, node_limit):
            for state in current_level_tgt:
                next_state_arr = cur_ts.next_states_of_wts(state, traveling_states=False)
                list_t = []
                for j in range(0, next_state_arr.__len__()):
                    list_t.append(next_state_arr[j][0])
                next_level_tgt = next_level_tgt + list_t
                target_node = target_node + next_level_tgt
            current_level_tgt = next_level_tgt
            next_level_tgt = []
        # move identical elements
        for s in target_node:
            if target_node.count(s) > 1:
                target_node.remove(s)

    return target_node

def safety_game(friendly_ts, foe_ts, friendly_state, foe_state, cur_team_run):
    '''
        Due to the foe ts is known, construct team ts is not realizable
        Assume it that the velocity is not faster than friendly
    '''
