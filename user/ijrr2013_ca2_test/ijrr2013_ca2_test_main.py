import networkx
import logging

from lomap import Ts


run_1 = [('u1', 'u2', '11'),  ('4', '10', ('11', '12', 2)),  ('5', '11', ('11', '12', 3)),  ('27', ('11', '12', 1), '12'),  (('27', '28', 1), ('11', '12', 2), '1'),  (('27', '28', 2), ('11', '12', 3), '2'),  ('28', '12', '21'),  (('28', '21', 1), '1', ('21', '22', 1)),  (('28', '21', 2), '2', '22'),  ('21', '21', ('22', '23', 1)),  (('21', '22', 1), ('21', '22', 1), '23'),  ('22', '22', '9'),  ('g1', 'g1', '10'),  (('g1', '22', 1), ('g1', '22', 1), '11'),  ('22', '22', ('11', '12', 1)),  ('23', '23', ('11', '12', 3)),  ('9', '9', '12'),  ('10', '10', '1'),  ('u2', 'u2', '2'),  (('u2', '10', 1), ('u2', '10', 1), '21'),  ('10', '10', ('21', '22', 1)),  ('11', '11', '22'),  (('11', '12', 2), ('11', '12', 2), '23'),  (('11', '12', 3), ('11', '12', 3), '9'),  ('12', '12', '10'),  ('1', '1', '11'),  ('2', '2', ('11', '12', 1)),  ('21', '21', ('11', '12', 2)),  ('22', '22', '12'),  ('g1', 'g1', '1'),  (('g1', '22', 1), ('g1', '22', 1), '2'),  ('22', '22', '21'),  ('23', '23', '22'),  ('9', '9', ('22', '23', 1)),  ('10', '10', '23'),  ('u2', 'u2', '9'),  (('u2', '10', 1), ('u2', '10', 1), '10'),  ('10', '10', '11'),  ('11', '11', ('11', '12', 1)),  (('11', '12', 3), ('11', '12', 3), '12'),  ('12', '12', '1'),  ('1', '1', '2'),  ('2', '2', '21'),  ('21', '21', ('21', '22', 1)),  (('21', '22', 1), ('21', '22', 1), '22'),  ('22', '22', ('22', '23', 1)),  ('g1', 'g1', '23'),  (('g1', '22', 1), ('g1', '22', 1), '9'),  ('22', '22', '10'),  (('22', '23', 1), ('22', '23', 1), '11'),  ('23', '23', ('11', '12', 1)),  ('9', '9', ('11', '12', 2)),  ('10', '10', ('11', '12', 3)),  ('u2', 'u2', '12'),  (('u2', '10', 1), ('u2', '10', 1), '1'),  ('10', '10', '2'),  ('11', '11', '21'),  (('11', '12', 2), ('11', '12', 2), '22'),  ('12', '12', '23'),  ('1', '1', '9'),  ('2', '2', '10'),  ('21', '21', '11'),  ('22', '22', ('11', '12', 2)),  ('g1', 'g1', ('11', '12', 3)),  (('g1', '22', 1), ('g1', '22', 1), '12'),  ('22', '22', '1'),  (('22', '23', 1), ('22', '23', 1), '2'),  ('23', '23', '21'),  ('9', '9', ('21', '22', 1)),  ('10', '10', '22'),  ('u2', 'u2', ('22', '23', 1)),  (('u2', '10', 1), ('u2', '10', 1), '23'),  ('10', '10', '9'),  ('11', '11', '10'),  (('11', '12', 1), ('11', '12', 1), '11'),  ('12', '12', ('11', '12', 3)),  ('1', '1', '12'),  ('2', '2', '1'),  ('21', '21', '2'),  (('21', '22', 1), ('21', '22', 1), '21'),  ('22', '22', ('21', '22', 1)),  ('g1', 'g1', '22'),  ('22', '22', '23'),  (('22', '23', 1), ('22', '23', 1), '9'),  ('23', '23', '10'),  ('9', '9', '11'),  ('10', '10', ('11', '12', 1)),  ('u2', 'u2', ('11', '12', 2)),  ('10', '10', '12'),  ('11', '11', '1'),  (('11', '12', 1), ('11', '12', 1), '2'),  (('11', '12', 2), ('11', '12', 2), '21'),  ('12', '12', '22'),  ('1', '1', ('22', '23', 1)),  ('2', '2', '23'),  ('21', '21', '9'),  (('21', '22', 1), ('21', '22', 1), '10'),  ('22', '22', '11'),  ('g1', 'g1', ('11', '12', 1)),  ('22', '22', ('11', '12', 3)),  (('22', '23', 1), ('22', '23', 1), '12'),  ('23', '23', '1'),  ('9', '9', '2'),  ('10', '10', '21'),  ('u2', 'u2', ('21', '22', 1)),  (('u2', '10', 1), ('u2', '10', 1), '22'),  ('10', '10', ('22', '23', 1)),  ('11', '11', '23'),  (('11', '12', 1), ('11', '12', 1), '9'),  (('11', '12', 2), ('11', '12', 2), '10'),  (('11', '12', 3), ('11', '12', 3), '11'),  ('12', '12', ('11', '12', 1)),  ('1', '1', ('11', '12', 2)),  ('2', '2', ('11', '12', 3)),  ('21', '21', '12'),  (('21', '22', 1), ('21', '22', 1), '1'),  ('22', '22', '2'),  ('g1', 'g1', '21'),  ('22', '22', '22'),  ('23', '23', '23'),  ('9', '9', '9'),  ('10', '10', '10'),  ('u2', 'u2', '11'),  ('10', '10', ('11', '12', 2)),  ('11', '11', ('11', '12', 3)),  (('11', '12', 1), ('11', '12', 1), '12'),  (('11', '12', 2), ('11', '12', 2), '1'),  (('11', '12', 3), ('11', '12', 3), '2'),  ('12', '12', '21'),  ('1', '1', ('21', '22', 1)),  ('2', '2', '22'),  ('21', '21', ('22', '23', 1))]

run_4_vehicles_case_2 = [('u1', 'u2', '2', '6'), ('4', '10', '21', '7'), ('5', 'u2', '12', '8'), ('27', '10', '1', '25'), (('27', '28', 2), '11', '2', '6'), ('28', ('11', '23', 1), ('2', '21', 1), ('6', '7', 1)), (('28', '21', 1), '23', '21', '7'), ('21', '9', '12', '8'), (('21', '22', 2), '10', '1', '25'), ('22', ('10', '11', 1), ('1', '2', 1), ('25', '6', 1)), (('22', '23', 1), '11', '2', '6'), ('23', '23', '21', '7'), (('23', '24', 2), ('23', '24', 2), '12', '8'), ('24', '24', ('12', '1', 1), ('8', '25', 1)), ('g2', 'g2', '1', '25'), ('24', '24', ('1', '2', 1), ('25', '6', 1)), (('24', '25', 1), ('24', '25', 1), '2', '6'), ('25', '25', '21', '7'), (('25', '26', 2), ('25', '26', 2), '12', '8'), ('26', '26', ('12', '1', 1), ('8', '25', 1)), (('26', '27', 1), ('26', '27', 1), '1', '25'), ('27', '27', '2', '6'), ('3', '3', '21', '7'), ('4', '4', '12', '8'), ('u1', 'u1', '1', '25'), ('4', '4', '2', '6'), ('5', '5', '21', '7'), ('27', '27', '12', '8'), (('27', '28', 2), ('27', '28', 2), '1', '25'), ('28', '28', ('1', '2', 1), ('25', '6', 1)), ('g4', 'g4', '2', '6'), ('28', '28', ('2', '21', 1), ('6', '7', 1)), (('28', '21', 1), ('28', '21', 1), '21', '7'), ('21', '21', '12', '8'), (('21', '22', 2), ('21', '22', 2), '1', '25'), ('22', '22', ('1', '2', 1), ('25', '6', 1)), (('22', '23', 1), ('22', '23', 1), '2', '6'), ('23', '23', '21', '7'), ('9', '9', '12', '8'), ('10', '10', '1', '25'), ('u2', 'u2', '2', '6'), ('10', '10', '21', '7'), ('11', '11', '12', '8'), ('23', '23', '1', '25'), (('23', '24', 2), ('23', '24', 2), '2', '6'), ('24', '24', ('2', '21', 1), ('6', '7', 1)), ('g2', 'g2', '21', '7'), ('24', '24', ('21', '12', 1), ('7', '8', 1)), (('24', '25', 1), ('24', '25', 1), '12', '8'), ('25', '25', '1', '25'), (('25', '26', 2), ('25', '26', 2), '2', '6'), ('26', '26', ('2', '21', 1), ('6', '7', 1)), (('26', '27', 1), ('26', '27', 1), '21', '7'), ('27', '27', '12', '8'), ('3', '3', '1', '25'), ('4', '4', '2', '6'), ('u1', 'u1', '21', '7'), ('4', '4', '12', '8'), ('5', '5', '1', '25'), ('27', '27', '2', '6'), (('27', '28', 2), ('27', '28', 2), '21', '7'), ('28', '28', ('21', '12', 1), ('7', '8', 1)), ('g4', 'g4', '12', '8'), ('28', '28', ('12', '1', 1), ('8', '25', 1)), (('28', '21', 1), ('28', '21', 1), '1', '25'), ('21', '21', '2', '6'), (('21', '22', 2), ('21', '22', 2), '21', '7'), ('22', '22', ('21', '12', 1), ('7', '8', 1)), (('22', '23', 1), ('22', '23', 1), '12', '8'), ('23', '23', '1', '25'), ('9', '9', '2', '6'), ('10', '10', '21', '7'), ('u2', 'u2', '12', '8'), ('10', '10', '1', '25'), ('11', '11', '2', '6'), ('23', '23', '21', '7')]


# Logger configuration
logger = logging.getLogger(__name__)

min_cost = 1
go_back_additional_cost = 1

def is_traveling_state(curr_run):
    if type(curr_run) == str:
        return False        # str
    else:
        return True         # traveling state: tuple


def is_traveling_states_intersect(state_1, state_2):
    if is_traveling_state(state_1) and is_traveling_state(state_2):
        state_1_src = list(state_1)[0]
        state_1_dst = list(state_1)[1]

        state_2_src = list(state_2)[0]
        state_2_dst = list(state_2)[1]
        if (state_1_src == state_2_src and state_1_dst == state_2_dst) or \
           (state_1_dst == state_2_src and state_1_src == state_2_dst):
           return True
    else:
        return False

def find_last_non_traveling_state(team_run, agent_id, current_run_seq):
    # find last indivdual state expect for travelling
    for k in range(1, current_run_seq):
        team_state_last = list(team_run[current_run_seq - k])
        if not is_traveling_state(team_state_last[agent_id]):
            return [team_state_last, current_run_seq - k]
    return [None, None]

def find_next_non_traveling_state(team_run, agent_id, current_run_seq):
    # find next indivdual state expect for travelling
    for k in range(1, team_run.__len__() - current_run_seq):
        team_state_next = list(team_run[current_run_seq + k])
        if not is_traveling_state(team_state_next[agent_id]):
            return [team_state_next, current_run_seq + k]
    return [None, None]

def check_remove_collisions(ts_tuple, is_modifible, team_run):
    ###
    ''' Check if collision '''
    '''
    # pop the start of suffix to remove repetance
    suffix_cycle_on_team_ts.pop(0)
    team_run = list(prefix_on_team_ts + suffix_cycle_on_team_ts)

    # according to definitions, collision should always be identified in those non-travelling points
    to_pop = []
    # calculate those all-travelling-state points
    for i in range(0, team_run.__len__()):
        travelling_state_num = 0
        for j in range(0, ts_tuple.__len__()):
            if is_traveling_state(team_run[i][j]):
                travelling_state_num += 1
        if travelling_state_num == ts_tuple.__len__():
            to_pop.append(team_run[i])
    # remove from team run
    for i in range(0, to_pop.__len__()):
        team_run.remove(to_pop[i])
    '''

    #
    is_singleton_collision = False
    is_pairwise_collision  = False
    is_rear_end_collision  = False
    singleton_collision_list = [ [False for i in range(ts_tuple.__len__())] for j in range(team_run.__len__()) ]
    pairwise_collision_list  = [ [False for i in range(ts_tuple.__len__())] for j in range(team_run.__len__()) ]
    rear_end_collision_list  = [[False for i in range(ts_tuple.__len__())] for j in range(team_run.__len__())]

    ''' singleton_collision '''
    num_singleton_collision = 0
    for i in range(0, team_run.__len__()):
        curr_run = list(team_run[i])
        for j in range(0, curr_run.__len__()):
            for k in range(0, curr_run.__len__()):
                if j != k and curr_run[j] == curr_run[k] and not is_traveling_state(curr_run[j]):
                    is_singleton_collision = True
                    singleton_collision_list[i][j] = True
                    singleton_collision_list[i][k] = True
                    num_singleton_collision += 1

    ''' pairwise_collision '''
    num_pairwise_collision = 0
    for i in range(0, team_run.__len__()):
        for j in range(0, ts_tuple.__len__()):              # agent j
            # find current non-travelling state for agent j
            curr_run = list(team_run[i])
            curr_run_j = curr_run[j]
            if is_traveling_state(curr_run_j):
                # if current run is traveling state

                for k in range(0, ts_tuple.__len__()):
                    if k == j:
                        continue
                    curr_run_k = curr_run[k]
                    if is_traveling_state(curr_run_k):
                        l = i + 1
                        next_run = list(team_run[l])
                        next_run_j = next_run[j]
                        next_run_k = next_run[k]
                        # find the next, non-travelling state for agent k
                        if is_traveling_state(next_run_j) and is_traveling_state(next_run_k):     # find next traveling state
                            if is_traveling_states_intersect(curr_run_j, next_run_k) and \
                               is_traveling_states_intersect(curr_run_k, next_run_j):
                                # now can confirm pairwise collision
                                [last_run_j_nt, last_seq_j] = find_last_non_traveling_state(team_run, j, i)
                                [last_run_k_nt, last_seq_k] = find_last_non_traveling_state(team_run, k, i)
                                [next_run_j_nt, next_seq_j] = find_next_non_traveling_state(team_run, j, i)
                                [next_run_k_nt, next_seq_k] = find_next_non_traveling_state(team_run, k, i)

                                if next_run_k_nt[k] == last_run_j_nt[j] and next_run_j_nt[j] == last_run_k_nt[k]:
                                    if last_run_j_nt != None and last_run_k_nt != None and next_run_j_nt != None and next_run_k_nt != None:
                                        is_pairwise_collision = True
                                        pairwise_collision_list[last_seq_j][j] = [next_run_k_nt[k], next_seq_k, k]
                                        pairwise_collision_list[last_seq_k][k] = [next_run_j_nt[j], next_seq_j, j]
                                        pairwise_collision_list[next_seq_j][j] = [last_run_k_nt[k], last_seq_k, k]
                                        pairwise_collision_list[next_seq_k][k] = [last_run_j_nt[j], last_seq_j, j]
                                        num_pairwise_collision += 1
            else:
                # if current run of both agent is actual state:

                for k in range(0, ts_tuple.__len__()):  # agent k
                    if k == j:
                        continue
                    curr_run_k = curr_run[k]
                    if not is_traveling_state(curr_run_k):
                        [next_run_j, next_seq_j] = find_next_non_traveling_state(team_run, j, i)
                        [next_run_k, next_seq_k] = find_next_non_traveling_state(team_run, k, i)
                        if next_run_j != None and next_run_k != None:
                            if curr_run_k == next_run_j[j] and next_run_k[k] == curr_run_j:   # next_seq_k == next_seq_j
                                is_pairwise_collision = True
                                pairwise_collision_list[i][j] = [next_run_k[k], next_seq_k, k]
                                pairwise_collision_list[i][k] = [next_run_j[j], next_run_j, j]
                                pairwise_collision_list[next_seq_j][j] = [curr_run_k, i, k]
                                pairwise_collision_list[next_seq_k][k] = [curr_run_j, i, j]
                                num_pairwise_collision += 1

    ''' rear-end collisions '''
    num_rear_end_collision = 0
    for i in range(0, team_run.__len__()):
        for j in range(0, ts_tuple.__len__()):              # agent j
            # find current non-travelling state for agent j
            curr_run = list(team_run[i])
            if is_traveling_state(curr_run[j]):
                # if current run is traveling state
                for k in range(0, ts_tuple.__len__()):          # agent k
                    if k == j:
                        continue
                    if not is_traveling_state(curr_run[k]) and list(curr_run[j])[0] == curr_run[k]:
                        # if the origin of traveling state is equal to the other non-traveling state
                        [last_run_j_nt, last_seq_j] = find_last_non_traveling_state(team_run, j, i)
                        [run_j, seq_j] = find_next_non_traveling_state(team_run, j, i)      # the end of the traveling state
                        [run_k, seq_k] = find_next_non_traveling_state(team_run, k, i)      # next non-traveling state of agent k from i

                        if run_j != None and run_k != None and last_run_j_nt != None and \
                           list(curr_run[j])[1] == run_k[k] and seq_j > seq_k:
                            # check wether agent k arrives the end earlier than agnent j
                            is_rear_end_collision = True
                            rear_end_collision_list[i][j] = [curr_run[k],      i,          k, 'lo']   # [the_state_of_the_agents_collited, state_seq_of_the_other_agents, the_other_agent_id, speed_of_the_current_agent]
                            pairwise_collision_list[i][k] = [last_run_j_nt[j], last_seq_j, j, 'hi']
                            rear_end_collision_list[seq_j][j] = [run_k, seq_k, k, 'lo']
                            rear_end_collision_list[seq_k][k] = [run_j, seq_j, j, 'hi']
                            num_rear_end_collision += 1  # only a half need to be done, because the other half will be completed in the whole process, just Fix this below

    print('[collision] Number singleton collision: ', num_singleton_collision)
    print('[collision] Number pairwise  collision: ', num_pairwise_collision)
    print('[collision] Number rear-end  collision: ', num_rear_end_collision)

    ''' singleton_collision '''
    if is_singleton_collision:
        # add stay motion for collision points
        for i in range(0, singleton_collision_list.__len__()):
            for j in range(0, ts_tuple.__len__()):
                if singleton_collision_list[i][j] == True:
                    team_state_curr = list(team_run[i])
                    team_state_last = None
                    team_state_next = None

                    # find last indivdual state expect for travelling
                    for k in range(1, i):
                        team_state_last = list(team_run[i - k])
                        if not is_traveling_state(team_state_last[j]):
                            break
                    # find next indivdual state expect for travelling
                    for k in range(1, team_run.__len__() - i):
                        team_state_next = list(team_run[i + k])
                        if not is_traveling_state(team_state_next[j]):
                            break

                    if not is_traveling_state(team_state_curr[j]) and is_modifible[j]:
                        if is_modifible[j]:
                            if team_state_last != None:
                                # avoid adding the same edge to reduce states
                                if ts_tuple[j].g.edge[team_state_last[j]].get(team_state_last[j]) == None:
                                    ts_tuple[j].g.add_edge(team_state_last[j], team_state_last[j],
                                                           attr_dict={'weight': min_cost, 'control': 's'})
                            if team_state_next != None:
                                if ts_tuple[j].g.edge[team_state_next[j]].get(team_state_next[j]) == None:
                                    ts_tuple[j].g.add_edge(team_state_next[j], team_state_next[j],
                                                           attr_dict={'weight': min_cost, 'control': 's'})

    ''' pairwise_collision '''
    if is_pairwise_collision:
        # add turn-back points
        for i in range(0, pairwise_collision_list.__len__()):
            for j in range(0, ts_tuple.__len__()):
                if pairwise_collision_list[i][j] != False:

                    team_state_curr = list(team_run[i])
                    team_state_last = None
                    team_state_next = None

                    # find last indivdual state expect for travelling
                    for k in range(1, i):
                        team_state_last = list(team_run[i - k])
                        if not is_traveling_state(team_state_last[j]):
                            break
                    # find next indivdual state expect for travelling
                    for k in range(1, team_run.__len__() - i):
                        team_state_next = list(team_run[i + k])
                        if not is_traveling_state(team_state_next[j]):
                            break

                    if is_modifible[j]:
                        ''' FIRST, add go-back points '''
                        # find ALL edges to target state
                        go_back_list = []       # for go_back_list[i], [0] for node and [1] for cost
                        for u in ts_tuple[j].g.edge:
                            if ts_tuple[j].g.edge[u].get(team_state_last[j]) != None:
                                go_back_list.append([u, ts_tuple[j].g.edge[u].get(team_state_last[j])[0]['weight'] + additional_goback_cost])   # record point and corresponding weight

                        min_cost_index = 0
                        for k in range(0, go_back_list.__len__()):
                            if go_back_list[k][1] <= go_back_list[min_cost_index][1]:
                                min_cost_index = k

                        # if the go back edge does not exist, add it
                        if ts_tuple[j].g.edge[team_state_last[j]].get(go_back_list[min_cost_index][0]) == None:
                            ts_tuple[j].g.add_edge(team_state_last[j], go_back_list[min_cost_index][0],
                                                   attr_dict={'weight': go_back_list[min_cost_index][1], 'control': 'go_back'})

                        ''' SECOND, add wait points '''
                        if team_state_next != None:
                            if ts_tuple[j].g.edge[team_state_next[j]].get(team_state_next[j]) == None:
                                ts_tuple[j].g.add_edge(team_state_next[j], team_state_next[j],
                                                       attr_dict={'weight': min_cost, 'control': 's'})

def main():
    r1 = Ts.load('./robot_1.yaml')  # robot_1_real.yaml
    r2 = Ts.load('./robot_2.yaml')  # robot_2_real.yaml
    r3 = Ts.load('./robot_3.yaml')  # robot_3_real.yaml

    # CASE 2
    ts_tuple = (r1, r2, r3)
    is_modifible = [True, True, False]
    formula = ('[]<>gather && [](gather->(r1gather && r2gather)) '
               '&& [](r1gather -> X(!r1gather U r1upload)) '
               '&& [](r2gather -> X(!r2gather U r2upload))')
    opt_prop = set(['r1gather', 'r2gather'])

    check_remove_collisions(ts_tuple, is_modifible, run_1)

if __name__ == '__main__':
    main()
    print(run_1)
