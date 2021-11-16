#!/usr/bin/env python
# coding=utf-8

import lomap
import networkx
from lomap import Ts
import copy

def sort_U(elem):
    return elem[1][0]

class incremental_A_star():
    def __init__(self, graph, start, goal, is_with_h = False, h_start = 0):
        self.U = []

        self.graph = graph
        self.g = dict()
        self.h = dict()
        self.rhs = dict()
        self.start = start
        self.goal  = goal

        self.predecessor = dict()

        self.path = []

        if not is_with_h:
            for state in graph.node.keys():
                self.h[state] = 0
        self.h_start = h_start

        self.re_init(graph)

    def re_init(self, graph):
        for state in graph.node:
            self.g[state]   = 1e6
            self.rhs[state] = 1e6

            self.predecessor[state]  = None

        self.rhs[self.start] = 0
        self.U.append([self.start, [self.h_start, 0]])

    def cycle(self, change):
        '''
        :param change:
                [from, to, cost] or
                [[from_1, to_1, cost_1], [from_2, to_2, cost_2], ...]
        :return:
        '''
        self.path = []

        if type(change[0]) != list:
            change = [change]

        for change_t in change:
            self.graph.edge[change_t[0]][change_t[1]][0]['weight'] = change_t[2]
            self.update_vertex(change_t[1])

        self.compute_shortest_path()


    def compute_shortest_path(self):

        while self.topkey() < self.calculate_key(self.goal) or \
              self.rhs[self.goal] != self.g[self.goal]:
            u = self.U.pop(0)[0]
            if self.g[u] > self.rhs[u]:
                self.g[u] = self.rhs[u]
                for state in self.graph.succ[u]:
                    self.update_vertex(state)
            else:
                self.g[u] = 1e6
                self.update_vertex(u)
                for state in self.graph.succ[u]:
                    self.update_vertex(state)

    def topkey(self):
        if self.U.__len__() == 0:
            return [1e6, 1e6]
        else:
            self.U.sort(key=sort_U)
            return self.U[0][1]

    def calculate_key(self, state):
        g_s   = self.g[state]
        rhs_s = self.rhs[state]
        return [min(g_s, rhs_s) + self.h[state], min(g_s, rhs_s)]

    def update_vertex(self, state):
        if state != self.start:
            min_val = 1e6
            min_index = None
            for pred_state in self.graph.pred[state]:
                if min_val > self.graph.pred[state][pred_state][0]['weight']:
                    min_val = self.graph.pred[state][pred_state][0]['weight']
                    min_index = pred_state
            self.rhs[state] = self.g[min_index] + min_val
            
            self.predecessor[state] = min_index

        for u in self.U:
            if u[0] == state:
                self.U.remove(u)
                break

        if self.g[state] != self.rhs[state]:
            self.U.append([state, self.calculate_key(state)])

    def extract_path(self):
        self.path.append(self.goal)
        state = self.goal

        while state != self.start:
            self.path.append(self.predecessor[state])
            state = self.predecessor[state]

        self.path = self.path[::-1]
        return copy.deepcopy(self.path)

def main():
    graph_1 = Ts.load('./graph_mit.yaml')

    graph_1_a_star = incremental_A_star(graph_1.g, 'S', 'G')

    graph_1_a_star.compute_shortest_path()
    path_t = graph_1_a_star.extract_path()

    graph_1_a_star.cycle(['B', 'D', 100])
    path_2 = graph_1_a_star.extract_path()

    graph_1_a_star.graph.add_edge('C', 'D', attr_dict={'weight': 3, 'control': 'u'})
    graph_1_a_star.cycle([['B', 'D', 100], ['A', 'D', 100]])
    path_3 = graph_1_a_star.extract_path()

    print(path_t)
    print(path_2)
    print(path_3)

    print(233)

if __name__ == '__main__':
    main()
