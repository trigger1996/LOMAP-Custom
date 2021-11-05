#!/usr/bin/env python
# coding=utf-8

import lomap
import networkx
from lomap import Ts

class incremental_A_star():
    def __init__(self, graph, start, goal, h_start = 0):
        self.U = []

        self.graph = graph
        self.g = dict()
        self.h = dict()
        self.rhs = dict()
        self.start = start
        self.goal  = goal

        self.h_start = h_start

        self.re_init(graph)

    def re_init(self, graph):
        for state in graph.node:
            self.g[state]   = 1e6
            self.rhs[state] = 1e6
        self.rhs[self.start] = 0
        self.U.append([self.start, [self.h_start, 0]])

    def compute_shortest_path(self):
        while self.topkey() < self.calculate_key(self.goal) or \
              self.rhs[self.goal] != self.g[self.goal]:
            u = self.U.pop(0)
            if self.g[u] > self.rhs[u]:
                self.g[u] = self.rhs[u]

    def topkey(self):
        if self.U.__len__() == 0:
            return [1e6, 1e6]
        else:
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

        for u in self.U:
            if u[0] == state:
                self.U.remove(u)
                break

        if self.g[state] != self.rhs[state]:
            self.U.append(state, self.calculate_key(state))

def main():
    graph_1 = Ts.load('./graph_mit.yaml')

    graph_1_a_star = incremental_A_star(graph_1.g, 'S', 'G')
    #graph_1_a_star.compute_shortest_path()
    graph_1_a_star.update_vertex('A')

    print(233)

if __name__ == '__main__':
    main()
