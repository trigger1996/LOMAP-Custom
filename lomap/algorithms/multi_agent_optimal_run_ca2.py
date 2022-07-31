#! /usr/bin/python

# ! /usr/bin/python

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

from __future__ import print_function

import sys
import traceback
import logging
from lomap.algorithms.product_ca import ts_times_ts
from lomap.algorithms.product_ca import ts_times_ts_ca
from lomap.algorithms.optimal_run2 import optimal_run as optimal_run_pp

# Logger configuration
logger = logging.getLogger(__name__)

try:
    import pp

    pp_installed = True
except ImportError:
    pp_installed = False
import networkx as nx

from lomap.classes import Buchi
from lomap.algorithms.product import ts_times_buchi
from lomap.algorithms.dijkstra import (source_to_target_dijkstra,
                                       subset_to_subset_dijkstra_path_value)

import igraph as ig
import copy

# Logger configuration
logger = logging.getLogger(__name__)
# logger.addHandler(logging.NullHandler())

# Cluster configuration
# SERVER_ADDR is the ip address of the computer running lomap
# pp_servers is a tuple of server ips that are running ppserver.py
# SERVER_ADDR = '107.20.62.59'father_path
# pp_servers = ('23.22.80.26', '50.17.177.92', '50.16.82.182', '50.16.116.126')
SERVER_ADDR = 'localhost'
pp_servers = ()
SERVER_PORT = 60001
data_source = (SERVER_ADDR, SERVER_PORT)


def optimal_run(t, formula, opt_prop):
    try:
        logger.info('T has %d states', len(t.g))
        # Convert formula to Buchi automaton
        b = Buchi()
        b.from_formula(formula)
        logger.info('B has %d states', len(b.g))
        # Compute the product automaton
        p = ts_times_buchi(t, b)
        logger.info('P has %d states', len(p.g))
        logger.info('Set F has %d states', len(p.final))
        # Find the set S of states w/ opt_prop
        s = p.nodes_w_prop(opt_prop)
        logger.info('Set S has %d states', len(s))
        # Compute the suffix_cycle* and suffix_cycle_cost*
        suffix_cycle_cost, suffix_cycle_on_p = min_bottleneck_cycle(p.g, s, p.final)
        # Compute the prefix: a shortest path from p.init to suffix_cycle
        prefix_length = float('inf')
        prefix_on_p = ['']
        i_star = 0
        for init_state in p.init.keys():
            for i in range(0, len(suffix_cycle_on_p)):
                length, prefix = source_to_target_dijkstra(p.g, init_state, suffix_cycle_on_p[i], degen_paths=True)
                if (length < prefix_length):
                    prefix_length = length
                    prefix_on_p = prefix
                    i_star = i

        if (prefix_length == float('inf')):
            raise Exception(__name__, 'Could not compute the prefix.')

        # Wrap suffix_cycle_on_p as required
        if i_star != 0:
            # Cut and paste
            suffix_cycle_on_p = suffix_cycle_on_p[i_star:] + suffix_cycle_on_p[1:i_star + 1]

        # Compute projection of prefix and suffix-cycle to T and return
        suffix_cycle = [x[0] for x in suffix_cycle_on_p]
        prefix = [x[0] for x in prefix_on_p]
        return (prefix_length, prefix, suffix_cycle_cost, suffix_cycle)
    except Exception as ex:
        if (len(ex.args) == 2):
            print("{}: {}".format(*ex.args))
        else:
            print("{}: Unknown exception {}: {}".format(__name__, type(ex), ex))
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_tb(exc_traceback)
            exit(1)


def find_best_cycle(f, s, d_f_to_s, d_s_to_f, d_bot):
    import itertools

    cost_star = float('inf')
    len_star = float('inf')
    cycle_star = None
    for triple in itertools.product(f, s, s):
        (ff, s1, s2) = triple
        f_s_cycle_cost = d_f_to_s[ff][s1] + d_s_to_f[s2][ff]
        # Cost and length of this triple
        if s1 == s2 and ff != s1:
            cost = f_s_cycle_cost
            this_len = f_s_cycle_cost
        else:
            cost = max(f_s_cycle_cost, d_bot[s1][s2][0])
            this_len = f_s_cycle_cost + d_bot[s1][s2][1]

        if (cost < cost_star or (cost == cost_star and this_len < len_star)):
            cost_star = cost
            len_star = this_len
            cycle_star = triple

    return (cost_star, len_star, cycle_star)

def ig_node_2_index(g, node_name):
    return g.vs.find(nx_name=node_name).index

def find_best_cycle_ig(f, s, d_f_to_s, d_s_to_f, d_bot):
    import itertools

    cost_star = float('inf')
    len_star = float('inf')
    cycle_star = None

    triple_list = itertools.product(f, s, s)

    for triple in triple_list:
        (ff, s1, s2) = triple
        # data structure of d_f_to_s and d_s_to_f is DIFFERENT
        f_s_cycle_cost = d_f_to_s[f.index(ff)][s.index(s1)] + d_s_to_f[s2][ff]                      # d_s_to_f[s.index(s2)][f.index(ff)]

        # Cost and length of this triple
        if s1 == s2 and ff != s1:
            cost = f_s_cycle_cost
            this_len = f_s_cycle_cost
        else:
            #cost = max(f_s_cycle_cost, d_bot[s.index(s1)][s.index(s2)][0])
            #this_len = f_s_cycle_cost + d_bot[s.index(s1)][s.index(s2)][1]
            cost = max(f_s_cycle_cost, d_bot[s1][s2][0])
            this_len = f_s_cycle_cost + d_bot[s1][s2][1]

        if (cost < cost_star or (cost == cost_star and this_len < len_star)):
            cost_star = cost
            len_star = this_len
            cycle_star = triple

    return (cost_star, len_star, cycle_star)

def chunks(l, n):
    return [l[i:i + n] for i in range(0, len(l), n)]


def job_worker(chunk, data_source, func_name):
    global data_id
    global my_data
    import socket
    import pickle

    need_data = False

    # Get new_data_id
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(data_source)
    sock.sendall('data_id')
    new_data_id = sock.recv(64)
    sock.close()

    if 'data_id' in globals():
        if new_data_id != data_id:
            need_data = True
    else:
        need_data = True

    if need_data:  # or 'my_data' not in globals():
        # Get new data from data_address
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(data_source)
        sock.sendall('data')
        new_data = ''
        while True:
            this_data = sock.recv(8192)
            if not this_data:
                break
            new_data += this_data
        sock.close()
        my_data = pickle.loads(new_data)
        # set the data_id
        data_id = new_data_id

    # Call the dijkstra routine for this source_chunk
    return eval(func_name + '(chunk, *my_data)')


def job_dispatcher(job_server, func, arg_to_split, chunk_size, data_id, data, data_source):
    import socket
    import threading
    from six.moves import socketserver
    import pickle

    pickled_data = pickle.dumps(data, pickle.HIGHEST_PROTOCOL)

    # Data Server Configuration
    class ThreadedTCPRequestHandler(socketserver.BaseRequestHandler):
        def handle(self):
            req = self.request.recv(32)
            if req == 'data_id':
                self.request.sendall(data_id)
                # logger.info('data_id req received from %s', self.client_address)
            elif req == 'data':
                self.request.sendall(pickled_data)
                logger.info('Served dataset %s to %s', data_id, self.client_address)
            else:
                assert (False)

    class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
        allow_reuse_address = True
        pass

    # Start the server
    server = ThreadedTCPServer(('0.0.0.0', data_source[1]), ThreadedTCPRequestHandler)
    server_thread = threading.Thread(target=server.serve_forever)
    # Exit the server thread when the main thread terminates
    server_thread.daemon = True
    server_thread.start()

    # Array of dispatched jobs
    jobs = []
    # Split sl to chunks of size chunk_size
    arg_chunks = chunks(arg_to_split, chunk_size)
    # Dispatch jobs, job_worker will load static data from data_source
    for chunk in arg_chunks:
        jobs.append(job_server.submit(job_worker, (chunk, data_source, func.__name__), (func,)))

    # Wait for all jobs to complete
    job_server.wait()
    job_server.print_stats()

    # Shutdown the data server
    server.shutdown()
    server.server_close()

    return jobs

class Graph2(ig.Graph):
    def __init__(self, *args, **kwds):
        super(Graph2, self).__init__(*args, **kwds)

    def add_edges_by_name(self, edge_list, attributes=None):
        es = []
        for edge_t in edge_list:
            #es.append((self.vs.find(nx_name=edge_t[0]).index, self.vs.find(nx_name=edge_t[1]).index))
            es.append((self.vs['nx_name'].index(edge_t[0]), self.vs['nx_name'].index(edge_t[1])))

        super(Graph2, self).add_edges(es, attributes)

    def shortest_paths_nx(self, source=None, target=None, weights=None, mode='out'):
        #s_index_list = [self.vs.find(nx_name=source_index).index for source_index in list(set(source))]
        #f_index_list = [self.vs.find(nx_name=f_index).index for f_index in list(set(target))]
        s_index_list = [ self.vs['nx_name'].index(source_index) for source_index in list(set(source)) ]
        f_index_list = [ self.vs['nx_name'].index(f_index)      for f_index in list(set(target)) ]

        dist_list = super(Graph2, self).shortest_paths_dijkstra(source=s_index_list, target=f_index_list, weights=weights, mode=mode)

        d = {}
        for source_t in source:
            d_t = {}
            index_i = source.index(source_t)
            for target_t in target:
                d_t.update({target_t: dist_list[index_i][target.index(target_t)]})

            d.update({source_t: d_t})

        return d

    def shortest_paths_nx_no_degenerated_path(self, source=None, target=None, weights=None, mode='out'):
        # s_index_list = [self.vs.find(nx_name=source_index).index for source_index in list(set(source))]
        # f_index_list = [self.vs.find(nx_name=f_index).index for f_index in list(set(target))]
        s_index_list = [self.vs['nx_name'].index(source_index) for source_index in list(set(source))]
        f_index_list = [self.vs['nx_name'].index(f_index) for f_index in list(set(target))]

        dist_list = super(Graph2, self).shortest_paths_dijkstra(source=s_index_list, target=f_index_list,
                                                                weights=weights, mode=mode)

        d = {}
        for source_t in source:
            d_t = {}
            index_i = source.index(source_t)
            for target_t in target:
                d_t.update({target_t: dist_list[index_i][target.index(target_t)]})

            d.update({source_t: d_t})


        # find parent vertices v in bfs results, then use min(s->v + v->s) to solve the min cycle, where s in source
        for source_id_t in s_index_list:
            parent_node_list = self.bfs(source_id_t)[2]
            parent_list_t = []
            for j in range(0, parent_node_list.__len__()):
                if parent_node_list[j] == source_id_t and j != source_id_t:
                    parent_list_t.append(j)
            dist_s_v = self.shortest_paths_dijkstra(source=source_id_t,   target=parent_list_t, weights='weight')
            dist_v_s = self.shortest_paths_dijkstra(source=parent_list_t, target=source_id_t,   weights='weight')

            for i in range(0, parent_list_t.__len__()):
                # list(source)[s_index_list.index(source_id_t)] node name for node id: source_id_t
                source_vertex = list(source)[s_index_list.index(source_id_t)]
                dist_cycle = dist_s_v[0][i] + dist_v_s[i][0]
                if d[source_vertex][source_vertex] == 0:
                    d[source_vertex][source_vertex] = dist_cycle
                else:
                    d[source_vertex][source_vertex] = min(d[source_vertex][source_vertex], dist_cycle)

        return d

    def shortest_path_vertex_2_vertex(self, source=None, target=None, weights=None, mode='out'):
        s_index = self.vs['nx_name'].index(source)
        f_index = self.vs['nx_name'].index(target)

        path_t = super(Graph2, self).get_shortest_paths(s_index, to=f_index, weights=weights, output='vpath')
        cost_t = super(Graph2, self).shortest_paths_dijkstra(source=s_index, target=f_index, weights=weights, mode=mode)

        path_name_list = []
        for vertex in path_t[0]:
            path_name_list.append(self.vs['nx_name'][vertex])

        return (cost_t[0][0], path_name_list)

    def shortest_path_vertex_2_vertex_no_degen(self, source=None, target=None, weights=None, mode='out'):
        if source != target:
            return self.shortest_path_vertex_2_vertex(source, target, weights, mode)
        else:
            s_index = self.vs['nx_name'].index(source)

            parent_node_list = self.bfs(s_index)[2]
            parent_list_t = []
            for j in range(0, parent_node_list.__len__()):
                if parent_node_list[j] == s_index and j != s_index:
                    parent_list_t.append(j)

            dist_list = []
            dist_min_index = 0
            for i in range(0, parent_list_t.__len__()):
                dist_s_v = self.shortest_paths_dijkstra(source=s_index,   target=parent_list_t, weights='weight')
                dist_v_s = self.shortest_paths_dijkstra(source=parent_list_t, target=s_index,   weights='weight')
                dist_list.append(dist_v_s[0][0] + dist_s_v[0][0])

                if dist_list[dist_min_index] > dist_list[i]:
                    dist_min_index = i

            path_t = super(Graph2, self).get_shortest_paths(s_index, to=parent_list_t[dist_min_index], weights=weights, output='vpath')
            path_name_list = []
            for vertex in path_t[0]:
                path_name_list.append(self.vs['nx_name'][vertex])
            #path_name_list.append(source)

        return (dist_list[dist_min_index], path_name_list)

    def s_bottleneck_length(self, source_set, target_set, weights='weight'):

        import heapq

        all_dist = {}  # dictionary of final distances from source_set to target_set

        # Path length is (max edge length, total edge length)
        for source in source_set:
            dist = {}  # dictionary of final distances from source
            fringe = []  # use heapq with (bot_dist,dist,label) tuples

            # Don't allow degenerate paths
            # Add all neighbors of source to start the algorithm
            seen = dict()
            for edge_t in self.es.select(_source=self.vs.find(nx_name=source)):
                vw_dist = edge_t[weights]
                seen[edge_t.target] = (vw_dist, vw_dist)
                heapq.heappush(fringe, (vw_dist, vw_dist, edge_t.target))


            # source: v     target: w -> edge_t.target
            while fringe:
                (d_bot, d_sum, v) = heapq.heappop(fringe)

                if v in dist:
                    continue  # Already searched this node.

                dist[self.vs[v]['nx_name']] = (d_bot, d_sum)  # Update distance to this node

                for edge_t in self.es.select(_source=v):
                    vw_dist_bot = max(dist[self.vs[v]['nx_name']][0], edge_t[weights])
                    vw_dist_sum = dist[self.vs[v]['nx_name']][1] + edge_t[weights]
                    if self.vs[edge_t.target]['nx_name'] in dist:
                        if vw_dist_bot < dist[self.vs[edge_t.target]['nx_name']][0]:
                            raise ValueError('Contradictory paths found:', 'negative weights?')
                    elif edge_t.target not in seen or vw_dist_bot < seen[edge_t.target][0] \
                            or (vw_dist_bot == seen[edge_t.target][0] \
                                and vw_dist_sum < seen[edge_t.target][1]):
                        seen[edge_t.target] = (vw_dist_bot, vw_dist_sum)
                        heapq.heappush(fringe, (vw_dist_bot, vw_dist_sum, edge_t.target))


            # Remove the entries that we are not interested in
            for key in dist.keys():
                if key not in target_set:
                    dist.pop(key)

            # Add inf cost to target nodes not in dist
            for t in target_set:
                if t not in dist.keys():
                    dist[t] = (float('inf'), float('inf'))

            # Save the distance info for this source
            all_dist[source] = dist

        return all_dist


def networkx_to_igraph(g):
    """Converts the graph from networkx

    Vertex names will be converted to "nx_name" attribute and the vertices
    will get new ids from 0 up (as standard in igraph).

    @param g: networkx Graph or DiGraph
    """
    import networkx as nx

    # Graph attributes
    gattr = dict(g.graph)

    # Nodes
    vnames = list(g.node)
    vattr = {'nx_name': vnames}
    vcount = len(vnames)
    vd = {v: i for i, v in enumerate(vnames)}

    # NOTE: we do not need a special class for multigraphs, it is taken
    # care for at the edge level rather than at the graph level.
    graph = Graph2(
        n=vcount,
        directed=g.is_directed(),
        graph_attrs=gattr,
        vertex_attrs=vattr)

    # Node attributes
    for v, datum in g.node.items():
        for key, val in datum.items():
            graph.vs[vd[v]][key] = val

    # Edges and edge attributes
    # NOTE: we need to do both together to deal well with multigraphs
    # Each e might have a length of 2 (graphs) or 3 (multigraphs, the
    # third element is the "color" of the edge)
    '''
    for start, end, datum in g.edges(data=True):
        eid = graph.add_edge(vd[start], vd[end])
        for key, val in datum.items():
            eid[key] = val
    '''

    edge_list = [(vd[start], vd[end]) for start, end, datum in g.edges(data=True)]
    graph.add_edges(edge_list)
    weight_list = [ datum['weight'] for start, end, datum in g.edges(data=True) ]
    graph.es['weight'] = weight_list

    return graph

def min_bottleneck_cycle(g, s, f, pp_workers=4):
    """ Returns the minimum bottleneck cycle from s to f in graph g.

    An implementation of the Min-Bottleneck-Cycle Algortihm
    in S.L. Smith, J. Tumova, C. Belta, D. Rus " Optimal Path Planning
    for Surveillance with Temporal Logic Constraints", in IJRR.

    Parameters
    ----------
    g : NetworkX graph

    s : A set of nodes
        These nodes satisfy the optimizing proposition.

    f : A set of nodes
        These nodes are the final states of B x T product.

    Returns
    -------
    cycle : List of node labels.
        The minimum bottleneck cycle S->F->S

    Examples
    --------

    Notes
    -----

    """
    '''
    global pp_installed
    if not pp_installed:
        raise Exception('This functionality is not enables because, '
                        'Parallel Python not installed!')

    # Start job server
    job_server = pp.Server(ppservers=pp_servers, secret='trivial')
    ####
    job_server.set_ncpus(pp_workers)
    print("pp local server workers:", job_server.get_ncpus())
    '''

    #
    product_automata_2 = networkx_to_igraph(g)
    #
    s_index_list = [ product_automata_2.vs.find(nx_name=s_index).index for s_index in list(set(s)) ]
    f_index_list = [ product_automata_2.vs.find(nx_name=f_index).index for f_index in list(set(f)) ]
    s_f = list(set(s) | set(f))
    s_f_index_list = [ product_automata_2.vs.find(nx_name=s_f_index).index for s_f_index in s_f ]

    # Compute shortest S->S and S->F paths
    logger.info('S->S+F')
    #
    #d = subset_to_subset_dijkstra_path_value(s, g, s | f, degen_paths = False)
    #
    #s_to_s_f_length = product_automata_2.shortest_paths_dijkstra(product_automata_2.vs.select(nx_name_in=list(s)), product_automata_2.vs.select(nx_name_in=list(s | f)))
    #s_to_s_f_length = product_automata_2.shortest_paths_nx_no_degenerated_path(list(s), list(s | f), weights='weight')
    d = product_automata_2.shortest_paths_nx_no_degenerated_path(list(s), list(s | f), weights='weight')

    '''
    num_matched = 0
    num_unmatched = 0
    for s_t in list(s):
        for s_f_t in list(s | f):
            if s_to_s_f_length[s_t][s_f_t] == d[s_t][s_f_t]:
                num_matched += 1
            else:
                num_unmatched += 1

    print(num_matched, num_unmatched)
    '''

    '''
    jobs = job_dispatcher(job_server, subset_to_subset_dijkstra_path_value, list(s), 1, '0',
                          (g, s | f, 'sum', False, 'weight'), data_source)
    print(type(s))
    d = dict()
    for i in range(0, len(jobs)):
        d.update(jobs[i]())
        jobs[i] = ''
    del jobs
    '''
    logger.info('Collected results for S->S+F')

    # Create S->S, S->F dict of dicts
    g_s_edges = []
    d_s_to_f = dict()
    for src in d.keys():
        for dest in d[src].keys():
            if dest in s:
                w = d[src][dest]
                g_s_edges.append((src, dest, w))
            if dest in f:
                # We allow degenerate S->F paths
                w = 0 if src == dest else d[src][dest]
                if src not in d_s_to_f:
                    d_s_to_f[src] = dict()
                d_s_to_f[src][dest] = w

    # Create the G_s graph
    g_s = nx.MultiDiGraph()
    g_s.add_weighted_edges_from(g_s_edges)
    # Remove d and g_s_edges to save memory
    #del d
    #del g_s_edges

    '''
    g_s_edges = []
    g_s_edge_weight = []
    d_s_to_f = dict()
    for src in s_to_s_f_length.keys():
        for dest in s_to_s_f_length[src].keys():
            if dest in s:
                w = s_to_s_f_length[src][dest]
                g_s_edges.append((src, dest))
                g_s_edge_weight.append(w)
            if dest in f:
                # We allow degenerate S->F paths
                w = 0 if src == dest else s_to_s_f_length[src][dest]
                if src not in d_s_to_f:
                    d_s_to_f[src] = dict()
                d_s_to_f[src][dest] = w

    vattr = {'nx_name': list(s)}
    vcount = len(s)
    g_s_ig = Graph2(
        n=vcount,
        directed=True,
        vertex_attrs=vattr)
    g_s_ig.add_edges_by_name(g_s_edges)
    g_s_ig.es['weight'] = g_s_edge_weight

    matched = 0
    unmatched = 0
    for edge_t in g_s_ig.es:
        source = g_s_ig.vs[edge_t.source]['nx_name']
        target = g_s_ig.vs[edge_t.target]['nx_name']
        weight = edge_t['weight']
        if weight != g_s.edge[source][target][0]['weight']:
            unmatched += 1
        else:
            matched += 1
    print(matched, unmatched)
    '''

    # Compute shortest F->S paths
    logger.info('F->S')
    #d_f_to_s = subset_to_subset_dijkstra_path_value(f, g, s, degen_paths = True)
    d_f_to_s = product_automata_2.shortest_paths_nx(list(f), list(s), weights='weight')        # f_to_s_length
    '''
    jobs = job_dispatcher(job_server, subset_to_subset_dijkstra_path_value, list(f), 1, '1',
                          (g, s, 'sum', True, 'weight'), data_source)
    d_f_to_s = dict()
    for i in range(0, len(jobs)):
        d_f_to_s.update(jobs[i]())
        jobs[i] = ''
    del jobs
    '''
    logger.info('Collected results for F->S')

    '''
    num_matched = 0
    num_unmatched = 0
    for s_t in list(f):
        for s_f_t in list(s):
            if f_to_s_length[s_t][s_f_t] == d_f_to_s[s_t][s_f_t]:
                num_matched += 1
            else:
                num_unmatched += 1

    print(num_matched, num_unmatched)
    '''

    # Compute shortest S-bottleneck paths between verices in s
    logger.info('S-bottleneck')
    d_bot = subset_to_subset_dijkstra_path_value(s, g_s, s, 'max', False, 'weight')   # s, g_s, s, combine_fn = (lambda a,b: max(a,b)), degen_paths = False
    #d_bot_ig = g_s_ig.s_bottleneck_length(s, s, weights='weight')

    '''
    num_matched = 0
    num_unmatched = 0
    for s_t in list(s):
        for s_f_t in list(s):
            if d_bot_ig[s_t][s_f_t] == d_bot[s_t][s_f_t]:
                num_matched += 1
            else:
                num_unmatched += 1

    print(num_matched, num_unmatched)
    '''

    '''
    jobs = job_dispatcher(job_server, subset_to_subset_dijkstra_path_value, list(s), 1, '2',
                          (g_s, s, 'max', False, 'weight'), data_source)

    d_bot = dict()
    for i in range(0, len(jobs)):
        d_bot.update(jobs[i]())
        jobs[i] = ''
    del jobs
    '''
    logger.info('Collected results for S-bottleneck')

    # Find the triple \in F x S x S that minimizes C(f,s1,s2)
    logger.info('Path*')
    (cost_star, len_star, cycle_star) = find_best_cycle(f, s, d_f_to_s, d_s_to_f, d_bot)
    #s(cost_star_ig, len_star_ig, cycle_star_ig) = find_best_cycle_ig(f_index_list, s_index_list, f_to_s_length, s_to_f_length, d_bot_ig)

    '''
    jobs = job_dispatcher(job_server, find_best_cycle, list(f), 1, '3', (s, d_f_to_s, d_s_to_f, d_bot), data_source)
    cost_star = float('inf')
    len_star = float('inf')
    cycle_star = None
    for i in range(0, len(jobs)):
        this_cost, this_len, this_cycle = jobs[i]()
        jobs[i] = ''
        if (this_cost < cost_star or (this_cost == cost_star and this_len < len_star)):
            cost_star = this_cost
            len_star = this_len
            cycle_star = this_cycle
    del jobs
    '''
    logger.info('Collected results for Path*')
    logger.info('Cost*: %d, Len*: %d, Cycle*: %s', cost_star, len_star, cycle_star)


    ##
    # product_automata_2.vs['nx_name'][s_index_list[0]]


    if cost_star == float('inf'):
        raise Exception(__name__, 'Failed to find a satisfying cycle, spec cannot be satisfied.')

    else:
        logger.info('Extracting Path*')
        (ff, s1, s2) = cycle_star
        # This is the F->S1 path
        '''
        (cost_ff_to_s1, path_ff_to_s1) = source_to_target_dijkstra(g, ff, s1, degen_paths=True, cutoff=d_f_to_s[ff][s1])
        (cost_ff_to_s1_, path_ff_to_s1_) = product_automata_2.shortest_path_vertex_2_vertex(source=ff, target=s1, weights='weight')
        # for debugging
        if cost_ff_to_s1_ != cost_ff_to_s1 or path_ff_to_s1 != path_ff_to_s1_:
            print('result of shortest_path_vertex_2_vertex does not match!')
            raise Exception
        '''
        (cost_ff_to_s1, path_ff_to_s1) = product_automata_2.shortest_path_vertex_2_vertex(source=ff, target=s1, weights='weight')

        # This is the S2->F path
        '''
        (cost_s2_to_ff, path_s2_to_ff) = source_to_target_dijkstra(g, s2, ff, degen_paths=True, cutoff=d_s_to_f[s2][ff])
        (cost_s2_to_ff_, path_s2_to_ff_) = product_automata_2.shortest_path_vertex_2_vertex(source=s2, target=ff, weights='weight')
        # for debugging
        if cost_s2_to_ff_ != cost_s2_to_ff or path_s2_to_ff != path_s2_to_ff_:
            print('result of shortest_path_vertex_2_vertex does not match!')
            raise Exception
        '''
        (cost_s2_to_ff, path_s2_to_ff) = product_automata_2.shortest_path_vertex_2_vertex(source=s2, target=ff, weights='weight')

        if s1 == s2 and ff != s1:
            # The path will be F->S1==S2->F
            path_star = path_ff_to_s1[0:-1] + path_s2_to_ff
            assert (cost_star == (cost_ff_to_s1 + cost_s2_to_ff))
            assert (len_star == (cost_ff_to_s1 + cost_s2_to_ff))
        else:
            # The path will be F->S1->S2->F
            # Extract the path from s_1 to s_2
            (bot_cost_s1_to_s2, bot_path_s1_to_s2) = source_to_target_dijkstra(g_s, s1, s2, combine_fn='max',
                                                                               degen_paths=False,
                                                                               cutoff=d_bot[s1][s2][0])
            assert (cost_star == max((cost_ff_to_s1 + cost_s2_to_ff), bot_cost_s1_to_s2))
            path_s1_to_s2 = []
            cost_s1_to_s2 = 0
            for i in range(1, len(bot_path_s1_to_s2)):
                source = bot_path_s1_to_s2[i - 1]
                target = bot_path_s1_to_s2[i]
                '''
                cost_segment, path_segment = source_to_target_dijkstra(g, source, target, degen_paths=False)
                (cost_segment_, path_segment_) = product_automata_2.shortest_path_vertex_2_vertex_no_degen(source=source, target=source, weights='weight')
                # for debugging
                if cost_segment != cost_segment_ or path_segment != path_segment_:
                    print('result of shortest_path_vertex_2_vertex_no_degen does not match!')
                    raise Exception
                '''
                (cost_segment, path_segment) = product_automata_2.shortest_path_vertex_2_vertex_no_degen(source=source, target=target, weights='weight')

                path_s1_to_s2 = path_s1_to_s2[0:-1] + path_segment
                cost_s1_to_s2 += cost_segment
            assert (len_star == cost_ff_to_s1 + cost_s1_to_s2 + cost_s2_to_ff)

            # path_ff_to_s1 and path_s2_to_ff can be degenerate paths,
            # but path_s1_to_s2 cannot, thus path_star is defined as this:
            # last ff is kept to make it clear that this is a suffix-cycle
            path_star = path_ff_to_s1[0:-1] + path_s1_to_s2[0:-1] + path_s2_to_ff

        return (cost_star, path_star)


def pretty_print(agent_cnt, prefix, suffix):
    import string
    # Pretty print the prefix and suffix_cycle on team_ts
    hdr_line_1 = ''
    hdr_line_2 = ''
    for i in range(0, agent_cnt):
        hdr_line_1 += string.ljust('Robot-%d' % (i + 1), 20)
        hdr_line_2 += string.ljust('-------', 20)
    logger.info(hdr_line_1)
    logger.info(hdr_line_2)

    logger.info('*** Prefix: ***')
    for s in prefix:
        line = ''
        for ss in s:
            line += string.ljust('%s' % (ss,), 20)
        logger.info(line)

    logger.info('*** Suffix: ***')
    for s in suffix:
        line = ''
        for ss in s:
            line += string.ljust('%s' % (ss,), 20)
        logger.info(line)


def ca_safety_game(ts_tuple, prefixes, suffix_cycles, obs_range = 3):
    # obs_range observation range for nodes

    # form run for each agent
    ts_run = []
    for i in range(0, ts_tuple.__len__()):
        ts_run.append(prefixes[i] + suffix_cycles[i])
        #del ts_run[i][0]

    # for each agent, checkout if there is any other agent in range

    print(ts_run)


def multi_agent_optimal_run(ts_tuple, formula, opt_prop):
    # Construct the team_ts
    team_ts = ts_times_ts(ts_tuple)

    # Find the optimal run and shortest prefix on team_ts
    prefix_length, prefix_on_team_ts, suffix_cycle_cost, suffix_cycle_on_team_ts = optimal_run(team_ts, formula,
                                                                                               opt_prop)
    # Pretty print the run
    pretty_print(len(ts_tuple), prefix_on_team_ts, suffix_cycle_on_team_ts)

    # Project the run on team_ts down to individual agents
    prefixes = []
    suffix_cycles = []
    for i in range(0, len(ts_tuple)):
        ts = ts_tuple[i]
        prefixes.append([x for x in [x[i] if x[i] in ts.g.node else None for x in prefix_on_team_ts] if x != None])
        suffix_cycles.append(
            [x for x in [x[i] if x[i] in ts.g.node else None for x in suffix_cycle_on_team_ts] if x != None])

    return (prefix_length, prefixes, suffix_cycle_cost, suffix_cycles, prefix_on_team_ts, suffix_cycle_on_team_ts)


def multi_agent_optimal_run_ca_pre(ts_tuple, formula, opt_prop, is_pp=False):
    '''

        Directly remove collision points

    '''
    # Construct the team_ts
    team_ts = ts_times_ts_ca(ts_tuple)

    # Find the optimal run and shortest prefix on team_ts
    if not is_pp:
        prefix_length, prefix_on_team_ts, suffix_cycle_cost, suffix_cycle_on_team_ts = optimal_run(team_ts, formula,
                                                                                                   opt_prop)
    else:
        prefix_length, prefix_on_team_ts, suffix_cycle_cost, suffix_cycle_on_team_ts = optimal_run_pp(team_ts, formula,
                                                                                                   opt_prop)

    # Pretty print the run
    pretty_print(len(ts_tuple), prefix_on_team_ts, suffix_cycle_on_team_ts)

    # Project the run on team_ts down to individual agents
    prefixes = []
    suffix_cycles = []
    for i in range(0, len(ts_tuple)):
        ts = ts_tuple[i]
        prefixes.append([x for x in [x[i] if x[i] in ts.g.node else None for x in prefix_on_team_ts] if x != None])
        suffix_cycles.append(
            [x for x in [x[i] if x[i] in ts.g.node else None for x in suffix_cycle_on_team_ts] if x != None])


    # verify
    #ca_safety_game(ts_tuple, prefixes, suffix_cycles)

    return (prefix_length, prefixes, suffix_cycle_cost, suffix_cycles, prefix_on_team_ts, suffix_cycle_on_team_ts)


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

def multi_agent_optimal_run_ca(ts_tuple, formula, opt_prop, is_modifible, min_cost = 1, additional_goback_cost = 1, is_pp = False):
    '''

        Check-and-remove collision points

    '''

    ###
    '''  First construct standard route with old algorithm '''
    # Construct the team_ts
    team_ts = ts_times_ts(ts_tuple)

    # Find the optimal run and shortest prefix on team_ts
    if not is_pp:
        prefix_length, prefix_on_team_ts, suffix_cycle_cost, suffix_cycle_on_team_ts = optimal_run(team_ts, formula,
                                                                                                   opt_prop)
    else:
        prefix_length, prefix_on_team_ts, suffix_cycle_cost, suffix_cycle_on_team_ts = optimal_run_pp(team_ts, formula,
                                                                                                   opt_prop)
    # Pretty print the run
    pretty_print(len(ts_tuple), prefix_on_team_ts, suffix_cycle_on_team_ts)


    ###
    ''' Check if collision '''
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

    logger.info('[original] team run: %s', team_run)
    logger.info('[original] suffix cycle len on team TS: %d', suffix_cycle_on_team_ts.__len__())
    logger.info('[original] Cost: %d', suffix_cycle_cost)

    #
    is_singleton_collision = False
    is_pairwise_collision  = False
    is_rear_end_collision  = False
    singleton_collision_list = [ [False for i in range(ts_tuple.__len__())] for j in range(team_run.__len__()) ]
    pairwise_collision_list  = [ [False for i in range(ts_tuple.__len__())] for j in range(team_run.__len__()) ]
    rear_end_collision_list  = [ [False for i in range(ts_tuple.__len__())] for j in range(team_run.__len__()) ]

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
    for i in range(0, team_run.__len__() - 1):
        for j in range(0, ts_tuple.__len__()):  # agent j
            # find current non-travelling state for agent j
            curr_run = list(team_run[i])
            curr_run_j = curr_run[j]

            if is_traveling_state(curr_run_j):
                # if current run is traveling state

                for k in range(0, ts_tuple.__len__()):
                    if k == j:
                        continue
                    curr_run_k = curr_run[k]
                    if not is_traveling_state(curr_run_k):  # is_traveling_state(curr_run_k):
                        l = i + 1
                        next_run = list(team_run[l])
                        next_run_j = next_run[j]
                        next_run_k = next_run[k]

                        # (25, 26, 2)   27
                        # (25, 26, 1)   26              <--
                        # 26            (26, 25, 2)
                        # find the next, non-travelling state for agent k
                        # if is_traveling_state(next_run_j) and is_traveling_state(next_run_k):     # find next traveling state
                        #    if is_traveling_states_intersect(curr_run_j, next_run_k) and \
                        #       is_traveling_states_intersect(curr_run_k, next_run_j):
                        if is_traveling_state(next_run_k):
                            if is_traveling_states_intersect(curr_run_j, next_run_k):
                                # now can confirm pairwise collision
                                [last_run_j_nt, last_seq_j] = find_last_non_traveling_state(team_run, j, i)
                                # [last_run_k_nt, last_seq_k] = find_last_non_traveling_state(team_run, k, i)
                                [next_run_j_nt, next_seq_j] = find_next_non_traveling_state(team_run, j, i)
                                [next_run_k_nt, next_seq_k] = find_next_non_traveling_state(team_run, k, i)
                                # fixed bugs
                                last_run_k_nt = curr_run
                                last_seq_k = i

                                if last_run_j_nt != None and last_run_k_nt != None and next_run_j_nt != None and next_run_k_nt != None:
                                    if next_run_k_nt[k] == last_run_j_nt[j] and next_run_j_nt[j] == last_run_k_nt[k]:
                                        is_pairwise_collision = True
                                        pairwise_collision_list[last_seq_j][j] = [next_run_k_nt[k], next_seq_k, k]
                                        pairwise_collision_list[last_seq_k][k] = [next_run_j_nt[j], next_seq_j, j]
                                        pairwise_collision_list[next_seq_j][j] = [last_run_k_nt[k], last_seq_k, k]
                                        pairwise_collision_list[next_seq_k][k] = [last_run_j_nt[j], last_seq_j, j]
                                        num_pairwise_collision += 1

                        # if the other agent is quick enough
                        #INFO lomap.algorithms.multi_agent_optimal_run_ca - 25    21     26
                        #INFO lomap.algorithms.multi_agent_optimal_run_ca - 26    22     ('26', '25', 2)
                        #INFO lomap.algorithms.multi_agent_optimal_run_ca - g3    g1     25
                        if curr_run_k == list(curr_run_j)[0]:
                            [last_run_j_nt, last_seq_j] = find_last_non_traveling_state(team_run, j, i)
                            [next_run_j_nt, next_seq_j] = find_next_non_traveling_state(team_run, j, i)
                            [last_run_k_nt, last_seq_k] = find_last_non_traveling_state(team_run, k, i)
                            # fixed bugs
                            next_run_k_nt = curr_run
                            next_seq_k = i
                            if last_run_j_nt != None and last_run_k_nt != None and next_run_j_nt != None and next_run_k_nt != None:
                                if next_run_k_nt[k] == last_run_j_nt[j] and next_run_j_nt[j] == last_run_k_nt[k]:
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
                                pairwise_collision_list[i][k] = [next_run_j[j], next_seq_j, j]
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

    logger.info('[collision] Number singleton collision: %d', num_singleton_collision)
    logger.info('[collision] Number pairwise  collision: %d', num_pairwise_collision)
    logger.info('[collision] Number rear-end  collision: %d', num_rear_end_collision)

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
                    for k in range(1, i + 1):
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
                            if team_state_last != None and not is_traveling_state(team_state_last[j]):
                                # avoid adding the same edge to reduce states
                                if ts_tuple[j].g.edge[team_state_last[j]].get(team_state_last[j]) == None:
                                    ts_tuple[j].g.add_edge(team_state_last[j], team_state_last[j],
                                                           attr_dict={'weight': min_cost, 'control': 's'})
                            if team_state_next != None and not is_traveling_state(team_state_next[j]):
                                if ts_tuple[j].g.edge[team_state_next[j]].get(team_state_next[j]) == None:
                                    ts_tuple[j].g.add_edge(team_state_next[j], team_state_next[j],
                                                           attr_dict={'weight': min_cost, 'control': 's'})

                            #if team_state_last != None and is_traveling_state(team_state_last[j]):
                                # if team_state_last != None and is_traveling_state(team_state_last[j]):

                            if team_state_next != None and is_traveling_state(team_state_next[j]):
                                if ts_tuple[j].g.edge[list(team_state_next[j])[1]].get(list(team_state_next[j])[1]) == None:
                                    ts_tuple[j].g.add_edge(list(team_state_next[j])[1], list(team_state_next[j])[1],
                                                           attr_dict={'weight': min_cost, 'control': 's'})

    ''' pairwise_collision '''
    if is_pairwise_collision:
        # add turn-back points
        for i in range(0, pairwise_collision_list.__len__()):
            for j in range(0, ts_tuple.__len__()):
                if pairwise_collision_list[i][j] != False:

                    curr_state_j = list(team_run[i])[j]
                    next_state_k = pairwise_collision_list[i][j][0]
                    next_seq_k   = pairwise_collision_list[i][j][1]
                    agent_k_id   = pairwise_collision_list[i][j][2]

                    if is_modifible[j]:
                        if next_seq_k >= i:
                            # if next sequence of the run larger than current, it is the start of collision

                            # FIRST, add go-back points
                            # find ALL edges to target state
                            go_back_list = []  # for go_back_list[i], [0] for node, [1] for cost, [2] for whether exist return transition, exist [3] for two-way cost
                            go_from_list = []

                            for u in ts_tuple[j].g.edge[curr_state_j]:
                                if u != curr_state_j:
                                    go_from_list.append([u, ts_tuple[j].g.edge[curr_state_j][u][0]['weight']])
                            for u in ts_tuple[j].g.edge:
                                if ts_tuple[j].g.edge[u].get(curr_state_j) != None and \
                                    u != curr_state_j:
                                        go_back_list.append([u, ts_tuple[j].g.edge[u].get(curr_state_j)[0][
                                            'weight']])  # record point and corresponding weight

                            min_cost_index_goback = 0
                            for k in range(0, go_back_list.__len__()):
                                # calculate two-way cost for nodes in go-back list
                                if ts_tuple[j].g.edge[curr_state_j].get(go_back_list[k][0]) == None:
                                    go_back_list[k].append(False)
                                    go_back_list[k].append(go_back_list[k][1] * 2 + additional_goback_cost)
                                else:
                                    go_back_list[k].append(True)
                                    go_back_list[k].append(go_back_list[k][1] + ts_tuple[j].g.edge[curr_state_j].get(go_back_list[k][0])[0]['weight'])
                                # find the minimum
                                if go_back_list[k][3] <= go_back_list[min_cost_index_goback][3]:
                                    min_cost_index_goback = k

                            min_cost_index_goform = 0
                            for k in range(0, go_from_list.__len__()):
                                # calculate two-way cost for nodes in go-from list
                                if ts_tuple[j].g.edge[go_from_list[k][0]].get(curr_state_j) == None:
                                    go_from_list[k].append(False)
                                    go_from_list[k].append(go_from_list[k][1] * 2 + additional_goback_cost)
                                else:
                                    go_from_list[k].append(True)
                                    go_from_list[k].append(go_from_list[k][1] + ts_tuple[j].g.edge[go_from_list[k][0]].get(curr_state_j)[0]['weight'])
                                # find the minimum
                                if go_from_list[k][3] <= go_from_list[min_cost_index_goform][3]:
                                    min_cost_index_goform = k

                            # pick the best for go-back
                            # 1 cost
                            # 2 is go-back edge need to append, no need is better
                            # 3 go-back first
                            min_cost_node = None
                            is_min_cost_goback = True
                            if go_back_list[min_cost_index_goback][3] < go_from_list[min_cost_index_goform][3]:
                                min_cost_node = go_back_list[min_cost_index_goback]
                                is_min_cost_goback = True
                            elif go_back_list[min_cost_index_goback][3] > go_from_list[min_cost_index_goform][3]:
                                min_cost_node = go_from_list[min_cost_index_goback]
                                is_min_cost_goback = False
                            else:
                                if go_from_list[min_cost_index_goform][2]:
                                    min_cost_node = go_from_list[min_cost_index_goform]
                                    is_min_cost_goback = False
                                else:
                                    min_cost_node = go_back_list[min_cost_index_goback]
                                    is_min_cost_goback = True

                                # for debugging
                                '''
                                if go_back_list[min_cost_index_goback][2]:
                                    min_cost_node = go_back_list[min_cost_index_goback]
                                    is_min_cost_goback = True
                                else:
                                    min_cost_node = go_from_list[min_cost_index_goform]
                                    is_min_cost_goback = False
                                '''

                            # if the go back edge does not exist, add it
                            if not min_cost_node[2]:
                                if is_min_cost_goback:
                                    ts_tuple[j].g.add_edge(curr_state_j, min_cost_node[0],
                                                           attr_dict={'weight': min_cost_node[1] + additional_goback_cost,
                                                                      'control': 'go_back'})
                                else:
                                    ts_tuple[j].g.add_edge(min_cost_node[0], curr_state_j,
                                                           attr_dict={'weight': min_cost_node[1] + additional_goback_cost,
                                                                      'control': 'go_back'})
                    else:
                        # SECOND, add wait points
                        if ts_tuple[agent_k_id].g.edge[next_state_k].get(next_state_k) == None:
                            ts_tuple[agent_k_id].g.add_edge(next_state_k, next_state_k,
                                                   attr_dict={'weight': min_cost, 'control': 's'})

    ''' rear_end_collision '''
    if is_rear_end_collision:
        pass

    ''' Re-try '''
    #if is_singleton_collision or is_pairwise_collision or is_rear_end_collision:
    if True:
        # Construct the team_ts while removing collision points and re-try
        team_ts = ts_times_ts_ca(ts_tuple)

        # Find the optimal run and shortest prefix on team_ts
        if not is_pp:
            prefix_length, prefix_on_team_ts, suffix_cycle_cost, suffix_cycle_on_team_ts = optimal_run(team_ts, formula,
                                                                                                       opt_prop)
        else:
            prefix_length, prefix_on_team_ts, suffix_cycle_cost, suffix_cycle_on_team_ts = optimal_run_pp(team_ts, formula,
                                                                                                       opt_prop)

        ''' print modified team run '''
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
        logger.info('CA team run: %s', team_run)

        # Pretty print the run
        pretty_print(len(ts_tuple), prefix_on_team_ts, suffix_cycle_on_team_ts)

    # Project the run on team_ts down to individual agents
    prefixes = []
    suffix_cycles = []
    for i in range(0, len(ts_tuple)):
        ts = ts_tuple[i]
        prefixes.append([x for x in [x[i] if x[i] in ts.g.node else None for x in prefix_on_team_ts] if x != None])
        suffix_cycles.append(
            [x for x in [x[i] if x[i] in ts.g.node else None for x in suffix_cycle_on_team_ts] if x != None])


    # verify
    #ca_safety_game(ts_tuple, prefixes, suffix_cycles)

    return (prefix_length, prefixes, suffix_cycle_cost, suffix_cycles, prefix_on_team_ts, suffix_cycle_on_team_ts)


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

def multi_agent_optrun_unknown_w_route(ts_tuple, formula, opt_prop, is_modifible, view_range = 2):
    '''
    foe vehicle with a certain NON-stop route but vehicle only observable within a certain range

    :param ts_tuple:
    :param formula:
    :param opt_prop:
    :param is_modifible:
    :param view_range:
    :return:
    '''


    cur_ts = ts_tuple[0]
    cur_state = '4'
    target_node = search_agent_route(cur_ts, cur_state)

    print(target_node)

    # Construct the team_ts
    team_ts = ts_times_ts(ts_tuple)

    # Find the optimal run and shortest prefix on team_ts
    prefix_length, prefix_on_team_ts, suffix_cycle_cost, suffix_cycle_on_team_ts = optimal_run(team_ts, formula,
                                                                                               opt_prop)
    ''' 1. Remove Collision within team of robots '''
    is_singleton_collision, is_pairwise_collision, singleton_collision_list, pairwise_collision_list = \
        is_collision_within_team(prefix_on_team_ts, suffix_cycle_on_team_ts, is_modifible)

    if is_singleton_collision:
        # add stay motion for collision pointsFATHER AND
        for i in range(0, singleton_collision_list.__len__()):
            for j in range(0, ts_tuple.__len__()):
                if singleton_collision_list[i][j] == True:
                    if i < prefix_length:
                        prefix = list(prefix_on_team_ts[i])
                    else:
                        prefix = list(suffix_cycle_on_team_ts[i - prefix_length])
                    if not is_traveling_state(prefix[j]) and is_modifible[j]:
                        ts_tuple[j].g.add_edge(prefix[j], prefix[j],
                                               attr_dict={'weight': 1, 'control': 's'})
    ''' BUGS here '''
    if is_pairwise_collision:
        # add turn-back pointssuffix_cycle_on_team_ts
        for i in range(0, pairwise_collision_list.__len__()):
            for j in range(0, ts_tuple.__len__()):
                if pairwise_collision_list[i][j] == True:
                    if i < prefix_length:
                        prefix = list(prefix_on_team_ts[i])
                    else:
                        prefix = list(suffix_cycle_on_team_ts[i - prefix_length])
                    if not is_traveling_state(prefix[j]) and is_modifible[j]:
                        ''''''
                        ts_tuple[j].g.add_edge(prefix[j], prefix[j],                        # add the min-cost point
                                               attr_dict={'weight': 1, 'control': 's'})

    if is_singleton_collision or is_pairwise_collision:
    #if 0:
        # Construct the team_ts while removing collision points and re-try
        team_ts = ts_times_ts_ca(ts_tuple)

        # Find the optimal run and shortest prefix on team_ts
        prefix_length, prefix_on_team_ts, suffix_cycle_cost, suffix_cycle_on_team_ts = optimal_run(team_ts, formula,
                                                                                                   opt_prop)
        # Pretty print the run
        pretty_print(len(ts_tuple), prefix_on_team_ts, suffix_cycle_on_team_ts)

    ''' 2. Represent runs with ALL robots '''
    # prefix
    if prefix_length != prefix_on_team_ts.__len__():
        prefix_length = prefix_on_team_ts.__len__()
    suffix_length = suffix_cycle_on_team_ts.__len__()
    for timestamp in range(0, prefix_length):
        for i in range(0, ts_tuple.__len__()):
            ''' 3. If an agent spots foe robot within range '''
            if is_modifible[i]:
                curr_states = prefix_on_team_ts[timestamp][i]
                ts = ts_tuple[i]
                ''' FIND SUCCESSOR '''
                target_node = search_agent_route(ts, curr_states, is_weight_based=False)        # is_weight_based = True
                print(target_node)
                ''' Check whether the other agents are in these nodes '''
                for j in range(0, ts_tuple.__len__()):
                    if not is_modifible[j]:
                        foe_states = prefix_on_team_ts[timestamp][j]
                        ts = ts_tuple[j]
                        if foe_states in set(target_node):
                            ''' SPOT A FOE VEHICLE '''
                            a = 0

    for timestamp in range(0, suffix_length):
        for i in range(0, ts_tuple.__len__()):
            ''' 3. If an agent spots foe robot within range '''
            if is_modifible[i]:
                curr_states = suffix_cycle_on_team_ts[timestamp][i]



    ''' 3. If an agent spots foe robot within range, prepare to maneuver with safety game '''


    # Pretty print the run
    pretty_print(len(ts_tuple), prefix_on_team_ts, suffix_cycle_on_team_ts)

    # Project the run on team_ts down to individual agents
    prefixes = []
    suffix_cycles = []
    for i in range(0, len(ts_tuple)):
        ts = ts_tuple[i]
        prefixes.append([x for x in [x[i] if x[i] in ts.g.node else None for x in prefix_on_team_ts] if x != None])
        suffix_cycles.append(
            [x for x in [x[i] if x[i] in ts.g.node else None for x in suffix_cycle_on_team_ts] if x != None])


    # verify
    #ca_safety_game(ts_tuple, prefixes, suffix_cycles)

    print(prefix_on_team_ts)
    print(suffix_cycle_on_team_ts)

    return (prefix_length, prefixes, suffix_cycle_cost, suffix_cycles, prefix_on_team_ts, suffix_cycle_on_team_ts)