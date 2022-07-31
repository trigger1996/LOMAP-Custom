#! /usr/bin/python

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

# Logger configuration
logger = logging.getLogger(__name__)
#logger.addHandler(logging.NullHandler())

# Cluster configuration
#SERVER_ADDR is the ip address of the computer running lomap
#pp_servers is a tuple of server ips that are running ppserver.py
#SERVER_ADDR = '107.20.62.59'
#pp_servers = ('23.22.80.26', '50.17.177.92', '50.16.82.182', '50.16.116.126')
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
            for i in range(0,len(suffix_cycle_on_p)):
                length, prefix = source_to_target_dijkstra(p.g, init_state, suffix_cycle_on_p[i], degen_paths = True)
                if(length < prefix_length):
                    prefix_length = length
                    prefix_on_p = prefix
                    i_star = i

        if(prefix_length == float('inf')):
            raise Exception(__name__, 'Could not compute the prefix.')

        # Wrap suffix_cycle_on_p as required
        if i_star != 0:
            # Cut and paste
            suffix_cycle_on_p = suffix_cycle_on_p[i_star:] + suffix_cycle_on_p[1:i_star+1]

        # Compute projection of prefix and suffix-cycle to T and return
        suffix_cycle = [x[0] for x in suffix_cycle_on_p]
        prefix = [x[0] for x in prefix_on_p]
        return (prefix_length, prefix, suffix_cycle_cost, suffix_cycle)
    except Exception as ex:
        if(len(ex.args) == 2):
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

        if(cost < cost_star or (cost == cost_star and this_len < len_star)):
            cost_star = cost
            len_star = this_len
            cycle_star = triple

    return (cost_star, len_star, cycle_star)

def chunks(l, n):
    return [l[i:i+n] for i in range(0, len(l), n)]

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

    if need_data:# or 'my_data' not in globals():
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
    return eval(func_name+'(chunk, *my_data)')

def job_dispatcher(job_server, func, arg_to_split, chunk_size, data_id, data, data_source):
    import socket
    import threading
    from six.moves import socketserver
    import pickle

    pickled_data = pickle.dumps(data,pickle.HIGHEST_PROTOCOL)

    # Data Server Configuration
    class ThreadedTCPRequestHandler(socketserver.BaseRequestHandler):
        def handle(self):
            req = self.request.recv(32)
            if req == 'data_id':
                self.request.sendall(data_id)
                #logger.info('data_id req received from %s', self.client_address)
            elif req == 'data':
                self.request.sendall(pickled_data)
                logger.info('Served dataset %s to %s', data_id, self.client_address)
            else:
                assert(False)

    class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
        allow_reuse_address = True
        pass

    # Start the server
    server = ThreadedTCPServer(('0.0.0.0',data_source[1]), ThreadedTCPRequestHandler)
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
            dist_s_v = self.shortest_paths_dijkstra(source=source_id_t, target=parent_list_t, weights='weight')
            dist_v_s = self.shortest_paths_dijkstra(source=parent_list_t, target=source_id_t, weights='weight')

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
                dist_s_v = self.shortest_paths_dijkstra(source=s_index, target=parent_list_t, weights='weight')
                dist_v_s = self.shortest_paths_dijkstra(source=parent_list_t, target=s_index, weights='weight')
                dist_list.append(dist_v_s[0][0] + dist_s_v[0][0])

                if dist_list[dist_min_index] > dist_list[i]:
                    dist_min_index = i

            path_t = super(Graph2, self).get_shortest_paths(s_index, to=parent_list_t[dist_min_index], weights=weights,
                                                            output='vpath')
            path_name_list = []
            for vertex in path_t[0]:
                path_name_list.append(self.vs['nx_name'][vertex])
            # path_name_list.append(source)

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

def shortest_paths_nx(source, g, target, weights=None, mode='out'):

    s_index_list = [ g.vs['nx_name'].index(source_index) for source_index in list(set(source)) ]
    f_index_list = [ g.vs['nx_name'].index(f_index)      for f_index in list(set(target)) ]

    dist_list = g.shortest_paths_dijkstra(source=s_index_list, target=f_index_list, weights=weights, mode=mode)

    d = {}
    for source_t in source:
        d_t = {}
        index_i = source.index(source_t)
        for target_t in target:
            d_t.update({target_t: dist_list[index_i][target.index(target_t)]})

        d.update({source_t: d_t})

    return d

def shortest_paths_nx_no_degenerated_path(source, g, target, weights=None, mode='out'):

    s_index_list = [ g.vs['nx_name'].index(source_index) for source_index in list(set(source))]
    f_index_list = [ g.vs['nx_name'].index(f_index) for f_index in list(set(target))]

    dist_list = g.shortest_paths_dijkstra(source=s_index_list, target=f_index_list, weights=weights, mode=mode)

    d = {}
    for source_t in source:
        d_t = {}
        index_i = source.index(source_t)
        for target_t in target:
            d_t.update({target_t: dist_list[index_i][target.index(target_t)]})

        d.update({source_t: d_t})

    # find parent vertices v in bfs results, then use min(s->v + v->s) to solve the min cycle, where s in source
    for source_id_t in s_index_list:
        parent_node_list = g.bfs(source_id_t)[2]
        parent_list_t = []
        for j in range(0, parent_node_list.__len__()):
            if parent_node_list[j] == source_id_t and j != source_id_t:
                parent_list_t.append(j)
        dist_s_v = g.shortest_paths_dijkstra(source=source_id_t, target=parent_list_t, weights='weight')
        dist_v_s = g.shortest_paths_dijkstra(source=parent_list_t, target=source_id_t, weights='weight')

        for i in range(0, parent_list_t.__len__()):
            # list(source)[s_index_list.index(source_id_t)] node name for node id: source_id_t
            source_vertex = list(source)[s_index_list.index(source_id_t)]
            dist_cycle = dist_s_v[0][i] + dist_v_s[i][0]
            if d[source_vertex][source_vertex] == 0:
                d[source_vertex][source_vertex] = dist_cycle
            else:
                d[source_vertex][source_vertex] = min(d[source_vertex][source_vertex], dist_cycle)

    return d

def min_bottleneck_cycle(g, s, f):
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

    global pp_installed
    if not pp_installed:
        raise Exception('This functionality is not enables because, '
                        'Parallel Python not installed!')

    # Start job server
    job_server = pp.Server(ncpus=14, ppservers=pp_servers, secret='trivial')

    #
    # for speeding up
    product_automata_2 = networkx_to_igraph(g)
    #
    #s_index_list = [ product_automata_2.vs.find(nx_name=s_index).index for s_index in list(set(s)) ]
    #f_index_list = [ product_automata_2.vs.find(nx_name=f_index).index for f_index in list(set(f)) ]


    # Compute shortest S->S and S->F paths
    logger.info('S->S+F')
    #d = subset_to_subset_dijkstra_path_value(g, s, s|f, degen_paths = False)
    #d = product_automata_2.shortest_paths_nx_no_degenerated_path(list(s), list(s | f), weights='weight')
    '''
    jobs = job_dispatcher(job_server, subset_to_subset_dijkstra_path_value, list(s), 1, '0', (g, s | f, 'sum', False, 'weight'), data_source)
    '''
    jobs = job_dispatcher(job_server, shortest_paths_nx_no_degenerated_path, list(s), 1, '0', (product_automata_2, list(s | f), 'weight'), data_source)
    d = dict()
    for i in range(0,len(jobs)):
        d.update(jobs[i]())
        jobs[i]=''
    del jobs
    logger.info('Collected results for S->S+F')


    # Create S->S, S->F dict of dicts
    g_s_edges = []
    d_s_to_f = dict()
    for src in d.keys():
        for dest in d[src].keys():
            if dest in s:
                w = d[src][dest]
                g_s_edges.append((src,dest,w))
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
    del d
    del g_s_edges

    # Compute shortest F->S paths
    logger.info('F->S')
    #d_f_to_s = product_automata_2.shortest_paths_nx(list(f), list(s), weights='weight')
    #d_f_to_s = subset_to_subset_dijkstra_path_value(g, f, s, degen_paths = True)
    '''
    jobs = job_dispatcher(job_server, subset_to_subset_dijkstra_path_value, list(f), 1, '1', (g, s, 'sum', True, 'weight'), data_source)
    '''
    jobs = job_dispatcher(job_server, shortest_paths_nx, list(f), 1, '1', (product_automata_2, list(s), 'weight'), data_source)
    d_f_to_s = dict()
    for i in range(0,len(jobs)):
        d_f_to_s.update(jobs[i]())
        jobs[i]=''
    del jobs

    logger.info('Collected results for F->S')

    # Compute shortest S-bottleneck paths between verices in s
    logger.info('S-bottleneck')
    #d_bot = subset_to_subset_dijkstra_path_value(g_s, s, s, combine_fn = (lambda a,b: max(a,b)), degen_paths = False)
    jobs = job_dispatcher(job_server, subset_to_subset_dijkstra_path_value, list(s), 1, '2', (g_s, s, 'max', False, 'weight'), data_source)

    d_bot = dict()
    for i in range(0,len(jobs)):
        d_bot.update(jobs[i]())
        jobs[i]=''
    del jobs
    logger.info('Collected results for S-bottleneck')

    # Find the triple \in F x S x S that minimizes C(f,s1,s2)
    logger.info('Path*')
    jobs = job_dispatcher(job_server, find_best_cycle, list(f), 1, '3', (s, d_f_to_s, d_s_to_f, d_bot), data_source)
    cost_star = float('inf')
    len_star = float('inf')
    cycle_star = None
    for i in range(0,len(jobs)):
        this_cost, this_len, this_cycle = jobs[i]()
        jobs[i]=''
        if (this_cost < cost_star or (this_cost == cost_star and this_len < len_star)):
            cost_star = this_cost
            len_star = this_len
            cycle_star = this_cycle
    del jobs
    logger.info('Collected results for Path*')
    logger.info('Cost*: %d, Len*: %d, Cycle*: %s', cost_star, len_star, cycle_star)

    if cost_star == float('inf'):
        raise Exception(__name__, 'Failed to find a satisfying cycle, spec cannot be satisfied.')

    else:
        logger.info('Extracting Path*')
        (ff, s1, s2) = cycle_star
        # This is the F->S1 path
        #(cost_ff_to_s1, path_ff_to_s1) = source_to_target_dijkstra(g, ff, s1, degen_paths = True, cutoff = d_f_to_s[ff][s1])
        (cost_ff_to_s1, path_ff_to_s1) = product_automata_2.shortest_path_vertex_2_vertex(source=ff, target=s1, weights='weight')
        # This is the S2->F path
        #(cost_s2_to_ff, path_s2_to_ff) = source_to_target_dijkstra(g, s2, ff, degen_paths = True, cutoff = d_s_to_f[s2][ff])
        (cost_s2_to_ff, path_s2_to_ff) = product_automata_2.shortest_path_vertex_2_vertex(source=s2, target=ff, weights='weight')

        if s1 == s2 and ff != s1:
            # The path will be F->S1==S2->F
            path_star = path_ff_to_s1[0:-1] + path_s2_to_ff
            assert(cost_star == (cost_ff_to_s1 + cost_s2_to_ff))
            assert(len_star == (cost_ff_to_s1 + cost_s2_to_ff))
        else:
            # The path will be F->S1->S2->F
            # Extract the path from s_1 to s_2
            (bot_cost_s1_to_s2, bot_path_s1_to_s2) = source_to_target_dijkstra(g_s, s1, s2, combine_fn = 'max', degen_paths = False, cutoff = d_bot[s1][s2][0])
            assert(cost_star == max((cost_ff_to_s1 + cost_s2_to_ff),bot_cost_s1_to_s2))
            path_s1_to_s2 = []
            cost_s1_to_s2 = 0
            for i in range(1,len(bot_path_s1_to_s2)):
                source = bot_path_s1_to_s2[i-1]
                target = bot_path_s1_to_s2[i]
                #cost_segment, path_segment = source_to_target_dijkstra(g, source, target, degen_paths = False)
                (cost_segment, path_segment) = product_automata_2.shortest_path_vertex_2_vertex_no_degen(source=source, target=target, weights='weight')
                path_s1_to_s2 = path_s1_to_s2[0:-1] + path_segment
                cost_s1_to_s2 += cost_segment
            assert(len_star == cost_ff_to_s1 + cost_s1_to_s2 + cost_s2_to_ff)

            # path_ff_to_s1 and path_s2_to_ff can be degenerate paths,
            # but path_s1_to_s2 cannot, thus path_star is defined as this:
            # last ff is kept to make it clear that this is a suffix-cycle
            path_star = path_ff_to_s1[0:-1] + path_s1_to_s2[0:-1] + path_s2_to_ff

        job_server.destroy()
        del job_server

        return (cost_star, path_star)
