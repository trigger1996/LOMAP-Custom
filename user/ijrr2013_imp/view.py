
import networkx as nx
from lomap.classes import ts
from  matplotlib import pyplot as plt

pos_ref = { "1" : [0,   0.1],
            "2" : [0,   0.2],
            "3" : [0,   0.8],
            "4" : [0,   0.9],
            "5" : [0.1, 0.9],
            "6" : [0.9, 0.9],
            "7" : [1,   0.9],
            "8" : [1,   0.8],
            "9" : [1,   0.2],
            "10" : [1,  0.1],
            "11" : [0.9, 0.1],
            "12" : [0.1, 0.1],
            "21" : [0.1, 0.2],
            "22" : [0.5, 0.2],
            "23" : [0.9, 0.2],
            "24" : [0.9, 0.5],
            "25" : [0.9, 0.8],
            "26" : [0.5, 0.8],
            "27" : [0.1, 0.8],
            "28" : [0.1, 0.5],
            "u1" : [0, 1],
            "u2" : [1, 0],
            "g1" : [0.5, 0.3],
            "g2" : [0.7, 0.5],
            "g3" : [0.5, 0.7],
            "g4" : [0.3, 0.5] }

def visualize_run(tran_sys, run, edgelabel='control', draw='matplotlib'):
    """
    Visualizes a LOMAP system model with run.
    """
    assert edgelabel is None or nx.is_weighted(tran_sys.g, weight=edgelabel)
    if draw == 'pygraphviz':
        nx.view_pygraphviz(tran_sys.g, edgelabel)
    elif draw == 'matplotlib':
        pos = nx.get_node_attributes(tran_sys.g, 'location')
        if len(pos) != tran_sys.g.number_of_nodes():
            pos = nx.spring_layout(tran_sys.g)

        # because the map is the same
        # add map (drawn before)
        pos = pos_ref

        # add color (set before)
        # https://blog.csdn.net/qq_26376175/article/details/67637151
        node_colors = dict([(v, 'yellowgreen') for v in tran_sys.g])
        node_colors['u1'] = node_colors['u2'] = 'tomato'
        node_colors['g1'] = node_colors['g2'] = node_colors['g3'] = node_colors['g4'] = 'cornflowerblue'
        node_colors = list(node_colors.values())

        # edge color
        color_map = []
        index = 0
        for u, v in tran_sys.g.edges():
            color_map.append('black')
            index += 1
            for i in range(1, run.__len__()):
                if u == run[i - 1] and v == run[i]:
                    color_map[index - 1] = 'blue'
                    break

        nx.draw(tran_sys.g, pos=pos, node_color=node_colors, edge_color=color_map)
        nx.draw_networkx_labels(tran_sys.g, pos=pos)
        edge_labels = nx.get_edge_attributes(tran_sys.g, edgelabel)

        #
        edge_labels_to_draw = []
        for (n1, n2) in edge_labels.items():
            edge_labels_to_draw.append(((n1[0], n1[1]), n2))
        edge_labels_to_draw = dict(edge_labels_to_draw)

        nx.draw_networkx_edge_labels(tran_sys.g, pos=pos,
                                     edge_labels=edge_labels_to_draw)  # edge_labels

        plt.show()
