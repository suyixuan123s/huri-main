""" 

Author: Hao Chen (chen960216@gmail.com)
Created: 20230702osaka
Install pygraphviz: https://zhuanlan.zhihu.com/p/624095174
"""
from huri.learning.method.AlphaZero.mcts import Node, DummyNode
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
import numpy as np

plt.switch_backend('TkAgg')


def vis_mcts(root_node: Node):
    nodes = find_all_nodes(root_node)
    graph = nx.DiGraph()
    cnt = 1
    node_ids = {}
    for n in nodes:
        if id(n) not in node_ids:
            node_ids[id(n)] = cnt
            cnt += 1
        graph.add_node(node_ids[id(n)], n=n.number_visits)
    for n in nodes:
        if isinstance(n.parent, DummyNode):
            start_node = 0
            action = None
            # if n.has_parent and not isinstance(n.parent, DummyNode):
        else:
            if id(n.parent) not in node_ids:
                node_ids[id((n.parent))] = cnt
                cnt += 1
            start_node = node_ids[id((n.parent))]
            action = n.move
        if id(n) not in node_ids:
            node_ids[id(n)] = cnt
            cnt += 1
        Q = n.total_value
        graph.add_edge(start_node, node_ids[id(n)], action=action, Q=Q)

    # print the best move
    print(root_node.child_number_visits)
    print(np.argmax(root_node.child_number_visits))
    print(graph.nodes)
    print(graph.edges)
    pos = graphviz_layout(graph, prog="dot")
    labels = {node: int(graph.nodes[node]['n']) for node in graph.nodes if 'n' in graph.nodes[node]}
    nx.draw(graph, pos=pos, labels=labels, with_labels=True)
    nx.draw_networkx_edge_labels(
        graph, pos,
        edge_labels={edge: f'a: [{graph.edges[edge]["action"]}] Q: [{graph.edges[edge]["Q"]:.2f}]' for edge in
                     graph.edges if
                     'action' in graph.edges[edge]},
        font_color='red'
    )
    plt.axis('off')
    plt.show()
    plt.show()


def find_all_nodes(root_node: Node):
    nodes = []
    stack = [root_node]
    while stack:
        current_node = stack.pop()
        nodes.append(current_node)
        stack.extend(list(current_node.children.values()))
    return nodes


def add_edges(graph, subgraph, parent, depth):
    for child in graph.successors(parent):
        if depth:
            add_edges(graph, subgraph, child, depth - 1)

        subgraph.add_node(parent)
        subgraph.add_node(child)
        for node in [parent, child]:
            subgraph.node[node]['n'] = graph.node[node]['n']
            subgraph.node[node]['w'] = graph.node[node]['w']
            subgraph.node[node]['uct'] = graph.node[node]['uct']
            subgraph.node[node]['state'] = graph.node[node]['state']
        subgraph.add_edge(parent, child)


if __name__ == '__main__':
    import huri.core.file_sys as fs

    root_node = fs.load_pickle('./root_node_parallel_20230704-235816.pkl')
    vis_mcts(
        root_node
    )
