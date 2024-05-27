import networkx as nx


class LinkGraph:
    """
    Use this class to implement the required graph in link analysis.
    You are free to modify this class according to your needs.
    You can add or remove methods from it.
    """

    def __init__(self):
        self.nodes = []
        self.edges = {}

    def add_edge(self, u_of_edge, v_of_edge):
        self.edges[u_of_edge].append(v_of_edge)
        # self.edges[v_of_edge].append(u_of_edge)

    def add_node(self, node_to_add):
        self.nodes.append(node_to_add)
        self.edges[node_to_add] = []

    def get_successors(self, node):
        return self.edges[node]

    def get_predecessors(self, node):
        preds = []
        for n in self.nodes:
            if node in self.edges[n]:
                preds.append(n)
        return preds

# lg = LinkGraph()
# lg.add_node(1)
# lg.add_node(2)
# lg.add_node(3)
# lg.add_node(4)
# lg.add_edge(1, 3)
# lg.add_edge(1, 4)
# lg.add_edge(2, 4)
# print(lg.get_predecessors(4))
# print(lg.get_successors(1))