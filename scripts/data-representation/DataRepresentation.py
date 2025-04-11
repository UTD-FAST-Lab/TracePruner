import networkx as nx
import matplotlib.pyplot as plt



class DataRepresentation:

    def __init__(self):
        pass

    def load_edge_trace(self, path):
        pass


    def create_graph(self, trace):
        # Create an empty graph
        G = nx.DiGraph()


        nodes, edges = self.format_trace()
        node_embeddings = self.get_embeddings(nodes)


        for node_id, emb in node_embeddings.items():
            G.add_node(node_id, embedding=emb)


        for src, dst, freq, info in edges:
            G.add_edge(src, dst, weight=freq, info=info)    

        return G
    

    def format_trace(self, trace):
        pass


    def get_embeddings(self, nodes):
        pass


if __name__ == '__main__':

    dr = DataRepresentation()

    # load the edge trace
    cg_trace = dr.load_edge_trace(cg_path)
    br_trace = dr.load_edge_trace(br_path)

    # create the graph

    graph = dr.create_graph(trace)