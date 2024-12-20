import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
import networkx as nx
from com.dataprep.PropertiesConfig import PropertiesConfig as PC


# Step 1: Define three simple graphs
# Graph 1: 4 nodes and 4 edges
class Graph:
    def __init__(self, edge_index, x, y):
        self.graph = Data(x=x, edge_index=edge_index, y=y)

# Step 2: Combine the graphs into a "graph of graphs"
# Define the meta-graph (graph of graphs)
class MetaGraph:
    def __init__(self):
        self.edge_index = torch.tensor([
            [0, 1, 1, 2],
            [1, 0, 2, 1]
        ], dtype=torch.long)  # Connectivity between graph1, graph2, graph3

        self.graph = Data(edge_index=self.edge_index)

        # Visualize the meta-graph
        self.g_meta = nx.Graph()
        for i in range(self.edge_index.size(1)):
            src, tgt = self.edge_index[:, i].tolist()
            self.g_meta.add_edge(src, tgt)

# Step 3: Visualize individual graphs and the "graph of graphs"
class VisualAction:
    def __init__(self):
        properties_config = PC()
        # Get properties as a dictionary
        properties = properties_config.get_properties_config()
        self.plot_path = properties['plot_path']

    def visualize_graph(self, graph_data, title, save_path):
        G = to_networkx(graph_data, to_undirected=True)
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(G)
        nx.draw_networkx_nodes(G, pos, node_size=500, node_color=graph_data.y.cpu(), cmap=plt.cm.Set1)
        nx.draw_networkx_labels(G, pos, labels={i: f"{i}" for i in range(len(graph_data.y))}, font_size=12, font_color="white")
        nx.draw_networkx_edges(G, pos, edge_color="gray", width=1.5)
        plt.title(title, fontsize=14)
        plt.axis("off")
        plt.savefig(save_path, format="png", dpi=300)
        plt.close()

    def visualise_meta_graph(self, g_meta, label, save_path):
        plt.figure(figsize=(6, 4))
        nx.draw(g_meta, with_labels=True, node_size=700, node_color="lightblue", font_size=10, font_color="black")
        plt.title(label, fontsize=14)
        plt.axis("off")
        plt.savefig(save_path, format="png", dpi=300)
        plt.close()
        print("Graphs and meta-graph visualizations saved as PNG files.")
    def print_graphs(self):
        edge_index_1 = torch.tensor([
            [0, 1, 2, 3],
            [1, 2, 3, 0]
        ], dtype=torch.long)
        x_1 = torch.tensor([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0]
        ], dtype=torch.float)
        y_1 = torch.tensor([0, 1, 0, 1], dtype=torch.long)
        graph1 = Graph(edge_index_1, x_1, y_1).graph

        edge_index_2 = torch.tensor([
            [0, 1, 2],
            [1, 2, 0]
        ], dtype=torch.long)
        x_2 = torch.tensor([
            [1, 0],
            [0, 1],
            [1, 1]
        ], dtype=torch.float)
        y_2 = torch.tensor([1, 0, 1], dtype=torch.long)
        graph2 = Graph(edge_index_2, x_2, y_2).graph

        edge_index_3 = torch.tensor([
            [0, 1, 2, 3, 4],
            [1, 2, 3, 4, 0]
        ], dtype=torch.long)
        x_3 = torch.tensor([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 1],
            [1, 0, 1]
        ], dtype=torch.float)
        y_3 = torch.tensor([0, 1, 1, 0, 1], dtype=torch.long)
        graph3 = Graph(edge_index_3, x_3, y_3).graph

        meta_graph = MetaGraph().g_meta
        # Visualize each graph
        self.visualize_graph(graph1, "Graph 1", f'{self.plot_path}/graph1.png')
        self.visualize_graph(graph2, "Graph 2", f'{self.plot_path}/graph2.png')
        self.visualize_graph(graph3, "Graph 3", f'{self.plot_path}/graph3.png')
        self.visualise_meta_graph(meta_graph, "Meta-Graph (Graph of Graphs)", f'{self.plot_path}/meta_graph.png')


visual_action = VisualAction()
visual_action.print_graphs()


