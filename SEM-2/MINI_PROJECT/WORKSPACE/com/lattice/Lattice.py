import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
import networkx as nx
from com.dataprep.PropertiesConfig import PropertiesConfig as PC
from com.lattice.graph_data import x_1, x_2, x_3, edge_index_1, edge_index_2, edge_index_3, y_1, y_2, y_3
# Step 1: Define Graph Class
from MetaGraph import MetaGraph
# Step 2: Define MetaGraph Class
from Graph import Graph
# Step 3: Define the Graph-Level GNN Model
from GraphLevelGNN import GraphLevelGNN
# Step 4: Define the Graph-of-Graphs Model
from GraphOfGraphsModel import GraphOfGraphsModel


# Step 5: Visualization Helper
class VisualAction:
    def __init__(self):
        properties_config = PC()
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

    def print_graphs(self):
        graph1 = Graph(edge_index_1, x_1, y_1).graph

        graph2 = Graph(edge_index_2, x_2, y_2).graph

        graph3 = Graph(edge_index_3, x_3, y_3).graph

        meta_graph1 = MetaGraph(edge_list=[(0, 1), (1, 2), (0, 2)])
        meta_graph = meta_graph1.g_meta
        meta_edge_index = meta_graph1.edge_index

        # Visualize each graph
        self.visualize_graph(graph1, "Graph 1", f'{self.plot_path}/graph1.png')
        self.visualize_graph(graph2, "Graph 2", f'{self.plot_path}/graph2.png')
        self.visualize_graph(graph3, "Graph 3", f'{self.plot_path}/graph3.png')
        self.visualise_meta_graph(meta_graph, "Meta-Graph (Graph of Graphs)", f'{self.plot_path}/meta_graph.png')

        return graph1, graph2, graph3, meta_graph, meta_edge_index


# Step 6: Run the Classification Model
def run_classification():
    visual_action = VisualAction()
    # Step 1: Define three simple graphs
    graph1, graph2, graph3, meta_graph, meta_edge_index = visual_action.print_graphs()

    # Initialize GNN models
    graph_model = GraphLevelGNN(in_channels=3, hidden_channels=4, out_channels=2)  # Graph-level model

    # Example prediction for individual graphs using GraphLevelGNN
    graph_model.eval()

    # Get node-level predictions
    prediction_1 = graph_model(graph1)
    print(f"Prediction for Graph 1: {prediction_1}")

    prediction_2 = graph_model(graph2)
    print(f"Prediction for Graph 2: {prediction_2}")

    prediction_3 = graph_model(graph3)
    print(f"Prediction for Graph 3: {prediction_3}")

    # Apply mean pooling to get graph-level embeddings (average of node predictions)
    graph_embedding1 = prediction_1.mean(dim=0)  # Mean pooling over nodes for Graph 1
    graph_embedding2 = prediction_2.mean(dim=0)  # Mean pooling over nodes for Graph 2
    graph_embedding3 = prediction_3.mean(dim=0)  # Mean pooling over nodes for Graph 3

    # Stack graph-level embeddings
    graph_embeddings = torch.stack([graph_embedding1, graph_embedding2, graph_embedding3])
    print(f"Graph-level embeddings: {graph_embeddings}")

    # Now, integrate graph-level embeddings into the meta-graph
    # For simplicity, we assume that each graph's prediction is part of the meta-graph's node features
    meta_graph_model = GraphOfGraphsModel(in_channels=2, hidden_channels=4, out_channels=2)  # Meta-graph model

    meta_graph_data = Data(x=graph_embeddings, edge_index=meta_edge_index)

    # Meta-graph model prediction
    meta_graph_model.eval()
    meta_pred = meta_graph_model(meta_graph_data, meta_graph_data)
    print(f"Prediction for Meta-Graph: {meta_pred}")


# Run the classification
run_classification()
