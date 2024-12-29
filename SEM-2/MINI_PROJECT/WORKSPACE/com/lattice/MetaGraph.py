import torch
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt
from PropertiesConfig import PropertiesConfig as PC

class MetaGraph:
    def __init__(self, edge_list=None):
        """
        Initialize the MetaGraph.

        Args:
            edge_list (list of tuples): List of edges defining the connectivity
                                        between graphs (e.g., [(0, 1), (1, 2)]).
        """
        # Dynamically set edges based on input or default to an empty list
        self.edge_list = edge_list if edge_list else []
        print(f'0. self.edge_list:{self.edge_list}')
        # Create edge_index from edge_list
        self.edge_index = torch.tensor(self.edge_list, dtype=torch.long).T \
            if self.edge_list else torch.empty((2, 0), dtype=torch.long)
        print(f'1. self.edge_list:{self.edge_list}')

        # Create PyTorch Geometric Data object for the meta-graph
        self.graph = Data(edge_index=self.edge_index)
        print(f'2. self.graph:{self.graph}')

        # Create and visualize the meta-graph using NetworkX
        self.g_meta = nx.Graph()
        print(f'3. self.g_meta:{self.g_meta}')

        self.g_meta.add_edges_from(self.edge_list)
        print(f'4. self.g_meta:{self.g_meta}')

    def add_edge(self, src, tgt):
        """
        Add a new edge to the meta-graph.

        Args:
            src (int): Source graph index.
            tgt (int): Target graph index.
        """
        self.edge_list.append((src, tgt))
        self.edge_index = torch.tensor(self.edge_list, dtype=torch.long).T
        self.graph.edge_index = self.edge_index
        self.g_meta.add_edge(src, tgt)

    def visualize(self, image_save_path):
        """
        Visualize the meta-graph.
        """
        nx.draw(self.g_meta, with_labels=True, node_color="lightblue", font_weight="bold")
        plt.savefig(f'{image_save_path}')
        plt.close()


properties_config = PC()
# Get properties as a dictionary
properties = properties_config.get_properties_config()
plot_path = properties['plot_path']
# Example usage
# Create a dynamic meta-graph
meta_graph = MetaGraph(edge_list=[(0, 1), (1, 2), (2, 3)])
meta_graph.visualize(f'{plot_path}/1.png')
# Add a new edge dynamically
meta_graph.add_edge(3, 0)

# Visualize the meta-graph
meta_graph.visualize(f'{plot_path}/2.png')
