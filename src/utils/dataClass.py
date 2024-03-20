# %%
import networkx as nx
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from geopandas import GeoSeries
import rasterio
from src.utils.tract_data import tract_data
import momepy
from rasterio.mask import mask
import sys
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt


class DataPreparation:
    def __init__(
        self,
        network_data,
        census_tract,
        digital_elevation,
        dataset,
        considered_feature,
        state_code,
        clean_graph,
        state_name=None,
    ) -> None:
        """This class is used to generate graph dataset for the training and testing process

        Args:
            network_data (shp): Original data in shp file
            census_tract (shp): Obtained census tract information in shp file
            digital_elevation (tif): digital elevation model as a raster data
            dataset (str): dataset name, e.g., 'PowerSystemCase'
            clean_graph (bool): weather remove all middle nodes
        """
        self.network_data = gpd.read_file(network_data).explode()
        if state_name is not None:
            self.network_data = self.network_data[
                self.network_data.STATE_NAME.isin(state_name)
            ]
        if census_tract is not None:
            self.census_tract = gpd.read_file(census_tract)
        else:
            self.census_tract = None

        self.elevation_data = rasterio.open(digital_elevation)

        if state_code is not None:
            self.census_data = tract_data(2021, f"{dataset}/data", state_code)
        else:
            self.census_data = None
        self.dataset = dataset

        self.considered_feature = considered_feature
        self.clean_graph = clean_graph

    def network_description(self):
        edge_length = self.network_data.length
        fig, ax = plt.subplots()
        ax.hist(edge_length, bins=12, edgecolor="black")
        ax.set_xlabel("Edge length (meters)")
        ax.set_ylabel("Counts")
        plt.savefig(
            f"{self.dataset}/model_results/edge_distribution.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

    def _get_census_information(self):
        """Combine the census tract and census information

        Returns:
            GeoDataFrame: GeoDataFrame contains social economic information
        """
        census_tract = pd.merge(self.census_tract, self.census_data, on="AFFGEOID")
        census_tract["area"] = census_tract.geometry.area

        census_tract["Population density"] = census_tract["Total population"].divide(
            census_tract["area"]
        )
        return census_tract

    def add_feature_to_nodes(self, G):
        """Get node features, including the elevation, social-economic, coordinate information

        Returns:
            GeoDataFrame: A geodataframe contains information
        """

        nodes_x = [node[0] for node in G.nodes]
        nodes_y = [node[1] for node in G.nodes]

        node_gdf = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(nodes_x, nodes_y)
        ).set_crs(3857)

        node_gdf["lon"] = node_gdf.geometry.get_coordinates().values[:, 0]
        node_gdf["lat"] = node_gdf.geometry.get_coordinates().values[:, 1]

        node_gdf["elevation"] = [
            x[0]
            for x in self.elevation_data.sample(
                node_gdf.geometry.get_coordinates().values
            )
        ]
        if self.census_tract is not None:
            census_tract = self._get_census_information()
            node_gdf = gpd.sjoin(node_gdf, census_tract, op="within")
        node_gdf = node_gdf[node_gdf["elevation"] >= 1]
        return node_gdf

    def get_node_region_info(self, considered_feature, G):
        """get node feature and elevation information

        Returns:
            dict, GeoPandas, : node elevation as a dict, and node other features
        """

        nodes = self.add_feature_to_nodes(G)
        nodes_buffer_proj = nodes.buffer(2400, cap_style=3)

        nodes_buffer_elevation = {}

        def crop_center(img, cropx, cropy):
            y, x = img.shape
            startx = x // 2 - cropx // 2
            starty = y // 2 - cropy // 2
            return img[starty : starty + cropy, startx : startx + cropx]

        nodes_feature = {
            (node.geometry.x, node.geometry.y): node[considered_feature].values
            for _, node in nodes.iterrows()
        }

        color = []
        for area, node in zip(nodes_buffer_proj, nodes.geometry):
            square_elevation, _ = mask(
                self.elevation_data,
                [area],
                crop=True,
                all_touched=True,
            )
            square_elevation = square_elevation[0]

            square_elevation[square_elevation < 0] = 0
            square_elevation = crop_center(square_elevation, cropx=50, cropy=50)
            if square_elevation.shape == (50, 50):
                color.append("b")
                nodes_buffer_elevation[(node.x, node.y)] = [
                    nodes_feature[(node.x, node.y)],
                    square_elevation,
                ]
            else:
                color.append("r")
        return nodes_buffer_elevation

    def get_initial_graph(self):
        """return the networkx, node feature as [elevation region, node feature]

        Returns:
            Graph: a networkx graph
        """
        G = momepy.gdf_to_nx(
            self.network_data, approach="primal", multigraph=False, directed=False
        )
        G.remove_edges_from(nx.selfloop_edges(G))
        # largest_cc = max(nx.connected_components(G), key=len)
        # G = G.subgraph(largest_cc).copy()
        if self.clean_graph:
            G = self._clean_middle_nodes(G)

        nx.set_edge_attributes(G, 1, "edge_attr")
        global_node_feature = self.get_node_region_info(self.considered_feature, G)
        keep_node = list(global_node_feature.keys())
        remaining = G.subgraph(keep_node).copy()

        nx.set_node_attributes(remaining, global_node_feature, "x")
        remaining = nx.convert_node_labels_to_integers(remaining)
        remaining.remove_nodes_from(list(nx.isolates(remaining)))

        return remaining

    def construct_node_by_graph(self):
        """Return the final built graph and geoDataFrame

        Returns:
            GeoDataFrame, Graph: A node geopandas dataframe, and a networkx graph
        """

        G = self.get_initial_graph()

        node_feature = nx.get_node_attributes(G, "x")

        pos_dict = {k: (v[0][0], v[0][1]) for k, v in node_feature.items()}

        nodes_pos = np.array(list(pos_dict.values()))
        nodes_id_list = np.array(list(pos_dict.keys()))

        points_pos_df = pd.DataFrame(
            list(zip(nodes_id_list, nodes_pos)), columns=["node_id", "geometry"]
        )

        points_pos_df[["x", "y"]] = pd.DataFrame(
            points_pos_df["geometry"].tolist(), index=points_pos_df.index
        )

        points_pos_df.sort_values(by="node_id", ascending=True, inplace=True)
        geometry = [Point(xy) for xy in zip(points_pos_df.x, points_pos_df.y)]
        points_pos_gdf = gpd.GeoDataFrame(points_pos_df, geometry=geometry)

        planar_graph = self._construct_planar_graph(G)
        return points_pos_gdf, G, planar_graph

    def _construct_planar_graph(self, G):
        node_feature = nx.get_node_attributes(G, "x")
        nodes_id = list(G.nodes())
        coords = np.array(
            [(node_feature[i][0][0], node_feature[i][0][1]) for i in G.nodes()]
        )
        tri = Delaunay(coords, qhull_options="QJ")

        edges = set()
        for simplex in tri.simplices:
            edges.update(
                {
                    tuple(sorted((nodes_id[simplex[i]], nodes_id[simplex[j]])))
                    for i in range(3)
                    for j in range(i + 1, 3)
                }
            )

        final_graph = nx.Graph(list(edges))
        nx.set_node_attributes(final_graph, node_feature, "x")
        planar_edge = {
            edge: 1 if G.has_edge(edge[0], edge[1]) else 0
            for edge in final_graph.edges()
        }
        nx.set_edge_attributes(final_graph, planar_edge, "edge_attr")
        return final_graph

    def _clean_middle_nodes(self, G):
        nodes = [i for i in list(G.nodes()) if G.degree[i] == 2]
        for node in nodes:
            neighbors = list(G.neighbors(node))
            if G.has_edge(neighbors[0], neighbors[1]):
                continue
            else:
                G.remove_node(node)
                G.add_edge(neighbors[0], neighbors[1])
        return G


class SamplingGraph:
    """sample from a large graph"""

    def __init__(
        self,
        graph,
        planar_graph,
        sample_method,
        dataset_type,
        window_size,
        total_node_min,
        total_node_max,
    ) -> None:
        """inputs of the graph sampler

        Args:
            graph (networkx): Orignal large network
            sample_method (str): 'bfs' or 'region'. `bfs` using broadth first search, and `region` use regional sampling method.
            dataset_type (str): '_normalized' or ''. `_normalized` means each subgraph are normalized with 0 as mean and 1 as std. empty means no normalization for subgraph.
            window_size (float): if using 'bfs', the window size is used for identify the findeing depth. If using 'region', the window size is used for identify the region size.
        """
        self.G = graph
        self.sample_method = sample_method
        self.dataset_type = dataset_type
        self.planar_graph = planar_graph
        self.window_size = window_size
        self.total_node_min = total_node_min
        self.total_node_max = total_node_max

    def sample_network(self, node, points_df):
        """sample one subgraph based on give node

        Args:
            node (networkx node): node id of a networkx graph
            points_df (geopands of all nodes): _description_

        Returns:
            networkx: sampled subgraph
        """
        if self.sample_method == "bfs":
            edges = list(nx.bfs_edges(self.G, node, depth_limit=int(self.window_size)))
            node_list = [node] + [v for u, v in edges]
        elif self.sample_method == "region":
            node_pos = GeoSeries(
                Point(
                    nx.get_node_attributes(self.G, "x")[node][0][0],
                    nx.get_node_attributes(self.G, "x")[node][0][1],
                )
            )

            nodes_buffer = node_pos.buffer(self.window_size, cap_style=3).values[0]
            node_list = points_df[points_df.within(nodes_buffer)].node_id.tolist()

            neighbors = []
            for node in node_list:
                neighbors += list(self.G.neighbors(node))

            node_list = set(node_list + neighbors)
        else:
            sys.exit(f"Used sample method is {self.sample_method}, it does not exit!")

        if len(node_list) > 2:
            node_feature = {
                v: nx.get_node_attributes(self.G, "x")[v][0] for v in node_list
            }
            node_elevation = {
                v: nx.get_node_attributes(self.G, "x")[v][1] for v in node_list
            }
            feature_list = np.array(list(node_feature.values()), dtype=float)

            if self.dataset_type == "Normalized":
                a = feature_list - np.mean(feature_list, axis=0)
                b = np.std(feature_list, axis=0)

                normalized_feature = np.divide(
                    a, b, out=np.zeros(a.shape, dtype=float), where=b != 0
                )

            elif self.dataset_type == "Standard":
                normalized_feature = (feature_list - self.total_node_min) / (
                    self.total_node_max - self.total_node_min
                )
            else:
                sys.exit(
                    f"Used normalization method is {self.dataset_type}, it does not exit!"
                )

            node_feature_normalized = {}
            for row, key in zip(normalized_feature, node_feature.keys()):
                node_feature_normalized[key] = [row, node_elevation[key]]

            subgraph = self.G.subgraph(node_list).copy()
            isolated = list(nx.isolates(subgraph))
            remaining_nodes = list(set(node_list) - set(isolated))
            subgraph = self.G.subgraph(remaining_nodes).copy()
            if self.planar_graph is not None:
                final_graph = self.planar_graph.subgraph(remaining_nodes).copy()
            else:
                final_graph = nx.complete_graph(remaining_nodes)

            edge_feature = {
                k: 1 if subgraph.has_edge(k[0], k[1]) else 0
                for k in final_graph.edges()
            }
            nx.set_edge_attributes(final_graph, edge_feature, name="edge_attr")
            nx.set_node_attributes(final_graph, node_feature_normalized, name="x")

            return final_graph, subgraph
        else:
            return None, None


# %%
