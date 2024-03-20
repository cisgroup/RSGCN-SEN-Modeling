# %%
import networkx as nx
import numpy as np
from scipy.spatial import distance_matrix
import random
import matplotlib.pyplot as plt
import multiprocessing
import pickle


# define Python user-defined exceptions
class InvalidCheckPoint(Exception):
    "This error means the code is free of bug, but there is some logistic error happening"
    pass


class random_geometric(object):
    """The edge is created if the distance between node is smaller than a thireshold *d*"""

    def __init__(
        self, node_number: int, dist: float, seed: int = None, mapSize: int = None
    ):
        """provide the number of node, and threshod distance.

           The network space is from (0,0) to (mapSize,mapSize)

        Args:
            node_number (int): the total number of the graph
            dist (float): the distance threshold
            clean_graph (book): if you want to remove the isolated points
        """
        self.node_number = node_number
        self.dist = dist
        self.seed = seed
        self.mapSize = mapSize

    def generate_graph(self):
        """generate a random graph

        Returns:
            Graph: non-direction graph
        """
        if self.seed:
            np.random.seed(self.seed)
        points = np.random.randint(50, self.mapSize - 50, size=(self.node_number, 2))
        pos = {i: points[i] for i in range(self.node_number)}
        matrix = distance_matrix(list(pos.values()), list(pos.values()))
        G = nx.from_numpy_array(
            matrix,
            parallel_edges=False,
        )

        nx.set_node_attributes(G, pos, name="x")
        w = nx.get_edge_attributes(G, "weight")

        w = {k: 1 if v < self.dist else 0 for k, v in w.items()}

        nx.set_edge_attributes(G, w, "edge_attr")

        return G, pos


class soft_random_geometric(object):
    """The edge is created if the distance between node is smaller than a thireshold *d* with a probability"""

    def __init__(self, node_number: int, dist: float, p_dist: any, seed: int = None):
        """provide the number of node, and distance distribution

           The network space is from (0,0) to (1,1)

        Args:
            node_number (int): the total number of the graph
            dist (float) : threshold
            p_dist (float): function.  A probability density function computing the probability of
        connecting two nodes that are of distance, dist, computed by the
        Minkowski distance metric.
        """
        self.node_number = node_number
        self.seed = seed
        self.dist = dist
        self.p_dist = p_dist
        self.G, self.pos = self._generate_graph()

    def generate_graph(self):
        """generate a random graph

        Returns:
            Graph: non-direction graph
        """
        if self.seed:
            np.random.seed(self.seed)
        points = np.random.uniform(0, 1, size=(self.node_number, 2))
        pos = {i: points[i] for i in range(self.node_number)}
        matrix = distance_matrix(list(pos.values()), list(pos.values()))
        G = nx.from_numpy_array(
            matrix,
            parallel_edges=False,
        )

        nx.set_node_attributes(G, pos, name="x")
        w = nx.get_edge_attributes(G, "weight")

        w = {
            k: 1 if (random.random() < self.p_dist(v) and v < self.dist) else 0
            for k, v in w.items()
        }

        nx.set_edge_attributes(G, w, "edge_attr")

        return G, pos


class spatial_random_geometric(object):
    """The edge is created if the distance between node is smaller than a thireshold *d*"""

    def __init__(self, node_number: int, dist: float = None, seed: int = None):
        """provide the number of node, and threshod distance.

           The network space is from (0,0) to (1,1)

        Args:
            node_number (int): the total number of the graph
            dist (float): the distance threshold
            clean_graph (book): if you want to remove the isolated points
        """
        self.node_number = node_number
        self.seed = seed
        self.dist = dist
        self.G, self.pos, self.pos_spatial = self._generate_graph()

    # Depreciated, we do not clean graph any more. Keep it temporarily
    # def clean_graph_function(self):
    #     largest_cc = max(nx.connected_components(self.G), key=len)
    #     self.G = self.G.subgraph(largest_cc)

    def _get_node_class(self, pos):
        if pos[0] < 0.5 and pos[1] < 0.5:
            dist = 0.2
        elif pos[0] < 0.5 and pos[1] >= 0.5:
            dist = 0.3
        elif pos[0] >= 0.5 and pos[1] < 0.5:
            dist = 0.1
        elif pos[0] >= 0.5 and pos[1] >= 0.5:
            dist = 0.2
        else:
            raise InvalidCheckPoint

        return dist

    def _generate_graph(self):
        """generate a random graph

        Returns:
            Graph: non-direction graph
        """
        if self.seed:
            np.random.seed(self.seed)
        points = np.random.uniform(0, 1, size=(self.node_number, 2))
        pos = {i: points[i] for i in range(self.node_number)}
        pos_spatial = {i: self._get_node_class(node_pos) for i, node_pos in pos.items()}

        matrix = distance_matrix(list(pos.values()), list(pos.values()))
        G = nx.from_numpy_array(
            matrix,
            parallel_edges=False,
        )

        nx.set_node_attributes(G, pos, name="x")
        w = nx.get_edge_attributes(G, "weight")

        w = {
            k: 1 if v < max(pos_spatial[k[0]], pos_spatial[k[1]]) else 0
            for k, v in w.items()
        }

        nx.set_edge_attributes(G, w, "edge_attr")

        return G, pos, pos_spatial


class spatial_regional_geometric(object):
    def __init__(
        self,
        node_number: int,
        k: int,
        dist: float = None,
        seed: int = None,
        mapsize: int = None,
        method: str = "area",
    ):
        self.node_number = node_number
        self.k = k
        self.seed = seed
        self.dist = dist
        self.mapSize = mapsize
        self.method = method

    def multivariate_gaussian(self, pos, mu, Sigma):
        """Return the multivariate Gaussian distribution on array pos.

        pos is an array constructed by packing the meshed arrays of variables
        x_1, x_2, x_3, ..., x_k into its _last_ dimension.

        """

        n = mu.shape[0]
        Sigma_det = np.linalg.det(Sigma)
        Sigma_inv = np.linalg.inv(Sigma)
        N = np.sqrt((2 * np.pi) ** n * Sigma_det)
        # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
        # way across all the input variables.
        fac = np.einsum("...k,kl,...l->...", pos - mu, Sigma_inv, pos - mu)

        return np.exp(-fac / 2) / N

    def getGaussianMap(self, diffusionVariance, locations):
        X = np.linspace(0, self.mapSize, self.mapSize)
        Y = np.linspace(0, self.mapSize, self.mapSize)
        X, Y = np.meshgrid(X, Y)
        pos = np.empty(X.shape + (2,))
        pos[:, :, 0] = X
        pos[:, :, 1] = Y

        Z = np.zeros(X.shape, dtype=np.float32)

        for foodID in np.arange(len(locations)):
            # we assume the food diffusion is independent in x and y direction, and the variance of x and y direction are same
            if len(diffusionVariance) > 1:
                diffusion = np.array(
                    [
                        [diffusionVariance[foodID][0], 0],
                        [0, diffusionVariance[foodID][1]],
                    ]
                )

            else:
                diffusion = np.array(
                    [[diffusionVariance[0][0], 0], [0, diffusionVariance[0][1]]]
                )

            Z += self.multivariate_gaussian(pos, np.array(locations[foodID]), diffusion)

        return Z / np.max(Z)

    def points_on_line(self, p1, p2):
        fx, fy = p1
        sx, sy = p2

        m = (sy - fy) / (sx - fx)
        b = (fx * sy - sx * fy) / (fx - sx)

        if fx == sx and fy == sy:
            return []
        elif fx == sx:
            return [(fx, y) for y in range(fy + 1, sy)]
        elif fy == sy:
            return [(x, fy) for x in range(fx + 1, sx)]
        elif abs(fx - sx) > abs(fy - sy):
            return [(x, int(m * x + b)) for x in range(min(fx, sx), max(fx, sx))]
        else:
            return [(int((y - b) / m), y) for y in range(min(fy, sy), max(fy, sy))]

    def region_by_points(self, Z, p):
        fx, fy = p
        area = Z[fy - 25 : fy + 25, fx - 25 : fx + 25]
        return area

    def generate_graph(self):
        np.random.seed(self.seed)

        num_locations = np.random.randint(int(0.5 * self.k), int(1.5 * self.k))

        locations = [
            np.random.randint(self.mapSize, size=(1, 2)) for i in range(num_locations)
        ]
        diffusion = [
            (np.random.randint(1000, 8000), np.random.randint(1000, 8000))
            for i in range(num_locations)
        ]

        Z = self.getGaussianMap(diffusionVariance=diffusion, locations=locations)

        initial_graph = random_geometric(
            node_number=np.random.randint(
                int(0.5 * self.node_number), int(1.5 * self.node_number)
            ),
            dist=self.dist,
            mapSize=self.mapSize,
            seed=24,
        )

        G, pos = initial_graph.generate_graph()
        node_feature = {}
        for node in G.nodes():
            node_feature[node] = [
                np.append(pos[node] / self.mapSize, Z[pos[node][1]][pos[node][0]]),
                self.region_by_points(Z, pos[node]),
            ]

        nx.set_node_attributes(G, node_feature, "x")

        for edge in list(G.edges()):
            if self.method == "line":
                value = self.check_connection(Z, pos[edge[0]], pos[edge[1]])
                G[edge[0]][edge[1]]["value"] = value
                if (
                    len(value) > 2
                    and max(abs(z - y) for z, y in zip(value[:-1], value[1:])) > 0.005
                ):
                    G[edge[0]][edge[1]]["edge_attr"] = 0
            elif self.method == "area":
                area_a = nx.get_node_attributes(G, "x")[edge[0]][1]
                area_b = nx.get_node_attributes(G, "x")[edge[1]][1]
                if np.std(area_a - area_b) > 0.1:
                    G[edge[0]][edge[1]]["edge_attr"] = 0
            else:
                raise InvalidCheckPoint

        return G, pos, Z

    def check_connection(self, Z, node_1, node_2):
        line = self.points_on_line(node_1, node_2)
        value = [Z[pos[1]][pos[0]] for pos in line]
        return value
