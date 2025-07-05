import numpy as np
from scipy import sparse as sp
import networkx as nx

from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph


def dopca(x, dim=50):
    pca = PCA(n_components=dim)
    pca.fit(x)
    return pca.transform(x)


def get_adj(count, k=15, pca=50, mode="connectivity"):
    if pca:
        countp = dopca(count, dim=pca)
    else:
        countp = count
    A = kneighbors_graph(countp, k, mode=mode, metric="euclidean", include_self=True)
    adj = A.toarray()
    adj_n = norm_adj(adj)
    return adj, adj_n


def degree_power(A, k):
    degrees = np.power(np.array(A.sum(1)), k).flatten()
    degrees[np.isinf(degrees)] = 0.
    if sp.issparse(A):
        D = sp.diags(degrees)
    else:
        D = np.diag(degrees)
    return D


def norm_adj(A):
    normalized_D = degree_power(A, -0.5)
    output = normalized_D.dot(A).dot(normalized_D)
    return output


def create_edge_index(x, k=5, pca_dim=50):
    x_p = dopca(x, dim=pca_dim)
    A = kneighbors_graph(x_p, k, mode='connectivity', metric="euclidean", include_self=True)
    A = A.toarray()
    A = np.float32(A)
    index = np.nonzero(A)
    e_index = np.concatenate((np.expand_dims(index[0], axis=0), np.expand_dims(index[1], axis=0)), axis=0)
    row_indices, col_indices = np.nonzero(A)
    indices_tuples = [(row, col) for row, col in zip(row_indices, col_indices)]
    return e_index, A, indices_tuples


def get_dist_matrix(node_num, edges):
    make_graph = nx.Graph()
    make_graph.add_edges_from(edges)
    dist_matrix = np.zeros((node_num, node_num))
    dist_matrix.fill(1e9)
    row, col = np.diag_indices_from(dist_matrix)
    dist_matrix[row, col] = 0
    graph_nodes = sorted(make_graph.nodes.keys())
    all_distance = dict(nx.all_pairs_shortest_path_length(make_graph))
    for dist in graph_nodes:
        node_relative_distance = dict(sorted(all_distance[dist].items(), key=lambda x: x[0]))
        temp_node_dist_dict = {i: node_relative_distance.get(i) if \
            node_relative_distance.get(i) != None else 1e9 for i in
                               graph_nodes}
        temp_node_dist_list = list(temp_node_dist_dict.values())
        dist_matrix[dist][graph_nodes] = temp_node_dist_list
    return dist_matrix.astype(np.float32)
