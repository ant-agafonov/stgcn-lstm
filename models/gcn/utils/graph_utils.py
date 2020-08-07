import pandas as pd
import numpy as np
import csv
from scipy.sparse.linalg import eigsh

def get_adjacency_matrix(distance_df_filename, num_of_vertices, file_type='lines',
                         use_cost=False, sigma2=0.1, epsilon=0.5, scaling=True):
    """
    Read and scale the adjacency matrix from the csv file with the following format: (from node, to node, cost)

    :param distance_df_filename: str, path of the csv file contains edges information
    :param num_of_vertices: int, the number of vertices
    :param file_type:
    :param use_cost: bool, whether use the cost in the adjacency matrix or just 1
    :param sigma2: float, scalar of matrix W.
    :param epsilon: float, thresholds to control the sparsity of matrix W.
    :param scaling: bool, whether applies numerical scaling on W.
    :return: np.ndarray, adjacency matrix
    """
    if file_type == 'matrix':
        # csv file contains matrix NxN
        adj_matrix = pd.read_csv(distance_df_filename, header=None).values
    else:
        # csv file contains lines "from,to,cost"
        with open(distance_df_filename, 'r') as csvfile:
            reader = csv.reader(csvfile)
            _ = csvfile.__next__()  # skip header
            edges = [(int(row[0]), int(row[1]), float(row[2])) for row in reader]  # from,to,cost

        adj_matrix = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)
        for i, j, cost in edges:
            adj_matrix[i, j] = cost if use_cost else 1
            if adj_matrix[j, i] == 0:
                adj_matrix[j, i] = cost if use_cost else 1

    # check whether adj_matrix is a 0/1 matrix.
    if scaling and set(np.unique(adj_matrix)) == {0, 1}:
        print('The input graph is a 0/1 matrix; set "scaling" to False.')
        scaling = False

    if not scaling:
        return adj_matrix

    # scaling according to STSGCN_IJCAI-18 eq.10
    n = adj_matrix.shape[0]
    adj_matrix = adj_matrix / 10000.
    adj_matrix_squared, adj_matrix_mask = adj_matrix * adj_matrix, np.ones([n, n]) - np.identity(n)
    return np.exp(-adj_matrix_squared / sigma2) * (np.exp(-adj_matrix_squared / sigma2) >= epsilon) * adj_matrix_mask


def normalize_adjacency_matrix(adjacency_matrix):
    """Symmetrically normalize adjacency matrix."""
    rowsum = np.array(adjacency_matrix.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return adjacency_matrix.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)


def calculate_scaled_laplacian(adjacency_matrix):
    """Calculate scaled Laplacian matrix."""
    adj_normalized = normalize_adjacency_matrix(adjacency_matrix)
    laplacian = np.eye(adjacency_matrix.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - np.eye(adjacency_matrix.shape[0])
    return scaled_laplacian


def chebyshev_polynomials_from_scaled_laplacian(scaled_laplacian, max_order):
    """
    Compute a list of Chebyshev polynomials from T_0 to T_{K-1} using a scaled Laplacian matrix
    :param scaled_laplacian: scaled Laplacian, np.ndarray, shape (N, N)
    :param max_order: the maximum order of chebyshev polynomials
    :return: list[np.ndarray], length: K, from T_0 to T_{K-1}
    """
    cheb_polynomials = [np.identity(scaled_laplacian.shape[0], dtype=np.float32),
                        np.ndarray.astype(scaled_laplacian.copy(), dtype=np.float32)]
    for i in range(2, max_order):
        cheb_polynomials.append(
            np.ndarray.astype(2 * scaled_laplacian * cheb_polynomials[i - 1] - cheb_polynomials[i - 2],
                              dtype=np.float32))
    return cheb_polynomials


def chebyshev_polynomials(adjacency_matrix, max_order):
    """
    Compute a list of Chebyshev polynomials from adjacency matrix
    :param adjacency_matrix: adjacency matrix, np.ndarray, shape (N, N)
    :param max_order: the maximum order of chebyshev polynomials
    :return: list[np.ndarray], length: K, from T_0 to T_{K-1}
    """
    print("Calculating Chebyshev polynomials up to order {}...".format(max_order))
    scaled_laplacian = calculate_scaled_laplacian(adjacency_matrix)
    cheb_polynomials = chebyshev_polynomials_from_scaled_laplacian(scaled_laplacian, max_order)
    return cheb_polynomials
