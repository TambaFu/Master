import numpy as np
import torch
from sklearn import cluster
from sklearn.preprocessing import normalize
from scipy.linalg import orth
import scipy.sparse as sparse
from munkres import Munkres
from sklearn.utils import check_random_state
from scipy.sparse.csgraph import laplacian as compute_laplacian


def regularizer_pnorm(c, p):
    return torch.pow(torch.abs(c), p).sum()


def sklearn_predict(A, n_clusters):
    spec = cluster.SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
    return spec.fit_predict(A)


def accuracy(pred, labels):
    mapped = best_map(labels, pred)
    acc = np.mean(labels == mapped)
    return acc


def subspace_preserving_error(A, labels, n_clusters):
    one_hot_labels = torch.zeros([A.shape[0], n_clusters], device=A.device)
    for i in range(A.shape[0]):
        one_hot_labels[i, labels[i]] = 1.0
    mask = one_hot_labels @ one_hot_labels.T
    l1_norm = torch.norm(A, p=1, dim=1)
    masked_l1_norm = torch.norm(mask * A, p=1, dim=1)
    e = torch.mean(1.0 - masked_l1_norm / (l1_norm + 1e-6)) * 100.0
    return e


def normalized_laplacian(A):
    D = torch.sum(A, dim=1)
    D_sqrt = torch.diag(1.0 / torch.sqrt(D + 1e-6))
    L = torch.eye(A.shape[0], device=A.device) - D_sqrt @ A @ D_sqrt
    return L


def connectivity(A, labels, n_clusters):
    c = []
    for i in range(n_clusters):
        idx = (labels == i)
        A_i = A[idx][:, idx]
        L_i = normalized_laplacian(A_i)
        eig_vals = torch.linalg.eigvalsh(L_i)
        c.append(eig_vals[1].item() if len(eig_vals) > 1 else 0.0)
    return np.min(c)


def topK(A, k, sym=True):
    val, indices = torch.topk(A, dim=1, k=k)
    Coef = torch.zeros_like(A).scatter(1, indices, val)
    if sym:
        Coef = (Coef + Coef.T) / 2.0
    return Coef


def best_map(L1, L2):
    Label1 = np.unique(L1)
    Label2 = np.unique(L2)
    nClass = max(len(Label1), len(Label2))
    G = np.zeros((nClass, nClass))
    for i, label1 in enumerate(Label1):
        idx1 = L1 == label1
        for j, label2 in enumerate(Label2):
            idx2 = L2 == label2
            G[i, j] = np.sum(idx1 & idx2)

    m = Munkres()
    mapping = np.array(m.compute(-G.T))
    reorder = mapping[:, 1]
    newL2 = np.zeros_like(L2)
    for i, label2 in enumerate(Label2):
        newL2[L2 == label2] = Label1[reorder[i]]
    return newL2


def err_rate(gt_s, s):
    c_x = best_map(gt_s, s)
    missrate = np.mean(gt_s != c_x)
    return missrate


def gen_union_of_subspaces(ambient_dim, subspace_dim, num_subspaces, num_points_per_subspace, noise_level=0.0):
    data = np.empty((num_subspaces * num_points_per_subspace, ambient_dim))
    label = np.empty(num_subspaces * num_points_per_subspace, dtype=int)

    for i in range(num_subspaces):
        basis = orth(np.random.randn(ambient_dim, subspace_dim))
        coeff = normalize(np.random.randn(subspace_dim, num_points_per_subspace), norm='l2', axis=0)
        subspace_points = (basis @ coeff).T
        start = i * num_points_per_subspace
        end = start + num_points_per_subspace
        data[start:end] = subspace_points
        label[start:end] = i

    data += np.random.randn(*data.shape) * noise_level
    return data, label


def dim_reduction(X, dim):
    if dim == 0:
        return X
    _, vecs = np.linalg.eigh(X.T @ X)
    return X @ vecs[:, -dim:]


def p_normalize(x, p=2):
    return x / (torch.norm(x, p=p, dim=1, keepdim=True) + 1e-6)


def minmax_normalize(x):
    rmax, _ = torch.max(x, dim=1, keepdim=True)
    rmin, _ = torch.min(x, dim=1, keepdim=True)
    return (x - rmin) / (rmax - rmin + 1e-6)


def spectral_clustering(affinity_matrix_, n_clusters, k, seed=1, n_init=20):
    affinity_matrix_ = (affinity_matrix_ + affinity_matrix_.T) / 2  # force symmetric
    random_state = check_random_state(seed)
    laplacian = compute_laplacian(affinity_matrix_, normed=True)
    _, vec = sparse.linalg.eigsh(sparse.identity(laplacian.shape[0]) - laplacian,
                                 k=k, which='LA')
    embedding = normalize(vec)
    _, labels_, _ = cluster.k_means(embedding, n_clusters, random_state=seed, n_init=n_init)
    return labels_
