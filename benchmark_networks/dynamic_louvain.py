import numpy as np
import itertools
from scipy.linalg import block_diag
from scipy.sparse import diags

def get_rng(seed=None):
    """
    By default, or if `seed` is np.random, return the global RandomState
    instance used by np.random.
    If `seed` is a RandomState instance, return it unchanged.
    Otherwise, use the passed (hashable) argument to seed a new instance
    of RandomState and return it.
    Parameters
    ----------
    seed : hashable or np.random.RandomState or np.random, optional
    Returns
    -------
    np.random.RandomState
    """
    if seed is None or seed == np.random:
        return np.random.mtrand._rand
    elif isinstance(seed, np.random.RandomState):
        return seed
    try:
        rstate =  np.random.RandomState(seed)
    except ValueError:
        rstate = np.random.RandomState(random.Random(seed).randint(0, 2**32-1))
    return rstate

def dynamic_louvain(adj_matrix, gamma=1, omega=1, ci=None, seed=None):
    """
    Calculates the time evolving community structure using the generalized 
    Louvain algorithm (see Jutla et al. 2011), using gamma as a spatial
    resolution parameter and omega as temporal resolution parameter.
    Parameters
    ----------
    adj_matrix : The adjacency matrix from which the algorithm will generate
    the community structure.
    gamma : Spatial resolution parameter.
    omega : Temporal resolution parameter.
    ci : Initial community structure, if not specified or None will start
    with each node in a different community.
    seed : Seed for the random number generator.
    Returns
    -------
    gamma : Spatial resolution value used.
    omega : Temporal resolution value used.
    ci : Final community structure generated.
    q : Quality metric value of the final community structure.
    """
    rng = get_rng(seed)
    time_len,nodes,_ = adj_matrix.shape
    n = time_len*nodes

    if ci is None:
        ci = np.arange(n) + 1
    else:
        if len(ci) != n:
            raise RuntimeError('initial ci vector size must equal N')
        _, ci = np.unique(ci, return_inverse=True)
        ci += 1
    Mb = ci.copy()
    renormalize = False

    if not(np.all(np.logical_not(adj_matrix < 0))):
            raise RuntimeError("Input connection matrix contains negative "
                'weights but objective function dealing with negative weights '
                'was not selected')

    l,N,_ = adj_matrix.shape
    Bs = []

    twomu = 0
    for s in range(l):
        W = adj_matrix[s]
        s = np.sum(W)
        twomu += s
        Bs.append(W - gamma * np.outer(np.sum(W, axis=1), np.sum(W, axis=0)) / s)

    twomu=twomu + 2.0*omega*N*(l-1)
    diagonals = np.ones((2,N*(l-1))).tolist()
    temporal_part = diags(diagonals, [-N,N]).toarray()
    B = block_diag(*Bs) + omega*temporal_part

    Hnm = np.zeros((n, n))
    for m in range(1, n + 1):
        Hnm[:, m - 1] = np.sum(B[:, ci == m], axis=1)  # node to module degree
    H = np.sum(Hnm, axis=1)  # node degree
    Hm = np.sum(Hnm, axis=0)  # module degree

    q0 = -np.inf
    # compute modularity
    q = np.trace(B) / twomu

    first_iteration = True

    while q - q0 > 1e-10:
        it = 0
        flag = True
        while flag:
            it += 1
            if it > 1000:
                raise RuntimeError('Modularity infinite loop style G. '
                                    'Please contact the developer.')
            flag = False
            for u in rng.permutation(n):
                ma = Mb[u] - 1
                dQ = Hnm[u, :] - Hnm[u, ma] + B[u, u]  # algorithm condition
                dQ[ma] = 0

                max_dq = np.max(dQ)
                if max_dq > 1e-10:
                    flag = True
                    mb = np.argmax(dQ)

                    Hnm[:, mb] += B[:, u]
                    Hnm[:, ma] -= B[:, u]  # change node-to-module strengths

                    Hm[mb] += H[u]
                    Hm[ma] -= H[u]  # change module strengths

                    Mb[u] = mb + 1

        _, Mb = np.unique(Mb, return_inverse=True)
        Mb += 1

        M0 = ci.copy()
        if first_iteration:
            ci = Mb.copy()
            first_iteration = False
        else:
            for u in range(1, n + 1):
                ci[M0 == u] = Mb[u - 1]  # assign new modules

        n = np.max(Mb)
        b1 = np.zeros((n, n))
        for i,j in itertools.combinations_with_replacement(range(1, n + 1),2):
            # pool weights of nodes in same module
            bm = np.sum(B[np.ix_(Mb == i, Mb == j)])
            b1[i - 1, j - 1] = bm
            b1[j - 1, i - 1] = bm
        B = b1.copy()

        Mb = np.arange(1, n + 1)
        Hnm = B.copy()
        H = np.sum(B, axis=0)
        Hm = H.copy()

        q0 = q

        q = np.trace(B)/twomu  # compute modularity
    ci = np.reshape(ci,(time_len,nodes))
    return gamma,omega,ci,q