# -*- coding: utf-8 -*-
"""
Created on Thu May 12 13:44:27 2022

@author: Italo'Ivo Lima Dias Pinto'
"""

import numpy as np
import itertools
from scipy.linalg import block_diag
from scipy.sparse import diags
from multiprocessing import Pool
import scipy.stats as stats
import math
import numpy.matlib
import random

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
    return ci,gamma,omega,q

def comm_time_calc(matrix):
    """
    Calculates the distribution of time nodes take to change community.
    Parameters
    ----------
    matrix : Adjacency matrix.
    Returns
    -------
    comm_times : Distribution of time a node takes to change community.
    """
    time_len,nodes = matrix.shape
    comm_times = []
    for i in range(nodes):
        continuity_sequence = matrix[1:,i]-matrix[:-1,i]==0
        length_series = []
        length = 1
        for i in continuity_sequence:
            if i:
                length += 1
            else:
                length_series.append(length)
                length = 1
        length_series.append(length)
        comm_times.extend(length_series)
    return comm_times

def cumulative_distances_calc(data,lower_threshold):
    """
    Calculates the distance between the curves of cumulative distributions of
    the data and the values generated by a power-law model with exponent
    calculated by the maximum likelihood estimator method.
    ----------
    data : Distribution of values to fit the model.
    lower_threshold : minimum value to consider for the comparison
    Returns
    -------
    ks_distances : Mean of the distances between the curves, the mean is
    calculated over the existing bins.
    """
    comm_sizes = data
    comm_sizes = np.array(comm_sizes)
    comm_sizes = comm_sizes[comm_sizes>=lower_threshold]
    
    a,_ = mle(comm_sizes,lower_threshold)
    
    ks_sample = comm_sizes
    bins = np.arange(np.amin(comm_sizes),np.amax(comm_sizes))
    
    empirical,_ = np.histogram(comm_sizes,bins=bins,density=True)
    empirical = np.cumsum(empirical)
    ks_distances = []
    for _ in range(100):
        synt_data = synt_data_generator(a,ks_sample,lower_threshold)
        synthetic,_ = np.histogram(synt_data,bins=bins,density=True)
        synthetic = np.cumsum(synthetic)
        ks_distances.append(np.mean(np.abs(empirical-synthetic)))
    ks_distances = np.mean(ks_distances)
    return ks_distances

def mle(x, xmin=None, xmax=None):
    """
    Calculates the exponent of a power law using the maximum likelihood 
    estimator method.
    ----------
    x : Distribution of values to fit the model.
    xmin : minimum value to consider for fitting the power law
    xmax : maximum value to consider for fitting the power law
    Returns
    -------
    tau : Estimated exponent calculated.
    Ln : Log-likelihood for the data to be sampled from the model obtained.
    """
    if (xmin==None):
        xmin = np.min(x)
    if (xmax==None):
        xmax = np.max(x)
    tauRange=np.array([1.000001,5])
    precision = 10**(-3)
    # Error check the precision
    if math.log10(precision) != round(math.log10(precision)):
        print('The precision must be a power of ten.')

    x = np.reshape(x, len(x))

    #Determine data type
    if np.count_nonzero(np.absolute(x - np.round(x)) > 3*(np.finfo(float).eps)) > 0:
        dataType = 'CONT'
    else:
        dataType = 'INTS'
        x = np.round(x)

    # print(dataType)
    #Truncate
    z = x[(x>=xmin) & (x<=xmax)]
    nZ = len(z)
    allZ = np.arange(xmin,xmax+1)
    nallZ = len(allZ)

    #MLE calculation

    r = xmin / xmax
    nIterations = int(-math.log10(precision))
    tauIdx=0

    for iIteration in range(1, nIterations+1):

        spacing = 10**(-iIteration)

        if iIteration == 1:
            taus = np.arange(tauRange[0], tauRange[1]+spacing, spacing)

        else: 
            if tauIdx == 0:
                taus = np.arange(taus[0], taus[1]+spacing, spacing)
            elif tauIdx == len(taus):    
                taus = np.arange(taus[-2], taus[-1]+spacing, spacing)#####
            else:
                taus = np.arange(taus[tauIdx-1], taus[tauIdx+1]+spacing, spacing)

        #return(dataType)        
        nTaus = len(taus)

        if dataType=='INTS':#this comes from Marshall et. al. 2016.
            #replicate arrays to equal size
            allZMat = np.matlib.repmat(np.reshape(allZ,(nallZ,1)),1,nTaus)
            tauMat = np.matlib.repmat(taus,nallZ,1)

            #compute the log-likelihood function
            L = - nZ*np.log(np.sum(np.power(allZMat,-tauMat),axis=0)) - (taus) * np.sum(np.log(z))

        elif dataType=='CONT':#this comes from Corral and Deluca 2013.
            #return (taus,r, nZ,z)
            L = np.log( (taus - 1) / (1 - r**(taus - 1)) )- taus * (1/nZ) * np.sum(np.log(z)) - (1 - taus) * np.log(xmin)

            if np.in1d(1,taus):
                L[taus == 1] = -np.log(np.log(1/r)) - (1/nZ) * np.sum(np.log(z))
        tauIdx=np.argmax(L)

    tau = taus[tauIdx]
    Ln = L[tauIdx]
    #it returns the exponent and the log-likelihood.
    return (tau, Ln)

def synt_data_generator(a,comm_sizes,lower_threshold):
    """
    Generates a synthetic series of values using the power law model
    calculated using MLE.
    ----------
    a : power law exponent
    comm_sizes : Distribution of values used to fit the model.
    lower_threshold : minimum value used for fitting the power law
    Returns
    -------
    synt_data : synthetic data obtained from the model within the same range
    as the original data.
    """
    comm_sizes = np.array(comm_sizes)
    comm_sizes = comm_sizes[comm_sizes>=lower_threshold]
    synt_data = np.random.zipf(a, len(comm_sizes))

    synt_data = synt_data[synt_data>=lower_threshold]
    synt_data = synt_data[synt_data<=np.amax(comm_sizes)]
    size_diff = len(comm_sizes) - len(synt_data)
    while size_diff > 0:
        add_data = np.random.zipf(a, size_diff)
        synt_data = np.append(synt_data,add_data)
        synt_data = synt_data[synt_data>=lower_threshold]
        synt_data = synt_data[synt_data<=np.amax(comm_sizes)]
        size_diff = len(comm_sizes) - len(synt_data)
    return synt_data

def comm_size_calc(matrix):
    """
    Calculates the community size distribution for all time layers.
    ----------
    matrix : dynamic community structure.
    Returns
    -------
    comm_sizes : list with the community sizes in the dynamic community
    structure.
    """
    no_tl,_ = matrix.shape
    comm_sizes = []
    for tl in range(no_tl):
        comm_labels = np.unique(matrix[tl,:])
        for label in comm_labels:
            comm_size = np.sum(matrix[tl,:]==label)
            comm_sizes.append(comm_size)
    return comm_sizes

def parameter_search(matrices,lower_gamma,higher_gamma,lower_omega,higher_omega,iterations,num_processors = 10,points=10):
    """
    Searches for the gamma and omega parameters that minimizes the spatial
    skewness and maximizes the scalefreeness for the adjacency matrices used.
    ----------
    matrices : adjacency matrices to use to obtain the community structures.
    lower_gamma : minimum value of gamma to consider in the parameter search.
    higher_gamma : maximum value of gamma to consider in the parameter search.
    lower_omega : minimum value of omega to consider in the parameter search.
    higher_omega : maximum value of omega to consider in the parameter search.
    iterations : number of iterations to perform.
    num_processors : Number of threads used to calculate the data points in 
    each iteration, the default number is 10
    points : Number of data points used to calculate skewnness and scalefreeness
    between the lower and higher values of gamma and omega for each iteration, 
    if not specified the default value of 10 is used.
    Returns
    -------
    gamma_omega_sequence : values of optimized gamma and omega for each 
    iteration.
    step_comm_sizes : values of gamma and absolute value of skewness for the 
    values used in each iteration
    step_durations : values of omega and mean distance between the cumulative
    distributions of the model and the data for each iteration.
    """
    step_comm_sizes = []
    current_omega = 0.0
    step_durations = []
    gamma_omega_sequence = []

    for _ in range(iterations):
        output_samples = []
        for adj_matrix in matrices:
            _,mat_size,_ = adj_matrix.shape
            gamma_values=np.linspace(lower_gamma,higher_gamma,points)
            parameters = [(adj_matrix, i, current_omega) for i in gamma_values]
    
            p=Pool(processes = num_processors)
            output_samples = []
            output = p.starmap(dynamic_louvain,parameters)
            output_samples.append(output)
    
        comm_sizes_samples = []
        for output in output_samples:
            comm_sizes_stats = []
            for entry in output:
                comm_structure = entry[0]
                comm_sizes = comm_size_calc(comm_structure)
                if np.mean(comm_sizes)>1 and np.mean(comm_sizes)<mat_size:
                    comm_sizes_stats.append([entry[1],np.abs(stats.skew(comm_sizes))])
            comm_sizes_samples.append(comm_sizes_stats)
        comm_sizes_samples = np.array(comm_sizes_samples)
    
        mean_comm_sizes = np.mean(comm_sizes_samples,axis=0)
        step_comm_sizes.append(np.transpose(mean_comm_sizes))

        minima_arg = np.argmin(mean_comm_sizes[:,1])
        current_gamma = mean_comm_sizes[minima_arg,0]
        step_size = mean_comm_sizes[1,0] - mean_comm_sizes[0,0]
        lower_gamma = current_gamma - step_size
        higher_gamma = current_gamma + step_size
    
        output_samples = []
        for adj_matrix in matrices:
            omega_values=np.linspace(lower_omega,higher_omega,points)
            parameters = [(adj_matrix, current_gamma, i) for i in omega_values]
    
            p=Pool(processes = num_processors)
            output = p.starmap(dynamic_louvain,parameters)
            output_samples.append(output)
        
        duration_distributions = []
        for sample in output_samples:
            sample_durations = []
            for entry in sample:
                comm_structure = entry[0]
                comm_sizes = comm_time_calc(comm_structure)
                sample_durations.append(comm_sizes)
            duration_distributions.append(sample_durations)
    
        concat_samples = [[] for i in omega_values]
        for index in range(len(omega_values)):
            for sample in duration_distributions:
                concat_samples[index].extend(sample[index])
            
        ks_distances = []
        lower_threshold = 2
        for entry in concat_samples:
            comm_sizes = entry
            ks_distances.append(cumulative_distances_calc(entry,lower_threshold))
        ks_distances = np.array(ks_distances)
        nans, x= np.isnan(ks_distances), lambda z: z.nonzero()[0]
        ks_distances[nans]= np.interp(x(nans), x(~nans), ks_distances[~nans])

        step_durations.append(np.array([omega_values,ks_distances]))
    
        minima_arg = np.argmin(ks_distances)
        current_omega = omega_values[minima_arg]
        step_size = omega_values[1] - omega_values[0]
        lower_omega = current_omega - step_size
        higher_omega = current_omega + step_size
    
        gamma_omega_sequence.append([current_gamma,current_omega])
    return gamma_omega_sequence,step_comm_sizes,step_durations

def flexibility_calc(comm_structure):
    """
    Calculates the flexibility metric for a community structure.
    ----------
    comm_structure : dynamic community structure as a numpy array with shape (time,nodes).
    Returns
    -------
    flexibility : Numpy array with the flexibility value for each network node.
    """
    discontinuities = np.logical_not((comm_structure[:-1,:] - comm_structure[1:,:]) == 0)
    L,N = discontinuities.shape
    flexibility = np.sum(discontinuities,axis=0)/L
    return flexibility

def disjointedness_calc(comm_structure):
    """
    Calculates the disjointedness metric for a community structure.
    ----------
    comm_structure : dynamic community structure as a numpy array with shape (time,nodes).
    Returns
    -------
    disjointedness : Numpy array with the flexibility value for each network node.
    """
    L,N = comm_structure.shape
    individual_transitions = []
    nodes_ind_transitions = np.zeros(N)
    for i in range(L-1):
        transitions_i = np.transpose(comm_structure[i:i+2,:])
        transitions_i = [(s,k) for s,k in transitions_i]
        unique_transitions = list(set(transitions_i))
        for transition in unique_transitions:
            occurrences = transitions_i.count(transition)
            if occurrences == 1:
                individual_transitions.append(transitions_i.index(transition))
    for transition in individual_transitions:
        nodes_ind_transitions[transition] += 1
    disjointedness = nodes_ind_transitions/(L-1)
    return disjointedness

def cohesion_calc(comm_structure):
    """
    Calculates the cohesion matrix and cohesion strength for a community structure.
    ----------
    comm_structure : dynamic community structure as a numpy array with shape (time,nodes).
    Returns
    -------
    cohesion_matrix : A 2-d numpy array with shape (nodes,nodes) and the pairwise cohesion for each node pair.
    cohesion_str : A numpy array with the cohesion strength for each node of the network.
    """
    L,N = comm_structure.shape
    cohesion_matrix = np.zeros((N,N))
    for i in range(L-1):
        transitions_i = np.transpose(comm_structure[i:i+2,:])
        transitions_i = [(s,k) for s,k in transitions_i]
        unique_transitions = list(set(transitions_i))
        for transition in unique_transitions:
            occurrences = transitions_i.count(transition)
            nodes = [i for i, x in enumerate(transitions_i) if x == transition]
            indexes_to_increment = list(itertools.permutations(nodes, 2))
            for i,j in indexes_to_increment:
                cohesion_matrix[i,j] += 1
    cohesion_matrix = cohesion_matrix/(L-1)
    cohesion_str = np.sum(cohesion_matrix,axis=0)
    return cohesion_matrix,cohesion_str

def promiscuity_calc(comm_structure):
    """
    Calculates the promiscuity for a community structure.
    ----------
    comm_structure : dynamic community structure as a numpy array with shape (time,nodes).
    Returns
    -------
    promiscuity : A numpy array with the promiscuity value for each node.
    """
    L,N = comm_structure.shape
    communities = np.amax(comm_structure)
    node_communities = []
    for i in range(N):
        node_communities.append(len(np.unique(comm_structure[:,i])))
    node_communities = np.array(node_communities)
    promiscuity = node_communities/communities
    return promiscuity

def dynamical_links(comm_structure):
    """
    Calculates a dynamical links 3d object used to calculate the pairwise allegiance and intermittence.
    ----------
    comm_structure : dynamic community structure as a numpy array with shape (time,nodes).
    Returns
    -------
    links_t : 3d object with shape (time,node,node) which stores the pair of nodes in the same community in each time layer.
    """
    time_range,nodes = comm_structure.shape
    links_t = []
    for t in range(time_range):
        struc_array = comm_structure[t,:]
        time_stamp = np.zeros((nodes,nodes))
        for i in range(nodes):
            for j in range(nodes):
                if struc_array[i] == struc_array[j]:
                    time_stamp[i,j] = 1
        links_t.append(time_stamp)
    links_t = np.array(links_t)
    return links_t

def allegiance_calc(comm_structure):
    """
    Calculates the pairwise allegiance of a community structure.
    ----------
    comm_structure : dynamic community structure as a numpy array with shape (time,nodes).
    Returns
    -------
    allegiance : matrix which stores the pairwise allegiances averaged on time layers.
    """
    instant_allegiances = dynamical_links(comm_structure)
    allegiance = np.mean(instant_allegiances, axis=0)
    return allegiance

def intermittence_calc(comm_structure):
    """
    Calculates the pairwise intermittence of a community structure.
    ----------
    comm_structure : dynamic community structure as a numpy array with shape (time,nodes).
    Returns
    -------
    intermittence : matrix which stores the pairwise intermittence averaged on time layers.
    """
    instant_allegiances = dynamical_links(comm_structure)
    alleg_differences = np.abs(instant_allegiances[1:,:,:] - instant_allegiances[:-1,:,:])
    intermitence = np.mean(alleg_differences, axis=0)
    return intermitence