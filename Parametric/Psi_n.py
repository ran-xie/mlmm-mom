import os
import torch
import pickle
import shutil
import itertools
import numpy as np

## enumerate non crossing pair partitions

def is_non_crossing(pairs):
    """
    Check if a given partition (list of pairs) is non-crossing.
    A partition is non-crossing if no pair crosses any other pair.
    """
    for i, (a1, a2) in enumerate(pairs):
        for b1, b2 in pairs[i+1:]:
            if (a1 < b1 < a2 < b2) or (b1 < a1 < b2 < a2):
                return False
    return True

def generate_pair_partitions(elements):
    """
    Recursively generate all pair partitions of a given set of elements.
    """
    if not elements:
        return [[]]
    
    first = elements[0]
    rest = elements[1:]
    
    partitions = []
    
    for i, second in enumerate(rest):
        pairs = (first, second)
        remaining_elements = rest[:i] + rest[i+1:]
        
        # Recursively generate partitions of the remaining elements
        for partition in generate_pair_partitions(remaining_elements):
            partitions.append([pairs] + partition)
    
    return partitions

def non_crossing_pair_partitions(n):
    """
    Generate all non-crossing pair partitions of {1, 2, ..., 2n}.
    """
    elements = list(range(1, 2*n + 1))
    all_pair_partitions = generate_pair_partitions(elements)
    
    # Filter the non-crossing partitions
    non_crossing_partitions = [p for p in all_pair_partitions if is_non_crossing(p)]

    return non_crossing_partitions

## compute kreweras complements
def permutation_to_partition(perm):
    """
    Given a permutation, returns the partition of its elements into cycles.
    
    Parameters:
    perm (list of int): A permutation of {1, 2, ..., n}, represented as a list.
    
    Returns:
    list of tuples: A partition of the permutation, where each tuple represents a cycle.
    """
    n = len(perm)
    visited = [False] * n  # Track visited elements
    partition = []         # Store the cycles (blocks of the partition)
    
    # Loop over each element to identify cycles
    for i in range(n):
        if not visited[i]:
            cycle = []
            current = i
            
            # Follow the cycle starting from element i
            while not visited[current]:
                visited[current] = True
                cycle.append(current + 1)  # Add 1 to convert 0-based to 1-based indexing
                current = perm[current] - 1  # Move to the next element in the cycle
            
            # Add the found cycle as a block in the partition
            partition.append(tuple(cycle))
    
    return partition

def partition_to_permutation(partition):
    """
    Given a partition of elements (as cycles), returns the corresponding permutation.
    
    Parameters:
    partition (list of tuples): A partition where each tuple represents a cycle.
    
    Returns:
    list of int: The corresponding permutation of {1, 2, ..., n}.
    """
    # Find the largest element to determine the size of the permutation
    n = max(max(block) for block in partition)
    
    # Initialize a permutation list of size n
    perm = [0] * n
    
    # Loop through each block (cycle) in the partition
    for block in partition:
        m = len(block)
        # For each element in the cycle, map it to the next one
        for i in range(m):
            perm[block[i] - 1] = block[(i + 1) % m]  # Connect element to the next in the cycle
    
    return perm

def compose_permutations(perm1, perm2):
    """
    Given two permutations perm1 and perm2, return their composition perm1 ◦ perm2.
    
    Parameters:
    perm1 (list of int): The first permutation σ (applied second).
    perm2 (list of int): The second permutation τ (applied first).
    
    Returns:
    list of int: The composition perm1 ◦ perm2, a new permutation.
    """
    n = len(perm1)
    composed_perm = [0] * n
    
    for i in range(n):
        composed_perm[i] = perm1[perm2[i] - 1]  # Apply perm2 first, then perm1
    
    return composed_perm

def cyclic_permutation(n):
    """
    Returns the cyclic permutation of {1, 2, ..., n}, where 1 -> 2 -> ... -> n -> 1.
    
    Parameters:
    n (int): The size of the set to permute.
    
    Returns:
    list of int: The cyclic permutation as a list.
    """
    return [i % n + 1 for i in range(1, n + 1)]

def kreweras_complement(pair_part):
    n = len(pair_part)*2
    gamma_n = cyclic_permutation(n)
    pair_part_perm = partition_to_permutation(pair_part)
    kre_perm = compose_permutations(pair_part_perm, gamma_n)
    kre = permutation_to_partition(kre_perm)
    return kre

cache_folder_default = f"cache_find_theta/cache_find_theta_fraction"

_trace_cache = {}
def clear_cache():
    _trace_cache.clear()
def get_from_cache(cache_key):
    return _trace_cache.get(cache_key)
def save_to_cache(cache_key, value):
    _trace_cache[cache_key] = value

def pair_part_trace(pair_part,p,n,B,U_list,key_to_pos_dict,Sig_mom_tensor,F_check_list=None,cache_folder = cache_folder_default):
    '''
    for each pair partition pi, we compute one component of Psi_n
    r_list = (r1,...,rl).
    m=max(I_1,...,I_k,p)
    key_to_pos_dict: a dict that maps key (number of times i appeared in r_list) to the position of that trace in Sig_mom_list.
    '''
    k = len(U_list)
    l = len(pair_part)
    kre_part = kreweras_complement(pair_part)

    sum_trace = torch.tensor(0.,requires_grad=True)
    for r_list in itertools.product(range(1,k+1), repeat=l):
        # k: Maximum value for each element 
        # l: Length of the tuple
        # kre_part = kreweras_complement(pair_part)
        prod = torch.tensor(1.,requires_grad=True)
        
        for block in kre_part:
            if block[0]%2==1:
                cache_key = (tuple(pair_part), tuple(r_list), tuple(block))
                cached_result = get_from_cache(cache_key)
                if cached_result is not None:
                    prod_block = cached_result
                else:
                    mat_block = np.eye(n)
                    if F_check_list is not None:
                        for i in block:
                            mat_block = mat_block @ F_check_list[r_list[int((i+1)/2-1)]-1] # each of these can be precomputed.
                    else:
                        for i in block:
                            mat_block = mat_block @ B @ U_list[r_list[int((i+1)/2-1)]-1] @ U_list[r_list[int((i+1)/2-1)]-1].T # each of these can be precomputed.
                    prod_block = torch.tensor(np.trace(mat_block)/p)
                    save_to_cache(cache_key, prod_block)
            else:
                r_list_sig = r_list[1:] + (r_list[0],)
                sigma_list = [r_list_sig[int(i/2-1)] for i in block] 
                count_array = [sigma_list.count(i) for i in range(1,k+1)]
                prod_block = Sig_mom_tensor[key_to_pos_dict[tuple(count_array)]]
            prod = prod * prod_block  
        sum_trace = sum_trace+ prod
    return sum_trace

def Psi_n(Sig_mom_list,L,p,n,B,U_list,key_to_pos_dict,from_find_theta=False,F_check_list=None,cache_folder = cache_folder_default):
    '''
    moments of sigma1,...,sigmak to moments of the sum of squares matrix
    L: max of L-th moment of sum of squares matrix
    '''
    if not from_find_theta:
        clear_cache()
    moments = torch.zeros(L) 
    if F_check_list is None:
        F_check_list = []
        for r in range(len(U_list)):
            F_check_list.append(B@U_list[r]@U_list[r].T)
    for l in range(1,L+1):
        partitions = non_crossing_pair_partitions(l)
        moment_l = torch.tensor(0.,requires_grad=True)
        for partition in partitions:
            trace_value = pair_part_trace(partition, p, n, B, U_list, key_to_pos_dict,Sig_mom_list,F_check_list,cache_folder=cache_folder)
            moment_l = moment_l + trace_value
        moment_l = moment_l 
        moments[l-1] = moment_l
    return moments
