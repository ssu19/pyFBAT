import pandas as pd
import numpy as np
import torch
import math
import scipy.stats
import multiprocessing as mp
from functools import partial
from tqdm.auto import tqdm
from .nuclearfamily import remap_index
import os


def compute_fbat_contributions_for_family_vectorized(
    family,
    genotype_tensor,
    marker_j,
    EX,  # scalar
    VX,  # scalar
    CX,  # scalar
    pheno_trait,
    offset=0.0
):
    """
    Vectorized calculation of FBAT contributions for a family.
    
    Parameters:
    family : NuclearFamily
        Family object
    genotype_tensor : torch.Tensor
        Tensor of genotypes
    marker_j : int
        Index of marker
    EX : float
        Expected value of genotype distribution
    VX : float
        Variance of genotype distribution
    CX : float
        Covariance between siblings' genotypes
    pheno_trait : str
        Name of phenotype trait
    offset : float, default 0.0
        Offset for phenotype
        
    Returns:
    tuple
        (numerator, variance)
    """
    x_list = []
    t_list = []
    for child in family.offspring:
        cidx = child['idx']
        if cidx < 0:
            continue
        gval = genotype_tensor[cidx, marker_j].item()
        if gval == -9:
            continue

        x_list.append(float(gval))
        # Subtract offset from phenotype if desired
        t_list.append(child['phenos'][pheno_trait] - offset)

    n_off = len(x_list)
    if n_off == 0:
        return (0.0, 0.0)

    # Convert to Torch Tensors
    X_t = torch.tensor(x_list, dtype=torch.float32)  # shape [n_off]
    T_t = torch.tensor(t_list, dtype=torch.float32)  # shape [n_off]

    # 1) numerator = T^T (X - EX)
    X_minus_EX = X_t - EX
    numerator = T_t.dot(X_minus_EX)  # dot product

    # 2) variance = T^T * C * T,  where C is (n_off x n_off) with diag=VX, offdiag=CX
    C = torch.full((n_off, n_off), CX, dtype=torch.float32)  # fill with CX
    C.fill_diagonal_(VX)  # replace diagonal entries with VX
    # Then variance = T^T * C * T
    #   shape: (1 x n_off) @ (n_off x n_off) @ (n_off x 1) => scalar
    variance = T_t @ C @ T_t

    # Return Python floats
    return (float(numerator.item()), float(variance.item()))

def fbat_for_one_marker(
        family,
        marker_j,
        marker_str,
        genotype_tensor,
        EX_array,
        VX_array,
        CX_array,
        n_max,
        pheno_trait,
        offset):
    """
    Calculate FBAT statistic for one marker in one family.
    
    Parameters:
    family : NuclearFamily
        Family object
    marker_j : int
        Index of marker
    marker_str : str
        Name of marker
    genotype_tensor : torch.Tensor
        Tensor of genotypes
    EX_array : torch.Tensor
        Tensor of expected values
    VX_array : torch.Tensor
        Tensor of variances
    CX_array : torch.Tensor
        Tensor of covariances
    n_max : int
        Maximum number of offspring to consider
    pheno_trait : str
        Name of phenotype trait
    offset : float
        Offset for phenotype
        
    Returns:
    list
        List containing a dictionary with FBAT contributions, or empty list
    """
    father_idx = family.father_idx
    mother_idx = family.mother_idx
    child_idxs = [c['idx'] for c in family.offspring if c['idx'] >= 0]
    if not child_idxs:
        return []

    # Parental genotype -> 0/1/2/3 index
    f_idx = remap_index(
        genotype_tensor[father_idx, marker_j].item() if father_idx >= 0 else -9)
    m_idx = remap_index(
        genotype_tensor[mother_idx, marker_j].item() if mother_idx >= 0 else -9)
    
    # Child genotypes => x0_obs, x1_obs, x2_obs
    child_genos = genotype_tensor[child_idxs, marker_j]
    valid = child_genos != -9
    if not valid.any():  # nothing observed
        return []

    valid_g = child_genos[valid]
    
    x0_obs = (valid_g == 0).sum().item()
    x1_obs = (valid_g == 1).sum().item()
    x2_obs = (valid_g == 2).sum().item()
    if max(x0_obs, x1_obs, x2_obs) > n_max:
        return []

    # EX / VX / CX lookup
    ex = EX_array[f_idx, m_idx, x0_obs, x1_obs, x2_obs].item()
    vx = VX_array[f_idx, m_idx, x0_obs, x1_obs, x2_obs].item()
    cx = CX_array[f_idx, m_idx, x0_obs, x1_obs, x2_obs].item()

    # Skip if NaN
    if math.isnan(ex) or math.isnan(vx) or math.isnan(cx):
        return []

    num, var = compute_fbat_contributions_for_family_vectorized(
        family,
        genotype_tensor,
        marker_j,
        ex, vx, cx,
        pheno_trait=pheno_trait,
        offset=offset
    )

    return [{
        "FID"         : family.fid,
        "marker"      : marker_str,
        "num_contrib" : num,
        "var_contrib" : var
    }]

def worker_marker_chunk(
        marker_slice,
        marker_list,
        families,
        genotype_tensor,
        EX_array, VX_array, CX_array,
        n_max, pheno_trait, offset):
    """
    Worker function for processing a chunk of markers in parallel.
    
    Parameters:
    marker_slice : slice or list
        Slice or list of marker indices to process
    marker_list : list
        List of marker names
    families : list
        List of NuclearFamily objects
    genotype_tensor : torch.Tensor
        Tensor of genotypes
    EX_array : torch.Tensor
        Tensor of expected values
    VX_array : torch.Tensor
        Tensor of variances
    CX_array : torch.Tensor
        Tensor of covariances
    n_max : int
        Maximum number of offspring to consider
    pheno_trait : str
        Name of phenotype trait
    offset : float
        Offset for phenotype
        
    Returns:
    list
        List of FBAT contribution dictionaries
    """
    if isinstance(marker_slice, slice):
        marker_slice = range(marker_slice.start, marker_slice.stop)
    out = []
    
    for mj in marker_slice:
        mstr = marker_list[mj]
        for fam in families:
            out.extend(
                fbat_for_one_marker(
                    fam, mj, mstr,
                    genotype_tensor,
                    EX_array, VX_array, CX_array,
                    n_max, pheno_trait, offset)
            )
    return out

def compute_fbat_parallel_markers(
    family_dict,
    genotype_tensor,
    marker_list,
    EX_array,
    VX_array,
    CX_array,
    n_max,
    pheno_trait,
    offset=0.0,
    n_jobs=4,
    chunk_size=500,
    show_progress=True 
):
    """
    Compute FBAT statistics in parallel for all markers.
    
    Parameters:
    family_dict : dict
        Dictionary of NuclearFamily objects
    genotype_tensor : torch.Tensor
        Tensor of genotypes
    marker_list : list
        List of marker names
    EX_array : torch.Tensor
        Tensor of expected values
    VX_array : torch.Tensor
        Tensor of variances
    CX_array : torch.Tensor
        Tensor of covariances
    n_max : int
        Maximum number of offspring to consider
    pheno_trait : str
        Name of phenotype trait
    offset : float, default 0.0
        Offset for phenotype
    n_jobs : int, default 4
        Number of parallel jobs
    chunk_size : int, default 500
        Size of marker chunks for parallel processing
    show_progress : bool, default True
        Whether to show progress bar
        
    Returns:
    list
        List of FBAT contribution dictionaries
    """
    if n_jobs == 1:
        chunk_size = len(marker_list)

    # Material to broadcast to workers
    families = list(family_dict.values())
    total_markers = len(marker_list)

    # Make equal-sized slices
    marker_chunks = [
        range(i, min(i + chunk_size, total_markers))
        for i in range(0, total_markers, chunk_size)
    ]

    # Partial function for worker
    _worker = partial(
        worker_marker_chunk,
        marker_list=marker_list,
        families=families,
        genotype_tensor=genotype_tensor,
        EX_array=EX_array,
        VX_array=VX_array,
        CX_array=CX_array,
        n_max=n_max,
        pheno_trait=pheno_trait,
        offset=offset
    )

    if n_jobs == 1:
        # Serial execution
        if show_progress:
            result_lists = [_worker(slc) for slc in tqdm(marker_chunks, desc="Processing markers")]
        else:
            result_lists = [_worker(slc) for slc in marker_chunks]
    else:
        # Parallel execution
        if show_progress:
            total_chunks = len(marker_chunks)
            print(f"Processing {total_markers} markers in {total_chunks} chunks using {n_jobs} processes")
            
            # Process-safe counter
            manager = mp.Manager()
            counter = manager.Value('i', 0)
            lock = manager.Lock()
            
            # Progress bar
            pbar = tqdm(total=total_chunks, desc="Processing marker chunks")
            
            # Callback function for progress updates
            def update_pbar(result):
                nonlocal counter
                with lock:
                    counter.value += 1
                    pbar.update(1)
            
            # Asynchronous map
            pool = mp.Pool(processes=n_jobs)
            async_results = []
            
            for chunk in marker_chunks:
                async_result = pool.apply_async(_worker, args=(chunk,), callback=update_pbar)
                async_results.append(async_result)
            
            # Wait for completion
            result_lists = [ar.get() for ar in async_results]
            
            # Clean up
            pool.close()
            pool.join()
            pbar.close()
        else:
            # Standard parallel execution
            with mp.Pool(processes=n_jobs) as pool:
                result_lists = pool.map(_worker, marker_chunks)

    # Flatten results
    results = [item for sub in result_lists for item in sub]
    return results

def compute_fbat_final_df(
    family_dict,
    genotype_tensor,
    marker_list,
    EX_array,
    VX_array,
    CX_array,
    n_max,
    pheno_trait,
    offset=0.0,
    num_cpu=os.cpu_count(),
    chunk_size=500,
    show_progress=True
):
    """
    Compute FBAT statistics and aggregate results into a dataframe.
    
    Parameters:
    family_dict : dict
        Dictionary of NuclearFamily objects
    genotype_tensor : torch.Tensor
        Tensor of genotypes
    marker_list : list
        List of marker names
    EX_array : torch.Tensor
        Tensor of expected values
    VX_array : torch.Tensor
        Tensor of variances
    CX_array : torch.Tensor
        Tensor of covariances
    n_max : int
        Maximum number of offspring to consider
    pheno_trait : str
        Name of phenotype trait
    offset : float, default 0.0
        Offset for phenotype
    num_cpu : int, default 1
        Number of CPU cores to use
    chunk_size : int, default 500
        Size of marker chunks for parallel processing
    show_progress : bool, default True
        Whether to show progress bar
        
    Returns:
    pd.DataFrame
        DataFrame with FBAT results
    """
    if show_progress:
        print(f"Starting FBAT computation for {len(marker_list)} markers using {num_cpu} processes")
        
    records = compute_fbat_parallel_markers(
        family_dict=family_dict,
        genotype_tensor=genotype_tensor,
        marker_list=marker_list,
        EX_array=EX_array,
        VX_array=VX_array,
        CX_array=CX_array,
        n_max=n_max,
        pheno_trait=pheno_trait,
        offset=offset,
        n_jobs=num_cpu,
        chunk_size=chunk_size,
        show_progress=show_progress
    )
    
    if show_progress:
        print("Processing results...")
     
    df = pd.DataFrame(records)

    # Family is informative if var_contrib > 1e-5
    df['is_informative'] = df['var_contrib'] > 1e-5

    # Group by marker
    grouped = df.groupby('marker', as_index=False).agg({
        'num_contrib': 'sum',
        'var_contrib': 'sum',
        'is_informative': 'sum'
    })
    
    grouped.rename(columns={
        'num_contrib': 'S-ES',
        'var_contrib': 'Var_S',
        'is_informative': 'fam_count'
    }, inplace=True)

    # Compute Z scores
    def compute_z(row):
        if row['Var_S'] < 1e-5:
            return float('nan')
        return row['S-ES'] / math.sqrt(row['Var_S'])
    
    grouped['Z'] = grouped.apply(compute_z, axis=1)

    # Compute P-values
    def z_to_p(z):
        if pd.isna(z):
            return float('nan')
        p = 2.0 * (1.0 - scipy.stats.norm.cdf(abs(z)))
        return p
    
    grouped['P'] = grouped['Z'].apply(z_to_p)

    # Final dataframe
    final = grouped[['marker', 'fam_count', 'S-ES', 'Var_S', 'Z', 'P']].copy()
    final.rename(columns={'fam_count': 'fam#'}, inplace=True)
    if show_progress:
        print("Process Complete!")
    
    return final