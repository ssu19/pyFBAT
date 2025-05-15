import torch
import itertools
from math import factorial, comb
from itertools import permutations, product
from collections import defaultdict

def combo_to_genotypes(combo, k):
    """Convert combination to genotype counts"""
    counts = [0]*k
    for item in combo:
        counts[item]+=1
    return counts

def enumerate_configurations(n):
    """Enumerate possible genotype configurations"""
    return [combo for combo in itertools.combinations_with_replacement(range(3), n)]

def multinomial_coefficient(x1, x2, x3):
    """Calculate multinomial coefficient"""
    return factorial(x1+x2+x3)//(factorial(x1)*factorial(x2)*factorial(x3))

def mendelian_prob(f, m, offspring):
    """Calculate Mendelian probability for a genotype configuration"""
    probs = torch.zeros(3, 3, 3)
    
    probs[0, 0, 0] = 1.0

    probs[0, 1, 0] = 0.5
    probs[0, 1, 1] = 0.5
    probs[1, 0, 0] = 0.5
    probs[1, 0, 1] = 0.5

    probs[0, 2, 1] = 1.0
    probs[2, 0, 1] = 1.0

    probs[1, 1, 0] = 0.25
    probs[1, 1, 1] = 0.5
    probs[1, 1, 2] = 0.25

    probs[2, 1, 1] = 0.5
    probs[2, 1, 2] = 0.5

    probs[1, 2, 1] = 0.5
    probs[1, 2, 2] = 0.5

    probs[2, 2, 2] = 1.0

    conf = [0, 0, 0]
    for geno in offspring:
        conf[geno] += 1
    factor = multinomial_coefficient(*conf)
    p_order = 1.0
    for geno in offspring:
        p_order *= probs[f, m, geno]
    return p_order*factor

def _build_mendelian_probabilities(max_offspring=10):
    """
    Internal function to build arrays of Mendelian probabilities.
    
    Parameters:
    max_offspring : int, default 10
        Maximum number of offspring to consider
        
    Returns:
    list
        List of probability records
    """
    all_mendel_probs = []
    genotypes = [0, 1, 2]
    states = [None, 0, 1, 2]

    for n_offspring in range(1, max_offspring+1):
        for father in states:
            for mother in states:
                offspring_combinations = enumerate_configurations(n_offspring)
                offspring_genotype_configs = [combo_to_genotypes(c, 3) for c in offspring_combinations]
                num_configs = len(offspring_genotype_configs)
                
                prob_tensor = torch.zeros((num_configs, 3, 3))
                for i, offspring_config in enumerate(offspring_genotype_configs):
                    offspring_genos = []
                    for j, count in enumerate(offspring_config):
                        offspring_genos.extend([genotypes[j]]*count)
                    for f in range(3):
                        for m in range(3):
                            prob_tensor[i, f, m] = mendelian_prob(f, m, offspring_genos)
                
                if father is not None:
                    invalid_fathers = [x for x in range(3) if x != father]
                    prob_tensor[:, invalid_fathers, :] = 0
                if mother is not None:
                    invalid_mothers = [x for x in range(3) if x != mother]
                    prob_tensor[:, :, invalid_mothers] = 0

                matrix = prob_tensor.reshape(num_configs, 9)

                norms = matrix.norm(dim=1, keepdim=True)
                normalized_matrix = (matrix/norms).round(decimals=8)
                fixed_matrix = torch.nan_to_num(normalized_matrix, nan=float('inf'))

                unique_rows, inverse_indices = torch.unique(fixed_matrix, dim=0, return_inverse=True)

                for eq_class in range(len(unique_rows)):
                    sub_matrix = matrix[inverse_indices == eq_class, :].reshape((inverse_indices == eq_class).sum().item(), 9)
                    
                    compatible_mating_types = ~torch.all(sub_matrix == 0, dim=0)
                    if torch.any(compatible_mating_types):
                        sub_matrix = sub_matrix[:, compatible_mating_types]
                        probabilities = sub_matrix[:, 0]/sub_matrix[:, 0].sum()
                        
                        idxs = torch.nonzero(inverse_indices == eq_class).squeeze()
                        if idxs.dim() == 0:  # in case there's only 1
                            idxs = idxs.unsqueeze(0)
                        geno_confs = [offspring_genotype_configs[i.item()] for i in idxs]

                        # Calculate moments
                        pgg = torch.zeros(3, 3)  # Expected proportion of offsprings who are geno i and j
                        pg = torch.zeros(3)      # Expected proportion of offspring who are genotype i
                        
                        for (config, prob) in zip(geno_confs, probabilities):
                            config_t = torch.tensor(config, dtype=torch.float32)
                            
                            pg = pg + prob * config_t / n_offspring
                            
                            if n_offspring >= 1:
                                for i in range(3):
                                    if config_t[i] > 1:
                                        pgg[i, i] += (prob * config_t[i] * (config_t[i] - 1) / 
                                                    (n_offspring * (n_offspring - 1)))
                                    for j in range(3):
                                        if i != j and config_t[j] > 0:
                                            pgg[i, j] += (prob * config_t[i] * config_t[j] /
                                                        (n_offspring * (n_offspring - 1)))
                        
                        # additive coding vector X
                        X = torch.tensor([[0.0], [1.0], [2.0]])  
                        
                        EX = (X.T @ pg).item()
                        CX = (X.T @ pgg @ X).item() - EX * EX
                        VX = (X.T @ torch.diag(pg) @ X).item() - EX * EX
                        
                        # Store in all_mendel_probs
                        for row_idx, pval in enumerate(probabilities):
                            config_counts = geno_confs[row_idx]
                            rec = {
                                "n_offspring": n_offspring,
                                "father": father,
                                "mother": mother,
                                "eq_class": eq_class,
                                "offspring_config": config_counts,
                                "probability": pval.item(),
                                "EX": EX,
                                "VX": VX,
                                "CX": CX
                            }
                            all_mendel_probs.append(rec)

    return all_mendel_probs

def remap_index(g):
    """
    Map genotype {0,1,2,None} -> {0,1,2,3}
    
    If father/mother genotype is in {0,1,2}, return 0,1,2.
    If it's -9 or we have -1 as the sample index, treat as 'missing' => 3
    We'll define -9 or any negative => 3.
    """
    if g in (9, -9, None):
        return 3  # missing
    else:
        return int(g)

def _build_5d_moment_arrays(all_mendel_probs, n_max=11):
    """
    Build 5D arrays for first and second moments of genotype distributions.
    
    Parameters:
    all_mendel_probs : list
        List of probability records from build_mendelian_probabilities
    n_max : int, default 11
        Maximum number of offspring to consider
        
    Returns:
    tuple
        (EX_tensor, VX_tensor, CX_tensor)
    """
    # 1 get maximum n_offspring if not provided
    if n_max is None:
        n_max = max(rec['n_offspring'] for rec in all_mendel_probs)

    # map parental geno {0,1,2,None} -> index in [0..3]
    # x0,x1,x2 each in [0..n_max]
    shape = (4, 4, n_max, n_max, n_max)

    # 2 create np arrays
    EX_tensor = torch.full(shape, float('nan'))
    VX_tensor = torch.full(shape, float('nan'))
    CX_tensor = torch.full(shape, float('nan'))

    # 3) populate from all_mendel_probs
    for rec in all_mendel_probs:
        f_geno = rec['father']   # 0,1,2, or None
        m_geno = rec['mother']   # 0,1,2, or None
        x0, x1, x2 = rec['offspring_config']
        EX = rec['EX']           # mean
        VX = rec['VX']           # variance
        CX = rec['CX']           # covariance

        # map father/mother genotype -> indices
        f_idx = remap_index(f_geno)
        m_idx = remap_index(m_geno)

        if (0 <= x0 < n_max) and (0 <= x1 < n_max) and (0 <= x2 < n_max):
            EX_tensor[f_idx, m_idx, x0, x1, x2] = EX
            VX_tensor[f_idx, m_idx, x0, x1, x2] = VX
            CX_tensor[f_idx, m_idx, x0, x1, x2] = CX

    return EX_tensor, VX_tensor, CX_tensor

def get_iid_mappings(genotype_counts):
    """
    Create mappings from IID to index.
    
    Parameters:
    genotype_counts : pd.DataFrame
        DataFrame with genotype counts
        
    Returns:
    dict
        Dictionary mapping IID to index
    """
    return {iid: i for i, iid in enumerate(genotype_counts.index)}

def get_marker_mappings(markers):
    """
    Create mappings from marker to index.
    
    Parameters:
    markers : list
        List of marker names
        
    Returns:
    dict
        Dictionary mapping marker to index
    """
    return {m: j for j, m in enumerate(markers)}

def build_mendelian_probs(max_offspring=10):
    """
    One-step function to build all necessary data structures for FBAT.
    
    This function runs:
    1. The Mendelian probability calculation
    2. The 5D moment array construction
    
    Parameters:
    max_offspring : int, default 10
        Maximum number of offspring in families
    n_max : int, default 11
        Maximum number of offspring to consider in arrays
        
    Returns:
    tuple
        (all_mendel_probs, EX_tensor, VX_tensor, CX_tensor)
    """
    
    print("Step 1/2: Building Mendelian probabilities...")
    all_mendel_probs = _build_mendelian_probabilities(max_offspring)
    print(f"Created {len(all_mendel_probs)} probability records")
    
    print("Step 2/2: Building moment arrays...")
    EX_tensor, VX_tensor, CX_tensor = _build_5d_moment_arrays(all_mendel_probs, max_offspring+1)
    print("Moment arrays built successfully")
    
    return all_mendel_probs, EX_tensor, VX_tensor, CX_tensor