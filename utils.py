import pandas as pd
import numpy as np
import torch

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

