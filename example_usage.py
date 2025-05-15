import pyFBAT
import pandas as pd
import numpy as np
import itertools
from pathlib import Path
import tqdm


def main():

    # Step 1: Load data and build families
    '''
    Parameters:
    prefix : str, PLINK2 prefix without extension: for 'plink2.pgen' pass 'plink2'.
    directory : str, Folder containing the PLINK 2 files.
    block_size : int, Number of variants to read at once.
    marker_prefix : str, default "chr", Prefix for marker IDs to keep.
    id_cols : tuple, default ('FID','IID','PAT','MAT','SEX'), Specify column ID names.
    sparse_fill : int, default -9, Value to use for missing genotypes.
    pheno_path : str, Path for phenotype data.
    build_family : bool, default True, Whether to build nuclear families.
    '''
    
    df, merged_psam, markers, genotype_counts, genotype_tensor, nuc_fams = fbat_package.load_plink2_dataset(
        prefix="your_plink_file_prefix",
        directory="/path/to/data/",
        pheno_file="your_phenotype.phe"
    )
    
    # Step 2: Build Mendelian probabilities
    # Max_offspring number defaults to 10
    all_mendel_probs, EX_tensor, VX_tensor, CX_tensor = pyFBAT.build_mendelian_probs(max_offspring=10)
    
    # Step 3: Compute FBAT
    '''
    Parameters:
    family_dict : dict, Dictionary of NuclearFamily objects
    genotype_tensor : torch.Tensor, Tensor of genotypes
    marker_list : list, List of marker names
    EX_array : torch.Tensor
    VX_array : torch.Tensor
    CX_array : torch.Tensor
    n_max : int, Maximum number of offspring to consider
    pheno_trait : str, Name of phenotype trait
    offset : float, default 0.0, Offset for phenotype
    num_cpu : int, default os.cpu_count(), Number of CPU cores to use
    chunk_size : int, Size of marker chunks for parallel processing
    show_progress : bool, default True, Whether to show progress bar
    '''
    
    results = pyFBAT.compute_fbat_final_df(
        family_dict=nuc_fams,  
        genotype_tensor=genotype_tensor,
        marker_list=markers,
        EX_array=EX_tensor,
        VX_array=VX_tensor,
        CX_array=CX_tensor,
        n_max=10,
        pheno_trait='trait2',
        offset=0.0,
        num_cpu=os.cpu_count(),
        chunk_size=1000,
        show_progress=True   
    )
        
    
if __name__ == "__main__":
    main()
