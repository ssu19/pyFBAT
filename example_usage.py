import pyFBAT
import pandas as pd
import numpy as np
import itertools
from pathlib import Path
import tqdm


def main():
    
    # load in PLINK2 data and phenotype data
    df1, merged_psam, markers, genotype_counts, genotype_tensor, nuc_fams = pyFBAT.load_plink2_dataset(prefix="EFIGA_update_qc1_subset1000", pheno_path="test4_standard.phe")

    # build mendelian probabilities for up to n_max offsprings
    all_mendel_probs, EX_tensor, VX_tensor, CX_tensor = pyFBAT.build_mendelian_probs(max_offspring=10)

    # Run FBAT with specified trait and save results as a df
    results = pyFBAT.compute_fbat_final_df(
        family_dict=nuc_fams,  
        genotype_tensor=genotype_tensor,
        marker_list=markers,
        EX_array=EX_tensor,
        VX_array=VX_tensor,
        CX_array=CX_tensor,
        n_max=10,
        pheno_trait='trait2'
    )
    print(results.head())
    
    
if __name__ == "__main__":
    main()