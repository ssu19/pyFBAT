# Adapted from plinkio from the following source
# Functions for reading dosages from PLINK pgen files based on the Pgenlib Python API:
# https://github.com/chrchang/plink-ng/blob/master/2.0/Python/python_api.txt

import numpy as np
import pandas as pd
import pgenlib as pg
import os
import bisect


def read_pvar(pvar_path):
    """Read pvar file as pd.DataFrame"""
    return pd.read_csv(pvar_path, sep='\t', comment='#',
                       names=['chrom', 'pos', 'id', 'ref', 'alt', 'qual', 'filter', 'info'],
                       dtype={'chrom':str, 'pos':np.int32, 'id':str, 'ref':str, 'alt':str,
                              'qual':str, 'filter':str, 'info':str})


def read_psam(psam_path):
    """Read psam file as pd.DataFrame"""
    psam_df = pd.read_csv(psam_path, delim_whitespace=True, index_col=None, header=0, names=['FID','IID','PAT','MAT','SEX'])
    psam_df.index = psam_df.index.astype(str)
    return psam_df





def get_reader(pgen_path, sample_subset=None):
    """"""
    if sample_subset is not None:
        sample_subset = np.array(sample_subset, dtype=np.uint32)
    reader = pg.PgenReader(pgen_path.encode(), sample_subset=sample_subset)
    if sample_subset is None:
        num_samples = reader.get_raw_sample_ct()
    else:
        num_samples = len(sample_subset)
    return reader, num_samples





def read_range(pgen_path, start_idx, end_idx, sample_subset=None, dtype=np.int8):
    """
    Get genotypes for a range of variants.

    Parameters
    ----------
    pgen_path : str
        Path of PLINK 2 pgen file
    start_idx : int
        Start index of the range to query.
    end_idx : int
        End index of the range to query (inclusive).
    sample_subset : array_like
        List of sample indexes to select. Must be sorted.
    dtype : np.int{8,32,64}
        Data type of the returned array.

    Returns
    -------
    dosages : ndarray
        Genotypes for the selected variants and samples.
    """
    reader, num_samples = get_reader(pgen_path, sample_subset=sample_subset)
    num_variants = end_idx - start_idx + 1
    genotypes = np.zeros([num_variants, num_samples], dtype=dtype)
    with reader as r:
        r.read_range(start_idx, end_idx+1, genotypes)
    return genotypes


class PgenReader(object):
    """
    Class for reading genotype data from PLINK 2 pgen files


    Requires pgenlib: https://github.com/chrchang/plink-ng/tree/master/2.0/Python
    """
    def __init__(self, plink_prefix_path):
        """
        plink_prefix_path: prefix to PLINK pgen,psam,pvar files
        select_samples: specify a subset of samples
        """


        self.pvar_df = read_pvar(f"{plink_prefix_path}.pvar")
        self.psam_df = read_psam(f"{plink_prefix_path}.psam")
        
        self.pgen_file = f"{plink_prefix_path}.pgen"

        self.num_variants = self.pvar_df.shape[0]
        self.variant_ids = self.pvar_df['id'].tolist()
        self.variant_idx_dict = {i:k for k,i in enumerate(self.variant_ids)}

        self.sample_id_list = self.psam_df.index.tolist()
        self.sample_ids = self.sample_id_list
        self.sample_idxs = None
        variant_df = self.pvar_df.set_index('id')[['chrom', 'pos']]
        variant_df['index'] = np.arange(variant_df.shape[0])
        self.variant_df = variant_df
        self.variant_dfs = {c:g[['pos', 'index']] for c,g in variant_df.groupby('chrom', sort=False)}
        self.genotypes = self.load_genotypes()





    def load_genotypes(self):
        """Load all genotypes as np.int8, without imputing missing values."""
        genotypes = read_range(self.pgen_file, 0, self.num_variants-1, sample_subset=self.sample_idxs)
        return pd.DataFrame(genotypes, index=self.variant_ids, columns=self.sample_ids)




