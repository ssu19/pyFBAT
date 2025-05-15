import pandas as pd
import numpy as np
import torch
import plinkio_reader
from pathlib import Path

class NuclearFamily:
    """
    Class representing a nuclear family with parents and offspring.
    """
    def __init__(self, fid, father_iid, mother_iid, psam_df, iid_to_index):
        """
        Initialize a nuclear family.
        
        Parameters:
        fid : str
            Family ID
        father_iid : str
            IID of father
        mother_iid : str
            IID of mother
        psam_df : pd.DataFrame
            Sample information dataframe
        iid_to_index : dict
            Mapping from IID to index in genotype data
        """
        self.fid = fid
        self.father = father_iid
        self.mother = mother_iid
        # store integer indices
        self.father_idx = iid_to_index.get(father_iid, -1)
        self.mother_idx = iid_to_index.get(mother_iid, -1)
        self.offspring = []

        # Determine genotype availability of parents
        self.father_genotype_available = self._check_genotype_avail(psam_df, father_iid)
        self.mother_genotype_available = self._check_genotype_avail(psam_df, mother_iid)

    def _check_genotype_avail(self, psam_df, iid):
        """
        Check if this IID has 'genotype_available' = True in psam_df.
        Return False if not found or if genotype_available is not True.
        """
        row = psam_df.loc[psam_df['IID'] == iid]
        if not row.empty:
            return bool(row['genotype_available'].values[0])
        return False
    
    ID_COLS = {'FID','IID','PAT','MAT','SEX','genotype_available'}

    def add_offspring(self, child_iid, psam_df, iid_to_index):
        """
        Add offspring to the family.
        
        Parameters:
        child_iid : str
            IID of child
        psam_df : pd.DataFrame
            Sample information dataframe
        iid_to_index : dict
            Mapping from IID to index in genotype data
        """
        ID_COLS = {'FID','IID','PAT','MAT','SEX','genotype_available'}
        child_idx = iid_to_index.get(child_iid, -1)
        row = psam_df.loc[psam_df['IID'] == child_iid]
        if row.empty:
            # no row => no phenotypes
            pheno_data = {}
            genotype_avail = False
        else:
            row0 = row.iloc[0]  # single row as a Series
            # Mark whether genotype is available
            genotype_avail = self._check_genotype_avail(psam_df, child_iid)

            # Build a dict of all phenotypes (i.e. columns not in ID_COLS)
            pheno_data = {}
            for col in row.columns:
                if col not in ID_COLS:
                    pheno_data[col] = row0[col]

        self.offspring.append({
            'iid': child_iid,
            'idx': child_idx,
            'genotype_available': genotype_avail,
            'phenos': pheno_data
        })

    def __repr__(self):
        return (f"NuclearFamily(FID={self.fid}, "
                f"Offspring={self.offspring}, "
                f"Father=({self.father},{self.father_idx}), "
                f"FatherAvail={self.father_genotype_available}, "
                f"Mother=({self.mother},{self.mother_idx}), "
                f"MotherAvail={self.mother_genotype_available})")


def build_nuclear_families(df, psam_df, iid_to_index):
    """
    Build nuclear families from pedigree data.
    
    Parameters:
    df : pd.DataFrame
        Pedigree dataframe
    psam_df : pd.DataFrame
        Sample information dataframe
    iid_to_index : dict
        Mapping from IID to index in genotype data
        
    Returns:
    dict
        Dictionary of nuclear families keyed by (fid, pat, mat)
    """
    for col in ['FID', 'IID', 'PAT', 'MAT']:
        df[col] = df[col].astype(str)
    
    # Mark genotype availability
    available_ids = set(psam_df['IID'])
    psam_df['genotype_available'] = psam_df['IID'].apply(lambda iid: iid in available_ids)
    
    grouped = df.groupby(['FID', 'PAT', 'MAT'])
    families = {}

    for (fid, pat, mat), group in grouped:
        family = NuclearFamily(fid, pat, mat, psam_df, iid_to_index)

        # Offspring = all individuals not father/mother
        offspring_iids = [iid for iid in group['IID'] if iid not in [pat, mat]]
        for child_iid in offspring_iids:
            family.add_offspring(child_iid, psam_df, iid_to_index)

        families[(fid, pat, mat)] = family

    return families

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

def load_plink2_dataset(
    prefix: str,
    directory: str | Path = ".",
    block_size: int = 10_000,
    marker_prefix: str = "chr",
    id_cols=("FID", "IID", "PAT", "MAT", "SEX"),
    torch_dtype: torch.dtype = torch.int64,
    sparse_fill: int = -9,
    pheno_path: str = "pheno_df.phe",
    build_family: bool = True):
    """
    Load PLINK2 dataset and merge with phenotype data.
    
    Parameters:
    prefix : str
        PLINK2 prefix without extension: for 'plink2.pgen' pass 'plink2'.
    directory : str | Path, default "."
        Folder containing the PLINK 2 files.
    block_size : int, default 10_000
        Number of variants to read at once.
    marker_prefix : str, default "chr"
        Prefix for marker IDs to keep.
    id_cols : tuple, default ('FID','IID','PAT','MAT','SEX')
        Specify column ID names.
    torch_dtype : torch.dtype, default torch.int64
        Data type for genotype tensor.
    sparse_fill : int, default -9
        Value to use for missing genotypes.
    pheno_path : str
        Path for phenotype data.
    build_family : bool, default True
        Whether to build nuclear families.
    Returns:
    pgr: plinkio.PgenReader
    merged_psam: pd.DataFrame
    markers: list[str]
    genotype_counts: pd.DataFrame (Sparse[int8]), index = IID
    genotype_tensor: torch.Tensor
    nuc_families: Optional, returned only if build_family=True
    """
    # import phenotype data 
    pheno_df = pd.read_csv(pheno_path, delim_whitespace=True)
    pheno_df["IID"] = pheno_df["IID"].astype(str)
    pheno_df["FID"] = pheno_df["FID"].astype(str)
    
    
    #  open metadata
    work_dir = Path(directory).expanduser().resolve()
    pgr = plinkio_reader.PgenReader(str(work_dir / prefix))
    psam_df = pgr.psam_df.copy()
    psam_df[list(id_cols)] = psam_df[list(id_cols)].astype(str)
    psam_df.reset_index(drop=True, inplace=True)
    pvar_df = pgr.pvar_df.copy()
    
    # Merge pheno data if provided
    if pheno_df is not None:
        if 'FID' in pheno_df.columns:
            pheno_df = pheno_df.drop(columns=["FID"])
            print('Duplicate FID column detected, dropping FID in phenotype data')
        merged_psam = pd.merge(pgr.psam_df, pheno_df, on="IID", how="left")
        merged_psam.reset_index(drop=True, inplace=True)
    else:
        merged_psam = psam_df.copy()
    #  row‑order map
    iid_to_row = {iid: i for i, iid in enumerate(psam_df["IID"])}
    row_order = np.fromiter(
        (iid_to_row[iid] for iid in merged_psam["IID"]),
        dtype=np.uint32, count=len(merged_psam)
    )
    # choose variants 
    keep_mask = pvar_df["id"].str.startswith(marker_prefix)
    variant_idx = np.flatnonzero(keep_mask)
    markers = pvar_df["id"][keep_mask].tolist()
    n_samples = len(row_order)
    n_variants = len(variant_idx)
    # pre‑allocate dense int8 matrix 
    geno_mat = np.empty((n_samples, n_variants), dtype=np.int8)
    pgen_file = str(work_dir / f"{prefix}.pgen")
    # stream in blocks 
    for blk_start in range(0, n_variants, block_size):
        blk_end = min(blk_start + block_size, n_variants)
        dest_slice = slice(blk_start, blk_end)
        vstart, vend = int(variant_idx[blk_start]), int(variant_idx[blk_end - 1])
        block = plinkio_reader.read_range(pgen_file, vstart, vend, dtype=np.int8)  # (vars, samp)
        geno_mat[:, dest_slice] = block.T[row_order, :]             # re‑order rows
    #  convert to outputs
    genotype_counts = (
        pd.DataFrame(geno_mat, index=merged_psam["IID"], columns=markers)
        .astype(pd.SparseDtype("int8", fill_value=sparse_fill))
    )
    
    iid_to_index = get_iid_mappings(genotype_counts)
    genotype_tensor = torch.as_tensor(geno_mat, dtype=torch_dtype)
    print('loading Plink2 files complete')
    
    if build_family:
        nuc_families = build_nuclear_families(pgr.psam_df, merged_psam, iid_to_index)
        return pgr, merged_psam, markers, genotype_counts, genotype_tensor, nuc_families
    else:
        return pgr, merged_psam, markers, genotype_counts, genotype_tensor