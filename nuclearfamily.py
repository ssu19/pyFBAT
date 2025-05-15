import pandas as pd

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