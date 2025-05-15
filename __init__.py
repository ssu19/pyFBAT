"""
FBAT Package - Family-Based Association Test Implementation

This package provides tools for performing family-based association tests on genetic data.
"""

# Import key functions for easy access
from .plink_loader import load_plink2_dataset
from .nuclearfamily import build_nuclear_families, NuclearFamily
from .mendelian_probs import (
    build_mendelian_probs, 
    get_iid_mappings,
    get_marker_mappings,
    remap_index
)
from .fbat_compute import compute_fbat_final_df
from .plinkio_reader import PgenReader, read_range
__all__ = [
    'load_plink2_dataset',
    'build_nuclear_families',
    'NuclearFamily',
    'build_mendelian_probs',
    'compute_fbat_final_df',
    'get_iid_mappings',
    'get_marker_mappings',
    'identify_non_founders'
]