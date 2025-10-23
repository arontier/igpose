from typing import Dict, List, Any, Optional
from abnumber import Chain  # conda install -c bioconda abnumber

def _default_scheme(mt: str) -> str:
    return "imgt" if mt.upper() == "TCR" else "chothia"

def _chain_letter(mt: str, which: str) -> str:
    mt = mt.upper()
    if mt == "TCR":
        return "B" if which == "heavy" else "A"   # beta ~ heavy-like, alpha ~ light-like
    # Ab / Nb / scFv
    return "H" if which == "heavy" else "L"

def _cdrs_for_chain(
    seq: str,
    orig_ids: List[Any],
    scheme: str,
) -> Dict[str, List[Any]]:
    """
    Map CDR regions (per Chothia or IMGT) to *original* residue IDs.

    Parameters
    ----------
    seq : str
        V-domain amino acid sequence you feed to AbNumber.
    orig_ids : list[Any]
        Original residue identifiers aligned 1:1 with `seq`.
        (e.g., PDB-style numbers 'H25','H25A','H26', or tuples, ints, etc.)
        len(orig_ids) must equal len(seq).
    scheme : {"chothia","imgt"}
        Numbering + CDR definition used by AbNumber.

    Returns
    -------
    dict
        {"CDR1": [...], "CDR2": [...], "CDR3": [...]}, where each list contains
        your original residue IDs corresponding to that region.
    """
    if len(orig_ids) != len(seq):
        raise ValueError(
            f"Length mismatch: len(orig_ids)={len(orig_ids)} vs len(seq)={len(seq)}"
        )

    # Build AbNumber chain with consistent numbering and CDR definition
    ch = Chain(seq, scheme=scheme, cdr_definition=scheme)

    # AbNumber exposes contiguous regions that partition the sequence in order.
    # We only need region lengths to slice `orig_ids` sequentially.
    ordered_regions = [r for r in ("FR1", "CDR1", "FR2", "CDR2", "FR3", "CDR3", "FR4") if r in ch.regions]

    idx = 0
    cdr_map: Dict[str, List[Any]] = {"CDR1": [], "CDR2": [], "CDR3": []}

    for r in ordered_regions:
        n = len(ch.regions[r])     # number of residues AbNumber assigns to this region
        seg = orig_ids[idx: idx + n]
        if r in cdr_map:
            cdr_map[r] = list(seg) # keep your original IDs
        idx += n

    # Optional sanity check (AbNumber regions should cover the full sequence)
    if idx != len(seq):
        # You can log or raise here if you want stricter guarantees.
        pass
    
    final_list = []
    for k in sorted(cdr_map.keys()):
        final_list.extend(cdr_map[k])
    return final_list

def extract_cdr_ids(
    heavy_ids: List,
    light_ids: Optional[List] = None,
    heavy_seq: Optional[str] = None,
    light_seq: Optional[str] = None,
    molecule_type: str = "Ab",              # {"Ab","Nb","scFv","TCR"}
    scheme: Optional[str] = None            # {"chothia","imgt"} to override defaults
) -> Dict[str, Optional[Dict[str, List[str]]]]:
    """
    Return dict with scheme and CDR residue IDs for each chain.
    Residue IDs look like 'H26', 'H27A', 'L91', etc.
    For TCR, IDs are 'A..' (alpha) and 'B..' (beta) by convention.
    """

    sc = (scheme or _default_scheme(molecule_type)).lower()
    result = {"scheme": sc, "heavy": None, "light": None}

    # Nb is single-chain; treat provided heavy_seq as VHH
    if molecule_type.upper() == "NB":
        if heavy_seq:
            result["heavy"] = _cdrs_for_chain(heavy_seq, heavy_ids, sc)
        return result

    if heavy_seq:
        result["heavy"] = _cdrs_for_chain(heavy_seq, heavy_ids, sc)
    if light_seq:
        result["light"] = _cdrs_for_chain(light_seq, light_ids, sc)

    return result