import numpy as np
from src.helpers.inext_gaussian_fjc_utils import w_c_inext_gaussian_fjc_func
from src.helpers.inext_kuhn_grun_fjc_utils import w_c_inext_kuhn_grun_fjc_func
from src.helpers.cufjc_utils import w_c_cufjc_func
from src.helpers.excl_vol_fjc_utils import w_c_log_barrier_clnk_vol_pnlty_func

def w_c_thermo_conn_func(p):
    return -np.log(p)

def w_c_zero_func(gamma, n) -> float:
    return 0.0

def master_w_c_func(w_c_dist: str):
    """Master polymer chain free energy function.

    This function returns the selected polymer chain free energy
    function.

    Args:
        w_c_dist (str): Short-hand name for the selected polymer chain free energy function.
    
    Returns:
        function: The selected polymer chain free energy function.
    
    """
    if w_c_dist == "inext_gaussian_fjc": return w_c_inext_gaussian_fjc_func
    elif w_c_dist == "inext_kuhn_grun_fjc": return w_c_inext_kuhn_grun_fjc_func
    elif w_c_dist == "cufjc": return w_c_cufjc_func
    else:
        error_str = (
            "The called-for polymer chain free energy function is not "
            + "implemented!"
        )
        raise NotImplementedError(error_str)

def master_w_c_dfrmtn_func(w_c_dfrmtn_dist: str):
    """Master polymer chain free energy function.

    This function returns the selected polymer chain free energy
    function.

    Args:
        w_c_dist (str): Short-hand name for the selected polymer chain free energy function.
    
    Returns:
        function: The selected polymer chain free energy function.
    
    """
    if w_c_dfrmtn_dist == "log_barrier_clnk_vol_pnlty":
        return w_c_log_barrier_clnk_vol_pnlty_func
    else: return w_c_zero_func
