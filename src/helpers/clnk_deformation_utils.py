import numpy as np
from src.helpers.chain_conformation_utils import gamma_func
from src.helpers.chain_conformation_dispersity_utils import p_thermo_conn_func
from src.helpers.chain_free_energy_utils import w_c_thermo_conn_func

def monodisperse_y_clnk(): return np.zeros(3)

def r_chn_func(
        ideal_numerics_form: bool,
        F: np.ndarray,
        X_chn: np.ndarray,
        Q_clnk: np.ndarray,
        y_clnk: np.ndarray) -> np.ndarray:
    r_chn = np.matmul(F, np.matmul(Q_clnk, X_chn))
    if ideal_numerics_form: r_chn -= np.matmul(Q_clnk, y_clnk) 
    else: r_chn -= y_clnk
    return r_chn

def gamma_clnk_func(
        ideal_numerics_form: bool,
        F: np.ndarray,
        n_clnk: np.ndarray,
        b: float,
        X_clnk: np.ndarray,
        Q_clnk: np.ndarray,
        y_clnk: np.ndarray) -> np.ndarray:
    k_num = np.shape(n_clnk)[0]
    gamma_clnk = np.zeros(k_num)
    for chn_indx in range(k_num):
        r_chn = r_chn_func(
            ideal_numerics_form, F, X_clnk[chn_indx], Q_clnk, y_clnk)
        gamma_clnk[chn_indx] = gamma_func(
            np.linalg.norm(r_chn), n_clnk[chn_indx], b)
    return gamma_clnk

def r_chn_approx_func(
        F_Lmbda: np.ndarray,
        X_chn: np.ndarray,
        Q_clnk_m: np.ndarray,
        y_clnk_m: np.ndarray,
        delta_Q_clnk: np.ndarray,
        delta_y_clnk: np.ndarray) -> np.ndarray:
    return (
        np.matmul(F_Lmbda, np.matmul(delta_Q_clnk, np.matmul(Q_clnk_m, X_chn)))
        - (y_clnk_m+delta_y_clnk)
    )

def gamma_clnk_approx_func(
        F_Lmbda: np.ndarray,
        n_clnk: np.ndarray,
        b: float,
        X_clnk: np.ndarray,
        Q_clnk_m: np.ndarray,
        y_clnk_m: np.ndarray,
        delta_Q_clnk: np.ndarray,
        delta_y_clnk: np.ndarray) -> np.ndarray:
    k_num = np.shape(n_clnk)[0]
    gamma_clnk_approx = np.zeros(k_num)
    for chn_indx in range(k_num):
        r_chn_approx = r_chn_approx_func(
            F_Lmbda, X_clnk[chn_indx], Q_clnk_m, y_clnk_m,
            delta_Q_clnk, delta_y_clnk)
        gamma_clnk_approx[chn_indx] = gamma_func(
            np.linalg.norm(r_chn_approx), n_clnk[chn_indx], b)
    return gamma_clnk_approx

def w_chn_func(
        gamma_chn: float,
        n_chn: float,
        w_c_func,
        w_c_args: tuple[float],
        w_c_dfrmtn_func,
        w_c_dfrmtn_args: tuple[float]) -> float:
    return (
        w_c_func(gamma_chn, n_chn, *w_c_args)
        + w_c_dfrmtn_func(gamma_chn, n_chn, *w_c_dfrmtn_args)
    )

def W_clnk_func(
        gamma_clnk: np.ndarray,
        n_clnk: np.ndarray,
        w_c_func,
        w_c_args: tuple[float],
        w_c_dfrmtn_func,
        w_c_dfrmtn_args: tuple[float]) -> float:
    W_clnk = 0.
    for chn_indx in range(np.shape(n_clnk)[0]):
        W_clnk += w_chn_func(
            gamma_clnk[chn_indx], n_clnk[chn_indx], w_c_func, w_c_args,
            w_c_dfrmtn_func, w_c_dfrmtn_args)
    return W_clnk

def W_flucts_clnk_func(
        ideal_numerics_form: bool,
        F: np.ndarray,
        Q_clnk: np.ndarray,
        n_clnk: np.ndarray,
        b: float,
        X_clnk: np.ndarray,
        vol_quad_clnk: np.ndarray,
        w_c_func,
        w_c_args: tuple[float],
        w_c_dfrmtn_func,
        w_c_dfrmtn_args: tuple[float]) -> float:
    vol_quad_clnk_num = np.shape(vol_quad_clnk)[0]
    p_flucts_quad_clnk = 0.
    for vol_quad_clnk_indx in range(vol_quad_clnk_num):
        y_fluct_clnk = vol_quad_clnk[vol_quad_clnk_indx, :-1]
        weight_chull = vol_quad_clnk[vol_quad_clnk_indx, -1]
        p_fluct_clnk = 1.
        for chn_indx in range(np.shape(n_clnk)[0]):
            n_fluct_chn = n_clnk[chn_indx]
            r_fluct_chn = r_chn_func(
                ideal_numerics_form, F, X_clnk[chn_indx], Q_clnk, y_fluct_clnk)
            gamma_fluct_chn = gamma_func(
                np.linalg.norm(r_fluct_chn), n_fluct_chn, b)
            w_c_fluct_chn = w_chn_func(
                gamma_fluct_chn, n_fluct_chn, w_c_func, w_c_args,
                w_c_dfrmtn_func, w_c_dfrmtn_args)
            p_fluct_clnk *= p_thermo_conn_func(w_c_fluct_chn)
        p_flucts_quad_clnk += weight_chull * p_fluct_clnk
    return w_c_thermo_conn_func(p_flucts_quad_clnk)

def W_flucts_clnk_approx_func(
        F_Lmbda: np.ndarray,
        Q_clnk_m: np.ndarray,
        delta_Q_clnk: np.ndarray,
        n_clnk: np.ndarray,
        b: float,
        X_clnk: np.ndarray,
        vol_quad_clnk: np.ndarray,
        w_c_func,
        w_c_args: tuple[float],
        w_c_dfrmtn_func,
        w_c_dfrmtn_args: tuple[float]) -> float:
    vol_quad_clnk_num = np.shape(vol_quad_clnk)[0]
    p_flucts_quad_clnk = 0.
    for vol_quad_clnk_indx in range(vol_quad_clnk_num):
        y_fluct_clnk = vol_quad_clnk[vol_quad_clnk_indx, :-1]
        weight_chull = vol_quad_clnk[vol_quad_clnk_indx, -1]
        p_fluct_clnk = 1.
        for chn_indx in range(np.shape(n_clnk)[0]):
            n_fluct_chn = n_clnk[chn_indx]
            r_fluct_chn_approx = r_chn_approx_func(
                F_Lmbda, X_clnk[chn_indx], Q_clnk_m, monodisperse_y_clnk(),
                delta_Q_clnk, y_fluct_clnk)
            gamma_fluct_chn = gamma_func(
                np.linalg.norm(r_fluct_chn_approx), n_fluct_chn, b)
            w_c_fluct_chn = w_chn_func(
                gamma_fluct_chn, n_fluct_chn, w_c_func, w_c_args,
                w_c_dfrmtn_func, w_c_dfrmtn_args)
            p_fluct_clnk *= p_thermo_conn_func(w_c_fluct_chn)
        p_flucts_quad_clnk += weight_chull * p_fluct_clnk
    return w_c_thermo_conn_func(p_flucts_quad_clnk)