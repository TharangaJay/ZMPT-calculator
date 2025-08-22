from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List, Union

import numpy as np
import pandas as pd

ArrayLike = Union[np.ndarray, List[float], pd.Series]


@dataclass
class ZMPTResult:
    C_ext: float
    C_int: float
    n_ext: float
    n_int: float
    history: pd.DataFrame
    cond_history: List[float]
    residual_history: List[float]
    converged: bool


def _normalize_colname(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _pick_col(df: pd.DataFrame, *candidates: str) -> str:
    norm_map = {_normalize_colname(c): c for c in df.columns}
    for cand in candidates:
        key = _normalize_colname(cand)
        if key in norm_map:
            return norm_map[key]
    raise KeyError(f"Could not find any of {candidates} in columns {list(df.columns)}")


def zmpt_solve(
    DeltaP_ext: ArrayLike,
    DeltaP_int: ArrayLike,
    Q_int: ArrayLike,
    Q_ext: ArrayLike,
    A_ext: Union[float, ArrayLike],
    A_int: Union[float, ArrayLike],
    C_ext0: float = 0.1,
    C_int0: float = 0.1,
    n_ext0: float = 0.6,
    n_int0: float = 0.6,
    iterations: int = 20,
    step: float = 0.01,
    damping: float = 1.0,
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    rtol: float = 1e-8,
) -> ZMPTResult:

    dpe = np.asarray(DeltaP_ext, dtype=float)
    dpi = np.asarray(DeltaP_int, dtype=float)
    qi = np.asarray(Q_int, dtype=float)
    qe = np.asarray(Q_ext, dtype=float)

    if np.isscalar(A_ext):
        aext = np.full_like(dpe, float(A_ext))
    else:
        aext = np.asarray(A_ext, dtype=float)
    if np.isscalar(A_int):
        aint = np.full_like(dpi, float(A_int))
    else:
        aint = np.asarray(A_int, dtype=float)

    nrows = len(dpe)
    assert len(dpi) == nrows and len(qi) == nrows and len(qe) == nrows and len(aext) == nrows and len(aint) == nrows, \
        "All inputs must be the same length."

    C_ext = C_ext0
    C_int  = C_int0
    n_ext = n_ext0
    n_int = n_int0

    hist_rows = []
    cond_hist: List[float] = []
    resid_hist: List[float] = []

    def residuals(Ce, Ci, ne, ni) -> np.ndarray:
        return Ce*aext*(dpe**ne) + qe - Ci*aint*(dpi**ni) - qi

    converged = False

    for it in range(1, iterations + 1):
        f = residuals(C_ext, C_int, n_ext, n_int)

        st = float(step)
        Ce_p = (C_ext + st*C_ext); Ce_m = (C_ext - st*C_ext)
        Ci_p = (C_int + st*C_int); Ci_m = (C_int - st*C_int)
        ne_p = (n_ext + st*n_ext); ne_m = (n_ext - st*n_ext)
        ni_p = (n_int + st*n_int); ni_m = (n_int - st*n_int)

        L1 = (residuals(Ce_p, C_int, n_ext, n_int) - residuals(Ce_m, C_int, n_ext, n_int)) / (2*st*C_ext)
        L2 = (residuals(C_ext, Ci_p, n_ext, n_int) - residuals(C_ext, Ci_m, n_ext, n_int)) / (2*st*C_int)
        L3 = (residuals(C_ext, C_int, ne_p, n_int) - residuals(C_ext, C_int, ne_m, n_int)) / (2*st*n_ext)
        L4 = (residuals(C_ext, C_int, n_ext, ni_p) - residuals(C_ext, C_int, n_ext, ni_m)) / (2*st*n_int)

        L = np.column_stack([L1, L2, L3, L4])
        b = L[:,0]*C_ext + L[:,1]*C_int + L[:,2]*n_ext + L[:,3]*n_int - f

        x, *_ = np.linalg.lstsq(L, b, rcond=None)

        C_ext_next = float(x[0])
        C_int_next = float(x[1])
        n_ext_next = float(x[2])
        n_int_next = float(x[3])

        if damping != 1.0:
            C_ext_next = C_ext + damping*(C_ext_next - C_ext)
            C_int_next = C_int + damping*(C_int_next - C_int)
            n_ext_next = n_ext + damping*(n_ext_next - n_ext)
            n_int_next = n_int + damping*(n_int_next - n_int)

        if bounds:
            if "C_ext" in bounds:
                lo, hi = bounds["C_ext"]; C_ext_next = float(np.clip(C_ext_next, lo, hi))
            if "C_int" in bounds:
                lo, hi = bounds["C_int"]; C_int_next = float(np.clip(C_int_next, lo, hi))
            if "n_ext" in bounds:
                lo, hi = bounds["n_ext"]; n_ext_next = float(np.clip(n_ext_next, lo, hi))
            if "n_int" in bounds:
                lo, hi = bounds["n_int"]; n_int_next = float(np.clip(n_int_next, lo, hi))

        condL = float(np.linalg.cond(L))
        cond_hist.append(condL)
        resid_hist.append(float(np.linalg.norm(f, ord=2)))
        hist_rows.append({
            "iter": it,
            "C_ext": C_ext_next,
            "C_int": C_int_next,
            "n_ext": n_ext_next,
            "n_int": n_int_next,
            "cond(L)": condL,
            "||f||2": float(np.linalg.norm(f, ord=2)),
        })

        denom = max(1e-12, abs(C_ext) + abs(C_int) + abs(n_ext) + abs(n_int))
        rel_change = abs(C_ext_next - C_ext) + abs(C_int_next - C_int) + abs(n_ext_next - n_ext) + abs(n_int_next - n_int)
        rel_change /= denom

        C_ext, C_int, n_ext, n_int = C_ext_next, C_int_next, n_ext_next, n_int_next

        if rel_change < rtol:
            converged = True
            break

    history = pd.DataFrame(hist_rows)
    return ZMPTResult(
        C_ext=C_ext, C_int=C_int, n_ext=n_ext, n_int=n_int,
        history=history, cond_history=cond_hist, residual_history=resid_hist, converged=converged
    )


def zmpt_solve_from_csv(csv_path: str, a_ext: float = None, a_int: float = None, **kwargs) -> ZMPTResult:
    df = pd.read_csv(csv_path)
    def pick(*c):
        return _pick_col(df, *c)
    col_dpe = pick("DeltaP_ext", "Pext", "DeltaPext", "delta_p_ext")
    col_dpi = pick("DeltaP_int", "Pint", "DeltaPint", "delta_p_int")
    col_qi  = pick("Q_int", "Qint", "Q_door", "Qdoor", "Q_d", "Q_door_panel")
    col_qe  = pick("Q_ext", "Qext", "Q_window", "Qwindow", "Q_w", "Q_window_panel")

    try:
        col_aext = pick("A_ext", "Aext")
        A_ext = df[col_aext].values
    except KeyError:
        if a_ext is None:
            raise ValueError("A_ext not found in CSV and no --a_ext provided.")
        A_ext = float(a_ext)

    try:
        col_aint = pick("A_int", "Aint")
        A_int = df[col_aint].values
    except KeyError:
        if a_int is None:
            raise ValueError("A_int not found in CSV and no --a_int provided.")
        A_int = float(a_int)

    return zmpt_solve(
        DeltaP_ext=df[col_dpe].values,
        DeltaP_int=df[col_dpi].values,
        Q_int=df[col_qi].values,
        Q_ext=df[col_qe].values,
        A_ext=A_ext, A_int=A_int,
        **kwargs
    )


def _cli():
    p = argparse.ArgumentParser(description="ZMPT inverse solver (Newton + numerical Jacobian).")
    p.add_argument("--csv", required=True, help="Path to CSV with DeltaP_ext, DeltaP_int, Q_int, Q_ext; optional A_ext/A_int.")
    p.add_argument("--a_ext", type=float, default=None, help="Exterior area (if not in CSV).")
    p.add_argument("--a_int", type=float, default=None, help="Interior area (if not in CSV).")
    p.add_argument("--C_ext0", type=float, default=0.1)
    p.add_argument("--C_int0", type=float, default=0.1)
    p.add_argument("--n_ext0", type=float, default=0.6)
    p.add_argument("--n_int0", type=float, default=0.6)
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--step", type=float, default=0.01)
    p.add_argument("--damping", type=float, default=1.0)
    p.add_argument("--rtol", type=float, default=1e-8)
    args = p.parse_args()

    res = zmpt_solve_from_csv(
        args.csv, a_ext=args.a_ext, a_int=args.a_int,
        C_ext0=args.C_ext0, C_int0=args.C_int0,
        n_ext0=args.n_ext0, n_int0=args.n_int0,
        iterations=args.iterations, step=args.step, damping=args.damping, rtol=args.rtol
    )

    print("Converged:", res.converged)
    print(f"C_ext = {res.C_ext:.6g}, n_ext = {res.n_ext:.6g}")
    print(f"C_int = {res.C_int:.6g}, n_int = {res.n_int:.6g}")
    print("Last cond(L):", res.cond_history[-1] if res.cond_history else np.nan)
    print("History:")
    print(res.history.to_string(index=False))


if __name__ == "__main__":
    _cli()
