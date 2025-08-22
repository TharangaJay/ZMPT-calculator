
# ZMPT Calculator (Blower Door Zonal Multipoint Pressure Test)

Open-source implementation of the ZMPT inverse solver and a Streamlit dashboard.
It estimates the normalized leakage parameters for exterior and interior boundaries:

- `C_ext, n_ext` for exterior leakage
- `C_int, n_int` for interior leakage

## Quick start (local)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## CSV format

Required columns (case-insensitive; spaces/underscores ignored):

- `DeltaP_ext`, `DeltaP_int` (Pa)
- `Q_door`, `Q_window` (consistent units)

Optional per-row: `A_ext`, `A_int`. Otherwise enter constants in the UI.

Residual equation solved:

```text
f = C_ext*A_ext*(DeltaP_ext)**n_ext + Q_window - C_int*A_int*(DeltaP_int)**n_int - Q_door = 0
```

## CLI

```bash
python -m zmpt.solver --csv examples/example.csv --a_ext 273 --a_int 2414
```

MIT Licensed.
