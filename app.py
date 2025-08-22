import streamlit as st
import pandas as pd
from zmpt.solver import zmpt_solve

# --- Force Light Theme ---
st.set_page_config(page_title="ZMPT Calculator", layout="wide")

# --- Title ---
st.title("ZMPT Blower Door Testing Calculator")

st.write(
    """
    This tool estimates leakage parameters (`C_ext`, `C_int`, `n_ext`, `n_int`)
    from blower door ZMPT test data.
    You can either upload a CSV file **or** enter the data manually.
    """
)

# --- Defaults ---
default_values = {
    "A_ext": 0.0,
    "A_int": 0.0,
    "C_ext": 0.1,
    "C_int": 0.1,
    "n_ext": 0.6,
    "n_int": 0.6,
    "iterations": 20,
    "step_frac": 0.01,
    "damping": 1.0,
}

def reset_app():
    """Reset all session state values and clear UI."""
    for k, v in default_values.items():
        st.session_state[k] = v
    st.session_state["manual_df"] = pd.DataFrame(
        {"DeltaP_ext": [0.0] * 4,
         "DeltaP_int": [0.0] * 4,
         "Q_ext": [0.0] * 4,
         "Q_int": [0.0] * 4}
    )
    st.session_state["uploaded_file"] = None
    st.session_state["results"] = None

# --- Sidebar Parameters ---
st.sidebar.header("Inputs")

A_ext = st.sidebar.number_input("Exterior Area (A_ext) [ft²]",
                                value=st.session_state.get("A_ext", default_values["A_ext"]), min_value=0.0)
A_int = st.sidebar.number_input("Interior Area (A_int) [ft²]",
                                value=st.session_state.get("A_int", default_values["A_int"]), min_value=0.0)

st.sidebar.subheader("Initial Guesses")
init_C_ext = st.sidebar.number_input("Initial C_ext",
                                     value=st.session_state.get("C_ext", default_values["C_ext"]))
init_C_int = st.sidebar.number_input("Initial C_int",
                                     value=st.session_state.get("C_int", default_values["C_int"]))
init_n_ext = st.sidebar.number_input("Initial n_ext",
                                     value=st.session_state.get("n_ext", default_values["n_ext"]))
init_n_int = st.sidebar.number_input("Initial n_int",
                                     value=st.session_state.get("n_int", default_values["n_int"]))

with st.sidebar.expander("Advanced Options"):
    iterations = st.number_input("Iterations", min_value=1, max_value=100,
                                 value=st.session_state.get("iterations", default_values["iterations"]))
    step_frac = st.number_input("Step fraction",
                                value=st.session_state.get("step_frac", default_values["step_frac"]))
    damping = st.number_input("Damping factor",
                              value=st.session_state.get("damping", default_values["damping"]))

# --- Input Mode ---
mode = st.radio("Select input method:", ["Upload CSV", "Enter manually"])
df = None

if mode == "Upload CSV":
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="uploaded_file")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

elif mode == "Enter manually":
    st.info("Enter test data below. Start with 4 rows, and you can add more.")
    if "manual_df" not in st.session_state:
        st.session_state.manual_df = pd.DataFrame(
            {"DeltaP_ext": [0.0] * 4,
             "DeltaP_int": [0.0] * 4,
             "Q_ext": [0.0] * 4,
             "Q_int": [0.0] * 4}
        )

    manual_df = st.data_editor(
        st.session_state.manual_df,
        num_rows="dynamic",
        use_container_width=True,
    )
    df = manual_df

# --- Buttons ---
col1, col2 = st.columns([1, 1])
with col1:
    calc_clicked = st.button("Calculate", use_container_width=True)
with col2:
    reset_clicked = st.button("🔄 Reset", type="secondary", use_container_width=True)

if reset_clicked:
    reset_app()
    st.experimental_rerun()

# --- Run Solver ---
if df is not None and calc_clicked:
    if A_ext <= 0 or A_int <= 0:
        st.error("Error: Both A_ext and A_int must be greater than zero.")
    else:
        try:
            res = zmpt_solve(
                DeltaP_ext=df["DeltaP_ext"].values,
                DeltaP_int=df["DeltaP_int"].values,
                Q_int=df["Q_int"].values,
                Q_ext=df["Q_ext"].values,
                A_ext=A_ext,
                A_int=A_int,
                C_ext0=init_C_ext,
                C_int0=init_C_int,
                n_ext0=init_n_ext,
                n_int0=init_n_int,
                iterations=iterations,
                step=step_frac,
                damping=damping,
            )

            # Save results in session state
            st.session_state["results"] = res

        except Exception as e:
            st.error(f"Error during calculation: {e}")

# --- Display Results ---
if st.session_state.get("results"):
    res = st.session_state["results"]

    st.success("Calculation complete")

    # --- Main Results ---
    st.write("### Estimated Parameters")
    st.write(f"- C_ext = **{res.C_ext:.6f}**")
    st.write(f"- n_ext = **{res.n_ext:.6f}**")
    st.write(f"- C_int = **{res.C_int:.6f}**")
    st.write(f"- n_int = **{res.n_int:.6f}**")

    # --- Airflow at 50 Pa ---
    Q50_ext = res.C_ext * (50.0 ** res.n_ext)
    Q50_int = res.C_int * (50.0 ** res.n_int)

    st.write("### Airflow at 50 Pa")
    st.write(f"- Exterior: **{Q50_ext:.3f} CFM/ft²**")
    st.write(f"- Interior: **{Q50_int:.3f} CFM/ft²**")

    # --- Advanced Results ---
    with st.expander("Advanced Results"):
        st.write(f"**Converged:** {res.converged}")
        st.write(f"**Last condition number:** {res.cond_history[-1] if res.cond_history else 'N/A'}")

        st.write("#### Iteration History")
        st.dataframe(res.history, use_container_width=True)

        st.line_chart(res.history[["C_ext", "C_int", "n_ext", "n_int"]])
        st.line_chart(pd.DataFrame({"Residual Norm ||f||2": res.residual_history}))
