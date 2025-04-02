import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.integrate import quad
import pandas as pd
import seaborn as sns
import math
import streamlit as st

# Increase the maximum number of elements that can be rendered
pd.set_option("styler.render.max_elements", 1000000)

# Model functions (keep all your existing model functions here)
# [Keep all your existing model functions exactly as they were]

def main():
    st.title("Dynamical System Model")
    st.latex(r"\frac{dC}{dt}=\alpha[aCP-b\frac{C{^2}}{P}]-dC+\beta[cCL]")
    st.latex(r"\frac{dP}{dt}=e-fCP-gP")

    # Parameters and inputs (keep all your existing parameter inputs here)
    # [Keep all your existing parameter inputs exactly as they were]
    
    # Run simulation
    dx = 0.0005
    t = np.arange(0, days + dx, dx)
    args = set_params(**params, Kc=Kc, Kt=Kt, z=z, q=q)
    
    if st.button("Run Simulation"):
        C, P, L, alpha_vals, beta_vals = run_simulation(t, C0, P0, I0, args)
        
        st.subheader("Simulation Results")
        
        # Create two columns for the plots
        col1, col2 = st.columns(2)
        
        with col1:
            # First plot: C, P, L
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            ax2 = ax1.twinx()
            ax3 = ax1.twinx()
            
            ax3.spines.right.set_position(("outward", 60))
            
            ax1.plot(t, C, color="g", label="Cyanobacteria (C)")
            ax2.plot(t, P, color="r", label="Phosphorus (P)")
            ax3.plot(t, L, color="b", linestyle="dashed", label="Light (L)")
            
            ax1.set_xlabel("Time (days)")
            ax1.set_ylabel("Cyanobacteria", color="g")
            ax2.set_ylabel("Phosphorus", color="r")
            ax3.set_ylabel("Light Intensity", color="b")
            
            ax1.legend(loc="upper left")
            ax2.legend(loc="upper right")
            ax3.legend(loc="center right")
            ax1.grid(True)
            ax1.set_title("Cyanobacteria, Phosphorus and Light Dynamics")
            st.pyplot(fig1)
        
        with col2:
            # Second plot: Alpha and Beta
            fig2, ax = plt.subplots(figsize=(10, 6))
            
            ax.plot(t, alpha_vals, color="purple", label="Alpha (1-switch)")
            ax.plot(t, beta_vals, color="orange", label="Beta (switch)")
            
            ax.set_xlabel("Time (days)")
            ax.set_ylabel("Value")
            ax.legend(loc="upper right")
            ax.grid(True)
            ax.set_title("Alpha and Beta Dynamics")
            ax.set_ylim(-0.1, 1.1)  # Since alpha/beta are between 0 and 1
            
            st.pyplot(fig2)
        
        # Show alpha and beta values in a downsampled dataframe
        st.subheader("Switch Values Over Time (Sampled)")
        sample_rate = max(1, len(t) // 1000)  # Show about 1000 points max
        df = pd.DataFrame({
            'Time': t[::sample_rate],
            'Alpha (1-switch)': alpha_vals[::sample_rate],
            'Beta (switch)': beta_vals[::sample_rate]
        })
        st.dataframe(df.style.format("{:.4f}"), height=300)

if __name__ == "__main__":
    main()
