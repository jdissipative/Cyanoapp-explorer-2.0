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

# Modelo conjunto
def dC(t, C, P, L, a, b, c, d, e, f, g, Kc, Kt, z, q, switch):
    PLF = a * C * P - b * C * C / P
    LLF = c * C * L
    alpha = 1 - switch
    beta = switch
    return alpha * PLF + beta * LLF - d * C

def dP(t, C, P, a, b, c, d, e, f, g):
    return e - f * C * P - g * P

def Lv(C, I0, Kc, Kt, z):
    I = I0 * math.exp(-z * (C * Kc + Kt))
    return quad(lambda x: I, 0, z)[0]

def saturation(P, C, q):
    return P / (P + q * C)

def set_params(a, b, c, d, e, f, g, Kc, Kt, z, q):
    return a, b, c, d, e, f, g, Kc, Kt, z, q

def run_simulation(t, C0, P0, I0, args):
    C = np.zeros(len(t))
    P = np.zeros(len(t))
    L = np.zeros(len(t))
    alpha_vals = np.zeros(len(t))
    beta_vals = np.zeros(len(t))
    C[0] = C0
    P[0] = P0
    L[0] = Lv(C0, I0, *args[7:10])

    dx = t[1] - t[0]
    for i in range(len(t) - 1):
        if P[i] <= 1e-8:
            P[i] = 1.1e-7
        if C[i] <= 0 and P[i] >= 1e-8:
            C[i] = 1e-7
        switch = saturation(P[i], C[i], args[10])
        alpha_vals[i] = 1 - switch
        beta_vals[i] = switch
        L[i] = Lv(C[i], I0, *args[7:10])
        
        k11 = dC(t[i], C[i], P[i], L[i], *args, switch)
        k21 = dP(t[i], C[i], P[i], *args[:7])
        k12 = dC(t[i] + 0.5 * dx, C[i] + 0.5 * k11 * dx, P[i] + 0.5 * k21 * dx, L[i], *args, switch)
        k22 = dP(t[i] + 0.5 * dx, C[i] + 0.5 * k11 * dx, P[i] + 0.5 * k21 * dx, *args[:7])
        k13 = dC(t[i] + 0.5 * dx, C[i] + 0.5 * k12 * dx, P[i] + 0.5 * k22 * dx, L[i], *args, switch)
        k23 = dP(t[i] + 0.5 * dx, C[i] + 0.5 * k12 * dx, P[i] + 0.5 * k22 * dx, *args[:7])
        k14 = dC(t[i] + dx, C[i] + k13 * dx, P[i] + k23 * dx, L[i], *args, switch)
        k24 = dP(t[i] + dx, C[i] + k13 * dx, P[i] + k23 * dx, *args[:7])

        C[i + 1] = C[i] + (1 / 6) * (k11 + 2 * k12 + 2 *k13 + k14) * dx
        P[i + 1] = P[i] + (1 / 6) * (k21 + 2 * k22 + 2 * k23 + k24) * dx

    L[-1] = Lv(C[-1], I0, *args[7:10])
    switch = saturation(P[-1], C[-1], args[10])
    alpha_vals[-1] = 1 - switch
    beta_vals[-1] = switch
    return C, P, L, alpha_vals, beta_vals

def main():
    st.title("Dynamical System Model")
    st.latex(r"\frac{dC}{dt}=\alpha[aCP-b\frac{C{^2}}{P}]-dC+\beta[cCL]")
    st.latex(r"\frac{dP}{dt}=e-fCP-gP")

    # Parameters
    col1, col2 = st.columns(2)
    with col1:
        params = {param: st.slider(param, 0.00, 20.0, 0.5, 0.001, format="%.5f") 
                 for param in ['a', 'b', 'c', 'd']}
    with col2:
        params.update({param: st.slider(param, 0.00, 20.0, 0.5, 0.001, format="%.5f") 
                      for param in ['e', 'f', 'g']})
    
    st.subheader("Additional Parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        Kc = st.number_input("Kc", value=0.0003, format="%.6f")
        Kt = st.number_input("Kt", value=0.9, format="%.6f")
    with col2:
        z = st.number_input("z", value=2.0, format="%.6f")
        q = st.slider("q", 0.00, 100.0, 0.1, 0.001, format="%.5f")
    with col3:
        I0 = st.slider("Initial Light Intensity (I0)", 0.00, 800.0, 10.0, 1.0, format="%.2f")
    
    st.subheader("Initial Conditions")
    C0 = st.number_input("Initial Cyanobacteria (C0)", value=0.005, format="%.6f")
    P0 = st.number_input("Initial Phosphorus (P0)", value=0.005, format="%.6f")
    days = st.number_input("Simulation Days", value=100)
    
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
