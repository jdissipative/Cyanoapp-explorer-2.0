import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.integrate import quad
import pandas as pd
import seaborn as sns
import math
import streamlit as st

# Modelo conjunto
def dC(t, C, P, L, a, b, c, d, e, f, g, Kc, Kt, z, q, switch):
    # Phosphorus as limitant factor expression (PLF)
    PLF = a * C * P - b * C * C / P
    # Light as limitant factor expression (LLF)
    LLF = c * C * L
    alpha = 1 - switch
    beta = switch
    # Model
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
    C[0] = C0
    P[0] = P0
    L[0] = Lv(C0, I0, *args[7:10])

    for i in range(len(t) - 1):
        if P[i] <= 1e-8:
            P[i] = 1.1e-7

        if C[i] <= 0 and P[i] >= 1e-8:
            C[i] = 1e-7

        switch = saturation(P[i], C[i], args[10])
        L[i] = Lv(C[i], I0, *args[7:10])

        k11 = dC(t[i], C[i], P[i], L[i], *args, switch)
        k21 = dP(t[i], C[i], P[i], *args[:7])

        k12 = dC(t[i] + 0.5 * dx, C[i] + 0.5 * k11 * dx, P[i] + 0.5 * k21 * dx, L[i], *args, switch)
        k22 = dP(t[i] + 0.5 * dx, C[i] + 0.5 * k11 * dx, P[i] + 0.5 * k21 * dx, *args[:7])

        k13 = dC(t[i] + 0.5 * dx, C[i] + 0.5 * k12 * dx, P[i] + 0.5 * k22 * dx, L[i], *args, switch)
        k23 = dP(t[i] + 0.5 * dx, C[i] + 0.5 * k12 * dx, P[i] + 0.5 * k22 * dx, *args[:7])

        k14 = dC(t[i] + dx, C[i] + k13 * dx, P[i] + k23 * dx, L[i], *args, switch)
        k24 = dP(t[i] + dx, C[i] + k13 * dx, P[i] + k23 * dx, *args[:7])

        C[i + 1] = C[i] + (1 / 6) * (k11 + 2 * k12 + 2 * k13 + k14) * dx
        P[i + 1] = P[i] + (1 / 6) * (k21 + 2 * k22 + 2 * k23 + k24) * dx

    L[-1] = Lv(C[-1], I0, *args[7:10])

    return C, P, L

st.title("Dynamical System Model")
st.latex(r"\frac{dC}{dt}=\alpha[aCP-b\frac{C{^2}}{P}]-dC+\beta[cCL]")
st.latex(r"\frac{dP}{dt}=e-fCP-gP")

params = {param: st.slider(param, 0.00, 20.0, 0.5, 0.001, format="%.5f") for param in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'Kc', 'Kt', 'z']}
q = st.slider("q", 0.00, 5.0, 0.1, 0.001, format="%.5f")
I0 = st.slider("Initial Light Intensity (I0)", 0.00, 50.0, 10.0, 0.01, format="%.2f")
C0 = st.number_input("Initial Cyanobacteria (C0)", value=0.005, format="%.6f")
P0 = st.number_input("Initial Phosphorus (P0)", value=0.005, format="%.6f")
days = st.number_input("Days", value=100)
dx = 0.0005
t = np.arange(0, days + dx, dx)
args = set_params(**params, q=q)

C, P, L = run_simulation(t, C0, P0, I0, args)

st.subheader("Simulation Results")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(t, C, color="g", label="Cyanobacteria (C)")
ax.plot(t, P, color="r", label="Phosphorus (P)")
ax.plot(t, L, color="b", linestyle="dashed", label="Light reaching C (L)")
ax.set_xlabel("Time (d)")
ax.set_ylabel("Concentration / Light")
ax.legend()
ax.grid(True)
st.pyplot(fig)

