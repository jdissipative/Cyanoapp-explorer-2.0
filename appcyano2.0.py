import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.integrate import quad
import pandas as pd
import seaborn as sns
import math
import streamlit as st

def dC(t, C, P, L, a, b, c, d, e, f, g, h, i, Kc, Kt, z, switch):
    PLF = a * C * P - b * C - c * C * C / P
    LLF = g * C * L - h * C - i * C / L
    return PLF if switch == 0 else LLF

def dP(t, C, P, a, b, c, d, e, f):
    return d - e * C * P - f * P

def Lv(C, I0, g, h, i, Kc, Kt, z):
    I = I0 * math.exp(-z * (C * Kc + Kt))
    return quad(lambda x: I, 0, z)[0]

def run_simulation(params, C0, P0, I0, UMBRAL, days=100, dx=0.001):
    t = np.arange(0, days + dx, dx)
    C, P, L = np.zeros(len(t)), np.zeros(len(t)), np.zeros(len(t))
    C[0], P[0] = C0, P0
    L[0] = Lv(C0, I0[0], *params[6:])
    
    for i in range(len(t) - 1):
        if P[i] <= 1e-8:
            C[i] = 0
            P[i] = 1.1e-7
        if C[i] <= 0 and P[i] >= 1e-8:
            C[i] = 1e-7
        switch = 1 if P[i] >= UMBRAL else 0
        Ii = int((i * dx - dx) // 1)
        L[i] = Lv(C[i], I0[Ii], *params[6:])
        
        k11 = dC(t[i], C[i], P[i], L[i], *params, switch)
        k21 = dP(t[i], C[i], P[i], *params[:6])
        
        k12 = dC(t[i] + 0.5 * dx, C[i] + 0.5 * k11 * dx, P[i] + 0.5 * k21 * dx, L[i], *params, switch)
        k22 = dP(t[i] + 0.5 * dx, C[i] + 0.5 * k11 * dx, P[i] + 0.5 * k21 * dx, *params[:6])
        
        k13 = dC(t[i] + 0.5 * dx, C[i] + 0.5 * k12 * dx, P[i] + 0.5 * k22 * dx, L[i], *params, switch)
        k23 = dP(t[i] + 0.5 * dx, C[i] + 0.5 * k12 * dx, P[i] + 0.5 * k22 * dx, *params[:6])
        
        k14 = dC(t[i] + dx, C[i] + k13 * dx, P[i] + k23 * dx, L[i], *params, switch)
        k24 = dP(t[i] + dx, C[i] + k13 * dx, P[i] + k23 * dx, *params[:6])
        
        C[i + 1] = C[i] + (1 / 6) * (k11 + 2 * k12 + 2 * k13 + k14) * dx
        P[i + 1] = P[i] + (1 / 6) * (k21 + 2 * k22 + 2 * k23 + k24) * dx
    
    L[-1] = Lv(C[-1], I0[Ii], *params[6:])
    return t, C, P, L

st.title("Dynamical System Model")

params = {param: st.slider(param, 0.0, 10.0, 3.0) for param in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'Kc', 'Kt', 'z']}
C0 = st.number_input("Initial Cyanobacteria (C0)", value=0.005, format="%.6f")
P0 = st.number_input("Initial Phosphorus (P0)", value=0.005, format="%.6f")
UMBRAL = st.number_input("Umbral", value=0.5, format="%.6f")
days = st.number_input("Days (P0)", value=100)
I0 = np.random.uniform(20, 40, days)

t, C, P, L = run_simulation(list(params.values()), C0, P0, I0, UMBRAL)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(t, C, color="g", label="Cyanobacteria (C)")
ax.set_xlabel("Time (d)")
ax.set_ylabel("Cyanobacteria (g/L)", color="g")
ax.tick_params(axis="y", labelcolor="g")
ax.legend()
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(t, P, color="r", label="Phosphorus (P)")
ax.set_xlabel("Time (d)")
ax.set_ylabel("Phosphorus")
ax.tick_params(axis="y", labelcolor="r")
ax.legend()
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(t, L, color="b", label="Light reaching C (L)")
ax.set_xlabel("Time (d)")
ax.set_ylabel("Light reaching C")
ax.tick_params(axis="y", labelcolor="b")
ax.legend()
st.pyplot(fig)
