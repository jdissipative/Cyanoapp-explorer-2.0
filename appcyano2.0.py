import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.integrate import quad
import pandas as pd
import seaborn as sns
import math
import streamlit as st

def dC(t, C, P, L, a, b, c, d, e, f, g, h, i, Kc, Kt, z, switch):
    #Phosphorus as limitant factor expression (PLF)
    PLF=a*C*P-b*C-c*C*C/P
    #Light as limitant factor expression (LLF)
    LLF=g*C*L-h*C-i*C/L
    return switch*PLF+(1-switch)*LLF

def dP(t, C, P, a, b, c, d, e, f):
    return (d-e*C*P-f*P)

def Lv( C, I0, g, h, i, Kc, Kt, z):
    I=I0*math.exp(-z*(C*Kc+Kt))
    return quad(lambda x: I, 0, z)[0] 

def set_params( a, b, c, d, e, f, g, h, i, Kc, Kt, z):
        a = a
        b = b
        c = c
        d = d
        e = e
        f = f
        g = g
        h = h
        i = i
        Kc = Kc
        Kt = Kt
        z = z
        return a, b, c, d, e, f, g, h, i, Kc, Kt, z

def run_simulation(t, C0, P0, I0, UMBRAL, args):
    C = np.zeros(len(t))
    P = np.zeros(len(t))
    L = np.zeros(len(t))
    C[0] = C0
    P[0] = P0
    L[0] = Lv(C0,I0[0], *args[6:] )

    for i in range(0,len(t)-1):

        if P[i]<=1e-8:
            C[i]=0
            P[i]=1.1e-7

        if C[i]<=0 and P[i]>=1e-8:
            C[i]=1e-7        
        if UMBRAL-P[i]<0:
            switch=0
        else:
            switch=(UMBRAL-P[i])/UMBRAL
            
            
        Ii=int((i*dx-dx)//1)
        L[i]=Lv(C[i],I0[Ii], *args[6:])

        k11=dC(t[i],C[i], P[i], L[i], *args, switch)
        k21=dP(t[i],C[i], P[i],*args[0:6])

        k12=dC(t[i]+0.5*dx,C[i]+0.5*k11*dx, P[i]+0.5*k21*dx, L[i], *args, switch)
        k22=dP(t[i]+0.5*dx,C[i]+0.5*k11*dx, P[i]+0.5*k21*dx,*args[0:6])

        k13=dC(t[i]+0.5*dx, C[i]+0.5*k12*dx, P[i]+0.5*k22*dx, L[i], *args, switch)
        k23=dP(t[i]+0.5*dx, C[i]+0.5*k12*dx, P[i]+0.5*k22*dx,*args[0:6])

        k14=dC(t[i]+dx, C[i]+k13*dx, P[i]+k23*dx, L[i], *args, switch)
        k24=dP(t[i]+dx, C[i]+k13*dx, P[i]+k23*dx,*args[0:6])

        C[i + 1]=C[i]+(1/6)*(k11+2*k12+2*k13+k14)*dx
        P[i + 1]=P[i]+(1/6)*(k21+2*k22+2*k23+k24)*dx

    L[len(t)-1]=Lv(C[len(t)-1],I0[Ii],*args[6:], )

    return C, P, L
data=pd.read_csv('solar.csv', sep=';')
GHI=data['GHI']
st.title("Dynamical System Model")
left_col, spacer, right_col = st.columns([1, 0.2, 2])
with left_col:
    params = {param: st.slider(param, 0.00, 10.0, 0.5, 0.001, format="%.5f") for param in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'Kc', 'Kt', 'z']}
    C0 = st.number_input("Initial Cyanobacteria (C0)", value=0.005, format="%.6f")
    P0 = st.number_input("Initial Phosphorus (P0)", value=0.005, format="%.6f")
    UMBRAL = st.number_input("Umbral", value=0.5, format="%.6f")
    days = st.number_input("Days (P0)", value=100)
    #I0 = np.random.uniform(20, 40, days)
    #x = np.arange(days)  # Create an array from 0 to days-1
    ##I0 = 10 * np.cos(x / (9 * np.pi)) + 30
    I0=GHI[0:days-1]
    size(I0)
    I0_time = np.linspace(0, days, len(I0))
    dx = 0.0005
    t = np.arange(0,days+ dx, dx)
    args=set_params(**params)

C, P, L = run_simulation(t, C0, P0, I0, UMBRAL, args)

with right_col:
    st.subheader("Simulation Results")
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # First axis: Cyanobacteria concentration
    ax1.plot(t, C, color="g", label="Cyanobacteria (C)")
    ax1.set_xlabel("Time (d)", fontsize=12)
    ax1.set_ylabel("Cyanobacteria (g/L)", color="g", fontsize=12)
    ax1.tick_params(axis="y", labelcolor="g")
    
    # Second axis: Light reaching cyanobacteria
    ax2 = ax1.twinx()
    ax2.plot(t, P, color="r", label="Phosphorus (P)")
    ax2.set_ylabel("Phosphorus", color="r", fontsize=12)
    ax2.tick_params(axis="y", labelcolor="r")
    
    # Third axis: Light intensity in the air
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("outward", 60))  # Offset third axis
    ax3.plot(I0_time, I0, color="b", linestyle="dashed", label="Light Intensity (I0)")
    ax3.set_ylabel("Air Light Intensity", color="b", fontsize=12)
    ax3.tick_params(axis="y", labelcolor="b")
    
    # Custom legend
    ax1_lines = [plt.Line2D([0], [0], color="g", lw=2, label="Cyanobacteria (C)")]
    ax2_lines = [plt.Line2D([0], [0], color="r", lw=2, label="Phosphorus (P)")]
    ax3_lines = [plt.Line2D([0], [0], color="b", linestyle="dashed", lw=2, label="Air Light (I0)")]
    
    ax1.legend(handles=ax1_lines + ax2_lines + ax3_lines, loc="upper right", title="Legend")
    ax1.grid(True)
    
    st.pyplot(fig)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # First axis: Cyanobacteria concentration
    ax1.plot(t, C, color="g", label="Cyanobacteria (C)")
    ax1.set_xlabel("Time (d)", fontsize=12)
    ax1.set_ylabel("Cyanobacteria (g/L)", color="g", fontsize=12)
    ax1.tick_params(axis="y", labelcolor="g")
    
    # Second axis: Light reaching cyanobacteria
    ax2 = ax1.twinx()
    ax2.plot(t, L, color="r", label="Light (L)")
    ax2.set_ylabel("Light reaching C", color="r", fontsize=12)
    ax2.tick_params(axis="y", labelcolor="r")
    
    # Third axis: Light intensity in the air
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("outward", 60))  # Offset third axis
    ax3.plot(I0_time, I0, color="b", linestyle="dashed", label="Light Intensity (I0)")
    ax3.set_ylabel("Air Light Intensity", color="b", fontsize=12)
    ax3.tick_params(axis="y", labelcolor="b")
    
    # Custom legend
    ax1_lines = [plt.Line2D([0], [0], color="g", lw=2, label="Cyanobacteria (C)")]
    ax2_lines = [plt.Line2D([0], [0], color="r", lw=2, label="Light reaching C (L)")]
    ax3_lines = [plt.Line2D([0], [0], color="b", linestyle="dashed", lw=2, label="Air Light (I0)")]
    
    ax1.legend(handles=ax1_lines + ax2_lines + ax3_lines, loc="upper right", title="Legend")
    ax1.grid(True)
    
    st.pyplot(fig)

