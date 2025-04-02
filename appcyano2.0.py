# Modified run_simulation function to track alpha and beta
def run_simulation(t, C0, P0, I0, args):
    C = np.zeros(len(t))
    P = np.zeros(len(t))
    L = np.zeros(len(t))
    alpha_vals = np.zeros(len(t))  # New array for alpha values
    beta_vals = np.zeros(len(t))   # New array for beta values
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
        alpha_vals[i] = 1 - switch  # Store alpha
        beta_vals[i] = switch       # Store beta
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
    # Store final alpha and beta values
    switch = saturation(P[-1], C[-1], args[10])
    alpha_vals[-1] = 1 - switch
    beta_vals[-1] = switch
    return C, P, L, alpha_vals, beta_vals

# Update the simulation call
C, P, L, alpha_vals, beta_vals = run_simulation(t, C0, P0, I0, args)

# Update the plotting code
st.subheader("Simulation Results")
fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()
ax3 = ax1.twinx()
ax3.spines.right.set_position(("outward", 60))
ax4 = ax1.twinx()
ax4.spines.right.set_position(("outward", 120))

ax1.plot(t, C, color="g", label="Cyanobacteria (C)")
ax2.plot(t, P, color="r", label="Phosphorus (P)")
ax3.plot(t, L, color="b", linestyle="dashed", label="Light reaching C (L)")
ax4.plot(t, alpha_vals, color="purple", linestyle=":", label="Alpha (1-switch)")
ax4.plot(t, beta_vals, color="orange", linestyle=":", label="Beta (switch)")

ax1.set_xlabel("Time (days)")
ax1.set_ylabel("Cyanobacteria Concentration", color="g")
ax2.set_ylabel("Phosphorus Concentration", color="r")
ax3.set_ylabel("Light Intensity", color="b")
ax4.set_ylabel("Alpha/Beta Values", color="purple")

ax1.legend(loc="upper left")
ax2.legend(loc="upper right")
ax3.legend(loc="center right")
ax4.legend(loc="lower right")
ax1.grid(True)

st.pyplot(fig)
