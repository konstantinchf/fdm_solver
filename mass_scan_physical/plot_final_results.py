#!/usr/bin/env python3
"""
FINAL PLOTS for ensemble_three_masses.csv
- Correct axes: mu (dimensionless) or mass (eV)
- Norm relative error: (norm_ratio / (1/3)) - 1
- Saves high-resolution PNGs for publication
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Загружаем данные
df = pd.read_csv('ensemble_three_masses.csv')

# Группируем по mu
mu_vals = sorted(df['mu'].unique())
mass_vals = np.array(mu_vals) * 1e-22  # физическая масса в эВ

# Вычисляем статистику
means = []
stds = []
norm_means = []
norm_stds = []

for mu in mu_vals:
    subset = df[df['mu'] == mu]
    means.append(subset['noise_fixed'].mean())
    stds.append(subset['noise_fixed'].std())
    norm_means.append(subset['norm_ratio'].mean())
    norm_stds.append(subset['norm_ratio'].std())

means = np.array(means)
stds = np.array(stds)
norm_means = np.array(norm_means)
norm_stds = np.array(norm_stds)

# Относительная ошибка нормы
expected = 1.0 / 3.0
norm_rel_err = norm_means / expected - 1.0
norm_rel_std = norm_stds / expected

# ============================================================
# 1. ГРАФИК: NOISE vs μ (среднее ± std)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))

ax.errorbar(mu_vals, means, yerr=stds, fmt='bo-', capsize=4,
            linewidth=2, markersize=8, label='Mean ± std (5 seeds)')
ax.axhline(y=0.02, color='r', linestyle='--', alpha=0.7,
           label='Illustrative threshold (not calibrated)')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$\mu \equiv m_{\rm FDM} / 10^{-22}\,\mathrm{eV}$', fontsize=12)
ax.set_ylabel('Fractional high-k potential RMS', fontsize=12)
ax.set_title('Mass dependence of granular noise (double precision)', fontsize=14)
ax.grid(True, which='both', ls='--', alpha=0.5)
ax.legend()

plt.tight_layout()
plt.savefig('final_noise_vs_mu.png', dpi=300)
plt.close()

# ============================================================
# 2. ГРАФИК: NOISE vs μ (индивидуальные точки + среднее)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))

# Индивидуальные точки
for mu in mu_vals:
    subset = df[df['mu'] == mu]
    x = [mu] * len(subset)
    y = subset['noise_fixed'].values
    ax.scatter(x, y, alpha=0.4, color='blue', s=30)

# Среднее с ошибками
ax.errorbar(mu_vals, means, yerr=stds, fmt='ro-', capsize=4,
            linewidth=2, markersize=8, label='Mean ± std')

ax.axhline(y=0.02, color='r', linestyle='--', alpha=0.7,
           label='Illustrative threshold')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$\mu \equiv m_{\rm FDM} / 10^{-22}\,\mathrm{eV}$', fontsize=12)
ax.set_ylabel('Fractional high-k potential RMS', fontsize=12)
ax.set_title('Individual realisations and mean (5 seeds)', fontsize=14)
ax.grid(True, which='both', ls='--', alpha=0.5)
ax.legend()

plt.tight_layout()
plt.savefig('final_noise_scatter.png', dpi=300)
plt.close()

# ============================================================
# 3. ГРАФИК: ОТНОСИТЕЛЬНАЯ ОШИБКА НОРМЫ
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))

ax.errorbar(mu_vals, norm_rel_err, yerr=norm_rel_std, fmt='go-',
            capsize=4, linewidth=2, markersize=8)
ax.axhline(0.0, color='r', linestyle='--', alpha=0.7,
           label='Expected (0.0)')

ax.set_xscale('log')
ax.set_yscale('symlog', linthresh=1e-11)
ax.set_xlabel(r'$\mu \equiv m_{\rm FDM} / 10^{-22}\,\mathrm{eV}$', fontsize=12)
ax.set_ylabel(r'$(N_f/N_i)/(1/3) - 1$', fontsize=12)
ax.set_title('Norm conservation: relative error (double precision)', fontsize=14)
ax.grid(True, which='both', ls='--', alpha=0.5)
ax.legend()

plt.tight_layout()
plt.savefig('final_norm_error.png', dpi=300)
plt.close()

print("\n✅ FINAL PLOTS SAVED:")
print("  - final_noise_vs_mu.png")
print("  - final_noise_scatter.png")
print("  - final_norm_error.png")
print("\nTable for publication:")
print("μ    | noise_mean ± std     | norm_rel_error")
for mu, m, s, err in zip(mu_vals, means, stds, norm_rel_err):
    print(f"{mu:4.0f}  | {m:.6f} ± {s:.6f}  | {err:.2e}")