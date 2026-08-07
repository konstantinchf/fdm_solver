#!/usr/bin/env python3
"""
Final ensemble: 5 seeds for μ = 1, 10, 100 in double precision.
- Correct initial field generation (IFFT)
- K2 and K2_poisson separated
- complex128 throughout
- Saves CSV, computes mean and std, plots with error bars
"""
import torch
import torch.fft
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ---------- ПАРАМЕТРЫ ----------
N = 128
L0 = 40.0                     # kpc
dx = L0 / N
dt0 = 0.01
t0 = 10.0
steps = 2000
g_nl = 0.1
scale_rho = 1.0
G = 1.0
flattening = 1.0
spin = 0.0

mu_list = [1.0, 10.0, 100.0]
seed_list = [42, 43, 44, 45, 46]   # 5 независимых реализаций

output_dir = './ensemble_three_masses'
os.makedirs(output_dir, exist_ok=True)

# ---------- ФУНКЦИИ ----------
def create_initial_state_double(N, L, seed, flattening=1.0, spin=0.0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    kx = torch.fft.fftfreq(N, d=L/N, device=device, dtype=torch.float64) * 2 * torch.pi
    ky = torch.fft.fftfreq(N, d=L/N, device=device, dtype=torch.float64) * 2 * torch.pi
    kz = torch.fft.fftfreq(N, d=L/N, device=device, dtype=torch.float64) * 2 * torch.pi

    K2_amp = kx[:, None, None]**2 + ky[None, :, None]**2 + (flattening * kz[None, None, :])**2
    K2_amp[0,0,0] = 1.0
    amp = torch.sqrt(K2_amp ** (-2))

    real = torch.randn(N, N, N, device=device, dtype=torch.float64)
    imag = torch.randn(N, N, N, device=device, dtype=torch.float64)
    noise_k = torch.complex(real, imag)
    psi_k = noise_k * amp.to(torch.float64)
    psi_k[0,0,0] = 0.0 + 0.0j
    psi = torch.fft.ifftn(psi_k)

    if spin != 0.0:
        X, Y, Z = torch.meshgrid(
            torch.linspace(-L/2, L/2, N, device=device, dtype=torch.float64),
            torch.linspace(-L/2, L/2, N, device=device, dtype=torch.float64),
            torch.linspace(-L/2, L/2, N, device=device, dtype=torch.float64),
            indexing='ij'
        )
        phase = spin * torch.atan2(Y, X)
        psi = psi * torch.exp(1j * phase)

    psi = psi / torch.sqrt(torch.mean(torch.abs(psi)**2))
    return psi

def compute_potential(rho, G, scale_rho, K2_poisson, a):
    rho_mean = torch.mean(rho)
    delta_rho = (rho - rho_mean) * scale_rho / a
    rho_k = torch.fft.fftn(delta_rho)
    phi_k = -4 * torch.pi * G * rho_k / K2_poisson
    phi_k[0,0,0] = 0.0 + 0.0j
    phi = torch.real(torch.fft.ifftn(phi_k))
    return phi

def run_simulation(psi_initial, mu):
    dt = dt0 / mu
    actual_steps = int(steps * mu)
    psi = psi_initial.clone()

    kx = torch.fft.fftfreq(N, d=dx, device=device, dtype=torch.float64) * 2 * torch.pi
    ky = torch.fft.fftfreq(N, d=dx, device=device, dtype=torch.float64) * 2 * torch.pi
    kz = torch.fft.fftfreq(N, d=dx, device=device, dtype=torch.float64) * 2 * torch.pi

    K2 = kx[:, None, None]**2 + ky[None, :, None]**2 + kz[None, None, :]**2
    K2_poisson = K2.clone()
    K2_poisson[0,0,0] = 1.0

    for step in range(actual_steps):
        t_mid = t0 + (step + 0.5) * dt
        a = (t_mid / t0) ** (2.0 / 3.0)
        H = (2.0 / 3.0) / t_mid

        rho = torch.abs(psi)**2
        phi = compute_potential(rho, G, scale_rho, K2_poisson, a)
        V = mu * phi + g_nl * rho * mu

        damp_half = torch.exp(torch.tensor(-0.375 * H * dt, device=device, dtype=torch.float64))

        psi = psi * torch.exp(-0.5j * dt * V) * damp_half
        kinetic = torch.exp(-1j * dt * K2 / (2 * mu * a**2))
        psi_k = torch.fft.fftn(psi)
        psi_k = psi_k * kinetic
        psi = torch.fft.ifftn(psi_k)
        rho = torch.abs(psi)**2
        phi = compute_potential(rho, G, scale_rho, K2_poisson, a)
        V = mu * phi + g_nl * rho * mu
        psi = psi * torch.exp(-0.5j * dt * V) * damp_half

    rho = torch.abs(psi)**2
    t_final = t0 + steps * dt0
    a_final = (t_final / t0) ** (2.0 / 3.0)
    phi = compute_potential(rho, G, scale_rho, K2_poisson, a_final)

    return psi, rho, phi

def measure_noise(phi, K_vals, k_cut):
    phi_depth = torch.std(phi).item()
    if phi_depth < 1e-12:
        phi_depth = 1.0
    mask = K_vals > k_cut
    phi_k = torch.fft.fftn(phi).cpu().numpy()
    phi_high = np.copy(phi_k)
    phi_high[~mask] = 0
    phi_high_real = np.real(np.fft.ifftn(phi_high))
    return np.std(phi_high_real) / phi_depth

# ---------- ОСНОВНОЙ ЦИКЛ ----------
results = []
all_noise = {mu: [] for mu in mu_list}
all_norm = {mu: [] for mu in mu_list}

total_runs = len(mu_list) * len(seed_list)
run_idx = 0

for seed in seed_list:
    print(f"\n=== Seed {seed} ===")
    psi_initial = create_initial_state_double(N, L0, seed, flattening, spin)
    norm_initial = torch.sum(torch.abs(psi_initial)**2).item()

    for mu in mu_list:
        run_idx += 1
        print(f"  μ = {mu:.1f}  (run {run_idx}/{total_runs})")
        start_time = time.time()

        psi, rho, phi = run_simulation(psi_initial, mu)
        elapsed = time.time() - start_time

        norm_final = torch.sum(torch.abs(psi)**2).item()
        norm_ratio = norm_final / norm_initial if norm_initial > 0 else 1.0

        kx = np.fft.fftfreq(N, d=dx) * 2 * np.pi
        ky = np.fft.fftfreq(N, d=dx) * 2 * np.pi
        kz = np.fft.fftfreq(N, d=dx) * 2 * np.pi
        K2_vals = kx[:, None, None]**2 + ky[None, :, None]**2 + kz[None, None, :]**2
        K_vals = np.sqrt(K2_vals)

        noise_fixed = measure_noise(phi, K_vals, k_cut=0.5)

        all_noise[mu].append(noise_fixed)
        all_norm[mu].append(norm_ratio)

        print(f"    noise = {noise_fixed:.6f}, norm = {norm_ratio:.10f}, time = {elapsed:.1f}s")

        results.append({
            'seed': seed,
            'mu': mu,
            'mass_eV': mu * 1e-22,
            'noise_fixed': noise_fixed,
            'norm_ratio': norm_ratio,
            'phi_std': torch.std(phi).item(),
            'elapsed': elapsed
        })

# ---------- СОХРАНЕНИЕ CSV ----------
csv_file = os.path.join(output_dir, 'ensemble_three_masses.csv')
with open(csv_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    for r in results:
        writer.writerow(r)
print(f"\nCSV saved to {csv_file}")

# ---------- СТАТИСТИКА ----------
mu_vals = sorted(all_noise.keys())
mass_vals = np.array(mu_vals) * 1e-22
means = [np.mean(all_noise[mu]) for mu in mu_vals]
stds = [np.std(all_noise[mu]) for mu in mu_vals]

norm_means = [np.mean(all_norm[mu]) for mu in mu_vals]
norm_stds = [np.std(all_norm[mu]) for mu in mu_vals]

# Вывод статистики
print("\n=== STATISTICS (5 seeds) ===")
for mu, mean, std, nmean, nstd in zip(mu_vals, means, stds, norm_means, norm_stds):
    print(f"μ={mu:.1f}: noise = {mean:.6f} ± {std:.6f}, norm = {nmean:.10f} ± {nstd:.2e}")

# ---------- ГРАФИКИ ----------
# 1. Noise with error bars
plt.figure(figsize=(8,6))
plt.errorbar(mass_vals, means, yerr=stds, fmt='bo-', capsize=4, linewidth=2, markersize=8)
plt.axhline(y=0.02, color='r', linestyle='--', alpha=0.7, label='Illustrative threshold')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\mu = m_{\rm FDM} / 10^{-22}\,\mathrm{eV}$')
plt.ylabel('Granular noise RMS (fixed k-cut)')
plt.title('Ensemble: 5 seeds, double precision')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.savefig(os.path.join(output_dir, 'noise_ensemble.png'), dpi=150)
plt.close()

# 2. Individual scatter
plt.figure(figsize=(8,6))
for mu in mu_vals:
    x = [mu] * len(all_noise[mu])
    y = all_noise[mu]
    plt.scatter(x, y, alpha=0.4, color='blue')
plt.errorbar(mu_vals, means, yerr=stds, fmt='ro-', capsize=4, linewidth=2, markersize=8,
             label='Mean ± std')
plt.axhline(y=0.02, color='r', linestyle='--', alpha=0.7, label='Illustrative threshold')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\mu = m_{\rm FDM} / 10^{-22}\,\mathrm{eV}$')
plt.ylabel('Granular noise RMS (fixed k-cut)')
plt.title('Individual realisations and mean (5 seeds)')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.savefig(os.path.join(output_dir, 'noise_ensemble_scatter.png'), dpi=150)
plt.close()

# ... (весь код до графиков без изменений) ...

# 3. Norm ratio
plt.figure(figsize=(8,6))
plt.errorbar(mu_vals, norm_means, yerr=norm_stds, fmt='go-', capsize=4, linewidth=2, markersize=8)
plt.axhline(y=1/3, color='r', linestyle='--', alpha=0.7, label='Analytical expectation (1/3)')
plt.xscale('log')
plt.yscale('linear')
plt.xlabel(r'$\mu = m_{\rm FDM} / 10^{-22}\,\mathrm{eV}$')
plt.ylabel(r'Norm ratio ($|\psi_f|^2 / |\psi_i|^2$)')
plt.title('Norm conservation (5 seeds, double precision)')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.savefig(os.path.join(output_dir, 'norm_ensemble.png'), dpi=150)
plt.close()

print(f"\n✅ Ensemble complete. Results saved in {output_dir}")