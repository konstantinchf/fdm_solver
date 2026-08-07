#!/usr/bin/env python3
"""
FINAL: Physical mass scan for FDM.
- Correct initial field generation (IFFT, P(k) ∝ k^{-4})
- Independent realisations (new field for each seed)
- Same initial field for all masses within a seed
- Fixed k-cut for noise measurement (k=0.5)
- Multiple random realisations (n_seeds) for statistics
- Mean power spectrum and variance
- Full diagnostic output to CSV
"""
import torch
import torch.fft
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import time
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ---------- ПАРАМЕТРЫ ----------
N = 128
L0 = 40.0                     # kpc (physical box size)
dx = L0 / N
dt0 = 0.01                    # base time step for mu = 1
t0 = 10.0                     # initial time
steps = 2000                  # number of steps for mu = 1
g_nl = 0.1
scale_rho = 1.0
G = 1.0
flattening = 0.8
spin = 0.3

# Mass grid (logarithmic, 9 points)
mu_list = np.logspace(0, 2, 9)   # 1.0, 1.78, 3.16, 5.62, 10, 17.8, 31.6, 56.2, 100
n_seeds = 5                      # number of independent realisations
seed_base = 42

output_dir = './mass_scan_final'
os.makedirs(output_dir, exist_ok=True)

# ---------- FUNCTIONS ----------
def create_initial_state(N, L, seed, flattening=1.0, spin=0.0):
    """Generate field with correct spectrum P(k) ∝ k^{-4}."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    kx = torch.fft.fftfreq(N, d=L/N, device=device) * 2 * torch.pi
    ky = torch.fft.fftfreq(N, d=L/N, device=device) * 2 * torch.pi
    kz = torch.fft.fftfreq(N, d=L/N, device=device) * 2 * torch.pi
    K2 = kx[:, None, None]**2 + ky[None, :, None]**2 + (flattening * kz[None, None, :])**2
    K2[0,0,0] = 1.0
    amp = torch.sqrt(K2 ** (-2))

    noise_k = (torch.randn(N, N, N, device=device) + 1j * torch.randn(N, N, N, device=device))
    psi_k = noise_k * amp
    psi_k[0,0,0] = 0.0
    psi = torch.fft.ifftn(psi_k)

    if spin != 0.0:
        X, Y, Z = torch.meshgrid(
            torch.linspace(-L/2, L/2, N, device=device),
            torch.linspace(-L/2, L/2, N, device=device),
            torch.linspace(-L/2, L/2, N, device=device),
            indexing='ij'
        )
        phase = spin * torch.atan2(Y, X)
        psi = psi * torch.exp(1j * phase)

    psi = psi / torch.sqrt(torch.mean(torch.abs(psi)**2))
    return psi

def compute_potential(rho, G, scale_rho, K2, a):
    rho_mean = torch.mean(rho)
    delta_rho = (rho - rho_mean) * scale_rho / a
    rho_k = torch.fft.fftn(delta_rho)
    phi_k = -4 * torch.pi * G * rho_k / K2
    phi_k[0,0,0] = 0.0
    phi = torch.real(torch.fft.ifftn(phi_k))
    return phi

def run_simulation(psi_initial, mu, dt, actual_steps):
    """Evolve field for a given mass."""
    dt = dt0 / mu
    actual_steps = int(steps * mu)
    psi = psi_initial.clone()

    kx = torch.fft.fftfreq(N, d=dx, device=device) * 2 * torch.pi
    ky = torch.fft.fftfreq(N, d=dx, device=device) * 2 * torch.pi
    kz = torch.fft.fftfreq(N, d=dx, device=device) * 2 * torch.pi
    K2 = kx[:, None, None]**2 + ky[None, :, None]**2 + kz[None, None, :]**2
    K2[0,0,0] = 1.0

    for step in range(actual_steps):
        t = t0 + step * dt
        a = (t / t0) ** (2.0/3.0)
        H = (2.0/3.0) / t
        rho = torch.abs(psi)**2
        phi = compute_potential(rho, G, scale_rho, K2, a)
        V = mu * phi + g_nl * rho * mu
        damp = torch.exp(torch.tensor(-0.75 * H * dt, device=device))
        psi = psi * torch.exp(-0.5j * dt * V) * damp
        kinetic = torch.exp(-1j * dt * K2 / (2 * mu * a**2))
        psi_k = torch.fft.fftn(psi)
        psi_k = psi_k * kinetic
        psi = torch.fft.ifftn(psi_k)
        rho = torch.abs(psi)**2
        phi = compute_potential(rho, G, scale_rho, K2, a)
        V = mu * phi + g_nl * rho * mu
        psi = psi * torch.exp(-0.5j * dt * V) * damp

    return psi, rho, phi

def measure_noise(phi, K_vals, k_cut):
    """Measure RMS noise for a given k-cut."""
    phi_depth = torch.std(phi).item()
    if phi_depth < 1e-12:
        phi_depth = 1.0
    mask = K_vals > k_cut
    phi_k = torch.fft.fftn(phi).cpu().numpy()
    phi_high = np.copy(phi_k)
    phi_high[~mask] = 0
    phi_high_real = np.real(np.fft.ifftn(phi_high))
    return np.std(phi_high_real) / phi_depth

def compute_power_spectrum(rho, L, N, dx):
    rho_cpu = rho.cpu().numpy()
    delta = (rho_cpu - np.mean(rho_cpu)) / np.mean(rho_cpu)
    rho_k = np.fft.fftn(delta)
    kx = np.fft.fftfreq(N, d=dx) * 2 * np.pi
    ky = np.fft.fftfreq(N, d=dx) * 2 * np.pi
    kz = np.fft.fftfreq(N, d=dx) * 2 * np.pi
    K2_vals = kx[:, None, None]**2 + ky[None, :, None]**2 + kz[None, None, :]**2
    K_vals = np.sqrt(K2_vals)
    k_min = 2 * np.pi / L
    k_max = min(2.0, 0.8 * np.pi / dx)
    k_bins = np.logspace(np.log10(k_min), np.log10(k_max), 25)
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
    P_k = np.zeros(len(k_centers))
    for i in range(len(k_bins)-1):
        mask = (K_vals >= k_bins[i]) & (K_vals < k_bins[i+1])
        if np.sum(mask) > 0:
            P_k[i] = np.mean(np.abs(rho_k[mask])**2)
        else:
            P_k[i] = np.nan
    valid = ~np.isnan(P_k)
    return k_centers[valid], P_k[valid], K_vals

# ---------- MAIN LOOP ----------
results = []
all_noise_fixed = {mu: [] for mu in mu_list}
all_noise_adaptive = {mu: [] for mu in mu_list}
all_spectra = {mu: {'k': None, 'Pk_list': []} for mu in mu_list}

seeds = [seed_base + i for i in range(n_seeds)]
print(f"Running scan: {len(mu_list)} masses × {n_seeds} realisations")
total_runs = len(mu_list) * n_seeds
run_idx = 0

for seed in tqdm(seeds, desc="Seeds"):
    print(f"\n=== Seed {seed} ===")
    psi_initial = create_initial_state(N, L0, seed, flattening, spin)

    for mu in mu_list:
        run_idx += 1
        dt = dt0 / mu
        actual_steps = int(steps * mu)
        print(f"  mu={mu:.2f} (m={mu*1e-22:.2e} eV)  run {run_idx}/{total_runs}")

        start_time = time.time()
        psi, rho, phi = run_simulation(psi_initial, mu, dt, actual_steps)
        elapsed = time.time() - start_time

        # Wave numbers
        kx = np.fft.fftfreq(N, d=dx) * 2 * np.pi
        ky = np.fft.fftfreq(N, d=dx) * 2 * np.pi
        kz = np.fft.fftfreq(N, d=dx) * 2 * np.pi
        K2_vals = kx[:, None, None]**2 + ky[None, :, None]**2 + kz[None, None, :]**2
        K_vals = np.sqrt(K2_vals)

        # ---- Noise ----
        # Fixed cut (k=0.5)
        noise_fixed = measure_noise(phi, K_vals, k_cut=0.5)
        all_noise_fixed[mu].append(noise_fixed)

        # Adaptive cut (diagnostic)
        k_nyquist = np.pi / dx
        k_db = min(0.5 * mu, 0.8 * k_nyquist)
        noise_adaptive = measure_noise(phi, K_vals, k_cut=k_db)
        all_noise_adaptive[mu].append(noise_adaptive)

        # ---- Power spectrum (store for all realisations) ----
        k_centers, P_k, _ = compute_power_spectrum(rho, L0, N, dx)
        if seed == seeds[0]:
            all_spectra[mu]['k'] = k_centers
        all_spectra[mu]['Pk_list'].append(P_k)

        # ---- Store results ----
        results.append({
            'seed': seed,
            'mu': mu,
            'mass_eV': mu * 1e-22,
            'noise_fixed': noise_fixed,
            'noise_adaptive': noise_adaptive,
            'k_db': k_db,
            'k_nyquist': k_nyquist,
            'phi_std': torch.std(phi).item(),
            'dx': dx,
            'L0': L0,
            'N': N,
            'elapsed': elapsed
        })

# ---------- SAVE CSV ----------
csv_file = os.path.join(output_dir, 'mass_scan_final.csv')
with open(csv_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    for r in results:
        writer.writerow(r)
print(f"\nCSV saved to {csv_file}")

# ---------- STATISTICS ----------
mu_vals = sorted(all_noise_fixed.keys())
means_fixed = [np.mean(all_noise_fixed[mu]) for mu in mu_vals]
stds_fixed = [np.std(all_noise_fixed[mu]) for mu in mu_vals]

means_adaptive = [np.mean(all_noise_adaptive[mu]) for mu in mu_vals]
stds_adaptive = [np.std(all_noise_adaptive[mu]) for mu in mu_vals]

# ---------- PLOTS ----------
# 1. Noise (fixed cut) with error bars
plt.figure(figsize=(8,6))
plt.errorbar(mu_vals, means_fixed, yerr=stds_fixed, fmt='bo-', capsize=4, label='Fixed k-cut (k=0.5)')
plt.axhline(y=0.02, color='r', linestyle='--', label='Fornax threshold (0.02)')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Mass $m$ (eV)')
plt.ylabel('Granular noise RMS')
plt.title('Granular noise vs FDM mass (with statistics)')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.savefig(os.path.join(output_dir, 'noise_vs_mass_final.png'), dpi=150)
plt.close()

# 2. Noise (adaptive) for diagnostics
plt.figure(figsize=(8,6))
plt.errorbar(mu_vals, means_adaptive, yerr=stds_adaptive, fmt='go-', capsize=4, label='Adaptive k-cut')
plt.axhline(y=0.02, color='r', linestyle='--', label='Fornax threshold (0.02)')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Mass $m$ (eV)')
plt.ylabel('Granular noise RMS (adaptive)')
plt.title('Granular noise with adaptive filter (diagnostic)')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.savefig(os.path.join(output_dir, 'noise_vs_mass_adaptive.png'), dpi=150)
plt.close()

# 3. Mean power spectra with variance
plt.figure(figsize=(8,6))
for mu in mu_vals:
    k = all_spectra[mu]['k']
    Pk_stack = np.array(all_spectra[mu]['Pk_list'])
    Pk_mean = np.mean(Pk_stack, axis=0)
    Pk_std = np.std(Pk_stack, axis=0)
    plt.errorbar(k, Pk_mean, yerr=Pk_std, label=f"m = {mu*1e-22:.0e} eV", capsize=2, elinewidth=0.5)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('k (1/Mpc)')
plt.ylabel('Power spectrum')
plt.title('Matter power spectra (mean ± std over realisations)')
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.savefig(os.path.join(output_dir, 'power_spectra_final.png'), dpi=150)
plt.close()

# 4. Individual realisations (scatter)
plt.figure(figsize=(8,6))
for mu in mu_vals:
    x = [mu] * len(all_noise_fixed[mu])
    y = all_noise_fixed[mu]
    plt.scatter(x, y, alpha=0.3, color='blue')
plt.errorbar(mu_vals, means_fixed, yerr=stds_fixed, fmt='ro-', capsize=4, label='Mean ± std')
plt.axhline(y=0.02, color='r', linestyle='--', label='Fornax threshold')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Mass $m$ (eV)')
plt.ylabel('Granular noise RMS (fixed k-cut)')
plt.title('Individual realisations and mean')
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.savefig(os.path.join(output_dir, 'noise_individual.png'), dpi=150)
plt.close()

print(f"\n✅ Scan complete. Results saved in {output_dir}")
print(f"   - mass_scan_final.csv       : all data")
print(f"   - noise_vs_mass_final.png   : main result with errors")
print(f"   - noise_individual.png      : scatter of individual runs")
print(f"   - power_spectra_final.png   : mean power spectra with variance")