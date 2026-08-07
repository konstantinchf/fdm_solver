#!/usr/bin/env python3
"""
CLEAN MODEL MASS SCAN FOR FDM
- Correct initial field generation (IFFT, P(ψ) ∝ k⁻⁴)
- Independent realisations (new field for each seed)
- Same initial field for all masses within a seed
- Fixed k-cut for noise measurement (k=0.5)
- Multiple realisations (5 seeds) for statistics
- Spin=0, Flattening=1 (no artefacts)
- Midpoint evaluation of a(t), H(t)
- Correct damping factor (half per half-step)
- Norm conservation check
- Delta_std diagnostic
"""
import torch
import torch.fft
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import time

# Попытка импорта tqdm с fallback
try:
    from tqdm import tqdm
except ImportError:
    # Простая замена, если tqdm не установлен
    class tqdm:
        def __init__(self, iterable, desc=""):
            self.iterable = iterable
            self.desc = desc
            self.total = len(iterable)
        def __iter__(self):
            print(f"{self.desc}: 0/{self.total}")
            for i, item in enumerate(self.iterable):
                yield item
                if (i+1) % max(1, self.total//5) == 0 or (i+1) == self.total:
                    print(f"{self.desc}: {i+1}/{self.total}")
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

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

mu_list = np.logspace(0, 2, 9)   # 1, 1.78, ..., 100
n_seeds = 5
seed_base = 42

output_dir = './clean_mass_scan'
os.makedirs(output_dir, exist_ok=True)

# ---------- ФУНКЦИИ ----------
def create_initial_state(N, L, seed, flattening=1.0, spin=0.0):
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

def run_simulation(psi_initial, mu):
    """Evolve field for a given mass. Uses midpoint for a(t), H(t)."""
    dt = dt0 / mu
    actual_steps = int(steps * mu)
    psi = psi_initial.clone()

    kx = torch.fft.fftfreq(N, d=dx, device=device) * 2 * torch.pi
    ky = torch.fft.fftfreq(N, d=dx, device=device) * 2 * torch.pi
    kz = torch.fft.fftfreq(N, d=dx, device=device) * 2 * torch.pi
    K2 = kx[:, None, None]**2 + ky[None, :, None]**2 + kz[None, None, :]**2
    K2[0,0,0] = 1.0

    for step in range(actual_steps):
        t_mid = t0 + (step + 0.5) * dt
        a = (t_mid / t0) ** (2.0 / 3.0)
        H = (2.0 / 3.0) / t_mid

        rho = torch.abs(psi)**2
        phi = compute_potential(rho, G, scale_rho, K2, a)
        V = mu * phi + g_nl * rho * mu

        # Правильное затухание за полушаг
        damp_half = torch.exp(torch.tensor(-0.375 * H * dt, device=device))

        # Strang: V/2 -> K -> V/2
        psi = psi * torch.exp(-0.5j * dt * V) * damp_half
        kinetic = torch.exp(-1j * dt * K2 / (2 * mu * a**2))
        psi_k = torch.fft.fftn(psi)
        psi_k = psi_k * kinetic
        psi = torch.fft.ifftn(psi_k)
        rho = torch.abs(psi)**2
        phi = compute_potential(rho, G, scale_rho, K2, a)
        V = mu * phi + g_nl * rho * mu
        psi = psi * torch.exp(-0.5j * dt * V) * damp_half

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

def compute_power_spectrum(rho, L, N, dx):
    rho_cpu = rho.cpu().numpy()
    rho_mean = np.mean(rho_cpu)
    delta = (rho_cpu - rho_mean) / rho_mean
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
    norm_factor = (L**3) / (N**6)
    power_3d = norm_factor * np.abs(rho_k)**2
    for i in range(len(k_bins)-1):
        mask = (K_vals >= k_bins[i]) & (K_vals < k_bins[i+1])
        if np.sum(mask) > 0:
            P_k[i] = np.mean(power_3d[mask])
        else:
            P_k[i] = np.nan
    valid = ~np.isnan(P_k)
    return k_centers[valid], P_k[valid], K_vals

# ---------- ОСНОВНОЙ ЦИКЛ ----------
results = []
all_noise_fixed = {mu: [] for mu in mu_list}
all_spectra = {mu: {'k': None, 'Pk_list': []} for mu in mu_list}

seeds = [seed_base + i for i in range(n_seeds)]
print(f"Running: {len(mu_list)} masses × {n_seeds} seeds")
total_runs = len(mu_list) * n_seeds
run_idx = 0

for seed in tqdm(seeds, desc="Seeds"):
    print(f"\n=== Seed {seed} ===")
    psi_initial = create_initial_state(N, L0, seed, flattening, spin)
    norm_initial = torch.sum(torch.abs(psi_initial)**2).item()

    for mu in mu_list:
        run_idx += 1
        print(f"  mu={mu:.2f} (m={mu*1e-22:.2e} eV)  run {run_idx}/{total_runs}")

        start_time = time.time()
        psi, rho, phi = run_simulation(psi_initial, mu)
        elapsed = time.time() - start_time

        norm_final = torch.sum(torch.abs(psi)**2).item()
        norm_ratio = norm_final / norm_initial if norm_initial > 0 else 1.0

        rho_mean = torch.mean(rho).item()
        delta_std = (torch.std(rho) / rho_mean).item()

        kx = np.fft.fftfreq(N, d=dx) * 2 * np.pi
        ky = np.fft.fftfreq(N, d=dx) * 2 * np.pi
        kz = np.fft.fftfreq(N, d=dx) * 2 * np.pi
        K2_vals = kx[:, None, None]**2 + ky[None, :, None]**2 + kz[None, None, :]**2
        K_vals = np.sqrt(K2_vals)

        noise_fixed = measure_noise(phi, K_vals, k_cut=0.5)
        all_noise_fixed[mu].append(noise_fixed)

        k_centers, P_k, _ = compute_power_spectrum(rho, L0, N, dx)
        if seed == seeds[0]:
            all_spectra[mu]['k'] = k_centers
        all_spectra[mu]['Pk_list'].append(P_k)

        print(f"    noise={noise_fixed:.5f}, norm={norm_ratio:.6f}, delta_std={delta_std:.4f}")

        results.append({
            'seed': seed,
            'mu': mu,
            'mass_eV': mu * 1e-22,
            'noise_fixed': noise_fixed,
            'phi_std': torch.std(phi).item(),
            'delta_std': delta_std,
            'norm_ratio': norm_ratio,
            'dx': dx,
            'L0': L0,
            'N': N,
            'elapsed': elapsed
        })

# ---------- СОХРАНЕНИЕ ----------
csv_file = os.path.join(output_dir, 'clean_mass_scan.csv')
with open(csv_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    for r in results:
        writer.writerow(r)
print(f"\nCSV saved to {csv_file}")

# ---------- СТАТИСТИКА ----------
mu_vals = sorted(all_noise_fixed.keys())
mass_vals = np.array(mu_vals) * 1e-22
means_fixed = [np.mean(all_noise_fixed[mu]) for mu in mu_vals]
stds_fixed = [np.std(all_noise_fixed[mu]) for mu in mu_vals]

# ---------- ГРАФИКИ ----------
# 1. Noise (fixed cut) with error bars
plt.figure(figsize=(8,6))
plt.errorbar(mass_vals, means_fixed, yerr=stds_fixed, fmt='bo-', capsize=4,
             label='Fixed k-cut (k=0.5)')
plt.axhline(y=0.02, color='r', linestyle='--', alpha=0.7,
            label='Illustrative threshold (not calibrated)')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\mu = m_{\rm FDM} / 10^{-22}\,\mathrm{eV}$')
plt.ylabel('Granular noise RMS (fixed k-cut)')
plt.title('Clean model mass scan (spin=0, flattening=1)')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.savefig(os.path.join(output_dir, 'noise_vs_mass_clean.png'), dpi=150)
plt.close()

# 2. Individual realisations
plt.figure(figsize=(8,6))
for mu in mu_vals:
    x = [mu] * len(all_noise_fixed[mu])
    y = all_noise_fixed[mu]
    plt.scatter(x, y, alpha=0.3, color='blue')
plt.errorbar(mu_vals, means_fixed, yerr=stds_fixed, fmt='ro-', capsize=4,
             label='Mean ± std')
plt.axhline(y=0.02, color='r', linestyle='--', alpha=0.7,
            label='Illustrative threshold')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\mu = m_{\rm FDM} / 10^{-22}\,\mathrm{eV}$')
plt.ylabel('Granular noise RMS (fixed k-cut)')
plt.title('Individual realisations and mean')
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.savefig(os.path.join(output_dir, 'noise_individual_clean.png'), dpi=150)
plt.close()

# 3. Power spectra with variance
plt.figure(figsize=(8,6))
for mu in mu_vals:
    k = all_spectra[mu]['k']
    Pk_stack = np.array(all_spectra[mu]['Pk_list'])
    Pk_mean = np.mean(Pk_stack, axis=0)
    Pk_std = np.std(Pk_stack, axis=0)
    plt.errorbar(k, Pk_mean, yerr=Pk_std,
                 label=rf"$\mu$={mu:.1f}", capsize=2, elinewidth=0.5)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$k$ [kpc$^{-1}$]')
plt.ylabel(r'$P(k)$ [kpc$^3$]')
plt.title('Matter power spectra (mean ± std)')
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.savefig(os.path.join(output_dir, 'power_spectra_clean.png'), dpi=150)
plt.close()

print(f"\n✅ Clean mass scan complete. Results saved in {output_dir}")