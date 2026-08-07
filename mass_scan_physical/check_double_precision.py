#!/usr/bin/env python3
"""
Quick check: double precision (complex128) for μ = 1, 10, 100.
Fixed:
- K2 and K2_poisson separated (no artificial phase for k=0)
- Initial field generated once, cloned for each mu
- rho and phi recomputed after final half-step
- Exact norm expectation computed
"""
import torch
import torch.fft
import numpy as np
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
seed = 42

# ---------- ФУНКЦИИ ----------
def create_initial_state_double(N, L, seed, flattening=1.0, spin=0.0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Все тензоры — float64
    kx = torch.fft.fftfreq(N, d=L/N, device=device, dtype=torch.float64) * 2 * torch.pi
    ky = torch.fft.fftfreq(N, d=L/N, device=device, dtype=torch.float64) * 2 * torch.pi
    kz = torch.fft.fftfreq(N, d=L/N, device=device, dtype=torch.float64) * 2 * torch.pi

    # Правильный K2 для амплитуды (с защитой от деления на ноль)
    K2_amp = kx[:, None, None]**2 + ky[None, :, None]**2 + (flattening * kz[None, None, :])**2
    K2_amp[0,0,0] = 1.0
    amp = torch.sqrt(K2_amp ** (-2))

    # Шум в double precision
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

    # Волновые числа
    kx = torch.fft.fftfreq(N, d=dx, device=device, dtype=torch.float64) * 2 * torch.pi
    ky = torch.fft.fftfreq(N, d=dx, device=device, dtype=torch.float64) * 2 * torch.pi
    kz = torch.fft.fftfreq(N, d=dx, device=device, dtype=torch.float64) * 2 * torch.pi

    # Два массива: один для Пуассона (с защитой), другой для кинетики (честный)
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

        # Strang: V/2
        psi = psi * torch.exp(-0.5j * dt * V) * damp_half

        # K (честный K2, без подмены нулевой моды)
        kinetic = torch.exp(-1j * dt * K2 / (2 * mu * a**2))
        psi_k = torch.fft.fftn(psi)
        psi_k = psi_k * kinetic
        psi = torch.fft.ifftn(psi_k)

        # V/2
        rho = torch.abs(psi)**2
        phi = compute_potential(rho, G, scale_rho, K2_poisson, a)
        V = mu * phi + g_nl * rho * mu
        psi = psi * torch.exp(-0.5j * dt * V) * damp_half

    # Пересчёт финальных rho и phi (после последнего полушага)
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

# ---------- ОСНОВНОЙ ЗАПУСК ----------
print("\n=== Double precision check (complex128) ===\n")

# Создаём начальное поле ОДИН РАЗ
psi_initial = create_initial_state_double(N, L0, seed, flattening, spin)
norm_initial = torch.sum(torch.abs(psi_initial)**2).item()

for mu in mu_list:
    print(f"μ = {mu:.1f}  (m = {mu*1e-22:.2e} eV)")
    start = time.time()

    psi, rho, phi = run_simulation(psi_initial, mu)

    norm_final = torch.sum(torch.abs(psi)**2).item()
    norm_ratio = norm_final / norm_initial if norm_initial > 0 else 1.0

    # Волновые числа для шума
    kx = np.fft.fftfreq(N, d=dx) * 2 * np.pi
    ky = np.fft.fftfreq(N, d=dx) * 2 * np.pi
    kz = np.fft.fftfreq(N, d=dx) * 2 * np.pi
    K2_vals = kx[:, None, None]**2 + ky[None, :, None]**2 + kz[None, None, :]**2
    K_vals = np.sqrt(K2_vals)

    noise_fixed = measure_noise(phi, K_vals, k_cut=0.5)

    # Точное ожидание нормы
    t_final = t0 + steps * dt0
    norm_expected = t0 / t_final  # = 1/3
    norm_relative = norm_ratio / norm_expected
    norm_error = norm_relative - 1.0

    print(f"  norm_ratio    = {norm_ratio:.10f}")
    print(f"  expected      = {norm_expected:.10f}")
    print(f"  ratio/theory  = {norm_relative:.10f}")
    print(f"  relative err  = {norm_error:+.3e}")
    print(f"  noise_fixed   = {noise_fixed:.6f}")
    print(f"  time          = {time.time()-start:.1f}s\n")

print("✅ Check complete.")