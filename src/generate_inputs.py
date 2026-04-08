import numpy as np
from pathlib import Path
from tqdm import tqdm


def generate_day_current(
    N: int = 10,
    T: int = 24,
    I0: float = 1.5,
    mean_reversion: float = 0.3,
    noise_std: float = 0.2,
    sparse_frac: float = 0.5,
    seed: int = 42,
):
    """
    Generate time-varying input current for one simulated day.

    The current drifts slowly (Ornstein-Uhlenbeck) around a fixed base pattern I_base.
    This represents naturalistic stimulation: the animal experiences stimuli that vary
    throughout the day but cluster around a central memory pattern.

    Returns
    -------
    I_m    : (T, N)  float32 — current at each time bin for each neuron
    I_base : (N,)   float32 — attractor of the OU process (the 'canonical' memory pattern)
    """
    rng = np.random.default_rng(seed)

    I_base = np.zeros(N, dtype=np.float32)
    n_active = max(1, int(N * sparse_frac))
    active_idx = rng.choice(N, n_active, replace=False)
    I_base[active_idx] = rng.uniform(0.5 * I0, I0, n_active).astype(np.float32)

    I_m = np.zeros((T, N), dtype=np.float32)
    I_m[0] = I_base.copy()

    for t in tqdm(range(1, T), desc="Generating OU current"):
        drift = mean_reversion * (I_base - I_m[t - 1])
        noise = rng.normal(0, noise_std, N).astype(np.float32)
        I_m[t] = np.clip(I_m[t - 1] + drift + noise, 0.0, I0)

    return I_m, I_base


def main():
    out_dir = Path(__file__).resolve().parent.parent / "data"
    out_dir.mkdir(exist_ok=True)

    N, T = 10, 24
    I_m, I_base = generate_day_current(N=N, T=T)

    np.save(out_dir / "I_m.npy",    I_m)
    np.save(out_dir / "I_base.npy", I_base)

    print(f"I_m    shape : {I_m.shape}  — (T={T} hourly bins × N={N} neurons)")
    print(f"I_base shape : {I_base.shape}")
    print(f"Range        : [{I_m.min():.3f}, {I_m.max():.3f}]")
    print(f"Saved to     : {out_dir}")


if __name__ == "__main__":
    main()
