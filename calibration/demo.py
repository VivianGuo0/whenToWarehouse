"""
demo.py — three usage patterns for the daily execution engine.

Run:  python demo.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from daily_engine import MultiDayEngine, DailyStrategy, DailySimulator, DayParams


# ══════════════════════════════════════════════════════════
# Pattern 1: Pre-market only — get strategy coefficients
#            No simulation needed. Runs in < 1ms.
# ══════════════════════════════════════════════════════════

def demo_premarket():
    print("=" * 55)
    print("Pattern 1: Pre-market — compute today's strategy")
    print("=" * 55)

    params = DayParams(beta=8.0, lam=0.2, eps=1e-2, theta=-1.0,
                       sigma=0.1, z=0.1, y=0.0)
    strategy = DailyStrategy(params)

    s = strategy.summary()
    print(f"  Opening block J0  = {s['J0']:+.4f}  ({s['J0']*100:+.2f}% ADV)")
    print(f"  f[0]  = {s['f0']:.4f}   (inventory coefficient, < 0)")
    print(f"  g[0]  = {s['g0']:.4f}   (impact coefficient,   < 0)")
    print(f"  h[0]  = {s['h0']:.4f}   (in-flow coefficient,  sign ~ θ)")
    print(f"  κ     = {s['kappa']:.4f}")

    # Show how q reacts at a hypothetical state midday
    X_mid, Y_mid, Z_mid = -0.05, 0.003, 0.08   # example midday state
    step_mid = params.N // 2
    q_mid = strategy.optimal_q(step_mid, X_mid, Y_mid, Z_mid)
    print(f"\n  Example midday state: X={X_mid}, Y={Y_mid}, Z={Z_mid}")
    print(f"  → optimal q = {q_mid:.4f}  ({q_mid*100:.2f}% ADV/day)")
    print()


# ══════════════════════════════════════════════════════════
# Pattern 2: Pre-market MC — cost quote for client
#            ~500 paths, takes ~0.1s
# ══════════════════════════════════════════════════════════

def demo_cost_quote():
    print("=" * 55)
    print("Pattern 2: Pre-market MC — cost quote")
    print("=" * 55)

    params   = DayParams(beta=8.0, lam=0.2, eps=1e-2, theta=0.0,
                         sigma=0.1, z=0.1, y=0.0)
    strategy = DailyStrategy(params)
    sim      = DailySimulator(strategy)

    quote = sim.cost_quote(n_samples=500)
    print("  Client quote (martingale in-flow, θ=0):")
    for k, v in quote.items():
        print(f"    {k:35s} = {v}")

    # Compare: momentum flow
    params_mom = DayParams(beta=8.0, lam=0.2, eps=1e-2, theta=-1.0,
                           sigma=0.1, z=0.1, y=0.0)
    quote_mom = DailySimulator(DailyStrategy(params_mom)).cost_quote(n_samples=500)
    print("\n  Client quote (momentum in-flow, θ=-1):")
    for k, v in quote_mom.items():
        print(f"    {k:35s} = {v}")
    print()


# ══════════════════════════════════════════════════════════
# Pattern 3: Multi-day back-test
#            Simulates N days, carries overnight impact state
# ══════════════════════════════════════════════════════════

def demo_multi_day(n_days: int = 10):
    print("=" * 55)
    print(f"Pattern 3: Multi-day back-test ({n_days} days)")
    print("=" * 55)

    # Simulate varying daily parameters (in practice: from calibration)
    np.random.seed(42)
    daily_params = [
        {
            "theta": np.random.choice([-1.0, 0.0, 1.0]),
            "sigma": 0.08 + 0.04 * np.random.rand(),
            "z":     0.08 + 0.04 * np.random.rand(),
        }
        for _ in range(n_days)
    ]

    engine = MultiDayEngine()

    for day, params in enumerate(daily_params):
        record = engine.run_simulated_day(params)
        y_carry = engine._y_carry
        print(
            f"  Day {day+1:2d}  θ={params['theta']:+.0f}"
            f"  z={params['z']:.3f}"
            f"  J0={record.J0:+.4f}"
            f"  JT={record.JT:+.4f}"
            f"  intern={record.internalization:.1f}%"
            f"  cost={( record.spread_cost + record.impact_cost) / record.tv_in * 1e4:.1f}bps"
            f"  y_carry→={y_carry*1e4:.1f}bps"
        )

    print()
    df = engine.history_df()
    print("  Aggregate across all days:")
    print(f"    Mean internalization : {df['internalization_%'].mean():.1f}%")
    print(f"    Mean cost per in-flow: {df['total_cost_per_in_bps'].mean():.1f} bps")
    print(f"    Mean |J0| (ADV%)     : {df['J0'].abs().mean()*100:.2f}%")
    print(f"    Mean |JT| (ADV%)     : {df['JT'].abs().mean()*100:.2f}%")

    _plot_multi_day(df)
    print()


def _plot_multi_day(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3))

    axes[0].bar(df["day"], df["internalization_%"], color="steelblue", alpha=0.8)
    axes[0].axhline(df["internalization_%"].mean(), color="red", linestyle="--", label="mean")
    axes[0].set_title("Internalization (%) per day")
    axes[0].set_xlabel("Day"); axes[0].legend()

    axes[1].bar(df["day"], df["total_cost_per_in_bps"], color="darkorange", alpha=0.8)
    axes[1].axhline(df["total_cost_per_in_bps"].mean(), color="red", linestyle="--", label="mean")
    axes[1].set_title("Total cost per in-flow (bps)")
    axes[1].set_xlabel("Day"); axes[1].legend()

    axes[2].plot(df["day"], df["y0"] * 1e4, marker="o", label="y₀ (impact carry-in)")
    axes[2].plot(df["day"], df["Y_terminal"] * 1e4, marker="s", linestyle="--",
                 label="Y_T (end of day)")
    axes[2].axhline(0, color="black", linewidth=0.5)
    axes[2].set_title("Impact state: carry-in vs terminal (bps)")
    axes[2].set_xlabel("Day"); axes[2].legend()

    plt.tight_layout()
    plt.savefig("multi_day_summary.png", dpi=120, bbox_inches="tight")
    print("  → saved multi_day_summary.png")
    plt.close()


# ══════════════════════════════════════════════════════════
# Pattern 4: Live-mode skeleton (no randomness — real dZ fed in)
# ══════════════════════════════════════════════════════════

def demo_live_mode():
    print("=" * 55)
    print("Pattern 4: Live-mode skeleton")
    print("=" * 55)

    engine = MultiDayEngine()

    # --- Pre-market ---
    today_params = {"theta": 0.0, "sigma": 0.1, "z": 0.1}
    J0 = engine.start_day(today_params)
    print(f"  Opening block J0 = {J0:+.4f}  ({J0*100:+.2f}% ADV)")

    # Pre-market cost quote
    quote = engine.cost_quote(today_params, n_samples=300)
    print(f"  Pre-market quote: expected cost = {quote['expected_cost_bps']} bps, "
          f"internalization ≈ {quote['expected_internalization_pct']}%")

    # --- Intraday: feed real observed dZ increments ---
    # In production this comes from your order management system.
    # Here we generate synthetic values as a placeholder.
    N = 200
    np.random.seed(0)
    fake_dZ = np.random.randn(N) * 0.002   # pretend these are real observations

    for step, dZ in enumerate(fake_dZ):
        q = engine.step(step, dZ)
        # → send q to market here

    # --- Close ---
    record = engine.end_day()
    print(f"  Closing block JT = {record.JT:+.4f}  ({record.JT*100:+.2f}% ADV)")
    print(f"  Realised internalization: {record.internalization:.1f}%")
    print(f"  Realised cost/in-flow:    "
          f"{(record.spread_cost+record.impact_cost)/record.tv_in*1e4:.1f} bps")
    print(f"  Impact carry to tomorrow: {engine._y_carry*1e4:.2f} bps")


# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    demo_premarket()
    demo_cost_quote()
    demo_multi_day(n_days=10)
    demo_live_mode()
