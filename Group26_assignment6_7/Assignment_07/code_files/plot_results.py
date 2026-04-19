#!/usr/bin/env python3
"""
Plot speedup, execution time, efficiency, and phase breakdown
from benchmark CSV results (clock() CPU-time data).

Derived wall-time speedup: speedup(N) = N * T1_clock / TN_clock
Because clock() = wall_time * N for a fully parallel program,
so wall_time = clock_time / N, and speedup = T1_wall / TN_wall.

Usage: python3 plot_results.py results_lab.csv "Lab-PC"
"""

import sys
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

def read_csv(filename):
    data = {}
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cfg = row['config']
            t = int(row['threads'])
            if cfg not in data:
                data[cfg] = {}
            data[cfg][t] = {
                'interp':  float(row['interp_time']),
                'norm':    float(row['norm_time']),
                'mover':   float(row['mover_time']),
                'denorm':  float(row['denorm_time']),
                'total':   float(row['total_time']),
                'voids':   int(row['voids']),
            }
    return data

CONFIG_LABELS = {
    'a': '(a) 250×100, 0.9M pts',
    'b': '(b) 250×100, 5M pts',
    'c': '(c) 500×200, 3.6M pts',
    'd': '(d) 500×200, 20M pts',
    'e': '(e) 1000×400, 14M pts',
}

COLORS = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0']

def derived_wall_speedup(data, cfg, t):
    """Derived wall-time speedup: N * T1_clock / TN_clock"""
    threads = sorted(data[cfg].keys())
    t1 = data[cfg][1]['total'] if 1 in data[cfg] else data[cfg][min(threads)]['total']
    tn = data[cfg][t]['total']
    if tn == 0:
        return 0
    return t * t1 / tn

def derived_wall_time(data, cfg, t):
    """Derived wall time: TN_clock / N  (in seconds)"""
    return data[cfg][t]['total'] / t

def plot_wall_execution_time(data, platform, outdir):
    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, cfg in enumerate(sorted(data.keys())):
        threads = sorted(data[cfg].keys())
        wall_times = [derived_wall_time(data, cfg, t) for t in threads]
        ax.plot(threads, wall_times, 'o-', label=CONFIG_LABELS.get(cfg, cfg),
                linewidth=2, markersize=7, color=COLORS[idx])
    ax.set_xlabel('Number of Threads', fontsize=12)
    ax.set_ylabel('Derived Wall Time (s)', fontsize=12)
    ax.set_title(f'Wall-Time Execution vs Threads — {platform}\n'
                 f'(Derived: wall_time = clock_time / N)', fontsize=13)
    ax.set_xticks(sorted(set(t for cfg in data for t in data[cfg])))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'execution_time.png'), dpi=150)
    plt.close()

def plot_wall_speedup(data, platform, outdir):
    fig, ax = plt.subplots(figsize=(10, 6))
    all_threads = sorted(set(t for cfg in data for t in data[cfg]))
    ax.plot(all_threads, all_threads, 'k--', label='Ideal Linear', linewidth=1.5, alpha=0.6)
    for idx, cfg in enumerate(sorted(data.keys())):
        threads = sorted(data[cfg].keys())
        speedups = [derived_wall_speedup(data, cfg, t) for t in threads]
        ax.plot(threads, speedups, 'o-', label=CONFIG_LABELS.get(cfg, cfg),
                linewidth=2, markersize=7, color=COLORS[idx])
    ax.set_xlabel('Number of Threads', fontsize=12)
    ax.set_ylabel('Speedup', fontsize=12)
    ax.set_title(f'Speedup vs Threads — {platform}\n'
                 f'(Derived: speedup = N × T₁_clock / Tₙ_clock)', fontsize=13)
    ax.set_xticks(all_threads)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'speedup.png'), dpi=150)
    plt.close()

def plot_efficiency(data, platform, outdir):
    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, cfg in enumerate(sorted(data.keys())):
        threads = sorted(data[cfg].keys())
        efficiencies = [derived_wall_speedup(data, cfg, t) / t * 100 for t in threads]
        ax.plot(threads, efficiencies, 'o-', label=CONFIG_LABELS.get(cfg, cfg),
                linewidth=2, markersize=7, color=COLORS[idx])
    ax.axhline(y=100, color='k', linestyle='--', alpha=0.5, label='Ideal (100%)')
    ax.set_xlabel('Number of Threads', fontsize=12)
    ax.set_ylabel('Parallel Efficiency (%)', fontsize=12)
    ax.set_title(f'Parallel Efficiency vs Threads — {platform}', fontsize=13)
    ax.set_xticks(sorted(set(t for cfg in data for t in data[cfg])))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'efficiency.png'), dpi=150)
    plt.close()

def plot_phase_breakdown(data, platform, outdir):
    """Stacked bar: derived wall-time per phase at each thread count for each config."""
    configs = sorted(data.keys())
    all_threads = sorted(set(t for cfg in data for t in data[cfg]))

    fig, axes = plt.subplots(1, len(all_threads), figsize=(16, 6), sharey=False)
    if len(all_threads) == 1:
        axes = [axes]

    for ax_idx, t in enumerate(all_threads):
        ax = axes[ax_idx]
        x = np.arange(len(configs))
        interp  = [data[c][t]['interp']  / t for c in configs if t in data[c]]
        norm    = [data[c][t]['norm']    / t for c in configs if t in data[c]]
        mover   = [data[c][t]['mover']   / t for c in configs if t in data[c]]
        denorm  = [data[c][t]['denorm']  / t for c in configs if t in data[c]]
        cfgs    = [c for c in configs if t in data[c]]
        xi      = np.arange(len(cfgs))

        ax.bar(xi, interp, label='Interpolation', color='#2196F3')
        ax.bar(xi, norm,   bottom=interp, label='Normalization', color='#4CAF50')
        b2 = [i+n for i,n in zip(interp, norm)]
        ax.bar(xi, mover,  bottom=b2, label='Mover', color='#FF9800')
        b3 = [b+m for b,m in zip(b2, mover)]
        ax.bar(xi, denorm, bottom=b3, label='Denorm', color='#F44336')
        ax.set_title(f'{t} Thread{"s" if t>1 else ""}', fontsize=11)
        ax.set_xticks(xi)
        ax.set_xticklabels([c.upper() for c in cfgs])
        ax.set_xlabel('Config')
        if ax_idx == 0:
            ax.set_ylabel('Wall Time (s)')
        if ax_idx == len(all_threads) - 1:
            ax.legend(fontsize=8, loc='upper right')

    fig.suptitle(f'Phase Breakdown (Derived Wall Time) — {platform}', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'phase_breakdown.png'), dpi=150)
    plt.close()

def plot_interp_vs_mover(data, platform, outdir):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for idx, cfg in enumerate(sorted(data.keys())):
        ax = axes[idx]
        threads = sorted(data[cfg].keys())
        interp = [data[cfg][t]['interp'] / t for t in threads]
        mover  = [data[cfg][t]['mover']  / t for t in threads]
        ax.plot(threads, interp, 'o-', label='Interpolation', linewidth=2, color='#2196F3')
        ax.plot(threads, mover,  's-', label='Mover',         linewidth=2, color='#FF9800')
        ax.set_title(CONFIG_LABELS.get(cfg, cfg), fontsize=11)
        ax.set_xlabel('Threads')
        ax.set_ylabel('Derived Wall Time (s)')
        ax.set_xticks(threads)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    if len(sorted(data.keys())) < 6:
        axes[-1].set_visible(False)
    fig.suptitle(f'Interpolation vs Mover Phase — {platform}', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'interp_vs_mover.png'), dpi=150)
    plt.close()

def print_summary_table(data, platform):
    print(f"\n{'='*90}")
    print(f"  Performance Summary — {platform}  (Derived Wall-Time Speedup = N × T1 / TN)")
    print(f"{'='*90}")
    for cfg in sorted(data.keys()):
        threads = sorted(data[cfg].keys())
        t1_clock = data[cfg][1]['total'] if 1 in data[cfg] else data[cfg][min(threads)]['total']
        t1_wall  = t1_clock  # single thread: wall == clock
        print(f"\nConfig {CONFIG_LABELS.get(cfg, cfg)}:")
        print(f"  {'Threads':>8} {'Clock(s)':>10} {'WallTime(s)':>12} {'Speedup':>10} {'Efficiency':>12} {'Voids':>8}")
        print(f"  {'-'*62}")
        for t in threads:
            clock  = data[cfg][t]['total']
            wall   = clock / t
            sp     = derived_wall_speedup(data, cfg, t)
            eff    = sp / t * 100
            voids  = data[cfg][t]['voids']
            print(f"  {t:>8} {clock:>10.4f} {wall:>12.4f} {sp:>10.3f}x {eff:>10.1f}% {voids:>8}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 plot_results.py <results.csv> [platform_name]")
        sys.exit(1)

    csv_file = sys.argv[1]
    platform = sys.argv[2] if len(sys.argv) > 2 else "Unknown"
    data     = read_csv(csv_file)
    outdir   = os.path.dirname(os.path.abspath(csv_file))

    print_summary_table(data, platform)
    plot_wall_execution_time(data, platform, outdir)
    plot_wall_speedup(data, platform, outdir)
    plot_efficiency(data, platform, outdir)
    plot_phase_breakdown(data, platform, outdir)
    plot_interp_vs_mover(data, platform, outdir)

    print(f"\nPlots saved to {outdir}/")
    for f in ['execution_time.png','speedup.png','efficiency.png',
              'phase_breakdown.png','interp_vs_mover.png']:
        print(f"  - {f}")
