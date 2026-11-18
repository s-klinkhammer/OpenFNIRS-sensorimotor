"""Scatter plot of decoding accuracy per participant, SD (x) vs. ID (y).

Points above the diagonal: ID better. Below: SD better. Point area scales with
the number of participants sharing an identical (SD, ID) value. Points with an
uncorrected paired p < .05 have a black outline.

Required CSV columns:
    acc_sd, acc_id, paired_p_id_vs_sd_sig_raw
Optional:
    participant, diff_id_minus_sd (used for a consistency check)

The figure is created at its final size (FIG_W_CM x FIG_H_CM), so do not
rescale it in the manuscript layout.

Usage:
    python scatter_sd_vs_id.py [path/to/accuracy_stats.csv]
(without an argument a file dialog opens)
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

FIG_W_CM, FIG_H_CM = 12.5, 8.0
S_UNIT = 22                 # marker area (pt^2) per participant
TICK_STEP = 0.2
SHOW_COUNTS_IN_PLOT = True  # print ID > SD / SD > ID counts inside the plot

C_ID, C_SD = '#486F69', '#5E6B82'
C_EQ = '#C4C4C4'           # no change
C_SIZE = '#9A9A9A'         # size legend

FS_TICK, FS_LABEL, FS_REGION, FS_LEGEND = 8, 9, 8, 8
LEGEND_MARKER = 6

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans'],
    'font.size': FS_TICK,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
})

if len(sys.argv) > 1:
    stats_file = sys.argv[1]
else:
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    stats_file = filedialog.askopenfilename(
        title="Select accuracy stats CSV",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )
    root.destroy()

if not stats_file:
    raise SystemExit("No file selected.")

output_dir = os.path.dirname(os.path.abspath(stats_file))
df = pd.read_csv(stats_file)
print("File:", stats_file)

# Round to avoid floating-point errors when testing for equality
df['sd'] = df['acc_sd'].round(3)
df['id'] = df['acc_id'].round(3)
d = np.round((df['id'] - df['sd']) * 1000).astype(int)
df['cat'] = np.where(d > 0, 'higher', np.where(d < 0, 'lower', 'equal'))
df['sig'] = df['paired_p_id_vs_sd_sig_raw'].astype(bool)  # uncorrected

n = len(df)
n_hi, n_lo, n_eq = [(df['cat'] == k).sum() for k in ('higher', 'lower', 'equal')]

if 'diff_id_minus_sd' in df.columns:
    dc = np.round(df['diff_id_minus_sd'] * 1000).astype(int)
    bad = df[dc != d]
    if len(bad):
        print("WARNING: diff_id_minus_sd does not match acc_id - acc_sd for:")
        print(bad[[c for c in ('participant', 'acc_sd', 'acc_id', 'diff_id_minus_sd') if c in df.columns]])

# Merge identical points
g = df.groupby(['sd', 'id', 'cat', 'sig']).size().reset_index(name='n')
COL = {'higher': C_ID, 'lower': C_SD, 'equal': C_EQ}

# Square axes, legend to the right
W, H = FIG_W_CM / 2.54, FIG_H_CM / 2.54
left_in, bottom_in, top_in = 0.60, 0.50, 0.10
side = H - bottom_in - top_in
fig = plt.figure(figsize=(W, H), facecolor='white')
ax = fig.add_axes([left_in / W, bottom_in / H, side / W, side / H])

lo = np.floor((min(df['sd'].min(), df['id'].min()) - 0.03) / 0.05) * 0.05
hi = 1.03
ax.fill_between([lo, hi], [lo, hi], hi, color=C_ID, alpha=0.08, zorder=0, lw=0)
ax.fill_between([lo, hi], lo, [lo, hi], color=C_SD, alpha=0.08, zorder=0, lw=0)
ax.plot([lo, hi], [lo, hi], color='#444444', linestyle='--', linewidth=0.8, zorder=1)

for sig_flag in (False, True):  # significant points on top
    sub = g[g['sig'] == sig_flag]
    ax.scatter(sub['sd'], sub['id'], s=sub['n'] * S_UNIT,
               c=sub['cat'].map(COL), alpha=0.95,
               edgecolor='black' if sig_flag else 'white',
               linewidth=1.0 if sig_flag else 0.4,
               zorder=3 if sig_flag else 2)

ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.set_aspect('equal')
ticks = np.arange(np.ceil(round(lo / TICK_STEP, 6)) * TICK_STEP, 1.001, TICK_STEP)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
fmt = FuncFormatter(lambda v, _: f"{v*100:.0f}%")
ax.xaxis.set_major_formatter(fmt)
ax.yaxis.set_major_formatter(fmt)
ax.set_xlabel("Accuracy SD", fontsize=FS_LABEL, labelpad=3)
ax.set_ylabel("Accuracy ID", fontsize=FS_LABEL, labelpad=3)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)

if SHOW_COUNTS_IN_PLOT:
    ax.text(0.03, 0.97, f"ID > SD: n = {n_hi} ({100*n_hi/n:.1f}%)", transform=ax.transAxes,
            color=C_ID, fontsize=FS_REGION, va='top')
    ax.text(0.97, 0.03, f"SD > ID: n = {n_lo} ({100*n_lo/n:.1f}%)", transform=ax.transAxes,
            color=C_SD, fontsize=FS_REGION, ha='right', va='bottom')

leg_x = (left_in + side + 0.25) / W
common = dict(frameon=False, fontsize=FS_LEGEND, handlelength=1.0, handletextpad=0.5,
              labelspacing=0.7, borderaxespad=0, borderpad=0, alignment='left',
              title_fontsize=FS_LEGEND)

h_status = [
    Line2D([], [], marker='o', ls='', markerfacecolor='white', markeredgecolor='black',
           markeredgewidth=1.0, markersize=LEGEND_MARKER, label='p < .05 (uncorrected)'),
    Line2D([], [], marker='o', ls='', color=C_EQ, markersize=LEGEND_MARKER,
           label=f'No change (n = {n_eq})'),
]
h_size = [
    Line2D([], [], marker='o', ls='', color=C_SIZE,
           markersize=np.sqrt(k * S_UNIT), label=str(k)) for k in (1, 2, 4)
]

leg1 = fig.legend(handles=h_status, loc='lower left', bbox_to_anchor=(leg_x, 0.52), **common)
fig.add_artist(leg1)
fig.legend(handles=h_size, loc='upper left', title='Participants per point',
           bbox_to_anchor=(leg_x, 0.46), **common)

# No bbox_inches='tight', to keep the exact size in cm
base = os.path.join(output_dir, "accuracy_scatter_sd_vs_id")
fig.savefig(base + ".png", dpi=600)
fig.savefig(base + ".svg")
fig.savefig(base + ".pdf")
plt.close(fig)

print(f"Saved ({FIG_W_CM} x {FIG_H_CM} cm): {base}.png / .svg / .pdf")
print(f"ID>SD: {n_hi}, SD>ID: {n_lo}, equal: {n_eq}, uncorrected significant: {int(df['sig'].sum())}")
