"""
05_quality_rejected_channels.py
Reads the Satori "rejectedChannels_SCI-*.txt" files (one per participant and run,
lines of the form "CH: <number> SCI: <value>") from one ID and one SD folder and
builds rejectedChannels_ID_<timestamp>.xlsx / rejectedChannels_SD_<timestamp>.xlsx
(columns: Participant, RunA_count, RunA_channels, RunB_count, RunB_channels,
high_quality), the input of 06_quality_channel_labels.py.

Also writes
  - high_quality_participants_<COND>_<timestamp>.txt (no rejected channel in either run)
  - QC histograms (most frequently rejected channels, average per participant)

A channel counts as rejected if SCI < SCI_CUTOFF. The txt files list ALL channels
with their SCI value, so the filtering is done here.

Usage: python 05_quality_rejected_channels.py --id-dir <folder> --sd-dir <folder>
Output goes to config.RESULTS_QUALITY_DIR.
"""

import os
import re
from datetime import datetime
from collections import Counter
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config

# Scalp coupling index (SCI) cutoff: a channel is rejected if SCI < SCI_CUTOFF.
# Set this to the cutoff used in the analysis.
SCI_CUTOFF = 0.75

CHANNEL_TOTALS = {"ID": 20, "SD": 13}  # must match ID_TOTAL / SD_TOTAL in 06_quality_channel_labels.py

parser = argparse.ArgumentParser(description="Satori SCI rejections -> rejectedChannels_ID/SD.xlsx")
parser.add_argument("--id-dir", required=True, help="Folder with the ID rejectedChannels_SCI txt files")
parser.add_argument("--sd-dir", required=True, help="Folder with the SD rejectedChannels_SCI txt files")
_args = parser.parse_args()
source_dirs = {"ID": _args.id_dir, "SD": _args.sd_dir}
for _c, _d in source_dirs.items():
    if not os.path.isdir(_d):
        raise SystemExit(f"{_c}: folder not found: {_d}")

config.ensure_output_dirs()
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
output_dir = config.RESULTS_QUALITY_DIR


# ---- Reading and preparing the data ----
def read_rejected_channels(source_dir):
    """Read all txt files in a folder (rejection: SCI < SCI_CUTOFF).
    File names must contain 'sub-<digits>' and 'run-1' or 'run-2' (mapped to RunA / RunB).
    Returns {participant: {run: {'count': int, 'channels': [int]}}}."""
    data = {}
    for filename in os.listdir(source_dir):
        if not filename.endswith('.txt'):
            continue
        file_path = os.path.join(source_dir, filename)

        match_p = re.search(r'^sub-(\d+)', filename)
        match_run = re.search(r'run-([12])', filename)

        if not (match_p and match_run):
            continue

        participant = f"sub-{match_p.group(1)}"
        run = {"1": "RunA", "2": "RunB"}[match_run.group(1)]

        rejected_channels = []
        n_listed = 0
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                m = re.search(r'CH:\s*(\d+)\s+SCI:\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)', line)
                if m:
                    n_listed += 1
                    ch, sci = int(m.group(1)), float(m.group(2))
                    if sci < SCI_CUTOFF:
                        rejected_channels.append(ch)
        if n_listed == 0:
            print(f"WARNING: no 'CH: .. SCI: ..' lines in {filename}")

        data.setdefault(participant, {})[run] = {
            'count': len(rejected_channels),
            'channels': rejected_channels
        }
    return data


def data_to_dataframe(data):
    rows = []
    for participant, runs in data.items():
        row = {'Participant': participant}
        for run, info in runs.items():
            row[f'{run}_count'] = info['count']
            row[f'{run}_channels'] = ','.join(map(str, info['channels']))
        rows.append(row)
        
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=['Participant', 'high_quality'])
        
    df = df.sort_values('Participant')

    # A participant is high quality if no channel was rejected in either run
    run_a = df['RunA_count'] if 'RunA_count' in df.columns else pd.Series(0, index=df.index)
    run_b = df['RunB_count'] if 'RunB_count' in df.columns else pd.Series(0, index=df.index)

    df['high_quality'] = ((run_a.fillna(0) == 0) & (run_b.fillna(0) == 0)).astype(int)
    return df


def get_both_run_rejected(df):
    both_rejected = []
    if df.empty:
        return Counter()
    for _, row in df.iterrows():
        if ('RunA_channels' in row and pd.notna(row['RunA_channels']) and
                'RunB_channels' in row and pd.notna(row['RunB_channels'])):
            runA_set = set(int(ch) for ch in str(row['RunA_channels']).split(',') if ch.strip())
            runB_set = set(int(ch) for ch in str(row['RunB_channels']).split(',') if ch.strip())
            both_rejected.extend(runA_set & runB_set)
    return Counter(both_rejected)


def plot_histogram(channel_counts, title, out_path, order_desc=False, bar_color='skyblue'):
    if not channel_counts:
        print(f"No data to plot for: {title}")
        return

    if order_desc:
        sorted_items = sorted(channel_counts.items(), key=lambda x: x[1], reverse=True)
        channels_sorted = [ch for ch, _ in sorted_items]
        counts_sorted = [count for _, count in sorted_items]
    else:
        channels_sorted = sorted(channel_counts.keys())
        counts_sorted = [channel_counts[ch] for ch in channels_sorted]

    x = np.arange(len(channels_sorted))
    plt.figure(figsize=(10, 6))
    bars = plt.bar(x, counts_sorted, color=bar_color)
    plt.xlabel('Channel Number')
    plt.ylabel('Number of Times Rejected')
    plt.title(title)
    plt.xticks(x, channels_sorted)
    for bar, count in zip(bars, counts_sorted):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 str(count), ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved histogram to: {out_path}")


def plot_avg_per_participant(df, label, out_path):
    if df.empty or 'RunA_count' not in df.columns or 'RunB_count' not in df.columns:
        print(f"Not enough data for per-participant averages ({label}).")
        return

    df = df.copy()
    df['avg_rejected'] = df[['RunA_count', 'RunB_count']].mean(axis=1)
    df_sorted = df.sort_values('avg_rejected', ascending=False)

    plt.figure(figsize=(max(10, len(df_sorted) * 0.5), 6))
    bars = plt.bar(df_sorted['Participant'], df_sorted['avg_rejected'],
                    color='skyblue' if label == 'ID' else 'lightgreen')
    plt.xlabel('Participant')
    plt.ylabel('Average Number of Rejected Channels')
    plt.title(f'Average Rejected Channels per Participant ({label})')
    plt.xticks(rotation=90)
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 f"{bar.get_height():.1f}", ha='center', va='bottom', rotation=90)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved average-rejected-per-participant ({label}) histogram to: {out_path}")


# ---- Process each condition (ID, SD) ----
dfs = {}
both_run_counts = {}
excel_paths = {}

os.makedirs(output_dir, exist_ok=True)

for cond in config.CONDITIONS:
    data = read_rejected_channels(source_dirs[cond])
    df = data_to_dataframe(data)
    dfs[cond] = df

    high_quality = df[df['high_quality'] == 1]['Participant'].tolist() if not df.empty else []
    if high_quality:
        hq_file = os.path.join(output_dir, f"high_quality_participants_{cond}_{timestamp}.txt")
        with open(hq_file, 'w') as f:
            for participant in high_quality:
                f.write(f"{participant}\n")
        print(f"Saved high quality participants ({cond}) to: {hq_file}")

    excel_path = os.path.join(output_dir, f"rejectedChannels_{cond}_{timestamp}.xlsx")
    df.to_excel(excel_path, index=False)
    excel_paths[cond] = excel_path
    print(f"Saved rejected channels ({cond}) to: {excel_path}")

    both_run_counts[cond] = get_both_run_rejected(df)

# ---- Histograms ----
colors = {"ID": "skyblue", "SD": "lightgreen"}
for cond in config.CONDITIONS:
    plot_histogram(
        both_run_counts[cond], f"Channels Rejected in Both RunA and RunB ({cond})",
        os.path.join(output_dir, f'rejected_channels_both_{cond}_{timestamp}.png'),
        bar_color=colors.get(cond, 'skyblue')
    )
    plot_histogram(
        both_run_counts[cond], f"Channels Rejected in Both Runs ({cond}) - Most to Least",
        os.path.join(output_dir, f'rejected_channels_both_{cond}_ordered_{timestamp}.png'),
        order_desc=True, bar_color=colors.get(cond, 'skyblue')
    )
    plot_avg_per_participant(
        dfs[cond], cond,
        os.path.join(output_dir, f'avg_rejected_channels_per_participant_{cond}_{timestamp}.png')
    )

# Note: there is intentionally no combined ID-vs-SD histogram. SD is a subset of ID
# but was renumbered (SD 1-13, ID 1-20), so equal channel numbers are NOT the same physical channels.

print("\nDone.")
print("-> Next step: run 06_quality_channel_labels.py with these files:")
for cond, path in excel_paths.items():
    print(f"   {cond}: {path}")