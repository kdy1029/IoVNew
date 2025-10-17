# ==========================================
# IoV IDS Runtime Bar Chart (i.MX8M Plus)
# ==========================================
import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)

# --- Runtime data from i.MX8M Plus experiments ---
models = [
    "LogReg (CPU)",
    "GaussianNB (CPU)",
    "ExtraTrees (CPU)",
    "MLP INT8 (CPU)",
    "MLP INT8 (NPU)"
]

avg_ms = [0.0002, 0.0013, 0.0045, 0.0803, 0.199]
fps = [4222104, 776248, 221640, 12451, 5024]

# --- Plot 1: Latency (log-scale) ---
plt.figure(figsize=(6, 5))
bars = plt.bar(models, avg_ms, color=['#4B8BBE','#306998','#FFE873','#FFD43B','#646464'])
plt.yscale('log')
plt.ylabel("Average Latency (ms / frame)")
plt.title("Runtime Latency of Models on i.MX8M Plus (log scale)")
plt.xticks(rotation=25, ha='right')
for i, v in enumerate(avg_ms):
    plt.text(i, v*1.2, f"{v:.4f}", ha='center', fontsize=8)
plt.tight_layout()
plt.savefig("figures/runtime_latency_bar.pdf", dpi=300)
plt.close()
colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e']

# --- Plot 2: Throughput (fps) ---
plt.figure(figsize=(6, 4.0))
bars = plt.bar(models, fps, color=plt.cm.Set2.colors)
plt.yscale('log')
plt.ylabel("Throughput (frames / s, log scale)")
plt.title("Runtime Throughput of Models on i.MX8M Plus")
plt.xticks(rotation=25, ha='right')
for i, v in enumerate(fps):
    plt.text(i, v*1.1, f"{v:,.0f}", ha='center', fontsize=8)
#plt.tight_layout()
plt.subplots_adjust(top=1, bottom=0.20)
plt.savefig("figures/runtime_fps_bar.pdf", dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.close()

print("✅ Saved: figures/runtime_latency_bar.pdf, figures/runtime_fps_bar.pdf")
