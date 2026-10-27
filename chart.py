import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# --- Graph 1: Loading Comparison ---
categories = ['Uncontrolled\n(Line Trip)', 'Physics PTDF\n(AC Flow)', 'ML Model\n(115% Load)', 'MATLAB\n(DC Flow)']
loadings = [148.50, 98.20, 95.90, 95.00]
colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']

bars = ax1.bar(categories, loadings, color=colors, width=0.55)
ax1.axhline(100, color='red', linestyle='--', linewidth=2, label='100% Thermal Limit')
ax1.set_ylabel('Line Loading (%)', fontsize=12, fontweight='bold')
ax1.set_title('9-Bus Post-Contingency Thermal Overload Relief', fontsize=14, fontweight='bold')
ax1.set_ylim(0, 170)

for bar in bars:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 3, f'{yval:.2f}%', ha='center', va='bottom', fontweight='bold')

ax1.legend(loc='upper right')

# --- Graph 2: Speed Comparison ---
methods = ['Physics PTDF\n(AC Solve)', 'ML Model\n(Inference)']
times = [118.5, 9.73] 

bars2 = ax2.bar(methods, times, color=['#e67e22', '#2ecc71'], width=0.45)
ax2.set_ylabel('Response Time (ms)', fontsize=12, fontweight='bold')
ax2.set_title('Control Execution Speed Comparison', fontsize=14, fontweight='bold')
ax2.set_yscale('log')
ax2.set_ylim(1, 300)

for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval * 1.2, f'{yval:.2f} ms', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('bess_control_performance_charts_9bus.png', dpi=300)
plt.show()