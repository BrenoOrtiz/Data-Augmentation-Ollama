import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# ── Dados ──────────────────────────────────────────────────────────────────────
models  = ['llama3.1', 'mistral:7b', 'phi3.5']
labels  = ['LLaMA 3.1', 'Mistral 7b', 'Phi 3.5']
colors  = ['#2166ac', '#1a9850', '#d6604d']
ratios  = ['10%', '25%', '50%', '75%']
baseline = 0.8930

aug = {
    'llama3.1':  [0.8759, 0.8912, 0.8658, 0.8495],
    'mistral:7b':[0.8848, 0.8864, 0.8763, 0.8542],
    'phi3.5':    [0.8874, 0.8746, 0.8821, 0.8632],
}
res = {
    'llama3.1':  [0.8868, 0.8450, 0.7988, 0.4837],
    'mistral:7b':[0.8895, 0.8552, 0.7988, 0.4837],
    'phi3.5':    [0.8895, 0.8552, 0.7988, 0.4837],
}

# ── Pasta de saída ──────────────────────────────────────────────────────────────
OUT = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
})

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURA 1 — Barras agrupadas por ratio (2×2)
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=False)
fig.suptitle('F1-macro: Augmented vs Restricted por Proporção de Dados Sintéticos',
             fontsize=13, fontweight='bold', y=1.01)

for ax, ratio, ri in zip(axes.flat, ratios, range(4)):
    x = np.arange(len(models))
    w = 0.35
    aug_vals = [aug[m][ri] for m in models]
    res_vals = [res[m][ri] for m in models]

    ymin = 0.40 if ri == 3 else 0.75

    bars1 = ax.bar(x - w/2, aug_vals, w,
                   color=[c + 'cc' for c in colors],
                   edgecolor=colors, linewidth=1.2, label='Augmented')
    bars2 = ax.bar(x + w/2, res_vals, w,
                   color='none', edgecolor=colors, linewidth=1.2,
                   hatch='//', label='Restricted')

    ax.axhline(baseline, color='black', linestyle='--', linewidth=1,
               label='Baseline (full)')

    ax.set_title(f'Ratio = {ratio}', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(ymin, 0.935)
    ax.set_ylabel('F1-macro')

    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.004,
                f'{bar.get_height():.3f}',
                ha='center', va='bottom', fontsize=7.5)

handles = [
    mpatches.Patch(facecolor='gray', alpha=0.7, label='Augmented'),
    mpatches.Patch(facecolor='none', edgecolor='gray', hatch='//', label='Restricted'),
    plt.Line2D([0], [0], color='black', linestyle='--', label='Baseline (full data)'),
]
fig.legend(handles=handles, loc='lower center', ncol=3,
           bbox_to_anchor=(0.5, -0.04), fontsize=10)

plt.tight_layout()
path_bar = os.path.join(OUT, 'grafico_barras.png')
path_bar_pdf = os.path.join(OUT, 'grafico_barras.pdf')
plt.savefig(path_bar,     bbox_inches='tight', dpi=300)
plt.savefig(path_bar_pdf, bbox_inches='tight')
print(f'✔ Barras salvo em {path_bar}')
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURA 2 — Linhas: F1-macro vs Ratio
# ═══════════════════════════════════════════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(8, 5))
markers = ['o', 's', '^']

for m, lbl, c, mk in zip(models, labels, colors, markers):
    ax2.plot(ratios, aug[m], color=c, marker=mk,
             linewidth=2, markersize=7, label=f'{lbl} (aug)')
    ax2.plot(ratios, res[m], color=c, marker=mk,
             linewidth=1.5, markersize=7, linestyle='--', alpha=0.55,
             label=f'{lbl} (res)')

ax2.axhline(baseline, color='black', linestyle=':', linewidth=1.5,
            label='Baseline (full data)')

ax2.set_xlabel('Proporção de dados sintéticos')
ax2.set_ylabel('F1-macro')
ax2.set_title('F1-macro vs Proporção de Dados Sintéticos',
              fontsize=12, fontweight='bold')
ax2.set_ylim(0.42, 0.91)
ax2.legend(fontsize=8, ncol=2, loc='lower left')
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
path_line = os.path.join(OUT, 'grafico_linhas.png')
path_line_pdf = os.path.join(OUT, 'grafico_linhas.pdf')
plt.savefig(path_line,     bbox_inches='tight', dpi=300)
plt.savefig(path_line_pdf, bbox_inches='tight')
print(f'✔ Linhas salvo em {path_line}')
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURA 3 — Δ vs Baseline (heatmap-like bar)
# ═══════════════════════════════════════════════════════════════════════════════
fig3, ax3 = plt.subplots(figsize=(13, 5))

group_labels, delta_aug, delta_res, bar_colors = [], [], [], []
for m, lbl, c in zip(models, labels, colors):
    for ri, r in enumerate(ratios):
        group_labels.append(f'{lbl}\n{r}')
        delta_aug.append(round(aug[m][ri] - baseline, 4))
        delta_res.append(round(res[m][ri] - baseline, 4))
        bar_colors.append(c)

x = np.arange(len(group_labels))
w = 0.38

b1 = ax3.bar(x - w/2, delta_aug, w,
             color=[c + 'bb' for c in bar_colors],
             edgecolor=bar_colors, linewidth=1, label='Augmented Δ')
b2 = ax3.bar(x + w/2, delta_res, w,
             color='none', edgecolor=bar_colors,
             linewidth=1.2, hatch='//', label='Restricted Δ')

ax3.axhline(0, color='black', linewidth=0.9)
ax3.set_xticks(x)
ax3.set_xticklabels(group_labels, fontsize=8)
ax3.set_ylabel('Δ F1-macro vs baseline')
ax3.set_title('Diferença em relação ao Baseline (0.893)',
              fontsize=12, fontweight='bold')

handles3 = [
    mpatches.Patch(facecolor='gray', alpha=0.7, label='Augmented Δ'),
    mpatches.Patch(facecolor='none', edgecolor='gray', hatch='//', label='Restricted Δ'),
]
ax3.legend(handles=handles3, fontsize=9)
ax3.grid(axis='y', alpha=0.25)

plt.tight_layout()
path_delta = os.path.join(OUT, 'grafico_delta.png')
path_delta_pdf = os.path.join(OUT, 'grafico_delta.pdf')
plt.savefig(path_delta,     bbox_inches='tight', dpi=300)
plt.savefig(path_delta_pdf, bbox_inches='tight')
print(f'✔ Delta salvo em {path_delta}')
plt.close()

print('\nTodos os gráficos salvos na pasta results/')
