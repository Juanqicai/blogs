import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.special import erf

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

x = np.linspace(-6, 6, 1000)

funcs = [
    (r"Sigmoid  $\sigma(x)=\frac{1}{1+e^{-x}}$",     lambda t: 1 / (1 + np.exp(-t))),
    (r"Tanh  $\tanh(x)$",                            lambda t: np.tanh(t)),
    (r"ReLU  $\max(0,x)$",                           lambda t: np.maximum(0, t)),
    (r"Leaky ReLU  ($\alpha=0.05$)",                 lambda t: np.where(t >= 0, t, 0.05 * t)),
    (r"Softplus  $\ln(1+e^x)$",                      lambda t: np.log1p(np.exp(t))),
    (r"ELU  ($\alpha=1$)",                           lambda t: np.where(t >= 0, t, np.expm1(t))),
    (r"GELU  $x\Phi(x)$",                            lambda t: t * 0.5 * (1 + erf(t / np.sqrt(2)))),
    (r"SiLU  $x\sigma(x)$",                          lambda t: t / (1 + np.exp(-t))),
    (r"Mish  $x\tanh(\ln(1+e^x))$",                 lambda t: t * np.tanh(np.log1p(np.exp(t)))),
    (r"SwiGLU  $SiLU(x)\cdot x$",                   lambda t: (t / (1 + np.exp(-t))) * t),
    (r"Hardswish",                                   lambda t: t * np.clip(t + 3, 0, 6) / 6),
    (r"Softmax  $p(x)=\frac{e^x}{1+e^x}$",          lambda t: 1 / (1 + np.exp(-t))),
]

n = len(funcs)
rows, cols = 4, 3

fig, axes = plt.subplots(rows, cols, figsize=(15, 16))
fig.suptitle("激活函数总览", fontsize=22, fontweight="bold", y=0.96)

for i, (title, f) in enumerate(funcs):
    ax = axes[i // cols, i % cols]
    y = f(x)
    ax.plot(x, y, lw=2.5, color="tab:blue")
    ax.axhline(0, color="k", lw=0.9, alpha=0.5)
    ax.axvline(0, color="k", lw=0.9, alpha=0.5)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(-6, 6)
    ymin, ymax = min(y.min(), -1), max(y.max(), 1)
    ax.set_ylim(ymin - 0.3, ymax + 0.3)
    ax.set_xticks([-6, -3, 0, 3, 6])
    ax.grid(alpha=0.3)

for i in range(n, rows * cols):
    axes[i // cols, i % cols].axis("off")

fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=3.0, w_pad=2.0)
fig.savefig("activate_functions.png", dpi=200, bbox_inches="tight")
plt.show()
print("saved: activate_functions.png")