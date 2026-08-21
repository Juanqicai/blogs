import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.special import erf

matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

x = np.linspace(-6, 6, 1000)

# 只画一元标量激活函数；SwiGLU（门控结构）与 Softmax（向量归一化）不适合一元曲线，已移除。
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
    (r"Hardswish",                                   lambda t: t * np.clip(t + 3, 0, 6) / 6),
]

n = len(funcs)
rows, cols = 2, 5

fig, axes = plt.subplots(rows, cols, figsize=(18, 9))
fig.suptitle("激活函数总览", fontsize=22, fontweight="bold", y=0.98)

for i, (title, f) in enumerate(funcs):
    ax = axes[i // cols, i % cols]
    y = f(x)
    ax.plot(x, y, lw=2.5, color="tab:blue")
    ax.axhline(0, color="k", lw=0.9, alpha=0.5)
    ax.axvline(0, color="k", lw=0.9, alpha=0.5)
    ax.set_title(title, fontsize=13)
    ax.set_xlim(-6, 6)
    ymin, ymax = min(y.min(), -1), max(y.max(), 1)
    ax.set_ylim(ymin - 0.3, ymax + 0.3)
    ax.set_xticks([-6, -3, 0, 3, 6])
    ax.grid(alpha=0.3)

fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=3.0, w_pad=2.0)
fig.savefig("激活函数总览.png", dpi=200, bbox_inches="tight")
print("saved: 激活函数总览.png")
