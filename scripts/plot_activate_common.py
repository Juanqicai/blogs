import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.special import erf

matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

x = np.linspace(-6, 6, 1000)

# 只画"一元标量激活函数"的曲线。
# SwiGLU 是门控结构（gate/up 两个线性映射 + 逐元素乘），不是一元函数；
# Softmax 是对整个向量的归一化，也不是标量函数。两者都不适合用一元曲线表示，
# 故不在此生成（SwiGLU 结构示意图见 assets/minimind/minimnd SwiGLU.png）。
common = [
    (r"Sigmoid  $\sigma(x)=\frac{1}{1+e^{-x}}$", lambda t: 1 / (1 + np.exp(-t))),
    (r"Tanh  $\tanh(x)$",                        lambda t: np.tanh(t)),
    (r"ReLU  $\max(0,x)$",                       lambda t: np.maximum(0, t)),
    (r"GELU  $x\Phi(x)$",                        lambda t: t * 0.5 * (1 + erf(t / np.sqrt(2)))),
    (r"SiLU  $x\sigma(x)$",                      lambda t: t / (1 + np.exp(-t))),
]

for i, (title, f) in enumerate(common, start=1):
    fig, ax = plt.subplots(figsize=(7, 5), dpi=200)
    y = f(x)
    ax.plot(x, y, lw=3, color="tab:red")
    ax.axhline(0, color="k", lw=1, alpha=0.5)
    ax.axvline(0, color="k", lw=1, alpha=0.5)
    ax.set_title(title, fontsize=18, pad=12)
    ax.set_xlim(-6, 6)
    ymin, ymax = min(y.min(), -1), max(y.max(), 1)
    ax.set_ylim(ymin - 0.3, ymax + 0.3)
    ax.set_xticks([-6, -3, 0, 3, 6])
    ax.grid(alpha=0.3)
    name = title.split()[0]
    fig.tight_layout()
    fig.savefig(f"activate_common_{i}_{name}.png", bbox_inches="tight")
    plt.close(fig)

print("saved common activation plots")
