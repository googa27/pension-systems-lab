"""Reproduce a 300-dpi dark README hero from actual synthetic public API runs."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
BG, PANEL, TEXT, MUTED = "#0d1117", "#161b22", "#e6edf3", "#9da7b3"
GREEN, CYAN, PURPLE, YELLOW = "#00e68a", "#44d7f4", "#9985ff", "#ffd43b"


def render() -> Path:
    is_afp = ROOT.name == "afp-operations"
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "text.color": TEXT,
            "axes.labelcolor": MUTED,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.edgecolor": "#30363d",
            "font.size": 11,
            "savefig.facecolor": BG,
        }
    )
    fig = plt.figure(figsize=(12, 6.75), facecolor=BG)
    fig.text(
        0.06,
        0.89,
        "AFP OPERATIONS" if is_afp else "PENSION SYSTEMS LAB",
        fontsize=29,
        weight="bold",
    )
    fig.text(
        0.06,
        0.825,
        "Typed operations library" if is_afp else "Chile today. International systems by design.",
        fontsize=16,
        color=MUTED,
    )
    fig.text(
        0.06,
        0.765,
        "McKendrick-von Foerster PDE  •  FEM / finite volume  •  Bayesian dynamics",
        fontsize=12,
        color=CYAN,
    )
    fig.text(
        0.06,
        0.69,
        r"$\partial_t n + \partial_a n = -\mu n + m,\qquad n(t,0)=b(t)$",
        fontsize=21,
        color=GREEN,
    )
    fig.text(
        0.06,
        0.625,
        "RESEARCH ALPHA  /  reproducible software, not certified policy advice",
        fontsize=10,
        color=YELLOW,
    )
    left = fig.add_axes((0.055, 0.14, 0.41, 0.415))
    left.set_axis_off()
    cards = (
        [
            ("POLICY + ACCOUNTS", "Dated fixtures · Decimal arithmetic", GREEN),
            ("TRANSPORT + UNCERTAINTY", "P1 SUPG FEM · hierarchical mortality", CYAN),
            ("EVIDENCE + INTEGRATION", "Typed API · JSON CLI · public contracts", PURPLE),
        ]
        if is_afp
        else [
            ("PUBLIC DATA", "National aggregates · units · provenance", GREEN),
            ("MODELS + SCENARIOS", "Continuous age · two-sex FEM · dynamics", CYAN),
            ("RESEARCH ECOSYSTEM", "AFP library · solver and data contracts", PURPLE),
        ]
    )
    for i, (heading, caption, color) in enumerate(cards):
        y = 0.70 - i * 0.30
        left.add_patch(
            FancyBboxPatch(
                (0.02, y),
                0.94,
                0.23,
                boxstyle="round,pad=0.012,rounding_size=0.025",
                facecolor=PANEL,
                edgecolor=color,
                linewidth=1.5,
            )
        )
        left.text(0.06, y + 0.14, heading, fontsize=12, weight="bold", color=color)
        left.text(0.06, y + 0.06, caption, fontsize=9.3, color=TEXT)
    ax = fig.add_axes((0.565, 0.18, 0.37, 0.35), facecolor=PANEL)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.12, color=MUTED)
    ax.set_axisbelow(True)
    if is_afp:
        from afp_operations import Account

        account = Account()
        balance = [float(account.balance)]
        for _ in range(36):
            account.apply_return(Decimal("0.002"))
            account.post_contribution(Decimal("10000"))
            balance.append(float(account.balance) / 1000)
        ax.plot(range(37), balance, color=CYAN, lw=2.5, label="Account balance")
        ax.plot(range(37), np.arange(37) * 10, color=GREEN, lw=2, ls="--", label="Contributions")
        ax.set(
            xlabel="Month",
            ylabel="Thousands of synthetic currency units",
            title="Synthetic account lifecycle",
        )
        filename = "afp_operations_hero.png"
    else:
        from chile_demographic_pde.pde.fem import AgeTransportFEM

        solver = AgeTransportFEM(maximum_age=6.0, nodes=61, artificial_diffusion=0.002)
        age = solver.age
        initial = 1000 * np.exp(-0.5 * ((age - 2.3) / 0.75) ** 2)
        current = initial.copy()
        for _ in range(4):
            current = solver.step(
                current,
                mortality=np.full_like(age, 0.02),
                inflow=0.0,
                source=np.zeros_like(age),
                dt=0.25,
            )
        ax.plot(age, initial, color=GREEN, lw=2, label="Initial density")
        ax.plot(age, current, color=CYAN, lw=2.5, label="After one model year")
        ax.set(
            xlabel="Age (scaled synthetic model grid)",
            ylabel="Density (synthetic persons / age unit)",
            title="Synthetic FEM age transport",
        )
        filename = "pension_systems_lab_hero.png"
    ax.legend(loc="upper left", frameon=False, fontsize=8, labelcolor=TEXT)
    ax.title.set_color(TEXT)
    ax.title.set_fontsize(13)
    fig.text(
        0.565, 0.095, "Synthetic inputs · actual public API computation", fontsize=9, color=MUTED
    )
    fig.text(
        0.06,
        0.055,
        "OPEN RESEARCH  /  no credentials or private member data",
        fontsize=9,
        color=MUTED,
    )
    output = ROOT / "docs" / "assets" / filename
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    plt.close(fig)
    return output


if __name__ == "__main__":
    print(render().relative_to(ROOT))
