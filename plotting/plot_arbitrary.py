import json, os, glob
from pathlib import Path
import matplotlib.pyplot as plt

def _format_settings(settings: dict) -> str:
    lines = ["Settings:"]
    if "Commutative Constraints" in settings:
        lines.append(f"- Commutative Constraints: {settings['Commutative Constraints']}")
    for k in sorted(settings.keys()):
        if k == "Commutative Constraints":
            continue
        new_k = k
        if k == "N_rounds":
            new_k = "Number of rounds"
        # if k == "n_shots":
        #     new_k = "Number of shots in circuit"
        if k == "visibility":
            new_k = r"Visibility $\nu$"
        if k == "p_long":
            new_k = r"$p_{long}$"
        if k == "decline":
            new_k = "Step Size"
        v = settings[k]
        if k == "Relaxation margin for constraints, Hoeffding Confidence":
            v = [f"{v[0]:.3g}", f"{v[1]:.2g}"]
        if isinstance(v, float):
            v = f"{v:.6g}"
        lines.append(f"- {new_k}: {v}")
    return "\n".join(lines)

def plot_keyrate_vs_etaBL(json_path: Path, out_dir: Path, threshold=0.8):
    with open(json_path, "r") as f:
        data = json.load(f)
    settings = data.get("settings", {})
    kr = data.get("keyrate_results", {})
    if not kr:
        return None, f"No keyrate_results in {json_path.name}"

    etaA_vals = sorted([float(k) for k in kr.keys()], reverse=True)
    if len(etaA_vals) < 2:
        # still plot whatever exists
        etaA_pick = etaA_vals
    else:
        etaA_pick = etaA_vals[:2]
    etaA_pick_str = [f"{v:g}" for v in etaA_pick]

    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    any_curve = False
    for i, etaA_s in enumerate(etaA_pick_str):
        pts = kr.get(etaA_s, [])
        xy = []
        for p in pts:
            x = p.get("etaBL", None)
            y = p.get("keyrate", None)
            if x is None or y is None:
                continue
            if x < threshold:
                continue
            xy.append((x, y))
        xy.sort(key=lambda t: t[0])
        # if xy[-1][0] < threshold:
        #     xy = xy[:-1]
        if not xy:
            continue
        x, y = zip(*xy)
        ax.plot(x, y, marker="o", label=rf"$\eta_A$ = {etaA_s}", color="#5dc084" if i == 0 else "#a172f9")
        any_curve = True

    if not any_curve:
        plt.close(fig)
        return None, f"No valid (etaBL,keyrate) points in {json_path.name}"

    ax.axhline(0.0, linewidth=1, linestyle="dashed", alpha=0.7)
    ax.invert_xaxis()
    # ax.set_xlim(1.0, 0.8)
    ax.set_xlabel("$\eta_{B}$ (long-path detection efficiency)")
    ax.set_ylabel("Key rate $r$")
    ax.grid(True, linestyle="dotted", alpha=0.7)
    ax.legend()

    comm = settings.get("Commutative Constraints", None)
    v = settings.get("visibility", None)
    p_long = settings.get("p_long", None)
    title_parts = []
    if v is not None: title_parts.append(r"$\nu$"+f"={v}")
    if p_long is not None: title_parts.append(r"$p_{LP}$"+f"={p_long:.3g}")
    if comm is not None: title_parts.append(f"Commutativity={comm}")
    if title_parts:
        ax.set_title(", ".join(title_parts))

    # settings_text = _format_settings(settings)
    # fig.tight_layout(rect=[0, 0.25, 1, 1])
    # fig.text(0.01, 0.01, settings_text, ha="left", va="bottom", fontsize=9)

    out_path = out_dir / f"{json_path.stem}__keyrate_vs_etaBL.png"
    fig.savefig(out_path.with_suffix(".svg"), dpi=200)
    plt.close(fig)
    return out_path, None

in_dir = Path("./results/")
out_dir = in_dir / "plots_keyrate_vs_etaBL"
out_dir.mkdir(parents=True, exist_ok=True)

json_files = sorted([Path(p) for p in glob.glob(str(in_dir / "*.json"))])
outputs = []
skipped = []

for fp in json_files:
    out, err = plot_keyrate_vs_etaBL(fp, out_dir)
    if out:
        outputs.append(out)
    else:
        skipped.append((fp.name, err))

outputs, skipped[:5], len(outputs), len(skipped)