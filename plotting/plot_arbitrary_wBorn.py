import json, os, glob, re
from pathlib import Path
import matplotlib.pyplot as plt

# -------------------------
# Helpers
# -------------------------

def _format_settings(settings: dict) -> str:
    lines = ["Settings:"]
    if "Commutative Constraints" in settings:
        lines.append(f"- Commutativity: {settings['Commutative Constraints']}")
    for k in sorted(settings.keys()):
        if k == "Commutative Constraints":
            continue
        new_k = k
        if k == "N_rounds":
            new_k = "Number of rounds"
        if k == "visibility":
            new_k = r"Visibility $\nu$"
        if k == "p_long":
            new_k = r"$p_{long}$"
        if k == "decline":
            new_k = "Step Size"
        v = settings[k]
        if k == "Relaxation margin for constraints, Hoeffding Confidence":
            # format list-like
            v = [f"{v[0]:.3g}", f"{v[1]:.2g}"]
        if isinstance(v, float):
            v = f"{v:.3g}"
        lines.append(f"- {new_k}: {v}")
    return "\n".join(lines)

def parse_vis_from_filename(name: str):
    """
    Extract visibility from filename patterns like:
      vis1_...
      vis0.99_...
      vis0.98_...
    Returns float or None.
    """
    m = re.search(r"vis(1(?:\.0+)?|0\.\d+)", name)
    if not m:
        return None
    return float(m.group(1))

def build_born_index(born_dir: Path):
    """
    Index BORN files by visibility float.
    If multiple BORN files exist for same vis, prefer:
      *Relaxed*  >  *Equality*  > anything else
    """
    idx = {}  # vis -> Path

    def score(p: Path):
        n = p.name.lower()
        if "relaxed" in n:
            return 2
        if "equality" in n:
            return 1
        return 0

    for p in born_dir.glob("vis*_BORN*.json"):
        v = parse_vis_from_filename(p.name)
        if v is None:
            continue
        if (v not in idx) or (score(p) > score(idx[v])):
            idx[v] = p
    # also include IdealBORN variants
    for p in born_dir.glob("vis*_IdealBORN*.json"):
        v = parse_vis_from_filename(p.name)
        if v is None:
            continue
        if (v not in idx) or (score(p) > score(idx[v])):
            idx[v] = p

    return idx

def extract_xy(points, threshold):
    xy = []
    for p in points:
        x = p.get("etaBL", None)   # plotted as η_B
        y = p.get("keyrate", None)
        if x is None or y is None:
            continue
        # if x < threshold or x > 0.99:
        if x < threshold:
            continue
        xy.append((x, y))
    xy.sort(key=lambda t: t[0])  # ascending; we invert axis later
    return xy

# -------------------------
# Plot function (with BORN overlay)
# -------------------------

def plot_keyrate_vs_etaBL(json_path: Path, out_dir: Path, threshold=0.85, born_index=None):
    with open(json_path, "r") as f:
        data = json.load(f)

    settings = data.get("settings", {})
    kr = data.get("keyrate_results", {})
    if not kr:
        return None, f"No keyrate_results in {json_path.name}"

    # choose first two etaA values (descending)
    etaA_vals = sorted([float(k) for k in kr.keys()], reverse=True)
    etaA_pick = etaA_vals[:2] if len(etaA_vals) >= 2 else etaA_vals
    etaA_pick_str = [f"{v:g}" for v in etaA_pick]

    fig, ax = plt.subplots(figsize=(6.0, 6.0))

    # -------------------------
    # 1) BORN overlay (background)
    # -------------------------
    vis = settings.get("visibility", None)
    born_path = None
    if born_index is not None and vis is not None:
        born_path = born_index.get(vis, None)

    if born_path is not None:
        with open(born_path, "r") as f:
            born = json.load(f)  # same structure: etaA -> list of points

        # plot BORN first, in grey, low alpha, behind
        born_etaA_vals = sorted([float(k) for k in born.keys()], reverse=True)
        born_etaA_pick = born_etaA_vals[:2] if len(born_etaA_vals) >= 2 else born_etaA_vals
        born_etaA_pick_str = [f"{v:g}" for v in born_etaA_pick]

        for i, etaA_s in enumerate(born_etaA_pick_str):
            xy = extract_xy(born.get(etaA_s, []), threshold)
            if not xy:
                continue
            x, y = zip(*xy)
            ax.plot(
                x, y,
                marker="o",
                linestyle="-",
                # alpha=0.5,
                linewidth=2,
                color="#364A3EFF" if i == 0 else "#644F89FF",
                zorder=1,
                label=rf"BORN ($\nu$={vis:.3g}, $\eta_A$={etaA_s})"
            )

    # -------------------------
    # 2) Your data (foreground)
    # -------------------------
    any_curve = False
    for i, etaA_s in enumerate(etaA_pick_str):
        pts = kr.get(etaA_s, [])
        xy = extract_xy(pts, threshold)
        if not xy:
            continue
        x, y = zip(*xy)
        ax.plot(
            x, y,
            marker="o",
            label=rf"$\eta_A$ = {etaA_s}",
            # keep your existing colors:
            color="#5dc084" if i == 0 else "#a172f9",
            zorder=3
        )
        any_curve = True

    if not any_curve:
        plt.close(fig)
        return None, f"No valid (etaBL,keyrate) points in {json_path.name}"

    # Styling
    ax.axhline(0.0, linewidth=1, linestyle="dashed", alpha=0.7)
    ax.invert_xaxis()
    ax.set_xlabel(r"$\eta_{B}$ (long-path detection efficiency)")
    ax.set_ylabel(r"Key rate $r$")
    ax.grid(True, linestyle="dotted", alpha=0.7)
    ax.legend()

    # Title
    comm = settings.get("Commutative Constraints", None)
    v = settings.get("visibility", None)
    p_long = settings.get("p_long", None)
    title_parts = []
    if v is not None: title_parts.append(r"$\nu$"+f"={float(v):.3g}")
    if p_long is not None: title_parts.append(r"$p_{long}$"+f"={float(p_long):.3g}")
    if comm is not None: title_parts.append(f"Commutativity={comm}")
    ax.set_title(", ".join(title_parts))

    # Bottom config text
    settings_text = _format_settings(settings)
    fig.tight_layout(rect=[0, 0.25, 1, 1])
    fig.text(0.01, 0.01, settings_text, ha="left", va="bottom", fontsize=9)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{json_path.stem}__keyrate_vs_etaBL_with_BORN.svg"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path, None

# -------------------------
# Batch runner
# -------------------------

if __name__ == "__main__":
    in_dir = Path("./results/")
    out_dir = in_dir / "plots_keyrate_vs_etaBL_with_BORN"

    # BORN files live alongside your script (or change this to wherever they are)
    born_dir = in_dir / "ideal_born/born_results/"
    born_index = build_born_index(born_dir)

    json_files = sorted([Path(p) for p in glob.glob(str(in_dir / "*.json"))])

    outputs, skipped = [], []
    for fp in json_files:
        out, err = plot_keyrate_vs_etaBL(fp, out_dir, threshold=0.85, born_index=born_index)
        if out:
            outputs.append(out)
        else:
            skipped.append((fp.name, err))

    print(f"Saved: {len(outputs)}  Skipped: {len(skipped)}")
    if skipped:
        print("Skipped examples:", skipped[:5])