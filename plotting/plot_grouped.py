import json
import math
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import textwrap

# ----------------------------
# Formatting helpers
# ----------------------------

def fmt3g(v):
    try:
        if isinstance(v, bool):
            return str(v)
        if isinstance(v, (int, float)):
            return f"{float(v):.3g}"
        return f"{float(v):.3g}"
    except Exception:
        return str(v)

def load_result_json(fp: Path) -> dict:
    with open(fp, "r") as f:
        data = json.load(f)
    data["_path"] = str(fp)
    return data

def pick_two_etaA_keys(keyrate_results: dict) -> list[str]:
    etaA_vals = sorted([float(k) for k in keyrate_results.keys()], reverse=True)
    if len(etaA_vals) == 0:
        return []
    if len(etaA_vals) == 1:
        return [f"{etaA_vals[0]:g}"]
    return [f"{etaA_vals[0]:g}", f"{etaA_vals[1]:g}"]

def extract_curve_points(points: list[dict], min_etaB: float = 0.85):
    xy = []
    for p in points:
        x = p.get("etaBL")   # labelled as η_B
        y = p.get("keyrate")
        if x is None or y is None:
            continue
        if x < min_etaB:
            continue
        xy.append((x, y))
    xy.sort(key=lambda t: t[0])
    if not xy:
        return [], []
    x, y = zip(*xy)
    return list(x), list(y)

def short_meta(settings: dict) -> str:
    comm = settings.get("Commutative Constraints", None)
    nu = settings.get("visibility", None)
    p_long = settings.get("p_long", None)

    parts = []
    if nu is not None:
        parts.append(rf"Visibility $\nu$={fmt3g(nu)}")
    if p_long is not None:
        parts.append(rf"$p_{{long}}$={fmt3g(p_long)}")
    if comm is not None:
        parts.append(f"Commutativity={comm}")
    return ", ".join(parts)

def group_by_setting(items: list[dict], setting_key: str) -> dict:
    groups = {}
    for d in items:
        s = d.get("settings", {})
        k = s.get(setting_key, None)
        groups.setdefault(k, []).append(d)
    return groups

def choose_ncols(n: int, max_cols: int = 3) -> int:
    """
    Dynamic columns:
      - for small n, avoid empty space (n=1 -> 1 col, n=2 -> 2 cols)
      - for larger n, keep <= max_cols
    """
    if n <= 0:
        return 1
    if n <= max_cols:
        return n
    return max_cols

def plot_grouped(
    items: list[dict],
    group_key: str,
    out_dir: Path,
    max_cols: int = 3,
    invert_x: bool = True,
    min_etaB: float = 0.85,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = group_by_setting(items, group_key)

    def _sort_key(x):
        return (x is None, float(x) if isinstance(x, (int, float)) or str(x).replace('.','',1).isdigit() else str(x))
    group_keys_sorted = sorted(groups.keys(), key=_sort_key)

    for gval in group_keys_sorted:
        group_items = groups[gval]
        if not group_items:
            continue

        # Sort within group (unchanged)
        def _item_sort(d):
            s = d.get("settings", {})
            if group_key == "visibility":
                return (s.get("p_long", 999), s.get("Commutative Constraints", False))
            if group_key == "p_long":
                return (s.get("visibility", 999), s.get("Commutative Constraints", False))
            return (s.get("visibility", 999), s.get("p_long", 999))
        group_items = sorted(group_items, key=_item_sort)

        n = len(group_items)
        ncols = choose_ncols(n, max_cols=max_cols)
        nrows = math.ceil(n / ncols)

        # --- Size in inches (key fix) ---
        plot_row_h = 3.3          # height per subplot row (inches)
        plot_col_w = 4.8          # width per subplot column (inches)

        # Caption height in inches: scale with number of lines, but clamp
        caption_lines_est = 3 + n  # suptitle + filter + "Subplots" + one per subplot
        caption_h = min(4.5, max(1.6, 0.18 * caption_lines_est))

        fig_w = plot_col_w * ncols
        fig_h = plot_row_h * nrows + caption_h

        fig = plt.figure(figsize=(fig_w, fig_h))

        # IMPORTANT: height_ratios uses real-ish weights, not tiny ratios
        height_ratios = [plot_row_h] * nrows + [caption_h]

        gs = GridSpec(
            nrows=nrows + 1,
            ncols=ncols,
            figure=fig,
            height_ratios=height_ratios,
            hspace=0.55,   # more vertical spacing between subplots
            wspace=0.35,   # more horizontal spacing between subplots
        )

        # Suptitle (same logic as before)
        if group_key == "visibility":
            supt = rf"Grouped by Visibility $\nu$ = {fmt3g(gval)}"
        elif group_key == "p_long":
            supt = rf"Grouped by $p_{{long}}$ = {fmt3g(gval)}"
        else:
            supt = f"Grouped by {group_key} = {fmt3g(gval)}"
        fig.suptitle(supt, y=0.995)

        # Create subplot axes in the top nrows
        axes = []
        for r in range(nrows):
            for c in range(ncols):
                axes.append(fig.add_subplot(gs[r, c]))

        caption_lines = [
            supt,
            "Subplots:",
        ]

        for idx, d in enumerate(group_items):
            ax = axes[idx]
            settings = d.get("settings", {})
            kr = d.get("keyrate_results", {})

            for i, etaA_s in enumerate(pick_two_etaA_keys(kr)):
                x, y = extract_curve_points(kr.get(etaA_s, []), min_etaB=min_etaB)
                if x and y:
                    ax.plot(x, y, marker="o", label=rf"$\eta_A$={fmt3g(etaA_s)}", color="#5dc084" if i == 0 else "#a172f9")

            ax.axhline(0.0, linewidth=1, alpha=0.8, linestyle="--")
            ax.grid(True, linestyle="dotted", alpha=0.7)
            ax.set_ylabel("Key rate")

            if invert_x:
                ax.invert_xaxis()
                ax.set_xlabel(r"$\eta_B$ (decreasing →)")
            else:
                ax.set_xlabel(r"$\eta_B$")

            ax.set_title(short_meta(settings), fontsize=10)
            ax.legend(fontsize=9)

            caption_lines.append(
                "  "
                + f"[{idx+1}] "
                + rf"Visibility $\nu$={fmt3g(settings.get('visibility'))}, "
                + rf"$p_{{long}}$={fmt3g(settings.get('p_long'))}, "
                + f"Commutativity={settings.get('Commutative Constraints')}"
            )

        # Turn off unused axes
        for j in range(n, len(axes)):
            axes[j].axis("off")

        # Caption axis spans full width on last row
        cap_ax = fig.add_subplot(gs[nrows, :])
        cap_ax.axis("off")

        wrapped = "\n".join(textwrap.fill(line, width=140) for line in caption_lines)
        cap_ax.text(0.0, 0.0, wrapped, ha="left", va="bottom", fontsize=9, transform=cap_ax.transAxes)

        # Extra margins so suptitle/caption don’t collide with figure edges
        # fig.subplots_adjust(top=0.94, bottom=0.05)
        top_margin = 0.86 if nrows == 1 else 0.92
        fig.subplots_adjust(top=top_margin, bottom=0.07)

        safe_gval = "None" if gval is None else str(fmt3g(gval)).replace("/", "_")
        out_path = out_dir / f"group_{group_key}_{safe_gval}.png"
        fig.savefig(out_path.with_suffix(".svg"), dpi=200)
        plt.close(fig)
        print(f"[saved] {out_path}")        
        
def main():
    in_dir = Path("./results")
    out_dir = in_dir / "grouped_plots_pretty"

    json_files = sorted(in_dir.glob("*.json"))
    items = [load_result_json(fp) for fp in json_files]

    plot_grouped(items, group_key="visibility", out_dir=out_dir / "by_visibility", invert_x=True)
    plot_grouped(items, group_key="p_long", out_dir=out_dir / "by_p_long", invert_x=True)

if __name__ == "__main__":
    main()