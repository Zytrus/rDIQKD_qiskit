import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

VIS_RE = re.compile(r"(?:^|[_\-])vis(?P<vis>1(?:\.0+)?|0\.\d+)(?:[_\-]|$)", re.IGNORECASE)

def gfmt(x, sig=3) -> str:
    try:
        return f"{float(x):.{sig}g}"
    except Exception:
        return str(x)

def parse_visibility_from_name(p: Path) -> str:
    m = VIS_RE.search(p.stem)
    if not m:
        return "unknown"
    vis = m.group("vis")
    try:
        vf = float(vis)
        if abs(vf - 1.0) < 1e-12:
            return "1"
        return f"{vf:g}"
    except Exception:
        return vis

def load_born_json(path: Path) -> Dict[str, List[dict]]:
    with open(path, "r") as f:
        data = json.load(f)
    # tolerate a wrapper key, just in case
    if isinstance(data, dict) and "keyrate_results" in data and isinstance(data["keyrate_results"], dict):
        return data["keyrate_results"]
    if isinstance(data, dict):
        return data
    raise ValueError(f"Unexpected JSON structure in {path.name}")

def pick_etaA_keys(kr: Dict[str, List[dict]], max_curves: int = 2) -> List[str]:
    numeric = []
    nonnum = []
    for k in kr.keys():
        try:
            numeric.append((float(k), k))
        except Exception:
            nonnum.append(k)
    numeric.sort(reverse=True, key=lambda t: t[0])
    keys = [k for _, k in numeric[:max_curves]]
    if len(keys) < max_curves:
        keys += nonnum[: max_curves - len(keys)]
    return keys

def extract_xy(points: List[dict], threshold: float) -> List[Tuple[float, float]]:
    xy: List[Tuple[float, float]] = []
    for p in points:
        x = p.get("etaBL", None)
        y = p.get("keyrate", None)
        if x is None or y is None:
            continue
        try:
            x = float(x); y = float(y)
        except Exception:
            continue
        if x < threshold:
            continue
        xy.append((x, y))
    xy.sort(key=lambda t: t[0])  # ascending; we invert axis later
    return xy

def classify_variant(fp: Path) -> str:
    name = fp.name.lower()
    if "relaxed" in name:
        return "Relaxed"
    if "equality" in name:
        return "Equality"
    return "Other"

def safe_vis_for_filename(vis: str) -> str:
    return str(vis).replace(".", "p")

def main():
    ap = argparse.ArgumentParser()
    results_dir = Path("./results/")
    in_dir = results_dir / "ideal_born/born_results/"
    out_dir = in_dir / "plots_keyrate_vs_etaBL/"
    out_dir.mkdir(parents=True, exist_ok=True)

    # collect Born json files (including IdealBORN naming variants)
    json_files = sorted(in_dir.glob("vis*_BORN_SDP_Results_*.json")) + sorted(in_dir.glob("vis*_IdealBORN_SDP_Results_*.json"))
    if not json_files:
        print(f"No Born JSON files found in {in_dir.resolve()}")
        return

    # group by visibility -> then by variant (Relaxed/Equality)
    by_vis: Dict[str, Dict[str, List[Path]]] = {}
    for fp in json_files:
        vis = parse_visibility_from_name(fp)
        var = classify_variant(fp)
        by_vis.setdefault(vis, {}).setdefault(var, []).append(fp)

    # For each vis, overlay Relaxed and Equality in one axes
    for vis in sorted(by_vis.keys(), key=lambda s: float(s) if s != "unknown" else -1.0, reverse=True):
        variants = by_vis[vis]

        # choose one file per variant (if multiple exist, pick last alphabetically = usually latest)
        chosen: Dict[str, Path] = {}
        for var in ["Relaxed", "Equality"]:
            fps = variants.get(var, [])
            if fps:
                chosen[var] = sorted(fps)[-1]

        # if neither exists, skip
        if not chosen:
            continue

        fig, ax = plt.subplots(figsize=(6.3, 6.0))

        # Styling: color by etaA; linestyle by variant
        linestyles = {"Relaxed": "-", "Equality": "--"}
        color = {"Relaxed": "#77E9AA", "Equality": "#FF6F61"}

        # We want consistent colors for the same etaA across variants.
        # Use a stable order of etaA keys: union of both variants' top keys.
        etaA_union: List[str] = []
        for var, fp in chosen.items():
            kr = load_born_json(fp)
            for k in pick_etaA_keys(kr, max_curves=1):
                if k not in etaA_union:
                    etaA_union.append(k)

        # plot in this order so legend is stable
        for etaA_k in etaA_union:
            for var in ["Relaxed", "Equality"]:
                if var not in chosen:
                    continue
                fp = chosen[var]
                kr = load_born_json(fp)
                pts = kr.get(etaA_k, [])
                if not isinstance(pts, list):
                    continue
                xy = extract_xy(pts, threshold=0.6)
                if not xy:
                    continue
                x, y = zip(*xy)
                ax.plot(
                    x, y,
                    marker="o",
                    linestyle=linestyles[var],
                    label=rf"{var}, $\eta_A$={gfmt(etaA_k,3)}",
                    color = color[var]
                )

        ax.axhline(0.0, linewidth=1, linestyle="dashed", alpha=0.7)
        ax.invert_xaxis()
        ax.set_xlabel(r"$\eta_{B}$ (long-path detection efficiency)")
        ax.set_ylabel(r"Key rate $r$")
        ax.grid(True, linestyle="dotted", alpha=0.7)
        ax.set_title(rf"Born SDP key rate vs $\eta_B$ (Visibility $\nu$={gfmt(vis,3)})")
        ax.legend(fontsize=9)

        safe_vis = safe_vis_for_filename(vis)
        # out_png = out_dir / f"vis{safe_vis}_BORN_relaxed_vs_equality_keyrate_vs_etaB.png"
        out_svg = out_dir / f"vis{safe_vis}_BORN_relaxed_vs_equality_keyrate_vs_etaB.svg"
        # fig.savefig(out_png, dpi=220)
        fig.savefig(out_svg)
        plt.close(fig)

        # print(f"[saved] {out_png}")
        print(f"[saved] {out_svg}")

if __name__ == "__main__":
    main()