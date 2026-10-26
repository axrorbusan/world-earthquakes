"""
World earthquakes 1975–2025 — a simple look at the USGS catalog (M ≥ 4.5)

Questions
  1. Where do large earthquakes happen?      → map + distance to plate boundaries
  2. How much rarer is each step in size?     → Gutenberg–Richter law, b-value (Aki–Utsu MLE)
  3. Are big quakes becoming more frequent?   → yearly M7+ counts vs a Poisson model
  4. How deep are they?                        → depth distribution

Input : data/raw/usgs_m45_1975_2025.csv.gz (USGS FDSN event service)
        data/raw/ne_110m_land.geojson (Natural Earth), data/raw/PB2002_boundaries.json (Bird, 2003)
Output: output/*.png, output/results.json
Run   : python analyze.py
"""
import glob
import json
import os

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.neighbors import BallTree

matplotlib.use("Agg")
HERE = os.path.dirname(os.path.abspath(__file__))
P = lambda *a: os.path.join(HERE, *a)
RNG = np.random.default_rng(0)

INK, MUTED, ACCENT = "#1F2933", "#6B7280", "#C2410C"
EARTH_R = 6371.0


def style():
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.titleweight": "bold", "axes.titlesize": 13, "font.size": 10})


def load():
    df = pd.read_csv(glob.glob(P("data", "raw", "usgs_m45_*.csv*"))[0])
    df = df[df["type"] == "earthquake"].copy()
    df["time"] = pd.to_datetime(df["time"], utc=True, format="ISO8601")
    df["year"] = df["time"].dt.year
    return df.dropna(subset=["mag", "latitude", "longitude", "depth"])


def b_value(m, mc, dm=0.1):
    """Aki (1965) / Utsu maximum-likelihood b-value with binning correction."""
    m = m[m >= mc - 1e-9]
    b = np.log10(np.e) / (m.mean() - (mc - dm / 2))
    return b, len(m)


def mc_maxc(m, dm=0.1):
    """Magnitude of completeness: maximum curvature (mode of the binned histogram) + 0.2."""
    bins = np.round(m / dm) * dm
    return float(bins.value_counts().idxmax()) + 0.2


def boundary_distance_km(df, pb):
    pts = []
    for geom in pb.geometry:
        lines = getattr(geom, "geoms", [geom])
        for ln in lines:
            # densify every ~0.1° so the nearest vertex ≈ nearest point on the line
            ln = ln.segmentize(0.1)
            pts.extend(ln.coords)
    pts = np.radians(np.array(pts)[:, ::-1])          # (lat, lon)
    tree = BallTree(pts, metric="haversine")
    q = np.radians(df[["latitude", "longitude"]].values)
    d, _ = tree.query(q, k=1)
    return d[:, 0] * EARTH_R


def main():
    style()
    os.makedirs(P("output"), exist_ok=True)
    df = load()
    land = gpd.read_file(P("data", "raw", "ne_110m_land.geojson"))
    pb = gpd.read_file(P("data", "raw", "PB2002_boundaries.json"))
    res = {"n_events_m45": int(len(df)), "years": [int(df.year.min()), int(df.year.max())]}

    # ---------------------------------------------------------------- 1. where
    df["dist_boundary_km"] = boundary_distance_km(df, pb)
    big = df[df["mag"] >= 6]
    res["share_within_100km_of_plate_boundary"] = {
        "m4.5+": float((df.dist_boundary_km <= 100).mean()),
        "m6+": float((big.dist_boundary_km <= 100).mean())}
    # Null comparison: random points in the same latitude band of the ocean+land surface
    lat = np.degrees(np.arcsin(RNG.uniform(-1, 1, 50000)))      # uniform on the sphere
    lon = RNG.uniform(-180, 180, 50000)
    rnd = pd.DataFrame({"latitude": lat, "longitude": lon})
    res["share_within_100km_random_points"] = float((boundary_distance_km(rnd, pb) <= 100).mean())

    recent = df[(df.year >= 2000) & (df.mag >= 5)].sort_values("mag")
    fig, ax = plt.subplots(figsize=(13, 6.6))
    ax.set_facecolor("#EEF3F7")
    land.plot(ax=ax, color="#F7F7F5", edgecolor="#C9CED6", linewidth=0.4)
    pb.plot(ax=ax, color="#9CA3AF", linewidth=0.6)
    sc = ax.scatter(recent.longitude, recent.latitude, c=recent.depth, cmap="plasma_r",
                    norm=matplotlib.colors.LogNorm(5, 700), s=(recent.mag - 4.6) ** 3 * 2.2,
                    alpha=0.75, linewidth=0)
    cb = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.01)
    cb.set_label("Depth (km, log scale)")
    ax.set_xlim(-180, 180); ax.set_ylim(-75, 85)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"Every M5+ earthquake, 2000–2025 ({len(recent):,} events). Grey lines = plate boundaries",
                 loc="left")
    ax.text(0, -0.03, "Dot size = magnitude. Source: USGS ComCat; plate boundaries: Bird (2003) PB2002; land: Natural Earth.",
            transform=ax.transAxes, fontsize=8.5, color=MUTED, va="top")
    fig.savefig(P("output", "fig1_world_map.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------------- 2. Gutenberg–Richter
    # The catalog mixes magnitude scales: most M4.5–5.5 events are body-wave mb, larger ones
    # moment magnitude Mw. mb saturates for big quakes, so a single fit across both is biased.
    # Main estimate: Mw-type magnitudes only, M ≥ 5.5 (where Mw is reported routinely).
    gr = df[df.year >= 2000]                       # modern network, more homogeneous
    mw = gr[gr["magType"].str.lower().str.startswith("mw")]
    mc_mw = 5.5
    b, n = b_value(mw["mag"].values, mc_mw)
    boots = [b_value(RNG.choice(mw["mag"].values, len(mw)), mc_mw)[0] for _ in range(500)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    mc_all = mc_maxc(gr["mag"])
    b_all, n_all = b_value(gr["mag"].values, mc_all)
    mb_share = gr.groupby(gr["mag"].round(0))["magType"].apply(
        lambda s: float(s.str.lower().str.startswith("mb").mean()))
    res["gutenberg_richter"] = {
        "period": "2000-2025",
        "main_estimate": {"magnitudes": "Mw only", "mc": mc_mw, "b": float(b),
                          "b_ci95": [float(lo), float(hi)], "n": int(n)},
        "naive_all_types": {"mc_max_curvature": mc_all, "b": float(b_all), "n": int(n_all)},
        "share_mb_by_magnitude": {str(k): v for k, v in mb_share.items()},
        "interpretation": f"each +1 magnitude is about {10 ** b:.0f}x rarer"}
    mags = np.round(np.arange(4.5, gr.mag.max() + 0.1, 0.1), 1)
    cum = np.array([(gr.mag >= m - 1e-9).sum() for m in mags])
    a = np.log10(n) + b * mc_mw
    a_all = np.log10(n_all) + b_all * mc_all
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(mags, cum / 26, "o", color=INK, ms=3.5, label="Observed (events per year with magnitude ≥ M)")
    fit_all = mags[mags >= mc_all]
    ax.semilogy(fit_all, 10 ** (a_all - b_all * fit_all) / 26, color="#9CA3AF", lw=1.3, ls="--",
                label=f"Naive fit, all magnitude types: b = {b_all:.2f}")
    fit_m = mags[mags >= mc_mw]
    ax.semilogy(fit_m, 10 ** (a - b * fit_m) / 26, color=ACCENT, lw=2.2,
                label=f"Fit on moment magnitudes (Mw ≥ 5.5): b = {b:.2f} (95% CI {lo:.2f}–{hi:.2f})")
    ax.axvline(mc_mw, color=MUTED, ls=":", lw=1)
    for m_ in (6, 7, 8):
        k = (gr.mag >= m_).sum() / 26
        ax.annotate(f"M{m_}+: {k:.1f}/yr" if k >= 1 else f"M{m_}+: 1 every {1/k:.1f} yr",
                    (m_, k), xytext=(8, 6), textcoords="offset points", fontsize=8.5, color=INK)
    ax.set_xlabel("Magnitude M"); ax.set_ylabel("Earthquakes per year (≥ M)")
    ax.set_title(f"Each step up in magnitude is about {10 ** b:.0f}× rarer", loc="left")
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(color="#E5E7EB", which="both", lw=0.5)
    fig.savefig(P("output", "fig2_gutenberg_richter.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------------- 3. are big quakes increasing?
    import statsmodels.api as sm

    def poisson_trend(counts, years):
        x = sm.add_constant(years - years.mean())
        m = sm.GLM(counts, x, family=sm.families.Poisson()).fit()
        return float((np.exp(m.params[1] * 10) - 1) * 100), float(m.pvalues[1])

    yrs = np.arange(1975, 2026)
    c7 = df[df.mag >= 7].groupby("year").size().reindex(yrs, fill_value=0)
    modern = c7[c7.index >= 1990]                    # moment magnitude (Mw) era
    mean7, var7 = modern.mean(), modern.var(ddof=1)
    disp = var7 / mean7
    p_disp = 1 - stats.chi2.cdf(disp * (len(modern) - 1), len(modern) - 1)
    t_all, p_all = poisson_trend(c7.values, yrs)
    t_mod, p_mod = poisson_trend(modern.values, modern.index.values)
    m7 = df[df.mag >= 7]
    ms_share = (m7.assign(dec=m7.year // 10 * 10)
                  .groupby("dec")["magType"].apply(lambda s: float(s.str.lower().isin(["ms", "mb"]).mean())))
    res["m7_per_year"] = {
        "mean_since_1990": float(mean7), "variance_since_1990": float(var7),
        "dispersion_index": float(disp), "p_overdispersion": float(p_disp),
        "trend_1975_2025_pct_per_decade": t_all, "trend_1975_2025_p": p_all,
        "trend_1990_2025_pct_per_decade": t_mod, "trend_1990_2025_p": p_mod,
        "share_ms_or_mb_magnitudes_by_decade": {int(k): v for k, v in ms_share.items()},
        "max_year": int(c7.idxmax()), "max_count": int(c7.max())}
    pois = stats.poisson(mean7)
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.fill_between([1989.5, 2025.5], pois.ppf(0.025), pois.ppf(0.975), color=ACCENT, alpha=0.08,
                    lw=0, zorder=0, label="95% range if quakes were purely random (Poisson)")
    ax.bar(yrs, c7.values, color=["#D1D5DB" if y < 1990 else "#6B7280" for y in yrs], width=0.8, zorder=2)
    ax.hlines(mean7, 1989.5, 2025.5, color=ACCENT, lw=2, zorder=3,
              label=f"1990–2025 average: {mean7:.1f} per year")
    ax.set_ylim(0, 31)
    ax.text(1982, 29.5, "before 1990: older magnitude\nscales (Ms) undercount M7+",
            ha="center", va="top", fontsize=8.5, color=MUTED)
    ax.set_title("M7+ earthquakes per year: steady since 1990", loc="left")
    ax.set_ylabel("Count")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.grid(axis="y", color="#E5E7EB")
    fig.savefig(P("output", "fig3_m7_per_year.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------------- 4. depth
    d = df["depth"].clip(lower=0)
    res["depth_share"] = {"shallow_<70km": float((d < 70).mean()),
                          "intermediate_70-300km": float(((d >= 70) & (d < 300)).mean()),
                          "deep_>=300km": float((d >= 300).mean()),
                          "deepest_km": float(d.max())}
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.hist(d, bins=np.arange(0, 720, 10), color=ACCENT)
    ax.set_yscale("log")
    ax.set_xlabel("Depth (km)"); ax.set_ylabel("Number of earthquakes (log)")
    ax.set_title("Most quakes are shallow, but some happen 600+ km down", loc="left")
    ax.axvspan(70, 300, color="#E5E7EB", alpha=0.5, lw=0)
    ax.text(185, ax.get_ylim()[1] * 0.3, "intermediate", ha="center", color=MUTED, fontsize=8.5)
    ax.text(500, ax.get_ylim()[1] * 0.3, "deep (subducting slabs)", ha="center", color=MUTED, fontsize=8.5)
    share10 = float((df["depth"] == 10).mean())
    res["depth_share"]["exactly_10km_default"] = share10
    ax.annotate(f"spike at 10 km = default depth\nwhen depth can't be resolved ({share10:.0%})",
                (16, (d.between(10, 19.99)).sum()), xytext=(60, -45), textcoords="offset points",
                fontsize=8.5, color=MUTED, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
    fig.savefig(P("output", "fig4_depth.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------------- largest events
    top = df.nlargest(10, "mag")[["time", "mag", "depth", "place"]]
    top["time"] = top["time"].dt.strftime("%Y-%m-%d")
    res["largest"] = top.to_dict(orient="records")
    top.to_csv(P("output", "largest_10.csv"), index=False)

    with open(P("output", "results.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
