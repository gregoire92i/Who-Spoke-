"""Résumés des CSV du banc RTF + figure RTF selon la durée."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

D = sys.argv[1]       # dossier des CSV
FIG = sys.argv[2]     # chemin de la figure
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)


def summ(df, keys):
    first = df[df.rep == 1].groupby(keys).agg(first_load_s=("load_s", "first"), first_diar_s=("diarize_s", "first"),
                                              first_RTF=("RTF_affiche", "first"))
    rest = df[df.rep > 1].groupby(keys).agg(n_rest=("rep", "size"), dur=("duration_s", "first"),
                                            load_med=("load_s", "median"), diar_med=("diarize_s", "median"),
                                            diar_min=("diarize_s", "min"), diar_max=("diarize_s", "max"),
                                            total_med=("total_s", "median"), RTF_aff_med=("RTF_affiche", "median"),
                                            RTF_aff_min=("RTF_affiche", "min"), RTF_aff_max=("RTF_affiche", "max"),
                                            RTF_tot_med=("RTF_total", "median"), startup_s=("startup_s", "first"),
                                            k_hyp=("k_hyp", "first"))
    return first.join(rest).reset_index()


f = pd.read_csv(f"{D}/formats.csv")
fs = summ(f, ["pipeline", "tag", "format"])
fs.to_csv(f"{D}/formats_summary.csv", index=False, float_format="%.4f")
print("=== FORMATS\n", fs.round(4).to_string())

r = pd.read_csv(f"{D}/repeat.csv")
print("=== REPEAT (chaque valeur)")
for (p, fm), g in r.groupby(["pipeline", "format"]):
    print(p, fm, "diar:", " ".join(f"{v:.3f}" for v in g.diarize_s), "| RTF:", " ".join(f"{v:.4f}" for v in g.RTF_affiche),
          "| load:", " ".join(f"{v:.3f}" for v in g.load_s))
    rest = g[g.rep > 1]
    # tendance : pente de RTF vs rang sur les appels 2..10 et corrélation de Spearman
    from scipy.stats import spearmanr
    rho, pv = spearmanr(rest.rep, rest.RTF_affiche)
    slope = np.polyfit(rest.rep, rest.RTF_affiche, 1)[0]
    print(f"   appels 2-10 : med {rest.RTF_affiche.median():.4f} min {rest.RTF_affiche.min():.4f} "
          f"max {rest.RTF_affiche.max():.4f} ; pente {slope:+.5f}/appel ; Spearman rho {rho:+.2f} p {pv:.2f}")
rs = summ(r, ["pipeline", "tag", "format"])
rs.to_csv(f"{D}/repeat_summary.csv", index=False, float_format="%.4f")

d = pd.read_csv(f"{D}/durations.csv")
ds = summ(d, ["pipeline", "tag"]).sort_values(["pipeline", "dur"])
ds.to_csv(f"{D}/durations_summary.csv", index=False, float_format="%.4f")
print("=== DURATIONS\n", ds.round(4).to_string())
fits = []
for p, g in d[d.rep > 1].groupby("pipeline"):
    for name, sub in (("extraits ipqqq (5-300 s)", g[g.tag != "ldnro"]), ("tout (ldnro compris)", g)):
        b, a = np.polyfit(sub.duration_s, sub.diarize_s, 1)
        print(f"fit {p} {name} : diarize_s = {a:.3f} s + {b:.5f} x durée")
        fits.append({"pipeline": p, "points": name, "cout_fixe_s": a, "cout_par_s_audio": b, "n": len(sub)})
pd.DataFrame(fits).to_csv(f"{D}/durations_fit.csv", index=False, float_format="%.5f")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
colors = {"baseline": "#0072B2", "pyannote": "#D55E00"}
for p, g in d.groupby("pipeline"):
    rest = g[g.rep > 1]
    first = g[g.rep == 1]
    med = rest.groupby("duration_s").RTF_affiche.median()
    ax[0].plot(med.index, med.values, "-o", color=colors[p], label=f"{p} : médiane des appels 2+")
    ax[0].scatter(rest.duration_s, rest.RTF_affiche, s=10, color=colors[p], alpha=.35)
    ax[0].scatter(first.duration_s, first.RTF_affiche, marker="x", s=40, color=colors[p], label=f"{p} : 1er appel")
    medd = rest.groupby("duration_s").diarize_s.median()
    ax[1].plot(medd.index, medd.values, "-o", color=colors[p], label=p)
    fp = [x for x in fits if x["pipeline"] == p and x["points"].startswith("extraits")][0]
    xs = np.linspace(1 if p == "baseline" else 30, 1100, 50)
    ax[1].plot(xs, fp["cout_fixe_s"] + fp["cout_par_s_audio"] * xs, ":", color=colors[p],
               label=f"ajustement 5-300 s : {fp['cout_fixe_s']:.2f} s + {fp['cout_par_s_audio']:.4f}·durée")
for a in ax:
    a.set_xscale("log")
    a.set_xlabel("durée audio (s, échelle log)")
    a.grid(alpha=.3, which="both")
    a.legend(fontsize=8)
ax[0].set_yscale("log")
ax[0].set_ylabel("RTF affiché (core.diarize / durée), log")
ax[0].set_title("RTF selon la durée (ipqqq 5-300 s, ldnro 1097 s ; dev)")
ax[1].set_yscale("log")
ax[1].set_ylabel("temps core.diarize (s), log")
ax[1].set_title("Temps de traitement : coût fixe + proportionnel")
fig.text(0.01, 0.01, "i5-11400H, RTX 3050 Laptop (GPU en mode idle : RTF pyannote non représentatif), WSL2 ; "
         "machine pas forcément au repos", fontsize=7)
fig.tight_layout(rect=(0, 0.03, 1, 1))
os.makedirs(os.path.dirname(FIG), exist_ok=True)
fig.savefig(FIG, dpi=130)
print("figure :", FIG)
