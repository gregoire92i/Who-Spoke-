"""Chiffres de la section « observations » de guide_figures.md (lus dans explore_runs_sample90.csv)."""
import pandas as pd, numpy as np
E = pd.read_csv("results/diag_k_oracle_vs_eigengap/explore_runs_sample90.csv")
def mic(g, c): return 100 * g[c].sum() / g.total_s.sum()
def der(g): return 100 * (g.FA_s + g.miss_s + g.conf_s).sum() / g.total_s.sum()
b = E[(E.p_neighbors == 0.1) & (E.seed == 42)]
for v in ["oracle", "energy", "stack_devtrain"]:
    for m in ["eigengap", "oracle", "eigengap_merged", "oracle_merged"]:
        g = b[(b.vad == v) & (b.k_mode == m)]
        print(f"{v:15s} {m:16s} DER {der(g):.2f} forgiving {100*g.f_err_s.sum()/g.f_total_s.sum():.2f} JER {100*g.jer_err.sum()/g.jer_count.sum():.2f} "
              f"FA {mic(g,'FA_s'):.2f} miss {mic(g,'miss_s'):.2f} conf {mic(g,'conf_s'):.2f} | impur {mic(g,'cf_impurity_s'):.2f} mismap {mic(g,'cf_mismapped_s'):.2f} unmap {mic(g,'cf_unmapped_s'):.2f} "
              f"| overlap {g.cf_overlap_s.sum():.0f}s bnd<0.5 {g['cf_bnd_0-0.5s_s'].sum():.0f}s >2s {g['cf_bnd_>2s_s'].sum():.0f}s disputed {g.cf_disputed_s.sum():.0f}s "
              f"| ref non appariés {g.n_ref_unmapped.sum()} ({g.time_ref_unmapped_s.sum():.0f} s)")
o = E[(E.vad == "oracle") & (E.p_neighbors == 0.1)]
for sd in sorted(o.seed.unique()):
    g = o[o.seed == sd]
    print(f"seed {sd}: DER eig {der(g[g.k_mode=='eigengap']):.2f} oracle {der(g[g.k_mode=='oracle']):.2f}")
q = E[(E.vad == "oracle") & (E.seed == 42)]
for p in sorted(q.p_neighbors.unique()):
    g = q[q.p_neighbors == p]
    print(f"p {p}: DER eig {der(g[g.k_mode=='eigengap']):.2f} oracle {der(g[g.k_mode=='oracle']):.2f}  conf eig {mic(g[g.k_mode=='eigengap'],'conf_s'):.2f} oracle {mic(g[g.k_mode=='oracle'],'conf_s'):.2f}")
pv = o[o.k_mode.isin(["eigengap","oracle"])].pivot_table(index=["uri","seed"], columns="k_mode", values="DER").reset_index()
pv["d"] = pv.oracle - pv.eigengap
r = pv.groupby("uri").d.agg(lambda s: s.max() - s.min())
print(f"amplitude max-min de ΔDER sur 5 seeds : médiane {100*r.median():.2f} pts, max {100*r.max():.2f} pts ({r.idxmax()}), fichiers > 1 pt : {(r>0.01).sum()}")
