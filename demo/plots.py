"""Figures de la démo.
- timeline_figure / bars_figure : plotly (interactif dans Gradio : zoom à la souris, déplacement, survol) ;
- save_timeline_png : même timeline en matplotlib (backend Agg, D-06) pour figures/demo/.
Couleurs : palette Okabe-Ito (lisible par les daltoniens), une couleur fixe par locuteur (ordre de temps de parole
décroissant) ; au-delà de 8 locuteurs, les couleurs se répètent avec des hachures différentes."""
import plotly.graph_objects as go

from demo.core import fmt_time

OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#000000"]
PATTERNS = ["", "/", "x", ".", "\\", "+", "-", "|"]
OVERLAP_LABEL = "overlap"
OVERLAP_COLOR = "#7f7f7f"
REF_COLOR = "#8b93a1"
# style commun des figures plotly : fond transparent (suit le thème clair ou sombre de Gradio), texte gris moyen
# lisible sur fond clair comme sur fond sombre, grille discrète, polices système (aucune police téléchargée)
FONT = dict(family="ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica Neue, Arial, sans-serif",
            size=13, color="#8b93a1")
GRID = "rgba(139, 147, 161, 0.22)"


def _style_layout(fig, title="", **kw):
    fig.update_layout(font=FONT, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      hoverlabel=dict(font=dict(family=FONT["family"], size=12)),
                      title=dict(text=title, font=dict(size=15), x=0.01, xanchor="left"), **kw)
    fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False, linecolor=GRID)
    fig.update_yaxes(showgrid=False, zeroline=False, linecolor=GRID)
    return fig


def speaker_order(annotation):
    """Locuteurs triés par temps de parole décroissant (le premier = le plus bavard)."""
    return sorted(annotation.labels(), key=lambda lab: -annotation.label_duration(lab))


def style(i):
    """(couleur, hachure) du i-ème locuteur."""
    return OKABE_ITO[i % len(OKABE_ITO)], PATTERNS[(i // len(OKABE_ITO)) % len(PATTERNS)]


def colors_for(annotation):
    return {lab: style(i) for i, lab in enumerate(speaker_order(annotation))}


def _bar_trace(name, segs, color, pattern, row_name):
    starts = [s.start for s in segs]
    durs = [s.duration for s in segs]
    text = [f"{name}<br>{fmt_time(s.start)} → {fmt_time(s.end)} ({s.duration:.2f} s)" for s in segs]
    return go.Bar(y=[row_name] * len(segs), x=durs, base=starts, orientation="h", name=name,
                  marker=dict(color=color, pattern=dict(shape=pattern, fgcolor="white" if pattern else None),
                              line=dict(width=0), cornerradius=3),
                  hovertext=text, hoverinfo="text", width=0.8)


def timeline_figure(annotation, duration, title="", reference=None, row_prefix=""):
    """Une ligne par locuteur (le plus bavard en haut) + une ligne 'overlap' (zones où 2 locuteurs ou plus
    parlent). Axe en secondes, survol en m:ss. reference : Annotation optionnelle, affichée en plus (lignes 'réf.')."""
    fig = go.Figure()
    rows = []
    order = speaker_order(annotation)
    for i, lab in enumerate(order):
        color, pattern = style(i)
        segs = list(annotation.label_timeline(lab, copy=False))
        row = f"{row_prefix}{lab}"
        rows.append(row)
        fig.add_trace(_bar_trace(str(lab), segs, color, pattern, row))
    ov = list(annotation.get_overlap())
    if ov:
        row = f"{row_prefix}{OVERLAP_LABEL}"
        rows.append(row)
        fig.add_trace(_bar_trace(OVERLAP_LABEL, ov, OVERLAP_COLOR, "/", row))
    if reference is not None:
        for i, lab in enumerate(speaker_order(reference)):
            segs = list(reference.label_timeline(lab, copy=False))
            row = f"réf. {lab}"
            rows.append(row)
            fig.add_trace(_bar_trace(f"réf. {lab}", segs, REF_COLOR, "", row))
    n = max(len(rows), 1)
    _style_layout(
        fig, title=title, barmode="overlay", showlegend=False, height=140 + 34 * n,
        margin=dict(l=10, r=10, t=48, b=40), bargap=0.25,
        xaxis=dict(title="temps (s)", range=[0, duration],
                   rangeslider=dict(visible=duration > 120, thickness=0.06, bgcolor="rgba(139,147,161,0.08)")),
        yaxis=dict(categoryorder="array", categoryarray=list(reversed(rows)), automargin=True),
        dragmode="zoom")
    if not rows:
        fig.add_annotation(text="aucune parole détectée", x=duration / 2, y=0.5, showarrow=False,
                           font=dict(size=15))
    return fig


def bars_figure(annotation, stats_rows):
    """Temps de parole par locuteur (barres horizontales, mêmes couleurs que la timeline)."""
    cols = colors_for(annotation)
    fig = go.Figure()
    for r in stats_rows:
        color, pattern = cols[r["label"]]
        fig.add_trace(go.Bar(x=[r["speech_s"]], y=[str(r["label"])], orientation="h",
                             marker=dict(color=color, pattern=dict(shape=pattern)),
                             text=[f"{r['speech_s']:.0f} s · {r['pct_speech']:.0f} %"], textposition="outside",
                             cliponaxis=False, textfont=dict(color=FONT["color"]),
                             hoverinfo="text", hovertext=[f"{r['label']} : {fmt_time(r['speech_s'])}"]))
    _style_layout(fig, showlegend=False, height=120 + 34 * max(len(stats_rows), 1),
                  margin=dict(l=10, r=40, t=40, b=40), xaxis_title="temps de parole (s)", bargap=0.3,
                  yaxis=dict(autorange="reversed", automargin=True), title="Répartition du temps de parole")
    return fig


def save_timeline_png(annotation, duration, path, title="", reference=None):
    """Timeline matplotlib (Agg) pour les figures du rapport (figures/demo/). Jamais plt.show()."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = speaker_order(annotation)
    rows = [(str(lab), list(annotation.label_timeline(lab, copy=False)), *style(i)) for i, lab in enumerate(order)]
    ov = list(annotation.get_overlap())
    if ov:
        rows.append((OVERLAP_LABEL, ov, OVERLAP_COLOR, "/"))
    if reference is not None:
        rows += [(f"réf. {lab}", list(reference.label_timeline(lab, copy=False)), REF_COLOR, "")
                 for lab in speaker_order(reference)]
    fig, ax = plt.subplots(figsize=(12, 1 + 0.4 * max(len(rows), 1)))
    for k, (name, segs, color, pattern) in enumerate(rows):
        y = len(rows) - 1 - k
        ax.broken_barh([(s.start, s.duration) for s in segs], (y - 0.4, 0.8), facecolors=color,
                       hatch=pattern * 2 if pattern else None, edgecolor="white" if pattern else color, linewidth=0)
    ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)])
    ax.set_xlim(0, duration)
    ax.set_xlabel("temps (s)")
    ax.set_title(title)
    ax.grid(axis="x", color="#e5e5e5")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
