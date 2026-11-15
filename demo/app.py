"""Interface Gradio de la démo. Lancement (depuis la racine du dépôt, .venv activé) :

    python -m demo.app

Application locale uniquement : http://127.0.0.1:7860 (WHO_SPOKE_PORT pour changer le port), jamais share=True,
analytics Gradio désactivées, fichiers temporaires dans demo/tmp/ (ignoré par git). Un fichier traité à la fois.
Apparence : thème Gradio à polices système (aucune police téléchargée) + demo/style.css."""
import html
import os
import time

import demo  # fixe HF_HUB_OFFLINE, GRADIO_* et MPLBACKEND avant les autres imports
import gradio as gr
import numpy as np
import pandas as pd

from demo import ROOT, TMP_DIR, core, plots

LABELS = {"baseline": "Baseline (MFCC + NME-SC, best_vad alldev)",
          "pyannote": "pyannote speaker-diarization-community-1"}
SHORT = {"baseline": "Baseline", "pyannote": "pyannote"}
FORMATS = ", ".join(core.AUDIO_EXTENSIONS)
DEFAULT_PORT = 7860
CSS_PATH = ROOT / "demo" / "style.css"

# polices système uniquement : aucune police web n'est chargée (démo hors ligne)
SYSTEM_FONTS = ["ui-sans-serif", "system-ui", "-apple-system", "Segoe UI", "Roboto", "Helvetica Neue", "Arial",
                "sans-serif"]
MONO_FONTS = ["ui-monospace", "SFMono-Regular", "Menlo", "Consolas", "Liberation Mono", "monospace"]


def make_theme():
    """Thème Soft, couleur principale bleue (#0072B2 = 1re couleur Okabe-Ito), coins arrondis, polices système."""
    return gr.themes.Soft(primary_hue="sky", secondary_hue="amber", neutral_hue="slate", radius_size="lg",
                          font=SYSTEM_FONTS, font_mono=MONO_FONTS).set(
        color_accent="#0072B2", color_accent_soft="rgba(0, 114, 178, 0.12)",
        button_primary_background_fill="#0072B2", button_primary_background_fill_hover="#005a8c",
        button_primary_background_fill_dark="#2a8fc7", button_primary_background_fill_hover_dark="#0072B2",
        button_primary_text_color="#ffffff", button_primary_text_color_dark="#ffffff",
        block_title_text_weight="600", block_label_text_weight="600")


# ------------------------------------------------------------------ mise en forme
def _section(step, title, hint=""):
    h = f"<div class='ws-hint'>{hint}</div>" if hint else ""
    return f"<div class='ws-section'><span class='ws-step'>{step}</span>{title}</div>{h}"


def _header(engines):
    badges = "".join(
        f"<span class='ws-badge'><span class='ws-dot {'ok' if ok else 'ko'}'></span>{SHORT.get(p, p)} : "
        f"{html.escape(msg)}</span>" for p, (ok, msg) in engines.status.items())
    return ("<div class='ws-header'><h1>Who Spoke?</h1><div class='ws-sub'>Qui parle, et quand ?</div>"
            "<div class='ws-desc'>Charger un audio, lancer la diarisation, lire la timeline des locuteurs, écouter les "
            "segments, exporter le RTTM. Application locale : l'audio ne quitte pas cette machine.</div>"
            f"<div class='ws-badges'>{badges}</div></div>")


EMPTY = ("<div class='ws-empty'>Chargez un fichier audio (ou enregistrez au micro), choisissez la pipeline, puis "
         "cliquez sur <b>Lancer l'analyse</b>.</div>")


def _card(k, v, s=""):
    return f"<div class='ws-card'><div class='ws-k'>{k}</div><div class='ws-v'>{v}</div><div class='ws-s'>{s}</div></div>"


def _legend(ann):
    cols = plots.colors_for(ann)
    return "<div class='ws-legend'>" + "".join(
        f"<span class='ws-chip'><span class='ws-swatch' style='background:{c}'></span>{html.escape(str(lab))}</span>"
        for lab, (c, _) in cols.items()) + "</div>"


def _info_html(infos, summary, ann):
    """Cartes de chiffres clés + légende des locuteurs + paramètres (repliés)."""
    p = infos["params"]
    cards = [_card("Durée audio", core.fmt_time(infos["duration_s"]), f"{infos['duration_s']:.1f} s"),
             _card("Locuteurs", infos["k_hyp"], "détectés"),
             _card("Parole", f"{summary['speech_s']:.0f} s", f"overlap {summary['overlap_s']:.1f} s"),
             _card("Traitement", f"{infos['processing_time_s']:.1f} s", "lecture du fichier comprise"),
             _card("RTF", f"{infos['RTF']:.3f}", "temps / durée audio")]
    if "device" in p:
        cards.append(_card("Appareil", html.escape(str(p["device"])), "pyannote"))
    out = (f"<div class='ws-title'>Résultats : <b>{html.escape(LABELS[infos['pipeline']])}</b></div>"
           f"<div class='ws-cards'>{''.join(cards)}</div>")
    if infos["k_hyp"] == 0:
        out += ("<div class='ws-warn'><b>Aucune parole détectée</b> : la timeline est vide et le RTTM exporté est vide "
                "(fichier valide).</div>")
    else:
        out += _legend(ann)
    params = " · ".join(f"<code>{html.escape(str(k))}</code> = {html.escape(str(v))}"
                        for k, v in p.items() if k != "device")
    out += f"<details class='ws-params'><summary>Paramètres utilisés</summary><p>{params}</p></details>"
    return out


def _text_color(hex_color):
    """Noir ou blanc selon la luminance du fond (lisibilité du texte sur la couleur du locuteur)."""
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return "#000000" if 0.299 * r + 0.587 * g + 0.114 * b > 150 else "#ffffff"


def _styled(df, ann, col):
    """Colonne `col` colorée avec la couleur du locuteur (mêmes couleurs que la timeline et les barres)."""
    cols = plots.colors_for(ann)

    def color(v):
        c = cols.get(v, cols.get(str(v)))
        if c is None:
            return ""
        return f"background-color: {c[0]}; color: {_text_color(c[0])}; font-weight: 600"
    return df.style.map(color, subset=[col])


STATS_HEADERS = ["locuteur", "temps (s)", "temps (m:ss)", "% de la parole", "% de l'audio", "segments"]
SEG_HEADERS = ["début (s)", "fin (s)", "durée (s)", "début (m:ss)", "locuteur"]


def _stats_df(rows, ann):
    df = pd.DataFrame([[r["label"], round(r["speech_s"], 1), core.fmt_time(r["speech_s"]), round(r["pct_speech"], 1),
                        round(r["pct_audio"], 1), r["n_segments"]] for r in rows], columns=STATS_HEADERS)
    return _styled(df, ann, "locuteur").format(precision=1, subset=["temps (s)", "% de la parole", "% de l'audio"])


def _seg_df(ann):
    df = pd.DataFrame([[round(s, 3), round(e, 3), round(d, 2), core.fmt_time(s), str(lab)]
                       for s, e, d, lab in core.segments_table(ann)], columns=SEG_HEADERS)
    return _styled(df, ann, "locuteur").format(precision=3, subset=["début (s)", "fin (s)"]) \
        .format(precision=2, subset=["durée (s)"])


def _metrics_html(state):
    ref = state.get("ref")
    if ref is None:
        return ""
    rows = ""
    for p, (ann, _) in state["results"].items():
        m = core.evaluate(ref, ann, state["audio"]["duration"])
        rows += (f"<tr><td>{SHORT[p]}</td><td><b>{m['DER']:.2%}</b></td><td>{m['DER_forgiving']:.2%}</td>"
                 f"<td>{m['JER']:.2%}</td><td>{m['false_alarm']:.2%}</td><td>{m['missed']:.2%}</td>"
                 f"<td>{m['confusion']:.2%}</td></tr>")
    return ("<div class='ws-metrics'><div class='ws-title'><b>Comparaison à la référence fournie</b></div>"
            f"<div class='ws-note'>Ce fichier uniquement ; collar 0, UEM = [0, durée de l'audio], overlap compris ; "
            f"{len(ref.labels())} locuteurs de référence, {ref.get_timeline().support().duration():.1f} s de parole.</div>"
            "<table><tr><th>pipeline</th><th>DER</th><th>DER forgiving</th><th>JER</th><th>FA</th><th>missed</th>"
            f"<th>confusion</th></tr>{rows}</table></div>")


def _current(state, view):
    """Annotation affichée pour la pipeline `view`, après renommage éventuel."""
    ann, infos = state["results"][view]
    return core.rename(ann, state["names"].get(view, {})), infos


def _view_outputs(state, view):
    """Tout ce qui dépend de la pipeline affichée : infos, stats, barres, segments, noms, RTTM."""
    ann, infos = _current(state, view)
    dur = state["audio"]["duration"]
    rows, summary = core.speaker_stats(ann, dur)
    rttm = core.to_rttm(ann, state["audio"]["uri"], TMP_DIR / "exports" / f"{state['audio']['uri']}_{view}.rttm")
    names = [[lab, state["names"].get(view, {}).get(lab, "")] for lab in plots.speaker_order(state["results"][view][0])]
    return (_info_html(infos, summary, ann), _stats_df(rows, ann), plots.bars_figure(ann, rows), _seg_df(ann), names,
            gr.update(value=str(rttm), visible=True, label=f"Télécharger le RTTM ({SHORT[view]})"), None)


def _timelines(state):
    figs = []
    for p in core.PIPELINES:
        if p in state["results"]:
            ann, _ = _current(state, p)
            figs.append(plots.timeline_figure(ann, state["audio"]["duration"],
                                              title=f"{SHORT[p]} · {state['audio']['uri']}",
                                              reference=state.get("ref")))
    if len(figs) == 1:
        return figs[0], gr.update(value=None, visible=False)
    return figs[0], gr.update(value=figs[1], visible=True)


def _to_int16(y):
    return (np.clip(y, -1.0, 1.0) * 32767).astype(np.int16)


# ------------------------------------------------------------------ application
def build_app(engines):
    available = engines.available()

    def analyse(file_path, mic_path, pipeline, compare, ref_path, progress=gr.Progress()):
        t0 = time.perf_counter()
        path = file_path or mic_path
        if not path:
            raise gr.Error("Choisir un fichier audio (ou enregistrer au micro) avant de lancer l'analyse.")
        progress(0.02, desc="Lecture de l'audio")
        try:
            audio = core.load_audio(path)
        except core.AudioLoadError as e:
            raise gr.Error(str(e))
        if not file_path:
            audio["uri"] = f"micro_{time.strftime('%Y%m%d_%H%M%S')}"
        todo = [p for p in core.PIPELINES if p in available] if compare else [pipeline]
        state = {"audio": audio, "results": {}, "names": {}, "ref": None}
        for i, p in enumerate(todo):
            progress(0.05 + 0.9 * i / len(todo), desc=f"Diarisation : {LABELS[p]}")
            try:
                state["results"][p] = core.diarize(audio, p, engines)
            except Exception as e:  # pas de crash de l'interface : message à l'utilisateur
                raise gr.Error(f"Échec de {p} : {type(e).__name__}: {e}")
        if ref_path:
            try:
                state["ref"] = core.load_reference(ref_path, audio["uri"])
            except Exception as e:
                gr.Warning(f"RTTM de référence ignoré : {e}")
        progress(1.0, desc="Affichage")
        view = todo[0]
        tl1, tl2 = _timelines(state)
        elapsed = time.perf_counter() - t0
        head = (f"Fichier <b>{html.escape(audio['uri'])}</b>"
                + (" · converti en wav 16 kHz mono par ffmpeg" if audio["converted"] else "")
                + f" · analyse terminée en {elapsed:.1f} s")
        return (state, f"<div class='ws-hint'>{head}</div>", tl1, tl2,
                gr.update(choices=[(LABELS[p], p) for p in todo], value=view, visible=len(todo) > 1),
                _metrics_html(state), *_view_outputs(state, view))

    def change_view(state, view):
        if not state or view not in state.get("results", {}):
            raise gr.Error("Lancer d'abord une analyse.")
        return _view_outputs(state, view)

    def play(state, view, evt: gr.SelectData):
        if not state or view not in state.get("results", {}):
            return None
        row = evt.row_value
        if not row:
            return None
        seg = (float(row[0]), float(row[1]))
        y = core.segment_audio(state["audio"]["y"], state["audio"]["sr"], seg)
        return (state["audio"]["sr"], _to_int16(y))

    def apply_names(state, view, names):
        if not state or view not in state.get("results", {}):
            raise gr.Error("Lancer d'abord une analyse.")
        rows = names.values.tolist() if hasattr(names, "values") else names
        mapping = {str(r[0]): str(r[1]) for r in rows if r and r[0] is not None and str(r[1] or "").strip()}
        try:
            core.rename(state["results"][view][0], mapping)  # vérifie l'absence de doublons
        except ValueError as e:
            raise gr.Error(str(e))
        state["names"][view] = mapping
        tl1, tl2 = _timelines(state)
        return (state, tl1, tl2, *_view_outputs(state, view))

    with gr.Blocks(title="Who Spoke? — démo", analytics_enabled=False) as app:
        state = gr.State(None)
        gr.HTML(_header(engines))

        with gr.Row(equal_height=False):
            # ---------- 1. entrée ----------
            with gr.Column(scale=4, min_width=320):
                with gr.Group():
                    gr.HTML(_section(1, "Audio", f"Formats : {FORMATS} (tout format lisible par ffmpeg)"),
                            padding=True)
                    file_in = gr.File(label="Fichier audio", file_types=core.AUDIO_EXTENSIONS, type="filepath",
                                      height=120)
                    with gr.Accordion("Enregistrer au micro (si aucun fichier n'est chargé)", open=False):
                        mic_in = gr.Audio(sources=["microphone"], type="filepath", format="wav", show_label=False)
                with gr.Group():
                    gr.HTML(_section(2, "Analyse"), padding=True)
                    pipe = gr.Radio([(LABELS[p], p) for p in available], value=available[0] if available else None,
                                    label="Pipeline")
                    compare = gr.Checkbox(label="Comparer les deux pipelines", value=False,
                                          interactive=len(available) == 2, visible=len(available) == 2)
                    with gr.Accordion("RTTM de référence (optionnel : DER / JER sur ce fichier)", open=False):
                        ref_in = gr.File(file_types=[".rttm"], type="filepath", show_label=False, height=90)
                run = gr.Button("Lancer l'analyse", variant="primary", size="lg", elem_id="ws-run",
                                interactive=bool(available))
            # ---------- 2. chiffres clés ----------
            with gr.Column(scale=7):
                head_md = gr.HTML(padding=False)
                view = gr.Radio(label="Résultats affichés", visible=False)
                info_md = gr.HTML(EMPTY)
                metrics_md = gr.HTML()

        # ---------- 3. timeline ----------
        gr.HTML(_section(3, "Timeline des locuteurs",
                         "Glisser pour zoomer, double-clic pour revenir ; survoler un segment pour ses bornes."))
        tl1 = gr.Plot(show_label=False)
        tl2 = gr.Plot(show_label=False, visible=False)

        # ---------- 4. temps de parole ----------
        gr.HTML(_section(4, "Temps de parole"))
        with gr.Row(equal_height=True):
            stats_df = gr.Dataframe(headers=STATS_HEADERS, show_label=False, interactive=False, scale=3)
            bars = gr.Plot(show_label=False, scale=2)

        # ---------- 5. segments, écoute, export ----------
        gr.HTML(_section(5, "Segments, écoute et export", "Cliquer sur une ligne pour écouter le segment."))
        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                seg_df = gr.Dataframe(headers=SEG_HEADERS, show_label=False, interactive=False, max_height=420,
                                      show_search="filter")
            with gr.Column(scale=2):
                seg_audio = gr.Audio(label="Segment sélectionné", type="numpy", interactive=False, autoplay=True)
                rttm_btn = gr.DownloadButton("Télécharger le RTTM", visible=False, variant="primary")
                with gr.Accordion("Renommer les locuteurs", open=False):
                    names_df = gr.Dataframe(headers=["locuteur", "nouveau nom"], interactive=True,
                                            static_columns=[0], column_count=2, show_label=False)
                    names_btn = gr.Button("Appliquer les noms", variant="secondary")

        view_outputs = [info_md, stats_df, bars, seg_df, names_df, rttm_btn, seg_audio]
        run.click(analyse, inputs=[file_in, mic_in, pipe, compare, ref_in],
                  outputs=[state, head_md, tl1, tl2, view, metrics_md, *view_outputs])
        view.change(change_view, inputs=[state, view], outputs=view_outputs)
        seg_df.select(play, inputs=[state, view], outputs=seg_audio)
        names_btn.click(apply_names, inputs=[state, view, names_df], outputs=[state, tl1, tl2, *view_outputs])
    return app


def main():
    os.chdir(ROOT)  # vad.py et les autres modules utilisent des chemins relatifs à la racine du dépôt
    (TMP_DIR / "gradio").mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    engines = core.Engines()
    print(f"[démo] modèles prêts en {time.perf_counter() - t0:.1f} s ; pipelines disponibles : {engines.available()}")
    app = build_app(engines)
    port = int(os.environ.get("WHO_SPOKE_PORT", DEFAULT_PORT))
    app.queue(default_concurrency_limit=1).launch(server_name="127.0.0.1", server_port=port, share=False,
                                                  inbrowser=False, show_error=True, ssr_mode=False, mcp_server=False,
                                                  theme=make_theme(), css_paths=[CSS_PATH])


if __name__ == "__main__":
    main()
