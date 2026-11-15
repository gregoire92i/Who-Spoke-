"""Bout en bout par l'API Gradio (serveur démo sur 127.0.0.1:7870). Mesure le temps réel côté client de chaque
analyse (upload compris) et relève le temps / RTF affichés par l'interface.
Usage : python e2e.py <csv_out> <pipeline> <n> <fichier>"""
import csv
import re
import sys
import time

from gradio_client import Client, handle_file

out, pipeline, n, f = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
client = Client("http://127.0.0.1:7870/", verbose=False)
rows = []
for i in range(1, n + 1):
    t0 = time.perf_counter()
    res = client.predict(handle_file(f), None, pipeline, False, None, api_name="/analyse")
    wall = time.perf_counter() - t0
    txt = " ".join(str(x) for x in res)
    elapsed = re.search(r"analyse terminée en ([\d.]+) s", txt)
    rtf = re.search(r"ws-k'>RTF</div><div class='ws-v'>([\d.]+)<", txt)
    proc = re.search(r"ws-k'>Traitement</div><div class='ws-v'>([\d.]+) s<", txt)
    rows.append({"pipeline": pipeline, "file": f.split("/")[-1], "call": i, "wall_client_s": wall,
                 "affiche_analyse_s": float(elapsed.group(1)) if elapsed else None,
                 "affiche_traitement_s": float(proc.group(1)) if proc else None,
                 "affiche_RTF": float(rtf.group(1)) if rtf else None})
    print(rows[-1], flush=True)
    if i == 1:
        print("N_OUTPUTS:", len(res), flush=True)
with open(out, "a", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    if fh.tell() == 0:
        w.writeheader()
    w.writerows(rows)
