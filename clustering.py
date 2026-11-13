
import numpy as np
from pyannote.database.util import load_rttm
from rttm_files import load_file_annotations
from sklearn.cluster import KMeans

def spectral_clustering(X, oracle=False, n_speakers=None, p_neighbors=0.1, min_neighbors=5,
                        min_speakers=1, max_speakers=10, standardize=True,
                        seed=42, return_info=False):
    """Clustering spectral des embeddings, nombre de locuteurs estimé par eigengap.
 
    X : (n_fenetres, dim), sortie de compute_embeddings.
    oracle : False (défaut) -> k estimé par eigengap entre min_speakers et max_speakers.
             True -> k = n_speakers, qui doit alors être donné (lu dans le RTTM).
    p_neighbors : fraction des fenêtres gardées comme voisines de chaque fenêtre (raffinement).
    min_neighbors : nombre minimal de voisins gardés, quelle que soit la taille du fichier.
    standardize : centre-réduit chaque dimension sur le fichier avant le cosinus.
    seed : graine du k-means, pour la reproductibilité.
    Renvoie labels (n_fenetres,) entiers 0..k-1, et si return_info=True un dict
    {"k", "eigenvalues", "affinity"} pour tracer la matrice et les valeurs propres."""
 
    # --- garde-fous sur les paramètres ---
    if oracle and (n_speakers is None or n_speakers < 1):
        raise ValueError(f"mode oracle : n_speakers doit être un entier >= 1, reçu {n_speakers}")
    if not 0 < p_neighbors <= 1:
        raise ValueError(f"p_neighbors doit être dans ]0, 1], reçu {p_neighbors}")
    if not 1 <= min_speakers <= max_speakers:
        raise ValueError(f"il faut 1 <= min_speakers <= max_speakers, reçu {min_speakers}, {max_speakers}")
 
    n = X.shape[0]
    # cas dégénérés : 0 ou 1 fenêtre (fichier sans parole, cas pqmho) -> rien à regrouper
    if n <= 1:
        labels = np.zeros(n, dtype=int)
        info = {"k": n, "eigenvalues": np.array([]), "affinity": np.ones((n, n))}
        return (labels, info) if return_info else labels
 
    # --- 1. normalisation des embeddings ---
    X = X.astype(float)
    if standardize:
        # sans ça, le cosinus est dominé par les dimensions de grande amplitude (c0, moyennes)
        # et toutes les fenêtres se ressemblent ; std nulle -> on divise par 1
        std = X.std(axis=0)
        X = (X - X.mean(axis=0)) / np.where(std > 0, std, 1.0)
    # norme L2 = 1 par ligne : le produit scalaire devient la similarité cosinus
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.where(norms > 0, norms, 1.0)
 
    # --- 2. matrice d'affinité : cosinus entre toutes les paires de fenêtres, dans [-1, 1] ---
    A = X @ X.T                                         # (n, n)
 
    # --- 3. raffinement : pour chaque ligne, on ne garde que les p % voisins les plus similaires ---
    # (binarisation, idée reprise de NME-SC ; pas NME-SC complet : ici Laplacien normalisé + p fixe,
    # NME-SC utilise le Laplacien non normalisé et choisit p automatiquement) : 1 = « probablement le même locuteur », 0 sinon
    # min_neighbors : sur un fichier court (peu de fenêtres), p_neighbors * n donnerait 1 ou 2 voisins,
    # le graphe se casserait en dizaines de morceaux et toutes les valeurs propres vaudraient 0
    p = min(n, max(min_neighbors, int(round(p_neighbors * n))))
    # argsort décroissant par ligne, on prend les p premiers indices de chaque ligne
    top = np.argsort(-A, axis=1)[:, :p]
    A_bin = np.zeros_like(A)
    np.put_along_axis(A_bin, top, 1.0, axis=1)
    # symétrisation : le graphe doit être non orienté (i voisin de j <=> j voisin de i)
    A_bin = 0.5 * (A_bin + A_bin.T)
 
    # --- 4. Laplacien normalisé L = I - D^-1/2 A D^-1/2 ---
    deg = A_bin.sum(axis=1)                             # degré de chaque fenêtre (>0 : elle est sa propre voisine)
    d_inv_sqrt = 1.0 / np.sqrt(deg)
    L = np.eye(n) - d_inv_sqrt[:, None] * A_bin * d_inv_sqrt[None, :]
 
    # --- 5. valeurs/vecteurs propres, triés par ordre croissant (eigh : matrice symétrique) ---
    eigvals, eigvecs = np.linalg.eigh(L)
 
    # --- 6. choix de k ---
    if oracle:
        k = n_speakers
    else:
        # k groupes bien séparés -> k valeurs propres proches de 0, puis un saut
        # gaps[i] = lambda_{i+1} - lambda_i ; le plus grand saut après la k-ième valeur propre donne k
        k_max = min(max_speakers, n - 1)
        gaps = np.diff(eigvals[:k_max + 1])             # gaps[0] = saut après 1 valeur propre
        k_min = min(min_speakers, k_max)
        k = int(np.argmax(gaps[k_min - 1:k_max])) + k_min
    k = min(k, n)                                       # on ne peut pas faire plus de groupes que de fenêtres
 
    if k == 1:
        labels = np.zeros(n, dtype=int)
    else:
        # --- 7. chaque fenêtre est représentée par ses coordonnées sur les k premiers vecteurs propres ---
        U = eigvecs[:, :k]                              # (n, k)
        u_norms = np.linalg.norm(U, axis=1, keepdims=True)
        U = U / np.where(u_norms > 0, u_norms, 1.0)     # normalisation par ligne (Ng, Jordan, Weiss) ; ligne nulle laissée à 0
        # --- 8. k-means dans cet espace, où les groupes sont presque des points ---
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(U)
 
    if return_info:
        return labels, {"k": k, "eigenvalues": eigvals[:max_speakers + 1], "affinity": A_bin}
    return labels

def nme_spectral_clustering(X, oracle=False, n_speakers=None, min_neighbors=5, max_frac=0.25, n_grid=30,
                            laplacian="normalized", min_speakers=1, max_speakers=10, standardize=True,
                            seed=42, return_info=False):
    """NME-SC (Park et al., 2019, « Auto-Tuning Spectral Clustering for Speaker Diarization Using
    Normalized Maximum Eigengap ») : p (nombre de voisins gardés par window) choisi FICHIER PAR FICHIER.

    Pour chaque p candidat : graphe binarisé à p voisins -> Laplacien -> valeurs propres ->
    g_p = (plus grand eigengap entre min_speakers et max_speakers) / (plus grande valeur propre),
    r(p) = p / g_p. On garde le p qui minimise r(p) (spectre net avec peu de voisins),
    puis k = position de l'eigengap maximal pour ce p (ou k = n_speakers si oracle).
    Les étapes 1 à 3 et 7-8 sont identiques à spectral_clustering ; seul le choix de p change.

    min_neighbors, max_frac, n_grid : grille de p = n_grid entiers répartis entre min_neighbors et
        max_frac * n (grille « clairsemée » comme dans l'implémentation NeMo, max 0.25, 30 valeurs).
    laplacian : "normalized" (I - D^-1/2 A D^-1/2, comme spectral_clustering) ou
        "unnormalized" (D - A, comme l'article).
    Renvoie labels, et si return_info=True un dict {"k", "p", "p_grid", "r", "eigenvalues", "affinity"}."""
    if oracle and (n_speakers is None or n_speakers < 1):
        raise ValueError(f"mode oracle : n_speakers doit être un entier >= 1, reçu {n_speakers}")
    if not 0 < max_frac <= 1:
        raise ValueError(f"max_frac doit être dans ]0, 1], reçu {max_frac}")
    if not 1 <= min_speakers <= max_speakers:
        raise ValueError(f"il faut 1 <= min_speakers <= max_speakers, reçu {min_speakers}, {max_speakers}")
    if laplacian not in ("normalized", "unnormalized"):
        raise ValueError(f"laplacian doit valoir 'normalized' ou 'unnormalized', reçu {laplacian}")

    n = X.shape[0]
    if n <= 1:
        labels = np.zeros(n, dtype=int)
        info = {"k": n, "p": n, "p_grid": np.array([]), "r": np.array([]),
                "eigenvalues": np.array([]), "affinity": np.ones((n, n))}
        return (labels, info) if return_info else labels

    # --- 1-2. normalisation + affinité cosinus (identique à spectral_clustering) ---
    X = X.astype(float)
    if standardize:
        std = X.std(axis=0)
        X = (X - X.mean(axis=0)) / np.where(std > 0, std, 1.0)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.where(norms > 0, norms, 1.0)
    A = X @ X.T
    order = np.argsort(-A, axis=1)                      # tri des voisins une seule fois pour toute la grille

    def graph_and_spectrum(p, vectors=False):
        """Graphe binarisé à p voisins, symétrisé, puis Laplacien et son spectre (croissant)."""
        A_bin = np.zeros_like(A)
        np.put_along_axis(A_bin, order[:, :p], 1.0, axis=1)
        A_bin = 0.5 * (A_bin + A_bin.T)
        deg = A_bin.sum(axis=1)
        if laplacian == "normalized":
            d_inv_sqrt = 1.0 / np.sqrt(deg)
            L = np.eye(n) - d_inv_sqrt[:, None] * A_bin * d_inv_sqrt[None, :]
        else:
            L = np.diag(deg) - A_bin
        if vectors:
            vals, vecs = np.linalg.eigh(L)
            return A_bin, vals, vecs
        return A_bin, np.linalg.eigvalsh(L), None

    # bornes de k (identiques à spectral_clustering)
    k_max = min(max_speakers, n - 1)
    k_min = min(min_speakers, k_max)

    # --- 3bis. grille de p et critère r(p) = p / g_p ---
    p_lo = min(n, min_neighbors)
    p_hi = min(n, max(p_lo, int(round(max_frac * n))))
    p_grid = np.unique(np.round(np.linspace(p_lo, p_hi, n_grid)).astype(int))
    r = np.full(len(p_grid), np.inf)
    for i, p in enumerate(p_grid):
        _, vals, _ = graph_and_spectrum(p)
        gaps = np.diff(vals[:k_max + 1])[k_min - 1:k_max]
        g = gaps.max() / (vals[-1] + 1e-10) if len(gaps) else 0.0
        r[i] = p / g if g > 0 else np.inf
    p_best = int(p_grid[np.argmin(r)]) if np.isfinite(r).any() else int(p_grid[-1])

    # --- 4-6. spectre complet au p retenu, choix de k ---
    A_bin, eigvals, eigvecs = graph_and_spectrum(p_best, vectors=True)
    if oracle:
        k = n_speakers
    else:
        gaps = np.diff(eigvals[:k_max + 1])
        k = int(np.argmax(gaps[k_min - 1:k_max])) + k_min
    k = min(k, n)

    # --- 7-8. identiques à spectral_clustering ---
    if k == 1:
        labels = np.zeros(n, dtype=int)
    else:
        U = eigvecs[:, :k]
        u_norms = np.linalg.norm(U, axis=1, keepdims=True)
        U = U / np.where(u_norms > 0, u_norms, 1.0)
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(U)

    if return_info:
        return labels, {"k": k, "p": p_best, "p_grid": p_grid, "r": r,
                        "eigenvalues": eigvals[:max_speakers + 1], "affinity": A_bin}
    return labels
