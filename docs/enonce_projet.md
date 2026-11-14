Who Spoke? Qui parle, et quand ?

Diarisation de locuteurs

Objectif

Construire en trois semaines un système qui détecte les périodes de parole, distingue les locuteurs et attribue à chacun des segments temporels. Le système doit fonctionner sur des conversations jamais vues.

Cadre
Format : individuel, 3 semaines, en autonomie.
Données fournies : aucune. Chaque participant recherche, sélectionne, télécharge ou constitue ses propres corpus.
Contraintes : documenter provenance, licences, consentement si nécessaire, répartition entraînement/validation/test et risques de fuite de données.
Outils : Python, GitHub et ressources de calcul accessibles au participant. Les modèles pré-entraînés sont autorisés, mais leurs usages doivent être documentés.
Évaluation : démonstration sur des enregistrements inconnus, préparés séparément par l'organisation. Aucun jeu d'entraînement ni de validation n'est fourni au démarrage.
Organisation indicative
Semaine 1 : recherche de corpus et technologies ; constitution des données ; baseline classique.
Semaine 2 : intégration de modèles de deep learning ; comparaison des approches.
Semaine 3 : améliorations mesurées, robustesse, démonstration et documentation.
Rendu
Dépôt GitHub avec code, dépendances, instructions de reproduction et scripts d'évaluation.
Fiche des corpus : source, accès, licence, volume, composition et limites.
Tableau comparatif des approches et courte analyse d'ablation.
Démonstration fonctionnelle sur un nouvel enregistrement.
Les données ne sont redistribuées que si leur licence et les droits des personnes le permettent.
Travail demandé
1. Rechercher et préparer les données
Identifier des corpus de conversations multi-locuteurs avec annotations temporelles, ou produire des enregistrements consentis et les annoter.
Explorer notamment AMI et VoxConverse ; vérifier les conditions d'accès et d'usage.
Construire un découpage de développement et de validation sans fuite entre enregistrements ou locuteurs pertinents.
Exporter les annotations au format RTTM ou équivalent.
2. Construire une baseline explicable
Charger l'audio, calculer spectrogrammes et MFCC.
Implémenter une détection de parole simple (énergie, seuils).
Segmenter puis regrouper les voix avec des caractéristiques acoustiques et un clustering.
Mesurer les erreurs sur les annotations collectées.
3. Explorer les modèles modernes
Comparer la baseline avec des embeddings de locuteurs et une méthode de clustering.
Tester au moins une pipeline existante : pyannote.audio ou NVIDIA NeMo.
Étudier au moins deux améliorations : VAD, seuils de clustering, nombre de locuteurs, bruit, segments courts ou chevauchements.
Produire une ablation : effet mesuré de chaque modification sur les mêmes fichiers de validation.
4. Démonstration

Application permettant de charger un audio, lancer l'analyse, afficher une timeline des locuteurs, écouter les segments et exporter JSON ou RTTM.

Évaluation technique
DER (Diarization Error Rate), protocole documenté, notamment gestion des chevauchements et collar.
JER en métrique complémentaire.
Temps d'inférence et cas d'échec.
Technologies et ressources à explorer
librosa
pyannote.audio
SpeechBrain
NVIDIA NeMo
AMI
VoxConverse
Point de vigilance

La constitution des données fait partie du travail : il n'y a ni corpus de départ, ni annotations, ni partition de validation fournis. Une première démo peut être construite sur un petit corpus légalement accessible ; la taille du corpus doit rester compatible avec trois semaines.