# Notes importantes et difficultés

## problemes rencontrés lors de la découverte du corpus :
La fiche officielle de pyannote/segmentation-3.0 indique un entraînement sur neuf corpus, dont AMI et VoxConverse. Elle ne précise pas quelles parties de chaque corpus ont été utilisées.

Conclusion : Pour la semaine 2 faire attention si le choix de faire du finetuning est fait -> n'utiliser ni AMI ni Voxconverse et verifier le corpus utilisé

Trois règles qu’on se fixe :
- Pour AMI, n’évaluer que sur des réunions dev ou test, jamais train. Pour VoxConverse, privilégier le test v0.3.
- Pour chaque modèle pré-entraîné, noter les corpus sur lesquels il a été entraîné et signaler tout recouvrement possible avec les fichiers d’évaluation.
Compléter l’évaluation par des enregistrements maison jamais publiés.