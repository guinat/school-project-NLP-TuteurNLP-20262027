"""Tuteur NLP : agent de révision pour les cours de NLP et de Transformers (ING3 IA, CY Tech).

- config        chemins, graine, paramètres de génération et d'entraînement
- model         chargement du tokenizer, du modèle et de l'adaptateur
- prompts       prompts comparés et requêtes de mise au point
- conversation  historique contrôlé, génération par lots et en flux
- metrics       indicateurs automatiques et détecteur de refus
- data          lecture, contrôles et séparation des dialogues
- training      encodage, boucle LoRA, courbes de perte
- evaluation    requêtes de test, tableaux, grille d'évaluation
- gradio_app    interface Gradio (partie 7)
- server        API de l'interface web (python -m nlp_tutor.server)
"""
