// API Python du tuteur (nlp_tutor/server.py)
export const TUTOR_API_URL = process.env.TUTOR_API_URL ?? "http://127.0.0.1:8000";

export const SERVER_DOWN_MESSAGE =
  "Le serveur du modèle ne répond pas. Lance « python -m nlp_tutor.server » dans le dossier du projet, attends la fin du chargement, puis réessaie.";
