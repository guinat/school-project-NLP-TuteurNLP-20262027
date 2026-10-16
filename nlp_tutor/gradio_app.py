"""Interface Gradio du tuteur (partie 7) : conversation, saisie, réinitialisation, rôle et limites."""
import threading

import gradio as gr

from nlp_tutor.conversation import build_messages, generation_config, start_stream
from nlp_tutor.prompts import TRAINING_SYSTEM

PRESENTATION = """## Tuteur NLP
Assistant de révision pour le cours **NLP — du texte aux modèles de langage** et le cours **Les Transformers** (ING3 IA, CY Tech).

**Ce qu'il fait** : expliquer une notion du cours, dérouler un calcul (softmax, TF-IDF, F1, perte…), comparer deux notions, t'interroger et corriger ta réponse.
**Ce qu'il ne fait pas** : répondre hors du cours, faire un travail noté à ta place, donner des informations administratives (dates, notes, contacts).

**Limites** : c'est un petit modèle (Qwen2.5-0.5B-Instruct adapté par LoRA) ; il se trompe souvent, invente des informations et peut répéter la même phrase jusqu'à la limite de longueur. Vérifie dans tes supports. Il ne garde en mémoire que les 3 derniers échanges. Sur CPU, compte une à deux minutes par réponse."""

EXAMPLES = ["C'est quoi le masque causal ?", "Refais le calcul de la softmax du cours",
            "Interroge-moi sur les métriques", "Écris le rapport de mon mini-projet"]


def create_app(model, tokenizer):
    config = generation_config(tokenizer)

    def answer(message, history):
        history = [{"role": m["role"], "content": m["content"]} for m in (history or [])]
        if not message.strip():
            yield "", history
            return
        messages, _ = build_messages(tokenizer, message, history, TRAINING_SYSTEM)
        stop = threading.Event()
        _, stream = start_stream(model, tokenizer, messages, config, stop)
        history += [{"role": "user", "content": message}, {"role": "assistant", "content": ""}]
        try:
            while (chunk := stream.queue.get()) is not None:
                history[-1]["content"] += chunk["text"]
                yield "", history
        finally:
            stop.set()      # réinitialisation ou page fermée : la génération s'arrête aussi

    with gr.Blocks(title="Tuteur NLP") as app:
        gr.Markdown(PRESENTATION)
        chat = gr.Chatbot(type="messages", height=420, label="Conversation")
        message = gr.Textbox(placeholder="Pose ta question sur le cours…", label="Ton message", lines=2)
        with gr.Row():
            send = gr.Button("Envoyer", variant="primary")
            reset = gr.Button("Réinitialiser la conversation")
        gr.Examples(EXAMPLES, inputs=message)
        submitted = message.submit(answer, [message, chat], [message, chat])
        sent = send.click(answer, [message, chat], [message, chat])
        reset.click(lambda: ([], ""), None, [chat, message], cancels=[submitted, sent])
    return app
