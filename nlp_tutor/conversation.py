"""Conversation avec le modèle : historique de taille contrôlée, génération par lots et en flux."""
import queue
import threading
import time

import torch
from transformers import GenerationConfig, StoppingCriteria, StoppingCriteriaList
from transformers.generation.streamers import BaseStreamer

from nlp_tutor.config import GENERATION_BATCH_SIZE, MAX_NEW_TOKENS, MAX_TURNS, PROMPT_BUDGET, REPETITION_PENALTY
from nlp_tutor.prompts import MINIMAL_SYSTEM


def build_messages(tokenizer, question, history=(), system=MINIMAL_SYSTEM, examples=(),
                   budget=PROMPT_BUDGET, max_turns=MAX_TURNS):
    """Messages envoyés au modèle et longueur du prompt en tokens.

    On garde au plus `max_turns` échanges passés, puis on retire les plus anciens, toujours entiers et
    jamais le message système, jusqu'à tenir dans `budget` tokens."""
    history = [{"role": m["role"], "content": m["content"]} for m in history]
    turns = [history[i:i + 2] for i in range(0, len(history), 2)][-max_turns:]
    while True:
        messages = [{"role": "system", "content": system}, *examples,
                    *[m for turn in turns for m in turn], {"role": "user", "content": question}]
        n = len(tokenizer.apply_chat_template(messages, add_generation_prompt=True))
        if n <= budget or not turns:
            return messages, n
        turns = turns[1:]


def end_token_ids(tokenizer):
    return [tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.eos_token_id]


def generation_config(tokenizer):
    return GenerationConfig(
        do_sample=False,
        repetition_penalty=REPETITION_PENALTY,
        max_new_tokens=MAX_NEW_TOKENS,
        eos_token_id=end_token_ids(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
    )


@torch.no_grad()
def generate(model, tokenizer, conversations, config, batch_size=GENERATION_BATCH_SIZE):
    """Une réponse par conversation, générées par lots. Renvoie les réponses et la durée totale."""
    answers, t0 = [], time.perf_counter()
    for i in range(0, len(conversations), batch_size):
        texts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                 for m in conversations[i:i + batch_size]]
        enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        out = model.generate(**enc, generation_config=config)
        answers += [tokenizer.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                    for o in out]
    return answers, time.perf_counter() - t0


class TokenStream(BaseStreamer):
    """File des tokens générés : {"ids": [...], "text": "..."} pour chacun, puis None à la fin.

    Un caractère accentué peut être coupé entre deux tokens : on attend alors le token suivant pour
    envoyer un texte lisible."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.end_ids = set(end_token_ids(tokenizer))
        self.queue = queue.Queue()
        self.ids, self.text, self.pending, self.prompt_seen = [], "", [], False

    def _decode(self):
        return self.tokenizer.decode(self.ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    def put(self, value):
        if not self.prompt_seen:            # le premier appel contient le prompt
            self.prompt_seen = True
            return
        for token_id in value.reshape(-1).tolist():
            if token_id in self.end_ids:
                continue
            self.ids.append(token_id)
            text = self._decode()
            if text.endswith("�"):
                self.pending.append(token_id)
                continue
            self.queue.put({"ids": self.pending + [token_id], "text": text[len(self.text):]})
            self.text, self.pending = text, []

    def end(self):
        if self.pending:
            self.queue.put({"ids": self.pending, "text": self._decode()[len(self.text):]})
        self.queue.put(None)


class StopOnEvent(StoppingCriteria):
    def __init__(self, event):
        self.event = event

    def __call__(self, input_ids, scores, **kwargs):
        return torch.full((input_ids.shape[0],), self.event.is_set(), dtype=torch.bool, device=input_ids.device)


def start_stream(model, tokenizer, messages, config, stop_event=None):
    """Lance la génération dans un fil séparé. Renvoie le fil et le flux de tokens à lire."""
    stream = TokenStream(tokenizer)
    enc = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                        return_tensors="pt", return_dict=True).to(model.device)
    criteria = StoppingCriteriaList([StopOnEvent(stop_event)]) if stop_event is not None else None
    thread = threading.Thread(target=model.generate, kwargs=dict(
        **enc, generation_config=config, streamer=stream, stopping_criteria=criteria))
    thread.start()
    return thread, stream
