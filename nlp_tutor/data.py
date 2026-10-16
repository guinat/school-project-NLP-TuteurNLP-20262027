"""Jeu de dialogues (partie 4) : lecture, contrôles de qualité, séparation et recherche de fuite."""
import json
import random
import re

import numpy as np

from nlp_tutor.config import DIALOGUE_TYPES, DIALOGUES_DIR, REDIRECT_TYPES, SEED, TEST_SHARE, VALIDATION_SHARE
from nlp_tutor.metrics import is_french

JSONL_FIELDS = ("id", "type", "topic", "should_redirect", "messages")

PERSONAL_DATA_PATTERNS = {
    "email address": r"[\w.+-]+@[\w-]+\.[\w.]+",
    "phone number": r"(?:\+33|\b0)[1-9](?:[ .-]?\d{2}){4}\b",
    "URL": r"https?://|www\.",
    "numeric identifier": r"\b\d{8,}\b",
}


def read_dialogues(directory=DIALOGUES_DIR):
    """Lit les dialogues écrits à la main ; le format est décrit en tête de chaque fichier."""
    dialogues = []
    for path in sorted(directory.glob("*.txt")):
        current = None
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("#"):
                continue
            if line.startswith("=== "):
                dialogue_id, type_, topic = (x.strip() for x in line[4:].split("|"))
                current = {"id": dialogue_id, "type": type_, "topic": topic, "messages": []}
                dialogues.append(current)
            elif re.match(r"^[UA]: ", line):
                role = "user" if line[0] == "U" else "assistant"
                current["messages"].append({"role": role, "content": line[3:]})
            elif current is not None and current["messages"]:
                current["messages"][-1]["content"] += "\n" + line
    for d in dialogues:
        for m in d["messages"]:
            m["content"] = m["content"].strip()
        d["should_redirect"] = d["type"] in REDIRECT_TYPES
        d["turns"] = len(d["messages"]) // 2
    return dialogues


def answers_and_questions(dialogues):
    answers = [m["content"] for d in dialogues for m in d["messages"] if m["role"] == "assistant"]
    questions = [m["content"] for d in dialogues for m in d["messages"] if m["role"] == "user"]
    return answers, questions


def quality_checks(dialogues, tokenizer):
    """Contrôles automatiques et longueurs des réponses en mots et en tokens."""
    answers, _ = answers_and_questions(dialogues)
    openings = [d["messages"][0]["content"].lower() for d in dialogues]
    words = np.array([len(a.split()) for a in answers])
    tokens = np.array([len(tokenizer.encode(a)) for a in answers])
    ids = [d["id"] for d in dialogues]
    checks = {
        "unique ids": len(ids) == len(set(ids)),
        "unknown types": sum(d["type"] not in DIALOGUE_TYPES for d in dialogues),
        "user/assistant alternation": all(
            [m["role"] for m in d["messages"]] == ["user", "assistant"] * d["turns"] and len(d["messages"]) % 2 == 0
            for d in dialogues),
        "dialogues with the same opening": len(openings) - len(set(openings)),
        "duplicate answers": len(answers) - len(set(answers)),
        "answers over 150 words": int((words > 150).sum()),
        "answers without final question": sum(not a.endswith("?") for a in answers),
        "answers not detected as French": sum(not is_french(a) for a in answers),
    }
    return checks, words, tokens


def find_personal_data(texts):
    return {name: [t for t in texts if re.search(pattern, t)] for name, pattern in PERSONAL_DATA_PATTERNS.items()}


def split_dialogues(dialogues, val_share=VALIDATION_SHARE, test_share=TEST_SHARE, seed=SEED):
    """Séparation stratifiée par type, avec au moins un dialogue de chaque type par partie."""
    rng = random.Random(seed)
    splits = {"train": [], "validation": [], "test": []}
    for type_ in DIALOGUE_TYPES:
        group = [d for d in dialogues if d["type"] == type_]
        rng.shuffle(group)
        n_test = max(1, round(test_share * len(group)))
        n_val = max(1, round(val_share * len(group)))
        splits["test"] += group[:n_test]
        splits["validation"] += group[n_test:n_test + n_val]
        splits["train"] += group[n_test + n_val:]
    return splits


def write_jsonl(dialogues, path):
    with open(path, "w", encoding="utf-8") as f:
        for d in dialogues:
            f.write(json.dumps({k: d[k] for k in JSONL_FIELDS}, ensure_ascii=False) + "\n")


def _words(text):
    return set(re.findall(r"\w+", text.lower()))


def closest_questions(splits):
    """Pour chaque dialogue de test, la question du train la plus proche (similarité de Jaccard).

    On compare les questions d'ouverture : les relances comme « oui » ne disent rien du sujet.
    Renvoie des tuples (similarité, id, question de test, question du train), du plus proche au moins proche."""
    train_questions = [d["messages"][0]["content"] for d in splits["train"]]
    closest = []
    for d in splits["test"]:
        q = d["messages"][0]["content"]
        sims = [len(_words(q) & _words(t)) / len(_words(q) | _words(t)) for t in train_questions]
        j = int(np.argmax(sims))
        closest.append((sims[j], d["id"], q, train_questions[j]))
    return sorted(closest, reverse=True)
