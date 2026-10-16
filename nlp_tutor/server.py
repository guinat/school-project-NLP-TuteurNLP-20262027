"""API locale du tuteur pour l'interface web (dossier web/), lancée par : python -m nlp_tutor.server

Même modèle, même adaptateur, même prompt et mêmes paramètres de génération que l'interface Gradio.
La réponse est envoyée token par token, une ligne JSON par événement :

    {"type": "start", "prompt_tokens": 54}
    {"type": "token", "ids": [8747], "text": "La"}
    {"type": "end", "tokens": 107, "seconds": 12.4, "interrupted": false, "truncated": false}
"""
import asyncio
import json
import queue
import threading
import time

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from nlp_tutor.config import ADAPTER_DIR, MAX_NEW_TOKENS, MAX_TURNS, MODEL_NAME
from nlp_tutor.conversation import build_messages, generation_config, start_stream
from nlp_tutor.model import load_adapted_model, pick_device
from nlp_tutor.prompts import TRAINING_SYSTEM

HOST, PORT = "127.0.0.1", 8000

device = pick_device()
tokenizer, model = load_adapted_model(device)
config = generation_config(tokenizer)


class Message(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(max_length=4000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    history: list[Message] = Field(default_factory=list)


app = FastAPI(title="Tuteur NLP")
lock = asyncio.Lock()                       # une seule génération à la fois sur le CPU


def ndjson(event):
    return json.dumps(event, ensure_ascii=False) + "\n"


@app.get("/health")
def health():
    return {"ready": True, "model": MODEL_NAME, "adapter": ADAPTER_DIR.name, "device": device,
            "max_turns": MAX_TURNS, "max_new_tokens": MAX_NEW_TOKENS}


@app.post("/chat")
async def chat(body: ChatRequest, request: Request):
    history = [m.model_dump() for m in body.history[-2 * MAX_TURNS:]]
    messages, prompt_tokens = build_messages(tokenizer, body.message.strip(), history, TRAINING_SYSTEM)

    async def events():
        async with lock:
            stop = threading.Event()
            t0, n = time.perf_counter(), 0
            thread, stream = start_stream(model, tokenizer, messages, config, stop)
            yield ndjson({"type": "start", "prompt_tokens": prompt_tokens})
            try:
                while True:
                    try:
                        chunk = stream.queue.get_nowait()
                    except queue.Empty:
                        if await request.is_disconnected():     # bouton « Arrêter » ou page fermée
                            stop.set()
                            break
                        await asyncio.sleep(0.02)
                        continue
                    if chunk is None:
                        break
                    n += len(chunk["ids"])
                    yield ndjson({"type": "token", **chunk})
                yield ndjson({"type": "end", "tokens": n, "seconds": round(time.perf_counter() - t0, 1),
                              "interrupted": stop.is_set(), "truncated": n >= MAX_NEW_TOKENS})
            finally:
                stop.set()
                thread.join(timeout=10)

    return StreamingResponse(events(), media_type="application/x-ndjson", headers={"Cache-Control": "no-store"})


if __name__ == "__main__":
    print(f"Tuteur NLP ready ({device}): http://{HOST}:{PORT}", flush=True)
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
