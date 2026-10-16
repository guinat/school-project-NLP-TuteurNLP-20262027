"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { About } from "@/components/About";
import { ServerStatus } from "@/components/ServerStatus";
import { Signature } from "@/components/Signature";
import { TokenTitle } from "@/components/TokenTitle";
import type { Exchange, Health, StreamEvent } from "@/lib/types";

const EXAMPLES = [
  "C'est quoi le masque causal ?",
  "Refais le calcul de la softmax du cours",
  "Interroge-moi sur les métriques",
  "Écris le rapport de mon mini-projet",
];

const MAX_LENGTH = 2000;
const MAX_NEW_TOKENS = 320;

const answerText = (e: Exchange) => e.tokens.map((t) => t.text).join("");
const formatNumber = (x: number) => x.toLocaleString("fr-FR");

function Answer({ exchange, tokensVisible }: { exchange: Exchange; tokensVisible: boolean }) {
  if (exchange.status === "waiting") {
    return (
      <p className="text-muted">
        Le modèle lit ta question <span className="cursor ml-0.5" aria-hidden="true" />
      </p>
    );
  }
  const streaming = exchange.status === "streaming";
  return (
    <div className="whitespace-pre-wrap break-words" aria-busy={streaming}>
      {exchange.tokens.map((t, i) => (
        <span
          key={i}
          className={streaming ? "token token-new" : "token"}
          title={tokensVisible ? `token ${t.ids.join(" + ")}` : undefined}
        >
          {t.text}
        </span>
      ))}
      {streaming && <span className="cursor ml-0.5" aria-hidden="true" />}
    </div>
  );
}

function AnswerFooter({ exchange, onRetry }: { exchange: Exchange; onRetry: () => void }) {
  if (exchange.status === "error") {
    return (
      <div className="mt-4 flex flex-wrap items-baseline gap-x-4 gap-y-1 text-sm">
        <p className="text-red">{exchange.error}</p>
        <button
          type="button"
          onClick={onRetry}
          className="font-medium text-foreground underline decoration-border-strong underline-offset-4 transition-colors hover:decoration-foreground"
        >
          Réessayer
        </button>
      </div>
    );
  }
  if (exchange.status === "interrupted") return <p className="mt-4 text-xs text-muted">Réponse interrompue.</p>;
  if (exchange.status !== "done" || !exchange.meta) return null;
  const { tokens, seconds, truncated } = exchange.meta;
  return (
    <p className="mt-4 text-xs text-muted tabular-nums">
      {formatNumber(tokens)} tokens en {formatNumber(seconds)} s.
      {truncated && ` Réponse coupée à la limite de ${MAX_NEW_TOKENS} tokens.`}
    </p>
  );
}

export function Tutor() {
  const [exchanges, setExchanges] = useState<Exchange[]>([]);
  const [input, setInput] = useState("");
  const [health, setHealth] = useState<Health | null>(null);
  const [tokensVisible, setTokensVisible] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const controller = useRef<AbortController | null>(null);
  const nextId = useRef(0);
  const textarea = useRef<HTMLTextAreaElement>(null);
  const scrollArea = useRef<HTMLDivElement>(null);
  const exchangeCount = useRef(0);
  const closeButton = useRef<HTMLButtonElement>(null);

  const busy = exchanges.some((e) => e.status === "waiting" || e.status === "streaming");

  // état du serveur Python : vérifié souvent tant qu'il est arrêté, de temps en temps ensuite
  useEffect(() => {
    let active = true;
    let timer: ReturnType<typeof setTimeout>;
    const check = async () => {
      let state: Health = { ready: false };
      try {
        const response = await fetch("/api/health", { cache: "no-store" });
        state = await response.json();
      } catch {}
      if (!active) return;
      setHealth(state);
      timer = setTimeout(check, state.ready ? 30000 : 4000);
    };
    check();
    return () => {
      active = false;
      clearTimeout(timer);
    };
  }, []);

  // descend sur chaque nouvelle question, puis suit la réponse sauf si l'utilisateur est remonté
  useEffect(() => {
    const el = scrollArea.current;
    const added = exchanges.length > exchangeCount.current;
    exchangeCount.current = exchanges.length;
    if (el && (added || el.scrollHeight - el.clientHeight - el.scrollTop < 240)) el.scrollTop = el.scrollHeight;
  }, [exchanges]);

  // menu mobile : Échap le referme, sauf quand la touche vise la fenêtre « À propos » ouverte par-dessus
  useEffect(() => {
    if (!menuOpen) return;
    closeButton.current?.focus();
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !(e.target as Element).closest("dialog")) setMenuOpen(false);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [menuOpen]);

  useEffect(() => {
    const el = textarea.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 220)}px`;
  }, [input]);

  const ask = useCallback(async (text: string, previous: Exchange[]) => {
    const question = text.trim();
    if (!question) return;
    const history = previous
      .filter((e) => e.status === "done")
      .flatMap((e) => [
        { role: "user", content: e.question },
        { role: "assistant", content: answerText(e) },
      ]);
    const id = ++nextId.current;
    const update = (f: (e: Exchange) => Exchange) => setExchanges((list) => list.map((e) => (e.id === id ? f(e) : e)));
    setExchanges([...previous, { id, question, tokens: [], status: "waiting" }]);
    setInput("");

    const ctrl = new AbortController();
    controller.current = ctrl;
    let reading = false;
    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: question, history }),
        signal: ctrl.signal,
      });
      if (!response.ok || !response.body) {
        const body = await response.json().catch(() => null);
        throw new Error(body?.error ?? `Le serveur a répondu ${response.status}.`);
      }
      reading = true;
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let ended = false;
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let i: number;
        while ((i = buffer.indexOf("\n")) >= 0) {
          const line = buffer.slice(0, i).trim();
          buffer = buffer.slice(i + 1);
          if (!line) continue;
          const event = JSON.parse(line) as StreamEvent;
          if (event.type === "token") {
            update((e) => ({ ...e, status: "streaming", tokens: [...e.tokens, { text: event.text, ids: event.ids }] }));
          } else if (event.type === "end") {
            ended = true;
            update((e) => ({
              ...e,
              status: event.interrupted ? "interrupted" : "done",
              meta: { tokens: event.tokens, seconds: event.seconds, truncated: event.truncated },
            }));
          }
        }
      }
      if (!ended) update((e) => ({ ...e, status: "interrupted" }));
    } catch (err) {
      if (ctrl.signal.aborted) {
        update((e) => ({ ...e, status: "interrupted" }));
      } else {
        const message = reading
          ? "La réponse du modèle s'est interrompue. Vérifie que le serveur du modèle tourne toujours."
          : err instanceof TypeError
            ? "L'interface web ne répond plus. Vérifie que « npm run dev » tourne toujours."
            : err instanceof Error
              ? err.message
              : String(err);
        update((e) => ({ ...e, status: "error", error: message }));
      }
    } finally {
      if (controller.current === ctrl) controller.current = null;
    }
  }, []);

  const send = (text: string) => {
    if (busy || !health?.ready || !text.trim()) return;
    ask(text, exchanges);
  };

  const stop = () => controller.current?.abort();

  const reset = () => {
    stop();
    setExchanges([]);
    setInput("");
    textarea.current?.focus();
  };

  const retry = (exchange: Exchange) => {
    if (busy) return;
    ask(
      exchange.question,
      exchanges.filter((e) => e.id !== exchange.id),
    );
  };

  return (
    <div className="flex min-h-0 flex-1">
      {/* colonne latérale sur ordinateur, menu plein écran sur mobile */}
      <aside
        id="menu"
        aria-label="Présentation et exemples"
        className={`${menuOpen ? "fixed inset-0 z-30 flex" : "hidden"} flex-col overflow-y-auto overscroll-contain bg-background px-4 pt-2.5 pb-[max(1rem,env(safe-area-inset-bottom))] [scrollbar-width:none] sm:px-6 lg:static lg:z-auto lg:flex lg:w-[22rem] lg:shrink-0 lg:border-r lg:border-border lg:px-8 lg:pt-8`}
      >
        <div className="flex justify-end lg:hidden">
          <button
            ref={closeButton}
            type="button"
            onClick={() => setMenuOpen(false)}
            aria-label="Fermer le menu"
            className="-mr-2 grid size-9 place-items-center rounded-lg text-muted transition-colors hover:bg-surface-raised hover:text-foreground"
          >
            <svg
              aria-hidden="true"
              viewBox="0 0 16 16"
              className="size-4"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.6"
              strokeLinecap="round"
            >
              <path d="M4 4l8 8M12 4l-8 8" />
            </svg>
          </button>
        </div>
        <div className="mt-4 lg:mt-0">
          <TokenTitle />
        </div>
        <p className="mt-5 text-[0.95rem] leading-relaxed text-pretty text-muted">
          Révise les cours NLP et Transformers d&apos;ING3 IA. Les réponses s&apos;écrivent token par token, comme dans
          les TP.
        </p>
        <h2 className="mt-8 text-sm font-medium">Exemples de questions</h2>
        <ul className="mt-3 mb-8 border-t border-border">
          {EXAMPLES.map((example) => (
            <li key={example} className="border-b border-border">
              <button
                type="button"
                onClick={() => {
                  setMenuOpen(false);
                  send(example);
                }}
                disabled={busy || !health?.ready}
                className="w-full py-2.5 text-left text-sm text-muted transition-colors hover:text-foreground disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:text-muted"
              >
                {example}
              </button>
            </li>
          ))}
        </ul>
        {/* même marge basse que la signature sous la saisie : les deux lignes restent alignées */}
        <div className="mt-auto">
          <About />
        </div>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="shrink-0">
          <div className="mx-auto flex h-14 max-w-2xl items-center gap-4 px-4 sm:px-6">
            <p className="text-[0.95rem] font-semibold tracking-tight lg:hidden">Tuteur NLP</p>
            <div className="ml-auto" aria-live="polite">
              <ServerStatus health={health} />
            </div>
            <button
              type="button"
              onClick={() => setMenuOpen(true)}
              aria-expanded={menuOpen}
              aria-controls="menu"
              aria-label="Ouvrir le menu"
              className="-mr-2 grid size-9 place-items-center rounded-lg text-muted transition-colors hover:bg-surface-raised hover:text-foreground lg:hidden"
            >
              <svg
                aria-hidden="true"
                viewBox="0 0 16 16"
                className="size-4"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.6"
                strokeLinecap="round"
              >
                <path d="M2.5 5.5h11M2.5 10.5h11" />
              </svg>
            </button>
          </div>
        </header>

        {/* conversation vide : question d'accueil et saisie centrées ; la saisie descend ensuite en bas */}
        <main className={`flex min-h-0 flex-1 flex-col ${exchanges.length === 0 ? "justify-center pb-[8vh]" : ""}`}>
          {exchanges.length === 0 ? (
            <p className="mb-8 px-4 text-center text-2xl font-medium tracking-tight text-balance sm:text-[1.75rem]">
              Qu&apos;est-ce qu&apos;on révise aujourd&apos;hui ?
            </p>
          ) : (
            <div ref={scrollArea} className="scroll-area min-h-0 flex-1 overflow-y-auto overscroll-contain">
              <ol
                role="log"
                aria-label="Conversation"
                className={`mx-auto w-full max-w-2xl px-4 py-4 sm:px-6 ${tokensVisible ? "tokens-visible" : ""}`}
              >
                {exchanges.map((exchange) => (
                  <li key={exchange.id} className="border-t border-border py-8 first:border-t-0">
                    <p className="text-lg leading-snug font-medium tracking-[-0.01em] text-pretty whitespace-pre-wrap break-words">
                      {exchange.question}
                    </p>
                    <div className="mt-4">
                      <Answer exchange={exchange} tokensVisible={tokensVisible} />
                      <AnswerFooter exchange={exchange} onRetry={() => retry(exchange)} />
                    </div>
                  </li>
                ))}
              </ol>
            </div>
          )}

          <form
            className="mx-auto w-full max-w-2xl shrink-0 px-4 pt-2 pb-3 sm:px-6"
            onSubmit={(e) => {
              e.preventDefault();
              if (busy) stop();
              else send(input);
            }}
          >
            {health && !health.ready && (
              <p className="mb-3 rounded-xl border border-red/20 bg-red/10 px-4 py-3 text-sm text-red">
                Le modèle n&apos;est pas lancé. Dans le dossier du projet, lance{" "}
                <code className="font-mono text-[0.85em] text-foreground">python -m nlp_tutor.server</code> et attends la
                fin du chargement.
              </p>
            )}
            <div className="rounded-2xl border border-border bg-surface transition-colors focus-within:border-border-strong">
              <label htmlFor="question" className="sr-only">
                Ta question
              </label>
              <textarea
                id="question"
                ref={textarea}
                rows={2}
                autoFocus
                maxLength={MAX_LENGTH}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
                    e.preventDefault();
                    send(input);
                  }
                }}
                placeholder="Pose ta question sur le cours…"
                className="block w-full resize-none bg-transparent px-4 pt-3.5 pb-1 leading-normal placeholder:text-subtle focus:outline-none"
              />
              <div className="flex items-center justify-between gap-2 px-2 pb-2">
                <label
                  className="flex cursor-pointer items-center gap-2 rounded-lg px-2 py-1.5 text-sm text-muted select-none hover:text-foreground"
                  title="Chaque surlignage est un token produit par le modèle : tu vois comment le tokenizer découpe le français."
                >
                  <input
                    type="checkbox"
                    role="switch"
                    className="peer sr-only"
                    checked={tokensVisible}
                    onChange={(e) => setTokensVisible(e.target.checked)}
                  />
                  <span
                    aria-hidden="true"
                    className="relative h-5 w-8 shrink-0 rounded-full bg-border-strong transition-colors peer-checked:bg-foreground peer-focus-visible:outline-2 peer-focus-visible:outline-offset-2 peer-focus-visible:outline-foreground after:absolute after:top-0.5 after:left-0.5 after:size-4 after:rounded-full after:bg-muted after:transition-[translate,background-color] peer-checked:after:translate-x-3 peer-checked:after:bg-background"
                  />
                  Afficher les tokens
                </label>
                <div className="flex items-center gap-1">
                  <button
                    type="button"
                    onClick={reset}
                    aria-label="Réinitialiser la conversation"
                    disabled={exchanges.length === 0 && !input}
                    className="rounded-lg px-3 py-1.5 text-sm text-muted transition-colors hover:bg-surface-raised hover:text-foreground disabled:pointer-events-none disabled:opacity-40"
                  >
                    Réinitialiser
                  </button>
                  <button
                    type="submit"
                    disabled={!busy && (!input.trim() || !health?.ready)}
                    className="grid size-9 place-items-center rounded-full bg-foreground text-background transition-colors hover:bg-white disabled:cursor-not-allowed disabled:bg-surface-raised disabled:text-subtle"
                  >
                    <span className="sr-only">{busy ? "Arrêter" : "Envoyer"}</span>
                    {busy ? (
                      <svg aria-hidden="true" viewBox="0 0 16 16" className="size-3" fill="currentColor">
                        <rect width="16" height="16" rx="3" />
                      </svg>
                    ) : (
                      <svg
                        aria-hidden="true"
                        viewBox="0 0 16 16"
                        className="size-4"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="1.8"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      >
                        <path d="M8 13V3M3.5 7.5 8 3l4.5 4.5" />
                      </svg>
                    )}
                  </button>
                </div>
              </div>
            </div>
            {input.length > MAX_LENGTH - 200 && (
              <p className="mt-2.5 text-center text-xs text-muted">
                {formatNumber(MAX_LENGTH - input.length)} caractères restants.
              </p>
            )}
          </form>
        </main>
        <Signature />
      </div>
    </div>
  );
}
