"use client";

import { useRef } from "react";

const SECTIONS = [
  {
    title: "Ce qu'il fait",
    text: "Il explique une notion du cours, déroule un calcul (softmax, TF-IDF, F1, perte…), compare deux notions, t'interroge et corrige ta réponse.",
  },
  {
    title: "Ce qu'il refuse",
    text: "Les questions hors du cours, les travaux notés à faire à ta place et les questions administratives (dates, notes, contacts).",
  },
  {
    title: "Ses limites",
    text: "C'est un petit modèle (Qwen2.5-0.5B-Instruct adapté par LoRA). Il se trompe souvent sur le fond, invente des informations et répète parfois la même phrase jusqu'à la limite de longueur. Vérifie dans tes supports.",
  },
];

export function About() {
  const dialog = useRef<HTMLDialogElement>(null);
  return (
    <>
      <button
        type="button"
        onClick={() => dialog.current?.showModal()}
        className="text-sm text-muted transition-colors hover:text-foreground"
      >
        À propos du tuteur
      </button>
      <dialog
        ref={dialog}
        aria-labelledby="about-title"
        onClick={(e) => e.target === e.currentTarget && e.currentTarget.close()}
        className="m-auto w-[calc(100%-2rem)] max-w-md rounded-2xl border border-border bg-surface text-foreground backdrop:bg-background/70 backdrop:backdrop-blur-sm"
      >
        <div className="p-6">
          <div className="flex items-center justify-between gap-4">
            <h2 id="about-title" className="font-semibold tracking-tight">
              À propos du tuteur
            </h2>
            <button
              type="button"
              onClick={() => dialog.current?.close()}
              aria-label="Fermer"
              className="-mr-2 grid size-8 place-items-center rounded-lg text-muted transition-colors hover:bg-surface-raised hover:text-foreground"
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
          <dl className="mt-5 space-y-5 text-sm leading-relaxed">
            {SECTIONS.map((s) => (
              <div key={s.title} className="grid gap-1">
                <dt className="font-medium text-foreground">{s.title}</dt>
                <dd className="text-muted">{s.text}</dd>
              </div>
            ))}
          </dl>
        </div>
      </dialog>
    </>
  );
}
