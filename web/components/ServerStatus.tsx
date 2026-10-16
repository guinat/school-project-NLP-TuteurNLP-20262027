import type { Health } from "@/lib/types";

export function ServerStatus({ health }: { health: Health | null }) {
  const [color, label] =
    health === null
      ? ["bg-subtle", "Connexion au modèle…"]
      : health.ready
        ? ["bg-green", "Modèle chargé"]
        : ["bg-red", "Modèle arrêté"];
  return (
    <p className="flex items-center gap-2 text-sm text-muted">
      <span className={`size-1.5 rounded-full ${color}`} aria-hidden="true" />
      {label}
      {health?.ready && (
        <span className="-ml-1 hidden sm:inline">
          sur {health.device === "cpu" ? "le processeur" : "la carte graphique"}
        </span>
      )}
    </p>
  );
}
