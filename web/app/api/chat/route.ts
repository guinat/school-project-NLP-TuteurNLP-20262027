import { SERVER_DOWN_MESSAGE, TUTOR_API_URL } from "@/lib/api";

// Relaie la question à l'API Python et renvoie son flux de tokens sans le mettre en tampon.
export async function POST(request: Request) {
  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return Response.json({ error: "La requête n'est pas du JSON valide." }, { status: 400 });
  }

  let upstream: Response;
  try {
    upstream = await fetch(`${TUTOR_API_URL}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: request.signal,
    });
  } catch {
    return Response.json({ error: SERVER_DOWN_MESSAGE }, { status: 502 });
  }

  if (!upstream.ok || !upstream.body) {
    const error =
      upstream.status === 422
        ? "Le message est vide ou trop long (2 000 caractères au plus)."
        : `Le serveur du modèle a répondu ${upstream.status}.`;
    return Response.json({ error }, { status: upstream.status === 422 ? 400 : 502 });
  }

  return new Response(upstream.body, {
    headers: {
      "Content-Type": "application/x-ndjson; charset=utf-8",
      "Cache-Control": "no-store, no-transform",
    },
  });
}
