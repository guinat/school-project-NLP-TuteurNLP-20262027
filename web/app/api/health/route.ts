import { TUTOR_API_URL } from "@/lib/api";

export async function GET() {
  try {
    const response = await fetch(`${TUTOR_API_URL}/health`, {
      cache: "no-store",
      signal: AbortSignal.timeout(3000),
    });
    if (!response.ok) throw new Error(String(response.status));
    return Response.json(await response.json(), { headers: { "Cache-Control": "no-store" } });
  } catch {
    return Response.json({ ready: false }, { status: 503, headers: { "Cache-Control": "no-store" } });
  }
}
