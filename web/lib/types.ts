export type Token = { text: string; ids: number[] };

export type Exchange = {
  id: number;
  question: string;
  tokens: Token[];
  status: "waiting" | "streaming" | "done" | "interrupted" | "error";
  meta?: { tokens: number; seconds: number; truncated: boolean };
  error?: string;
};

export type StreamEvent =
  | { type: "start"; prompt_tokens: number }
  | { type: "token"; text: string; ids: number[] }
  | { type: "end"; tokens: number; seconds: number; interrupted: boolean; truncated: boolean };

export type Health =
  | { ready: true; model: string; adapter: string; device: string; max_turns: number; max_new_tokens: number }
  | { ready: false };
