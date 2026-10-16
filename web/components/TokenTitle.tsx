// « Tuteur NLP » tel que le découpe le tokenizer de Qwen2.5, avec les identifiants des tokens
const TITLE_TOKENS = [
  { text: "T", id: 51 },
  { text: "uteur", id: 38995 },
  { text: " N", id: 451 },
  { text: "LP", id: 12567 },
];

export function TokenTitle() {
  return (
    <h1 aria-label="Tuteur NLP">
      <span
        aria-hidden="true"
        className="flex flex-wrap items-start gap-x-[0.07em] text-[2.25rem] leading-none font-semibold tracking-[-0.035em]"
        style={{ fontVariationSettings: '"wdth" 112' }}
      >
        {TITLE_TOKENS.map((t, i) => {
          const delay = { animationDelay: `${150 + i * 140}ms` };
          return (
            <span key={t.id} className="token inline-flex flex-col">
              <span className="whitespace-pre">{t.text}</span>
              <span className="token-bar mt-[0.2em] h-[3px] rounded-full bg-(--hue)" style={delay} />
              <span
                className="token-id mt-2.5 text-xs font-normal tracking-normal text-muted tabular-nums"
                style={{ ...delay, fontVariationSettings: "normal" }}
              >
                {t.id}
              </span>
            </span>
          );
        })}
      </span>
    </h1>
  );
}
