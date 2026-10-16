// Logo « guinat.ai », repris de www.guinat.ai
const LOGO = [
  "M11.6364 46.5455H23.2727V58.1818H0V0H34.9091V11.6364H11.6364V46.5455ZM23.2727 23.2727H34.9091V46.5455H23.2727V23.2727Z",
  "M58.1364 46.5455H46.5V11.6364H58.1364V46.5455ZM58.1364 46.5455H69.7727V11.6364H81.4091V58.1818H58.1364V46.5455Z",
  "M93 0H104.636V11.6364H93V0ZM93 23.2727H104.636V58.1818H93V23.2727Z",
  "M139.523 23.2727H127.886V58.1818H116.25V11.6364H139.523V23.2727ZM139.523 23.2727H151.159V58.1818H139.523V23.2727Z",
  "M174.386 11.6364H197.659V58.1818H174.386V46.5455H186.023V23.2727H174.386V11.6364ZM162.75 23.2727H174.386V46.5455H162.75V23.2727Z",
  "M220.886 46.5455H232.523V58.1818H209.25V0H220.886V11.6364H232.523V23.2727H220.886V46.5455Z",
  "M244.125 46.5455H255.761V58.1818H244.125V46.5455Z",
  "M279.011 11.6364H302.284V58.1818H279.011V46.5455H290.648V23.2727H279.011V11.6364ZM267.375 23.2727H279.011V46.5455H267.375V23.2727Z",
  "M313.875 0H325.511V11.6364H313.875V0ZM313.875 23.2727H325.511V58.1818H313.875V23.2727Z",
];

// Sous la saisie, avec la même marge basse que la colonne latérale : les deux dernières lignes restent alignées
export function Signature() {
  return (
    <footer className="flex shrink-0 items-center justify-center gap-2.5 px-4 pb-[max(1rem,env(safe-area-inset-bottom))] text-sm text-muted">
      Site réalisé par
      <a
        href="https://www.guinat.ai"
        target="_blank"
        rel="noopener noreferrer"
        aria-label="guinat.ai"
        className="text-foreground transition-opacity hover:opacity-60"
      >
        <svg aria-hidden="true" viewBox="0 0 326 59" className="h-3.5 w-auto" fill="currentColor">
          {LOGO.map((d) => (
            <path key={d.slice(0, 24)} d={d} />
          ))}
        </svg>
      </a>
    </footer>
  );
}
