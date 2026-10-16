"""Indicateurs automatiques de format et détecteur de refus (parties 3, 4 et 6)."""
import re

FRENCH_FUNCTION_WORDS = set("le la les de des du un une et est en que qui pour dans sur pas ce il on tu avec "
                            "par au aux ne se sa son ses ton ta tes mais ou donc plus cette ces".split())

# Formulations de refus ou de redirection, validées en partie 4 sur les réponses de référence
REFUSAL_PATTERNS = [
    r"je ne (peux|vais|donne|fais|change|te donne|réponds) pas",
    r"je n'ai pas (accès|d'information|cette information|de tarif|ce chiffre|d'avis)",
    r"\b(sort|sortent) (du cadre|de mon (rôle|périmètre)|complètement de mon rôle|du périmètre)",
    r"\bne (fait|font) pas partie (du cours|de mon rôle)",
    r"\b(est|sont|reste|restent) hors (du |de mon )?(périmètre|cadre|sujet)",
    r"\brelèvent? (d'un|du|de la) ",
    r"\bje (garde|reste dans|vais rester dans) mon rôle|je ne change pas de rôle",
    r"\bje préfère (ne pas|te renvoyer|te guider)",
    r"\b(je suis|mon rôle se) limit",
    r"\ble cours ne (donne|précise|parle) pas",
    r"\bdépasse le cours",
    r"pas en mesure de",
    r"\bdésolé",
    r"i'm sorry|i cannot|i can't",
]


def is_french(text):
    """Pas d'écriture chinoise ou japonaise, et au moins 15 % de mots-outils français."""
    if re.search(r"[぀-ヿ一-鿿]", text):
        return False
    words = re.findall(r"[a-zàâçéèêëîïôûùüÿœ']+", text.lower())
    return bool(words) and sum(w in FRENCH_FUNCTION_WORDS for w in words) / len(words) >= 0.15


def detects_refusal(text):
    text = text.lower().replace("’", "'")
    return any(re.search(p, text) for p in REFUSAL_PATTERNS)


def answer_metrics(answer, should_redirect):
    refusal = detects_refusal(answer)
    return {
        "french": is_french(answer),
        "≤ 150 words": len(answer.split()) <= 150,
        "final question": answer.rstrip().endswith("?"),
        "correct scope": refusal == should_redirect,
        "missed refusal": should_redirect and not refusal,
        "unwarranted refusal": refusal and not should_redirect,
        "words": len(answer.split()),
    }
