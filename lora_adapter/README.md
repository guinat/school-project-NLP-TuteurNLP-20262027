---
base_model: Qwen/Qwen2.5-0.5B-Instruct
library_name: peft
language: fr
---

# Tuteur NLP : adaptateur LoRA

Adaptateur LoRA de **Qwen2.5-0.5B-Instruct** pour le tuteur de révision du cours de NLP et de
Transformers (ING3 IA, CY Tech). Entraînement : `agent.ipynb`, partie 5 ; évaluation : partie 6 et
`rapport.pdf`.

## Entraînement

| | |
|---|---|
| modèle de base | Qwen/Qwen2.5-0.5B-Instruct, gelé, float32 (pas de quantification) |
| modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` (24 couches) |
| r, α, dropout | 16, 32, 0,05 (8,8 M de paramètres entraînables, 1,75 %) |
| données | 240 dialogues écrits à la main (`data/splits/train.jsonl`), perte sur les seuls tokens de réponse |
| optimisation | AdamW, 2·10⁻⁴, 10 % d'échauffement puis décroissance linéaire, lots de 8, 512 tokens au plus |
| fichier livré | point du pas 10 sur 90, perte de validation 2,421 ; le meilleur point (pas 60, 2,123) n'a pas été conservé |

## Utilisation

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = PeftModel.from_pretrained(base, "lora_adapter")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
```

Prompt système à utiliser (celui de l'entraînement) : « Tu es Tuteur NLP, un assistant qui aide les
étudiants d'ING3 IA de CY Tech à réviser leur cours de NLP et de Transformers. »

## Limites

Mesurées sur les 39 requêtes d'évaluation, en greedy (`rapport.pdf`, section 5) :

- 32 réponses sur 39 bouclent sur la même phrase jusqu'à la limite de 320 tokens ;
- la grille d'évaluation passe de 0,35 pour la baseline à 0,27, et aucune des 14 demandes à refuser
  ou rediriger n'est traitée correctement ;
- le modèle ne connaît pas le contenu du cours et invente des définitions, des chiffres et des
  coordonnées.

## Versions

- PEFT 0.13.2
