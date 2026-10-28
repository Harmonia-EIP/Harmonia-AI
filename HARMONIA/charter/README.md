# Charte Harmonia v3

`harmonia_v3.json` est la **seule source** de ce qu'est un preset Harmonia v3 : chaque paramètre (identifiant, module, plage, courbe, valeur par défaut, choix, unité, libellés anglais et français, description), le format des presets, le passage depuis la v2 et la fiche de son. Le plugin (Harmonia-APP), le backend et l'IA lisent tous ce fichier ou ce qui en est généré. On ne modifie que le JSON, puis on régénère.

| Généré depuis la charte | Pour | Comment |
|---|---|---|
| `generated/HarmoniaCharterV3.h` | plugin : table C++17 de tous les paramètres et fonctions de normalisation, sans dépendance à JUCE | `scripts/v3/charter_cpp.py` |
| `generated/HarmoniaParametersV3.h` | plugin : layout JUCE (`AudioProcessorValueTreeState`), un groupe par module et par opérateur DX7, passage d'un état v2 | `scripts/v3/charter_cpp.py` |
| `src/charter_v3.py` | IA et backend : lecture de la charte, presets, validation, passage v2 (bibliothèque standard seulement) | lu au chargement |
| `src/synth/v3_params.py` | moteur v3 et modèles de l'IA : mêmes 46 paramètres, même ordre | construit depuis `src/charter_v3.py` |

## Contenu

- **192 paramètres** :
  - `engine` (analogique ou DX7) ;
  - les **46 du moteur analogique**, dans l'ordre des vecteurs de l'IA (`src/v3/codecs.py`) : on ne réordonne jamais, on ajoute à la fin ;
  - les **145 du mode DX7** : 21 par opérateur × 6, plus 19 pour la voix. Ils sont décrits une fois par opérateur, puis déployés en `dx7_op1_eg_rate_1` … `dx7_transpose`.
- **19 groupes** pour l'interface : oscillateurs, unisson et stéréo, filtre, enveloppes, LFO, effets, un groupe par opérateur DX7, la voix DX7.
- **Normalisation** : les presets transportent des valeurs entre 0 et 1, comme les hôtes VST, avec les courbes de `juce::NormalisableRange`. Une valeur `centre` place cette valeur à 0,5 (par exemple 1 000 Hz pour la coupure).
- **Arrondi** : il suit exactement les paramètres JUCE du plugin. Les entiers et les choix vont à l'entier le plus proche, une demie allant au nombre pair (`juce::roundToInt` : 8,5 → 8). Un booléen passe à 1 à partir de la moitié. C'est vérifié avec JUCE 8.0.8 : les 192 paramètres créés par le layout généré donnent les mêmes valeurs que la charte.
- **Passage depuis la v2** : `v3 = offset + scale × v2[from]`, sur les valeurs physiques. Un état JUCE stocke des valeurs physiques, et quatre paramètres ont changé d'unité : `filter_env_amount` (× 4 octaves), `lfo_to_pitch` (× 50 cents), `lfo_to_cutoff` (× 2 octaves), et `reverb_size`, qui se déduit de `reverb_mix`. Côté plugin, c'est `HarmoniaParamsV3::applyV2State`.
- **Fiche de son** (`sound_brief`, brouillon v0.1) : ce qu'un prompt demande, en mots qu'on peut vérifier sur un son. Les champs sont la famille, l'attaque, la brillance, la durée, le mouvement, la largeur, l'espace, des références (noms d'instruments, de synthés, d'époques) et le moteur préféré.
  - Pour un vrai preset, la fiche se lit dans ses étiquettes humaines et se **mesure** sur son rendu.
  - Les seuils sont des quantiles des vrais presets de la banque : aucun seuil n'est inventé.
  - Elle est à valider à l'écoute avant d'entraîner un modèle dessus.

## Format d'un preset

```json
{
  "metadata": { "name": "Exemple", "charter_version": "3.0.0", "engine": "analog", "generated_by": "harmonia_v3" },
  "parameters": { "engine": 0.0, "osc_1_waveform": 0.6667, "filter_cutoff": 0.5, "unison_voices": 0.2, "…": 0.0 },
  "brief": { "family": "bass", "attack": "instant" }
}
```

C'est la même forme qu'en v1/v2 (`metadata` + `parameters`). Le `PresetCharterSchema` du backend et le `PresetLoader` du plugin la lisent donc sans changement. `engine` est un paramètre et pas seulement une métadonnée : le backend ignore les champs de `metadata` qu'il ne connaît pas, alors qu'il transmet `parameters` tel quel. Un paramètre absent prend sa valeur par défaut, et les paramètres de l'autre moteur sont ignorés.

## Commandes

```bash
python scripts/v3/charter_cpp.py              # régénère generated/ après une modification du JSON
python scripts/v3/charter_cpp.py --check      # échoue si generated/ n'est plus à jour (aussi testé par pytest)
python scripts/v3/charter_cpp.py --out ../../Harmonia-APP/src/parameters   # écrit directement dans l'app
python -m pytest tests/test_charter_v3.py     # charte, DX7, normalisation, presets, v2, C++ compilé
```

## Intégration dans le plugin (à faire)

1. Copier les deux en-têtes dans `Harmonia-APP/src/parameters/`.
2. Remplacer `HarmoniaParams::createLayout()` par `HarmoniaParamsV3::createLayout()`.
3. Porter `src/synth/engine_v3.py`, qui sert de référence, et brancher le mode DX7 (`native/dx7`, msfa Apache-2.0).
4. Dans `setStateInformation`, appeler `applyV2State` pour les états enregistrés avant la v3.

Écarts relevés le 2026-10-28 :
- le moteur de `dev` calcule la résonance avec `16^(r/0.95)`, alors que la charte et les modèles v3 utilisent `17^(r/0.95)` (Harmonia-App#40, pas encore dans `dev`) ;
- la route `/refine-preset` appelée par l'app et le backend n'existe pas encore côté IA.
