# Instruments échantillonnés (#41, épopée #45)

Le moteur analogique et le DX7 font de bons sons de synthé, mais jamais un vrai piano ou un vrai orchestre.
La source « échantillon » joue de vrais instruments enregistrés. Le preset choisit l'instrument et règle tout
le reste (filtre, enveloppes, effets) : l'IA génère toujours **un preset**, jamais un fichier audio.

Les enregistrements sont la matière première jouée par le moteur. Ils ne sont **jamais dans git** (GitHub
refuse les fichiers de plus de 100 Mo, et il y en a plusieurs Go) : ils sont téléchargés depuis leurs sources
officielles, puis publiés en paquets de release, comme les modèles IA.

## Bibliothèques

| Bibliothèque | Licence | Ce qu'on en garde |
|---|---|---|
| Salamander Grand Piano V3 (Alexander Holm) | CC BY 3.0 : l'auteur est crédité | le piano complet (16 couches de vélocité, bruits de marteau) |
| VSCO 2 Community Edition (Versilian Studios) | CC0 | cordes, cuivres, bois, claviers, percussions mélodiques |
| VCSL (Versilian Studios) | CC0 | pianos, clavecins, orgues, harpes, flûtes à bec, harmonicas, saxophones, percussions mélodiques |

Les versions sont épinglées dans [sources.json](sources.json) (adresse et sha256). Versilian Studios demande
de ne pas vendre les échantillons eux-mêmes : Harmonia ne les vend pas, ils accompagnent le plugin avec leurs
crédits (`CREDITS.md` dans chaque paquet).

## Catalogue ([instruments.json](instruments.json))

Un instrument est une articulation d'un instrument enregistré (« Violin Section · susVib »). Pour chacun : les
notes enregistrées, les couches de vélocité, la tessiture, et quel fichier répond à quelle touche et quelle
vélocité (dans `catalog.json`, livré avec les paquets).

- Salamander est lu depuis sa carte SFZ (`src/samples/sfz.py`).
- VSCO 2 et VCSL nomment leurs fichiers d'après la note (`VlnEns_susVib_A2_v1.wav`, `src/samples/names.py`),
  mais pas avec la même numérotation d'octave. Chaque enregistrement est donc écouté (`src/samples/pitch.py`) :
  la note du nom est comparée à ce que joue l'audio, octave par octave. Un instrument n'est gardé que si au
  moins trois quarts de ses enregistrements s'accordent sur l'octave : mieux vaut l'écarter que le jouer faux.

Résultat : **125 instruments**, 4 713 enregistrements :

| Famille | Instruments |
|---|---|
| claviers (5 pianos, 7 clavecins, clavinet et pianos FM) | 18 |
| cordes (violon, alto, violoncelle, contrebasse ; tenues, pizzicati, trémolos, spiccatos) | 21 |
| cuivres (trompette, cor, trombones, tuba) | 17 |
| bois (flûtes, hautbois, clarinette, basson, flûtes à bec, ocarinas, harmonicas, saxophones) | 42 |
| cordes pincées (harpes, dan tranh, psaltérion, strumstick) | 10 |
| percussions mélodiques (marimba, xylophone, vibraphone, cloches tubulaires…) | 10 |
| orgues | 7 |

Laissés de côté pour l'instant (165 dossiers, raison notée dans `instruments.json`) : les sons de relâchement
(ils viendront avec le moteur), les percussions sans hauteur (batteries, effets), et les instruments dont
l'octave reste douteuse (glockenspiels, kalimbas, piccolos, balafons).

## Paquets ([release.json](release.json))

| Paquet | Pour | Contenu | Taille |
|---|---|---|---|
| `full-salamander`, `full-vsco2`, `full-vcsl` | le plugin | tous les enregistrements du catalogue, en FLAC sans perte (même fréquence et même résolution qu'enregistrés) | 0,72 + 0,62 + 1,57 Go |
| `listen` | le serveur de l'IA | par instrument et par note enregistrée, une prise de vélocité moyenne, mono, 32 kHz, 4 s | 0,14 Go |

Pendant une génération, l'IA joue chaque preset candidat et l'écoute pour garder le plus proche du prompt :
elle a besoin des sons, mais pas en qualité studio. Le petit paquet `listen` suffit au VPS ; le plugin
télécharge les paquets complets.

Chaque archive fait moins de 1,9 Go (la limite d'un fichier de release GitHub est 2 Go) et contient
`catalog.json` et `CREDITS.md`. Les paquets ne sont pas encore publiés : ils le seront quand le plugin saura
les lire (#58).

## Commandes

```bash
python scripts/samples/fetch_samples.py      # ~/Datasets/harmonia-samples/<bibliothèque>/ (7 Go à télécharger)
python scripts/samples/build_catalog.py      # catalog.json + samples/instruments.json
python scripts/samples/build_packs.py --tag samples-1.0.0   # dist/samples/*.tar + samples/release.json
```

La CI ne télécharge jamais d'échantillons : les tests fabriquent leurs propres petits sons.
