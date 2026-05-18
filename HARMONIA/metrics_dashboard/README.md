# Harmonia Metrics Dashboard

Site live des métriques du moteur IA Harmonia.
URL prod : <https://harmonia.mcoet.com/>

## Architecture

| Fichier                       | Rôle                                                              |
| ----------------------------- | ----------------------------------------------------------------- |
| `index.html`                  | SPA dashboard (Tailwind + Chart.js, palette synthwave)            |
| `receiver.php`                | Endpoint **POST** sécurisé par token. Multi-kind, écrit `events/` |
| `api.php`                     | Endpoint **GET** public. Lit `events/`, `models.json`, etc.       |
| `events/`                     | Un fichier JSON par évènement (training/generation/command/...)   |
| `index.json`                  | Index sommaire de tous les évènements (kind + résumé)             |
| `models.json`                 | Index par `model_version` avec compteurs cumulés                  |
| `presets.json`                | Catalogue des générations (prompt + 20 paramètres)                |
| `latest_metrics.json`         | Dernier rapport training brut (compat legacy)                     |
| `history_metrics.json`        | Historique training (compat legacy)                               |
| `stats_snapshot.json`         | Snapshot complet produit par `scripts/dashboard_stats.py`         |
| `push_metrics.{sh,py}`        | Push manuel d'un rapport (legacy)                                 |

## Vues côté navigateur

L'`index.html` est une SPA avec 6 vues, navigables via la sidebar :

- **Overview** : modèle actif, compteurs, loss + erreur par paramètre, flux temps réel.
- **Modèles** : tuiles cliquables par `model_version`. Cliquer une tuile ouvre le détail (loss au fil des runs, radar erreur par paramètre, liste des trainings + presets).
- **Entraînement** : table complète + comparateur (filter par modèle).
- **Presets générés** : grille avec sparkline 20 paramètres + recherche par prompt.
- **Activité** : journal des évènements (training, generation, command, system).
- **Charte 20 paramètres** : description officielle des contrôles HARMONIA.

Chaque ligne/évènement est cliquable et ouvre une fenêtre avec le payload complet + bargraph 20 paramètres si applicable.

## Push vers le site

Les scripts Harmonia poussent automatiquement à chaque action :

| Action                                | Évènement poussé      |
| ------------------------------------- | --------------------- |
| `python scripts/train.py`             | `training` + `command`|
| `python scripts/generate.py "..."`    | `generation` + `command` |
| `python scripts/prepare_dataset.py`   | `command` (dataset)   |
| `python scripts/server.py` `/generate`| `generation` (kind=http) |
| `python scripts/dashboard_stats.py`   | `system` (snapshot)   |

### Wrapper "tout terminal"

```bash
scripts/harmonia.sh <ta-commande> [args...]
```

Encadre n'importe quelle commande : capture exit code, durée, et publie un event `command`. Exemple :

```bash
scripts/harmonia.sh make check
scripts/harmonia.sh python scripts/train.py
```

### Désactiver les push (ex : tests, hors-ligne)

```bash
export HARMONIA_PUSH_METRICS=0
```

Les évènements continuent d'être miroirés en local (`metrics_dashboard/events/`).

## Commandes locales

```bash
make dashboard-stats     # build snapshot + push
make dashboard-snapshot  # build snapshot local (no push)
make dashboard-serve     # php -S 127.0.0.1:8080 sur metrics_dashboard/
```

`make dashboard-serve` requiert PHP installé localement (mac : `brew install php`).
Sans PHP, ouvrir directement `metrics_dashboard/index.html` fonctionne : la SPA bascule automatiquement en mode "fichiers statiques" (lit `*.json` locaux).

## Déploiement serveur

1. Pousser les fichiers du dossier sur `harmonia.mcoet.com` (PHP 8+).
2. Configurer le token :

```apache
SetEnv METRICS_PUSH_TOKEN "TON_TOKEN_TRES_FORT"
```

3. Donner le droit d'écriture à PHP sur `events/`, `index.json`, `models.json`, `presets.json`, `history_metrics.json`, `latest_metrics.json`.

4. Le dashboard JS pointe automatiquement sur `api.php` (même domaine) ; cross-origin et `Access-Control-Allow-Origin: *` sont gérés.

## Token côté client

env :

```bash
HARMONIA/metrics_dashboard/.env
# METRICS_TOKEN=ton_token
```

## Format des évènements

Chaque event POST a la forme :

```json
{
  "event_kind": "training" | "generation" | "command" | "system" | "dataset",
  "timestamp": "2026-05-12T12:34:56Z",
  "model_version": "charter_v1",
  "model_hash": "...",
  "...": "champs spécifiques selon kind"
}
```

Le receiver détermine le kind à partir du champ `event_kind` ou du query string `?kind=...`.
