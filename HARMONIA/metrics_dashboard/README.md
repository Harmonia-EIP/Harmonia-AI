# Metrics Dashboard (Static + PHP)

Petit tableau de bord statique pour visualiser les metriques de ton modele IA sur un hebergement Apache/PHP.

## Fichiers

- `index.html` : dashboard (Tailwind CDN + Chart.js CDN)
- `receiver.php` : endpoint `POST` securise par token, sauvegarde dans `latest_metrics.json`
- `push_metrics.sh` : envoi des metriques via `curl`
- `push_metrics.py` : envoi des metriques via Python `requests`

## 1) Deploiement sur ton domaine

Place les fichiers de ce dossier dans ton vhost (`harmonia.mcoet.com` ou `metrics.mcoet.com`).

Configure un token cote serveur (Apache):

```apache
SetEnv METRICS_PUSH_TOKEN "TON_TOKEN_TRES_FORT"
```

Si cette variable n'est pas configuree, `receiver.php` refuse les requetes (HTTP 500).

## 1b) Token local pour l'envoi automatique depuis Harmonia

Dans le repo Harmonia, le token peut etre stocke localement dans un fichier ignore par git:

```bash
HARMONIA/metrics_dashboard/.env.local
```

Contenu:

```env
METRICS_TOKEN=TON_TOKEN_TRES_FORT
```

Le helper Python et le training loop lisent ce token automatiquement si aucune variable d'environnement n'est fournie.

## 2) Test local rapide

Lancer un serveur PHP local:

```bash
cd HARMONIA/metrics_dashboard
php -S 127.0.0.1:8080
```

Dans un autre terminal, pousser un JSON de test:

```bash
cd HARMONIA/metrics_dashboard
cat > sample_eval_report.json <<'JSON'
{
  "timestamp": "2026-04-26T17:15:00Z",
  "model_version": "train-test-v1",
  "metrics": {
    "mse": 0.0321,
    "mae": 0.1104,
    "final_loss": 0.0289,
    "accuracy": 0.91,
    "per_param_mse": {
      "Cutoff": 0.024,
      "Resonance": 0.041,
      "Attack": 0.018
    }
  },
  "loss_history": [0.18, 0.14, 0.11, 0.08, 0.06, 0.04, 0.03]
}
JSON

METRICS_URL="http://127.0.0.1:8080/receiver.php" \
METRICS_TOKEN="TON_TOKEN_TRES_FORT" \
./push_metrics.sh sample_eval_report.json
```

Puis ouvre:

- `http://127.0.0.1:8080/index.html`

## 3) Envoi automatique depuis ton training PyTorch

### Option A - commande `curl`

```bash
curl -X POST "https://harmonia.mcoet.com/receiver.php" \
  -H "Authorization: Bearer TON_TOKEN" \
  -F "metrics_file=@/chemin/vers/eval_report.json;type=application/json"
```

### Option B - script shell

```bash
METRICS_URL="https://harmonia.mcoet.com/receiver.php" \
METRICS_TOKEN="TON_TOKEN" \
./push_metrics.sh /chemin/vers/eval_report.json
```

### Option C - script Python `requests`

```bash
python3 -m pip install requests
python3 push_metrics.py /chemin/vers/eval_report.json --url "https://harmonia.mcoet.com/receiver.php" --token "TON_TOKEN"
```

### Option D - envoi automatique integre au projet Harmonia

- `scripts/train.py` pousse automatiquement le dernier `eval_*.json` apres l'entraienement.
- `make test` pousse aussi le dernier rapport local si un benchmark existe deja.
- Le comportement peut etre coupe avec `HARMONIA_PUSH_METRICS=0`.
- Le workflow CI pousse aussi un payload de statut a chaque `push` quand `METRICS_PUSH_TOKEN` est configure dans les secrets GitHub.

## Notes

- `index.html` tente de charger `latest_metrics.json`.
- Si le fichier n'existe pas encore, il affiche automatiquement des donnees simulees.
- Le dashboard sait aussi parser des structures proches de ton `eval_report` actuel (`metrics`, `latest_evaluation_report.metrics`, `latest_benchmark.eval_metrics`).

