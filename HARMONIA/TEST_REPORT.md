# Test Report - Projet HARMONIA IA

**Execution DAte:** 09/11/2026
**Target Environment:** Windows Pc

## Test Execution Results

| ID | Module | Test Scenario | Expected Result | Actual Result | Status | Ticket |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ENV-01** | Setup | Exécuter `make setup` ou l'installation manuelle via l'environnement virtuel (`requirements-ci.txt`, `requirements-dev.txt`) | L'environnement virtuel `.venv` est créé et toutes les dépendances sont installées sans erreur. | Installation réussie | ✅ SUCCESS | - |
| **CI-01** | Checks | Exécuter la suite de tests et d'analyse statique via `make check` | `pytest`, `bandit` et `compileall` s'exécutent avec succès et ne remontent aucune erreur bloquante. | Tests et vérifications de sécurité OK | ✅ SUCCESS | - |
| **DATA-01** | Dataset | Préparer les données avec `python3 scripts/prepare_dataset.py` depuis un fichier dump textuel brut | Le script génère correctement le fichier de sortie `data/processed/presets.json` ou `.npy`. | Fichier généré avec succès | ✅ SUCCESS | - |
| **TRAIN-01** | Training | Lancer un test d'entraînement rapide (smoke test) via `make train-fast` | L'entraînement effectue 1 epoch avec succès et sauvegarde les fichiers (`.pth`, `.meta.json`) dans `saved_models/`. | 1 epoch complétée, modèles sauvegardés | ✅ SUCCESS | - |
| **TRAIN-02** | Training | Vérifier la reproductibilité avec un seed fixe (`HARMONIA_SEED=123 python3 scripts/train.py`) | Les résultats de l'entraînement et les poids générés sont identiques pour deux exécutions avec le même seed. | Reproductibilité confirmée | ✅ SUCCESS | - |
| **CLI-01** | Inference | Générer un preset via la CLI : `python3 scripts/generate.py "Soft Piano"` | Un fichier JSON valide contenant le preset est créé dans le chemin de sortie spécifié. | Fichier JSON généré | ✅ SUCCESS | - |
| **API-01** | Server | Démarrer l'API (`scripts/server.py`) et appeler `GET /health` | L'API répond 200 OK et expose correctement `model_version` et `model_hash`. | Serveur UP, statut 200 | ✅ SUCCESS | - |
| **API-02** | Server | Appeler `POST /generate` avec un payload valide (`{"prompt": "Dark Reese Bass"}`) | L'API répond 200 OK et retourne le preset généré. | Preset généré via API | ✅ SUCCESS | - |
| **API-03** | Validation | Envoyer un prompt vide ou contenant uniquement des espaces à `POST /generate` | L'API rejette la requête et retourne un code HTTP 400. | Rejet correct (Erreur 400) | ✅ SUCCESS | - |
| **API-04** | Validation | Envoyer un prompt dépassant la limite de caractères (512) ou de tokens (32) | L'API rejette la requête et retourne un code HTTP 400. | Rejet correct (Erreur 400) | ✅ SUCCESS | - |
| **METRICS-01** | Pipeline | Exécuter le calcul des métriques locales avec `make metrics-local` | Les fichiers d'évaluation `eval_*.json` et `history.json` sont mis à jour dans `benchmarks/`. | Métriques générées et stockées | ✅ SUCCESS | - |
