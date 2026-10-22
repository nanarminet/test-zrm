# ZRM Agent IA — Guide de lancement

## Ce que fait cette application
Un parcours guidé en 4 étapes pour construire ton Motto d'intention
selon la méthode ZRM, assisté par une IA (modèles hébergés par Ollama).

---

## Installation (à faire une seule fois)

### Étape 1 — Installe Python
Télécharge Python 3.11+ sur https://www.python.org/downloads/
Coche "Add Python to PATH" pendant l'installation.

### Étape 2 — Ouvre un terminal dans le dossier du projet
- **Windows** : PowerShell, puis `cd chemin\vers\le\dossier`
- **Mac / Linux** : Terminal, puis `cd chemin/vers/le/dossier`

### Étape 3 — Installe les dépendances
```
pip install -r requirements.txt
```

### Étape 4 — Crée ta clé API Ollama (gratuite)
1. Crée un compte sur https://ollama.com
2. Génère une clé sur https://ollama.com/settings/keys
3. Garde-la pour toi : chaque personne utilise sa propre clé, elle ne se partage pas et ne se met pas dans le code.

---

## Lancement (à chaque fois)

Définis ta clé dans le terminal, puis lance l'app :

**Windows (PowerShell)**
```
$env:OLLAMA_API_KEY = "ta_cle"
streamlit run app.py
```

**Mac / Linux**
```
export OLLAMA_API_KEY="ta_cle"
streamlit run app.py
```

La page s'ouvre dans ton navigateur sur http://localhost:8501 (sinon, tape l'adresse à la main).
Pour arrêter l'application : Ctrl+C dans le terminal.

La variable n'est valable que pour le terminal en cours : à refaire à chaque nouvelle fenêtre.

---

## Les 4 étapes du parcours

| Étape | Ce qui se passe |
|-------|-----------------|
| 1. Exploration | Tu uploades une image, l'IA génère 20 mots-clés |
| 2. Composition | Tu choisis des mots, l'IA propose 5 Mottos |
| 3. Audit | Tu choisis un Motto, l'IA vérifie sa conformité ZRM |
| 4. Ressources | Ton Motto validé, l'IA propose 10 ancrages |

À la fin, tu peux télécharger ton bilan complet en format `.md` (Markdown).

---

## Problèmes fréquents

**"ModuleNotFoundError"** : relance `pip install -r requirements.txt`.

**"Clé API Ollama manquante"** : la variable `OLLAMA_API_KEY` n'est pas définie dans ce terminal (voir "Lancement").

**Erreur 401 / "unauthorized"** : la clé est incorrecte ou révoquée, génère-en une nouvelle.

**Erreur "model not found"** : le modèle indiqué dans `VISION_MODEL` / `TEXT_MODEL` en haut de `app.py` n'est pas disponible sur ton compte, remplace-le par un modèle de https://ollama.com/search (avec vision pour `VISION_MODEL`).

**La page ne charge pas** : vérifie que le terminal affiche "You can now view your Streamlit app".
