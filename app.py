import os
import re
from io import BytesIO

import streamlit as st
from ollama import Client
from PIL import Image

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
# Clé API gratuite à créer sur https://ollama.com/settings/keys, puis à définir avec :
#   PowerShell : $env:OLLAMA_API_KEY = "ta_cle"
#   puis : streamlit run app.py pour lancer l'app avec la clé en environnement
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", "")
OLLAMA_HOST = "https://ollama.com"  # remplacer par http://localhost:11434 pour un Ollama local (sans clé)
REQUEST_TIMEOUT = 120  # secondes
MAX_IMAGE_SIDE = 1024  # px, l'image est réduite avant envoi

VISION_MODEL = "gemma4:31b"
TEXT_MODEL = "gemma4:31b"

st.set_page_config(page_title="Parcours ZRM")

if OLLAMA_HOST.startswith("https://ollama.com") and not OLLAMA_API_KEY:
    st.error(
        "Clé API Ollama manquante. Crée-en une sur https://ollama.com/settings/keys, "
        "puis définis-la avant de lancer l'app :\n\n"
        '`$env:OLLAMA_API_KEY = "ta_cle"` (PowerShell) ou `export OLLAMA_API_KEY="ta_cle"` (Mac/Linux)'
    )
    st.stop()


@st.cache_resource
def get_client() -> Client:
    return Client(
        host=OLLAMA_HOST,
        headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"} if OLLAMA_API_KEY else None,
        timeout=REQUEST_TIMEOUT,
    )


def prepare_image(image_bytes: bytes) -> bytes:
    """Réduit l'image (côté max MAX_IMAGE_SIDE) et la convertit en JPEG pour un envoi plus rapide."""
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def extract_mottos(text: str) -> list[str]:
    """Extrait les lignes numérotées ('1. Je ...') sans le numéro ni les crochets/guillemets."""
    mottos = []
    for line in text.splitlines():
        match = re.match(r"^\s*\d+[.)]\s*(.+)$", line)
        if match:
            mottos.append(match.group(1).strip().strip('[]"«» ').strip())
    return mottos

# ─────────────────────────────────────────
# SYSTEM PROMPTS
# ─────────────────────────────────────────
SYSTEM_UC1 = """
Tu es un poète-sémiologue. À partir d'une image, extrais 20 mots inspirants en 3 catégories :
- DESCRIPTIF (ce qu'on voit littéralement) : 6 mots
- SYMBOLIQUE (ce que ça évoque, métaphores) : 8 mots
- ÉMOTIONNEL (ce que ça fait ressentir) : 6 mots

Règles strictes :
- Aucun verbe. Uniquement des noms ou adjectifs.
- Pas de redondance entre les catégories.
- Privilégie les mots rares, poétiques, inattendus.
- Format de sortie : liste numérotée avec la catégorie entre crochets.
Exemple : 1. [DESCRIPTIF] Brume
"""

SYSTEM_UC2 = """
Tu es un coach en formulation d'intention. L'utilisateur te donne 3 à 5 mots-clés et une thématique de projet.

Ta mission : générer exactement 5 propositions de "Motto" selon la méthode ZRM, avec ces contraintes ABSOLUES :
1. Structure : commence par "Je" + verbe d'action au présent
2. Zéro négation (jamais "ne...pas", "sans", "éviter")
3. Le sujet "Je" est acteur de l'action, pas spectateur
4. Phrase courte : 5 à 12 mots maximum
5. Les mots-clés de l'utilisateur doivent apparaître ou être sémantiquement proches dans chaque proposition

Format de sortie :
1. [Motto]
2. [Motto]
...
Pas d'explication. Juste les 5 phrases.
"""

SYSTEM_UC3 = """
Tu es un auditeur linguistique spécialisé dans la méthode ZRM. Tu analyses la conformité d'un "Motto" selon 4 critères non négociables.

CRITÈRES D'AUDIT :
[C1] POSITIVITÉ — Aucune négation (ne...pas, sans, jamais, éviter, refuser, empêcher).
[C2] AGENTIVITÉ — Le sujet "Je" est l'acteur principal.
[C3] PRÉSENT — Verbe conjugué au présent de l'indicatif.
[C4] CONCRÈTUDE — Pas de concepts abstraits, uniquement des actions observables.

Format de sortie :
[C1] : OK/KO + explication si KO
[C2] : OK/KO + explication si KO
[C3] : OK/KO + explication si KO
[C4] : OK/KO + explication si KO
"""

SYSTEM_UC4 = """
Tu es un expert en ancrage sensoriel. À partir d'un Motto ZRM, propose 10 ressources concrètes pour l'ancrer dans le quotidien :
- 3 objets symboliques
- 3 musiques/ambiances sonores
- 2 rituels quotidiens
- 2 citations inspirantes

Format de sortie :
### Objets symboliques
1. [Objet] : [Explication courte]

### Musiques/Ambiance
1. [Titre/Artiste] : [Description]

### Rituels quotidiens
1. [Rituel] : [Description]

### Citations
1. "[Citation]" — [Auteur]
"""

# ─────────────────────────────────────────
# FONCTIONS D'APPEL API
# ─────────────────────────────────────────
def call_ollama_vision(image_bytes: bytes, prompt: str, system: str):
    try:
        response = get_client().chat(
            model=VISION_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt, "images": [prepare_image(image_bytes)]},
            ],
        )
        return response["message"]["content"]
    except Exception as e:
        st.error(f"Erreur API Ollama : {e}")
        return None

def call_ollama_text(prompt: str, system: str):
    try:
        response = get_client().chat(
            model=TEXT_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        return response["message"]["content"]
    except Exception as e:
        st.error(f"Erreur API Ollama : {e}")
        return None

# ─────────────────────────────────────────
# FONCTIONS PAR ÉTAPE
# ─────────────────────────────────────────
def etape_1_mots_cles(image_bytes: bytes):
    return call_ollama_vision(
        image_bytes,
        "Analyse cette image et génère les 20 mots-clés selon les 3 catégories.",
        SYSTEM_UC1
    )

def etape_2_motto(mots_cles: str, thematique: str):
    return call_ollama_text(
        f"Mots-clés : {mots_cles}\nThématique : {thematique}",
        SYSTEM_UC2
    )

def etape_3_audit(motto: str):
    return call_ollama_text(
        f"Motto à auditer : '{motto}'",
        SYSTEM_UC3
    )

def etape_4_ressources(motto: str):
    return call_ollama_text(
        f"Motto : '{motto}'",
        SYSTEM_UC4
    )

# ─────────────────────────────────────────
# INTERFACE STREAMLIT
# ─────────────────────────────────────────
if "etape" not in st.session_state:
    st.session_state.etape = 1

st.title("Parcours ZRM — Génère ton Motto et tes ressources d'ancrage")

# ─── ÉTAPE 1 ────────────────
if st.session_state.etape == 1:
    st.header("Étape 1 — Analyse visuelle et mots-clés")
    uploaded_file = st.file_uploader("Télécharge une image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        st.image(uploaded_file, caption="Image téléchargée", width="stretch")
        if st.button("Analyser l'image et générer les mots-clés", type="primary"):
            with st.spinner("Analyse visuelle en cours..."):
                result = etape_1_mots_cles(uploaded_file.getvalue())
                if result:
                    st.session_state.mots_cles = result
                    st.session_state.etape = 2
                    st.rerun()

# ─── ÉTAPE 2 ────────────────
elif st.session_state.etape == 2:
    st.header("Étape 2 — Création de ton Motto")
    st.markdown(st.session_state.mots_cles)
    mots_cles_selection = st.text_input("Sélectionne 3 à 5 mots-clés :", value="")
    thematique = st.text_input("Thématique de ton projet :", value="")
    if st.button("Générer mes Mottos", type="primary"):
        if not (mots_cles_selection.strip() and thematique.strip()):
            st.warning("Renseigne les mots-clés et la thématique avant de continuer.")
        else:
            with st.spinner("Génération des Mottos..."):
                result = etape_2_motto(mots_cles_selection, thematique)
                if result:
                    st.session_state.mottos = result
                    st.session_state.etape = 3
                    st.rerun()

# ─── ÉTAPE 3 ────────────────
elif st.session_state.etape == 3:
    st.header("Étape 3 — Audit & Correction")
    st.markdown(st.session_state.mottos)
    lignes = extract_mottos(st.session_state.mottos) or [
        l.strip() for l in st.session_state.mottos.splitlines() if l.strip()
    ]
    motto_selection = st.selectbox("Sélectionne ton Motto préféré :", lignes)
    if st.button("Auditer ce Motto", type="primary"):
        with st.spinner("Audit en cours..."):
            result = etape_3_audit(motto_selection)
            if result:
                st.session_state.audit_result = result
                st.session_state.motto_final = motto_selection
                st.session_state.etape = 4
                st.rerun()

# ─── ÉTAPE 4 ────────────────
elif st.session_state.etape == 4:
    st.header("Étape 4 — Ressources & Ancrages")
    st.write(f"Ton Motto final : **{st.session_state.motto_final}**")
    st.subheader("Audit de conformité")
    st.markdown(st.session_state.audit_result)
    if st.button("Générer mes ressources", type="primary"):
        with st.spinner("Recherche d'ancrages..."):
            result = etape_4_ressources(st.session_state.motto_final)
            if result:
                st.session_state.ressources = result
    if "ressources" in st.session_state:
        st.subheader("Tes ressources-ancrages personnalisées")
        st.markdown(st.session_state.ressources)
        export_data = f"""# Mon Parcours ZRM

## Mon Motto
{st.session_state.motto_final}

## Audit de conformité
{st.session_state.audit_result}

## Mes ressources & ancrages
{st.session_state.ressources}
"""
        st.download_button(
            label="Télécharger mon bilan ZRM",
            data=export_data,
            file_name="mon_bilan_zrm.md",
            mime="text/markdown",
        )
        if st.button("Recommencer un nouveau parcours"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()