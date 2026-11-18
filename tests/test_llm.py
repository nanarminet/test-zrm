import json

import pytest

import llm


# ------------------------------------------------------------------ configuration
def env(**valeurs):
    return lambda nom: valeurs.get(nom, "")


def test_cle_ollama_seule():
    c = llm.resoudre_config(env(OLLAMA_API_KEY="k1"))
    assert (c.fournisseur, c.base_url, c.modele, c.cle) == ("ollama", "https://ollama.com/v1", "gemma4:31b", "k1")


def test_cle_mistral_seule_suffit_a_basculer():
    c = llm.resoudre_config(env(MISTRAL_API_KEY="k2"))
    assert (c.fournisseur, c.base_url, c.modele, c.cle) == (
        "mistral", "https://api.mistral.ai/v1", "mistral-medium-3-5", "k2")


def test_deux_cles_sans_fournisseur_est_ambigu():
    with pytest.raises(llm.ConfigurationError, match="LLM_PROVIDER"):
        llm.resoudre_config(env(OLLAMA_API_KEY="a", MISTRAL_API_KEY="b"))


def test_deux_cles_avec_fournisseur():
    c = llm.resoudre_config(env(OLLAMA_API_KEY="a", MISTRAL_API_KEY="b", LLM_PROVIDER="Mistral"))
    assert (c.fournisseur, c.cle) == ("mistral", "b")


def test_aucune_cle():
    with pytest.raises(llm.ConfigurationError, match="Aucune clé"):
        llm.resoudre_config(env())


def test_fournisseur_choisi_mais_cle_absente():
    with pytest.raises(llm.ConfigurationError, match="MISTRAL_API_KEY"):
        llm.resoudre_config(env(LLM_PROVIDER="mistral", OLLAMA_API_KEY="a"))


def test_fournisseur_inconnu():
    with pytest.raises(llm.ConfigurationError, match="inconnu"):
        llm.resoudre_config(env(LLM_PROVIDER="openai", OLLAMA_API_KEY="a"))


def test_mock_sans_cle():
    assert llm.resoudre_config(env(LLM_PROVIDER="mock")).fournisseur == "mock"


def test_cle_generique_avec_fournisseur_et_surcharges():
    c = llm.resoudre_config(env(LLM_PROVIDER="mistral", LLM_API_KEY=" kk ", LLM_MODEL="mistral-small-2603",
                                LLM_BASE_URL="https://proxy.exemple/v1", LLM_REASONING_EFFORT="low"))
    assert (c.cle, c.modele, c.base_url, c.effort_raisonnement) == (
        "kk", "mistral-small-2603", "https://proxy.exemple/v1", "low")


def test_cle_generique_seule_est_ambigue():
    with pytest.raises(llm.ConfigurationError):
        llm.resoudre_config(env(LLM_API_KEY="x"))


# ------------------------------------------------------------------ utilitaires
def test_extraire_json_variantes():
    assert llm.extraire_json('{"a": 1}') == {"a": 1}
    assert llm.extraire_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert llm.extraire_json('Voici : {"a": {"b": 2}} voilà') == {"a": {"b": 2}}
    for mauvais in ("", "pas du json", "[1, 2]", '{"a": '):
        with pytest.raises(llm.LLMSortieInvalide):
            llm.extraire_json(mauvais)


def test_message_utilisateur_texte_et_image():
    assert llm.message_utilisateur("bonjour") == {"role": "user", "content": "bonjour"}
    m = llm.message_utilisateur("voir", b"\xff\xd8abc")
    assert m["content"][0] == {"type": "text", "text": "voir"}
    assert m["content"][1]["type"] == "image_url"
    assert m["content"][1]["image_url"].startswith("data:image/jpeg;base64,")


def test_resumer_messages_masque_l_image():
    r = llm.resumer_messages([{"role": "system", "content": "S"}, llm.message_utilisateur("voir", b"x" * 4096)])
    assert r[0]["content"] == "S"
    assert "data:image" not in r[1]["content"] and "[image jointe" in r[1]["content"]


# ------------------------------------------------------------------ appel HTTP simulé
class FauxRetour:
    def __init__(self, code=200, corps=None, texte="", entetes=None):
        self.status_code, self._corps, self.text, self.headers = code, corps, texte, entetes or {}

    def json(self):
        return self._corps


def ok(contenu, finish="stop", usage=None):
    return FauxRetour(200, {"choices": [{"message": {"content": contenu}, "finish_reason": finish}],
                            "usage": usage or {"prompt_tokens": 10, "completion_tokens": 5}})


@pytest.fixture
def reseau(monkeypatch):
    class Faux:
        def __init__(self):
            self.reponses, self.appels = [], []

        def __call__(self, url, headers=None, json=None, timeout=None):
            self.appels.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
            r = self.reponses.pop(0)
            if isinstance(r, Exception):
                raise r
            return r
    faux = Faux()
    monkeypatch.setattr(llm.requests, "post", faux)
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    return faux


CONF = llm.Config("mistral", "Mistral", "https://api.mistral.ai/v1", "mistral-medium-3-5", "SECRET")
SCHEMA = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"], "additionalProperties": False}
MSG = [{"role": "user", "content": "x"}]


def test_appel_nominal(reseau):
    reseau.reponses = [ok('{"a": 1}')]
    r = llm.appeler(CONF, nom="test", messages=MSG, schema=SCHEMA, temperature=0.3, max_tokens=50)
    assert r.donnees == {"a": 1} and r.format_utilise == "json_schema" and r.usage["prompt_tokens"] == 10
    a = reseau.appels[0]
    assert a["url"] == "https://api.mistral.ai/v1/chat/completions"
    assert a["headers"]["Authorization"] == "Bearer SECRET"
    p = a["json"]
    assert p["model"] == "mistral-medium-3-5" and p["temperature"] == 0.3 and p["max_tokens"] == 50
    assert p["response_format"]["type"] == "json_schema"
    assert p["response_format"]["json_schema"] == {"name": "test", "schema": SCHEMA, "strict": True}
    assert "reasoning_effort" not in p


def test_effort_de_raisonnement_envoye_seulement_si_demande(reseau):
    reseau.reponses = [ok('{"a": 1}')]
    c = llm.Config("ollama", "Ollama", "https://ollama.com/v1", "m", "k", effort_raisonnement="low")
    llm.appeler(c, nom="t", messages=MSG, schema=SCHEMA)
    assert reseau.appels[0]["json"]["reasoning_effort"] == "low"


def test_repli_json_object_si_schema_refuse(reseau):
    reseau.reponses = [FauxRetour(400, texte="schema not supported"), ok('{"a": 2}')]
    r = llm.appeler(CONF, nom="t", messages=MSG, schema=SCHEMA)
    assert r.donnees == {"a": 2} and r.format_utilise == "json_object"
    assert [a["json"]["response_format"]["type"] for a in reseau.appels] == ["json_schema", "json_object"]


def test_repli_sans_format_impose_si_schema_et_json_object_refuses(reseau):
    reseau.reponses = [FauxRetour(400, texte="no"), FauxRetour(400, texte="no"), ok('Voici : {"a": 3}')]
    r = llm.appeler(CONF, nom="t", messages=MSG, schema=SCHEMA)
    assert r.donnees == {"a": 3} and r.format_utilise == "aucun"
    formats = [a["json"].get("response_format", {}).get("type") for a in reseau.appels]
    assert formats == ["json_schema", "json_object", None]


def test_refus_des_trois_formats(reseau):
    reseau.reponses = [FauxRetour(422, texte="no")] * 3
    with pytest.raises(llm.LLMSortieInvalide, match="refusée"):
        llm.appeler(CONF, nom="t", messages=MSG, schema=SCHEMA)


@pytest.mark.parametrize("texte", [
    '{"error": {"code": 400, "message": "API key not valid. Please pass a valid API key.", "status": "INVALID_ARGUMENT"}}',
    '[{"error": {"details": [{"reason": "API_KEY_INVALID"}]}}]'])
def test_cle_invalide_renvoyee_en_400_est_traitee_comme_une_cle_refusee(reseau, texte):
    reseau.reponses = [FauxRetour(400, texte=texte)]
    with pytest.raises(llm.LLMIndisponible, match="clé API refusée"):
        llm.appeler(CONF, nom="t", messages=MSG, schema=SCHEMA)
    assert len(reseau.appels) == 1


@pytest.mark.parametrize("code", [401, 403])
def test_cle_refusee(reseau, code):
    reseau.reponses = [FauxRetour(code, texte="nope")]
    with pytest.raises(llm.LLMIndisponible, match="clé API refusée"):
        llm.appeler(CONF, nom="t", messages=MSG, schema=SCHEMA)
    assert len(reseau.appels) == 1          # pas de nouvel essai inutile


def test_modele_introuvable(reseau):
    reseau.reponses = [FauxRetour(404, texte="")]
    with pytest.raises(llm.LLMIndisponible, match="mistral-medium-3-5"):
        llm.appeler(CONF, nom="t", messages=MSG, schema=SCHEMA)


def test_429_puis_succes(reseau):
    reseau.reponses = [FauxRetour(429, entetes={"Retry-After": "1"}), ok('{"a": 3}')]
    assert llm.appeler(CONF, nom="t", messages=MSG, schema=SCHEMA).donnees == {"a": 3}
    assert len(reseau.appels) == 2


def test_indisponible_apres_essais(reseau):
    reseau.reponses = [FauxRetour(503)] * llm.MAX_ESSAIS_RESEAU
    with pytest.raises(llm.LLMIndisponible, match="injoignable"):
        llm.appeler(CONF, nom="t", messages=MSG, schema=SCHEMA)
    assert len(reseau.appels) == llm.MAX_ESSAIS_RESEAU


def test_timeout_puis_succes(reseau):
    reseau.reponses = [llm.requests.Timeout(), ok('{"a": 4}')]
    assert llm.appeler(CONF, nom="t", messages=MSG, schema=SCHEMA).donnees == {"a": 4}


def test_reponse_tronquee(reseau):
    reseau.reponses = [ok('{"a": ', finish="length")]
    with pytest.raises(llm.LLMSortieInvalide, match="tronquée"):
        llm.appeler(CONF, nom="t", messages=MSG, schema=SCHEMA)


def test_reponse_sans_choices(reseau):
    reseau.reponses = [FauxRetour(200, {"error": "x"})]
    with pytest.raises(llm.LLMSortieInvalide):
        llm.appeler(CONF, nom="t", messages=MSG, schema=SCHEMA)


def test_contenu_en_blocs(reseau):
    blocs = [{"type": "thinking", "thinking": "..."}, {"type": "text", "text": '{"a": 5}'}]
    reseau.reponses = [FauxRetour(200, {"choices": [{"message": {"content": blocs}, "finish_reason": "stop"}]})]
    r = llm.appeler(CONF, nom="t", messages=MSG, schema=SCHEMA)
    assert r.donnees == {"a": 5} and r.usage == {}


def test_mock_n_appelle_pas_le_reseau(reseau):
    c = llm.Config("mock", "Simulation", "", "mock", "")
    r = llm.appeler(c, nom="mouvements", messages=MSG, schema=SCHEMA)
    assert r.format_utilise == "simulation" and reseau.appels == []
    json.dumps(r.donnees)


def test_envoi_de_l_image_actif_par_defaut_et_desactivable():
    base = {"LLM_PROVIDER": "mock"}
    assert llm.resoudre_config(base.get).envoyer_image is True
    for valeur in ("non", "false", "0", "No"):
        assert llm.resoudre_config({**base, "LLM_ENVOYER_IMAGE": valeur}.get).envoyer_image is False


# ------------------------------------------------------------------ Gemini
def test_cle_gemini_seule():
    c = llm.resoudre_config(env(GEMINI_API_KEY="g"))
    assert (c.fournisseur, c.modele, c.cle) == ("gemini", "gemini-2.5-flash", "g")
    assert c.base_url == "https://generativelanguage.googleapis.com/v1beta/openai"
    assert c.effort_raisonnement == "none"            # coupe la réflexion : elle consommerait max_tokens
    assert c.envoyer_image is True and c.image_objet is True      # image envoyée, au format objet de Google


def test_gemini_image_desactivable():
    assert llm.resoudre_config(env(GEMINI_API_KEY="g", LLM_ENVOYER_IMAGE="non")).envoyer_image is False


def test_image_en_objet_pour_gemini_et_en_chaine_pour_les_autres(reseau):
    msgs = [{"role": "system", "content": "S"}, llm.message_utilisateur("voir", b"\xff\xd8abc")]
    reseau.reponses = [ok('{"a": 1}'), ok('{"a": 1}')]
    llm.appeler(llm.resoudre_config(env(GEMINI_API_KEY="g")), nom="t", messages=msgs, schema=SCHEMA)
    img = reseau.appels[0]["json"]["messages"][1]["content"][1]["image_url"]
    assert isinstance(img, dict) and img["url"].startswith("data:image/jpeg;base64,")
    assert isinstance(msgs[1]["content"][1]["image_url"], str)               # message d'origine non modifié
    llm.appeler(CONF, nom="t", messages=msgs, schema=SCHEMA)
    assert isinstance(reseau.appels[1]["json"]["messages"][1]["content"][1]["image_url"], str)


def test_gemini_autre_modele_n_herite_pas_de_l_effort_par_defaut():
    c = llm.resoudre_config(env(GEMINI_API_KEY="g", LLM_MODEL="gemini-3.5-flash"))
    assert c.modele == "gemini-3.5-flash" and c.effort_raisonnement == ""
    c = llm.resoudre_config(env(GEMINI_API_KEY="g", LLM_MODEL="gemini-3.5-flash", LLM_REASONING_EFFORT="low"))
    assert c.effort_raisonnement == "low"


def test_effort_explicite_remplace_celui_du_preset():
    assert llm.resoudre_config(env(GEMINI_API_KEY="g", LLM_REASONING_EFFORT="low")).effort_raisonnement == "low"


def test_ollama_et_mistral_gardent_l_image_et_pas_d_effort():
    for cle in ("OLLAMA_API_KEY", "MISTRAL_API_KEY"):
        c = llm.resoudre_config(env(**{cle: "k"}))
        assert c.envoyer_image is True and c.effort_raisonnement == ""


def test_cle_ollama_perimee_plus_cle_gemini_demande_le_fournisseur():
    with pytest.raises(llm.ConfigurationError, match="LLM_PROVIDER"):
        llm.resoudre_config(env(OLLAMA_API_KEY="vieille", GEMINI_API_KEY="g"))
    assert llm.resoudre_config(env(OLLAMA_API_KEY="vieille", GEMINI_API_KEY="g", LLM_PROVIDER="gemini")).fournisseur == "gemini"


def test_gemini_requete_sur_l_url_openai_compatible_avec_effort_none(reseau):
    conf = llm.resoudre_config(env(GEMINI_API_KEY="g"))
    reseau.reponses = [ok('{"a": 1}')]
    llm.appeler(conf, nom="t", messages=MSG, schema=SCHEMA)
    appel = reseau.appels[0]
    assert appel["url"] == "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    assert appel["headers"]["Authorization"] == "Bearer g"
    assert appel["json"]["model"] == "gemini-2.5-flash" and appel["json"]["reasoning_effort"] == "none"
