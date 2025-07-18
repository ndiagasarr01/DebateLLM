import streamlit as st
import requests
import json
import time
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# --- Configuration des Personnalités (avec modèles HF par défaut) ---
PERSONNALITES = {
    "Analyste Pragmatique": {
        "prompt": "Tu es un analyste expert. Réponds de manière factuelle, en te basant sur des données et des preuves logiques. Décompose les problèmes de manière structurée. Ne laisse pas l'émotion influencer tes arguments. Sois précis et objectif.",
        "avatar": "📊",
        "default_model_ollama": "llama3",
        "default_model_hf": "meta-llama/Llama-3-8B-Instruct"
    },
    "Visionnaire Créatif": {
        "prompt": "Tu es un penseur visionnaire et créatif. Propose des idées audacieuses et regarde au-delà des contraintes actuelles. Utilise des analogies et des métaphores pour illustrer tes points. Inspire les autres à penser différemment.",
        "avatar": "🎨",
        "default_model_ollama": "mistral",
        "default_model_hf": "mistralai/Mistral-7B-Instruct-v0.2"
    },
    "Sceptique Critique": {
        "prompt": "Tu es un esprit critique et sceptique. Ton rôle est de questionner les affirmations, d'identifier les risques et les faiblesses dans les arguments des autres. Sois direct, concis et ne prends rien pour acquis.",
        "avatar": "🧐",
        "default_model_ollama": "phi3",
        "default_model_hf": "microsoft/Phi-3-mini-4k-instruct"
    },
    "Éthicien Humaniste": {
        "prompt": "Tu es un éthicien humaniste. Analyse chaque argument sous l'angle de son impact sur la société, l'individu et les principes moraux. Fais preuve d'empathie et rappelle constamment l'importance des valeurs humaines.",
        "avatar": "❤️",
        "default_model_ollama": "gemma",
        "default_model_hf": "google/gemma-1.1-7b-it"
    }
}

# --- Configuration des Thématiques ---
THEMATIQUES = [
    "L'IA va-t-elle créer plus d'emplois qu'elle n'en détruira ?",
    "L'humanité doit-elle investir dans la colonisation de Mars avant de résoudre les problèmes sur Terre ?",
    "Les réseaux sociaux sont-ils une menace ou un outil pour la démocratie ?",
]

# --- Fonctions de Backend ---

# --- Ollama Backend ---
@st.cache_data(ttl=60)
def get_ollama_models():
    try:
        response = requests.get("http://localhost:11434/api/tags")
        response.raise_for_status()
        return [model["name"] for model in response.json().get("models", [])]
    except (requests.exceptions.RequestException, json.JSONDecodeError):
        return []

def call_ollama_model(model_name, system_prompt, history_str, agent_name):
    prompt = f"System: {system_prompt}\n\n{history_str}\n\n**C'est à ton tour, {agent_name}.**"
    try:
        response = requests.post("http://localhost:11434/api/generate", json={"model": model_name, "prompt": prompt, "stream": False}, timeout=60)
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur Ollama: {e}")
        return None

# --- Hugging Face Backend ---
@st.cache_resource(max_entries=4) # Cache pour les modèles et tokenizers
def get_hf_pipeline(model_id):
    st.write(f"Chargement du modèle : {model_id}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        # Utilisation de torch_dtype pour optimiser la mémoire (surtout pour les gros modèles)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
        return pipeline("text-generation", model=model, tokenizer=tokenizer)
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle {model_id}: {e}")
        return None

def call_hf_model(pipe, system_prompt, history, agent_name):
    messages = [
        {"role": "system", "content": system_prompt},
        *history, # Dépaquette l'historique des messages
        {"role": "user", "content": f"C'est à ton tour, {agent_name}. Continue le débat."}
    ]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_p=0.95)
    return outputs[0]["generated_text"].split("<|assistant|>")[-1].strip()

# --- Interface Streamlit ---
st.set_page_config(layout="wide")
st.title("DebateLLM 🤖💬")
st.markdown("Simulez un débat entre agents IA avec des modèles locaux (Ollama) ou distants (Hugging Face).")

# --- Panneau de Configuration ---
with st.sidebar:
    st.header("Configuration du Débat")

    backend_choice = st.radio("1. Choisissez le backend du modèle :", ["Ollama (Local)", "Hugging Face (En ligne)"], horizontal=True)

    theme = st.selectbox("2. Choisissez une thématique :", THEMATIQUES)
    num_agents = st.slider("3. Nombre d'agents :", 2, 4, 2)
    num_tours = st.slider("4. Nombre de tours de parole par agent :", 1, 5, 2)

    st.markdown("---")
    st.header("Configuration des Agents")
    agents_config = []
    personnalites_disponibles = list(PERSONNALITES.keys())

    if backend_choice == "Ollama (Local)":
        local_models = get_ollama_models()
        if not local_models:
            st.warning("Aucun modèle Ollama détecté. Assurez-vous qu'Ollama est lancé.")
        else:
            st.success(f"Modèles locaux : {len(local_models)}")

    for i in range(num_agents):
        st.markdown(f"**Agent {i+1}**")
        perso = st.selectbox(f"Personnalité", personnalites_disponibles, key=f"perso_{i}")
        
        if backend_choice == "Ollama (Local)":
            if local_models:
                model = st.selectbox("Modèle", local_models, key=f"model_{i}")
            else:
                model = st.text_input("Modèle", PERSONNALITES[perso]["default_model_ollama"], key=f"model_{i}")
        else: # Hugging Face
            model = st.text_input("Repo ID Hugging Face", PERSONNALITES[perso]["default_model_hf"], key=f"model_{i}")
        
        agents_config.append({"nom": perso, "modele": model})

    start_button = st.button("🚀 Lancer le Débat", use_container_width=True)

# --- Zone de Débat ---
if start_button:
    st.header(f"Débat sur : *{theme}*")
    conversation_log = []
    
    # Initialisation de l'historique pour les deux backends
    hf_history = []
    ollama_history_str = f"Thème du débat : {theme}"

    with st.spinner("Le débat est en cours... Chargement des modèles si nécessaire."):
        # Pré-chargement des pipelines HF pour éviter de le faire dans la boucle
        if backend_choice == "Hugging Face (En ligne)":
            pipelines = {agent["modele"]: get_hf_pipeline(agent["modele"]) for agent in agents_config}
            if any(p is None for p in pipelines.values()):
                st.error("Un ou plusieurs modèles Hugging Face n'ont pas pu être chargés. Le débat est annulé.")
                st.stop()

        for tour in range(num_tours):
            st.subheader(f"Tour de parole n°{tour + 1}")
            for agent in agents_config:
                agent_name = agent["nom"]
                model_name = agent["modele"]
                personality = PERSONNALITES[agent_name]

                with st.chat_message(name=agent_name, avatar=personality["avatar"]):
                    placeholder = st.empty()
                    placeholder.markdown("...")

                    if backend_choice == "Ollama (Local)":
                        response = call_ollama_model(model_name, personality["prompt"], ollama_history_str, agent_name)
                    else: # Hugging Face
                        pipe = pipelines[model_name]
                        response = call_hf_model(pipe, personality["prompt"], hf_history, agent_name)

                    if response:
                        placeholder.markdown(response)
                        # Mise à jour des historiques
                        ollama_history_str += f"\n- **{agent_name}**: {response}"
                        hf_history.append({"role": "assistant", "content": response})
                        conversation_log.append({"agent": agent_name, "modele": model_name, "reponse": response})
                    else:
                        st.error(f"L'agent {agent_name} n'a pas pu répondre.")
                        break
                time.sleep(1)
            if not response: break

    st.success("Débat terminé !")
    st.download_button("📥 Télécharger en JSON", json.dumps(conversation_log, indent=2), "debat.json", "application/json")