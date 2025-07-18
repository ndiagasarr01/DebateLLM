# DebateLLM 🤖💬

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/NdiagaSarr/DebateLLM)

DebateLLM est une application Python développée avec Streamlit qui simule un débat entre plusieurs agents dotés de Grands Modèles de Langage (LLM). Chaque agent peut se voir attribuer une personnalité distincte et un modèle de langage différent, permettant aux utilisateurs d'explorer comment différentes IA argumenteraient sur une variété de sujets.

## ✨ Fonctionnalités

- **Simulation Multi-Agents :** Configurez jusqu'à 4 agents IA pour participer à un débat.
- **Personnalités Personnalisables :** Attribuez des personnalités uniques à chaque agent (ex: "Analyste Pragmatique", "Visionnaire Créatif", "Sceptique Critique").
- **Double Système de Backend :**
    - **Ollama (Local) :** Utilisez n'importe quel LLM installé localement via [Ollama](https://ollama.com/) pour des débats rapides et hors ligne.
    - **Hugging Face (En ligne) :** Utilisez n'importe quel modèle de génération de texte compatible directement depuis le [Hugging Face Hub](https://huggingface.co/models).
- **Interface Utilisateur Interactive :** Une interface web simple et intuitive construite avec Streamlit.
- **Export des Dialogues :** Téléchargez la transcription complète du débat au format JSON pour analyse.

## 🚀 Lancement en Local

### Prérequis

- Python 3.8+
- Git
- (Optionnel) [Ollama](https://ollama.com/) installé pour utiliser le mode local.

### 1. Cloner le Dépôt

```bash
git clone https://github.com/ndiagasarr01/DebateLLM.git
cd DebateLLM
```

### 2. Installer les Dépendances

Il est conseillé d'utiliser un environnement virtuel.

```bash
# Créer un environnement virtuel (optionnel mais recommandé)
python -m venv venv
# Activer l'environnement
# Sur Windows : venv\Scripts\activate
# Sur macOS/Linux : source venv/bin/activate

# Installer les bibliothèques requises
pip install -r requirements.txt
```

### 3. Lancer l'Application

```bash
streamlit run debate_app.py
```

Votre navigateur web devrait s'ouvrir avec l'application en cours d'exécution.

## 🔧 Utilisation

1.  **Choisir un Backend :**
    - Sélectionnez **Ollama (Local)** pour utiliser les modèles que vous avez installés sur votre machine (ex: `ollama pull llama3`). L'application les détectera automatiquement.
    - Sélectionnez **Hugging Face (En ligne)** pour utiliser des modèles du Hub. Vous devrez fournir l'identifiant du dépôt (ex: `mistralai/Mistral-7B-Instruct-v0.2`). *Note : La première fois que vous utilisez un modèle, il sera téléchargé, ce qui peut prendre du temps et des ressources.*

2.  **Configurer le Débat :**
    - Choisissez un sujet de débat.
    - Définissez le nombre d'agents et de tours de parole.
    - Pour chaque agent, attribuez une personnalité et sélectionnez le modèle souhaité.

3.  **Lancer :** Cliquez sur "🚀 Lancer le Débat" et observez la conversation !
