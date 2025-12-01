import requests
import streamlit as st

# -----------------------
# CONFIGURATION MISTRAL
# -----------------------

MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_API_KEY = "uXm9QqcIgCCylmcePxiZacnYdICgSouW"
MISTRAL_MODEL = "mistral-tiny"

def get_mistral_api_key():
    return MISTRAL_API_KEY

def call_mistral_api(context_chunks_with_metadata, question, conversation_history=[]):
    """Appelle l'API Mistral avec un prompt amélioré pour une meilleure intelligence"""
    
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Construction du prompt ultra-optimisé pour l'analyse documentaire
    system_prompt = """Tu es un expert en analyse documentaire avec des capacités de raisonnement avancées.

🎯 TON RÔLE :
Analyser profondément les documents pour fournir des réponses intelligentes, contextuelles et précises.

🔍 MÉTHODOLOGIE :
1. ANALYSE MULTI-NIVEAU : Comprendre le contexte global et les détails spécifiques
2. RAISONNEMENT DÉDUCTIF : Faire des liens entre les informations
3. SYNTHÈSE INTELLIGENTE : Résumer sans perdre l'essentiel
4. CONTEXTE DOCUMENTAIRE : Utiliser les métadonnées (sources, documents)

📝 RÈGLES STRICTES :
✅ UTILISE exclusivement le contexte fourni
✅ SOIS précis, détaillé et contextuel
✅ FAIS des déductions logiques basées sur les informations
✅ STRUCTURE tes réponses de manière claire
✅ MENTIONNE les documents sources quand c'est pertinent
✅ ADAPTE ton style à la complexité de la question

🚫 INTERDICTIONS :
❌ JAMAIS d'inventions ou d'hallucinations
❌ JAMAIS d'informations extérieures au contexte
❌ JAMAIS de réponses vagues ou génériques

Ton objectif : être l'analyste documentaire le plus compétent et fiable."""

    messages = [{"role": "system", "content": system_prompt}]
    
    # Ajout de l'historique récent avec contexte conversationnel
    for msg in conversation_history[-3:]:  # Garde les 3 derniers échanges
        if msg["type"] == "question":
            messages.append({"role": "user", "content": msg["text"]})
        else:
            messages.append({"role": "assistant", "content": msg["text"]})
    
    # Préparation du contexte enrichi avec métadonnées
    context_parts = []
    for i, chunk_data in enumerate(context_chunks_with_metadata[:6]):  # 6 chunks max
        source_info = f"[Source: {chunk_data['source']}]"
        context_parts.append(f"{source_info}\n{chunk_data['text']}")
    
    context_text = "\n\n" + "="*50 + "\n".join(context_parts) + "\n" + "="*50
    
    user_content = f"""## 📚 CONTEXTE DOCUMENTAIRE COMPLET :
{context_text}

## ❓ QUESTION À ANALYSER :
{question}

## 🎯 INSTRUCTIONS :
En tant qu'expert en analyse documentaire, fournis une réponse :
- Basée UNIQUEMENT sur le contexte ci-dessus
- Précise, détaillée et bien structurée
- Avec un raisonnement logique et clair
- Adaptée à la complexité de la question
- Mentionnant les sources documentaires quand c'est pertinent"""

    messages.append({"role": "user", "content": user_content})
    
    payload = {
        "model": MISTRAL_MODEL,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 1200,
        "top_p": 0.9
    }
    
    try:
        response = requests.post(MISTRAL_API_URL, headers=headers, json=payload, timeout=45)
        
        if response.status_code == 429:
            return "🔄 Trop de requêtes. Veuillez patienter quelques instants avant de réessayer."
        elif response.status_code == 401:
            return "🔐 Problème d'authentification. Clé API invalide."
        elif response.status_code == 403:
            return "🚫 Accès non autorisé. Vérifiez vos permissions API."
        elif response.status_code == 400:
            return "⚠️ Requête mal formée. Le service peut être temporairement surchargé."
        
        response.raise_for_status()
        
        result = response.json()
        return result["choices"][0]["message"]["content"]
        
    except requests.exceptions.Timeout:
        return "⏰ Délai de réponse dépassé. Le service met plus de temps à répondre en raison de la complexité de l'analyse."
    except Exception as e:
        return f"❌ Erreur technique: {str(e)}"

def smart_text_analysis_with_mistral(context_chunks_with_metadata, question, conversation_history):
    """Analyse intelligente avec Mistral pour plusieurs documents"""
    if not context_chunks_with_metadata:
        return "🔍 Aucune information pertinente trouvée dans les documents pour répondre à cette question. Essayez de reformuler ou vérifiez que les documents contiennent bien ces informations."
    
    # Questions très simples qu'on peut traiter sans LLM
    question_lower = question.lower()
    
    if any(word in question_lower for word in ['bonjour', 'salut', 'hello', 'coucou']):
        return "👋 Bonjour ! Je suis votre assistant IA spécialisé dans l'analyse documentaire. Importez vos documents et posez-moi toutes vos questions !"
    
    elif any(word in question_lower for word in ['merci', 'thanks']):
        return "✨ Je vous en prie ! N'hésitez pas si vous avez d'autres questions sur vos documents."
    
    elif any(word in question_lower for word in ['aide', 'help', 'comment ça marche']):
        return """
### 🎯 Guide d'Utilisation Complet

**📤 IMPORTATION :**
- Allez dans l'onglet 'Documents'
- Uploader jusqu'à 5 fichiers PDF ou DOCX
- Cliquez sur 'Indexer les documents'

**💬 CONVERSATION INTELLIGENTE :**
- Posez des questions complexes sur le contenu
- Demandez des analyses, résumés, comparaisons
- Interrogez sur des points spécifiques ou généraux

**🔍 TECHNOLOGIES UTILISÉES :**
- ✅ **FAISS** : Recherche vectorielle avancée
- ✅ **LangChain** : Framework d'IA professionnel
- ✅ **Sentence Transformers** : Embeddings sémantiques
- ✅ **Mistral AI** : Modèle de langage performant

**FONCTIONNALITÉS AVANCÉES :**
- Recherche sémantique par similarité vectorielle
- Analyse multi-documents intelligente
- Raisonnement contextuel approfondi
- Réponses détaillées et structurées
        """
    
    else:
        # Pour toutes les autres questions, on utilise Mistral avec le contexte enrichi
        return call_mistral_api(context_chunks_with_metadata, question, conversation_history)