import streamlit as st
from datetime import datetime
import os

# Import des modules séparés
from mistral_client import call_mistral_api, smart_text_analysis_with_mistral
from rag_system import FAISSIndex, process_multiple_files
from utils import extract_text, split_text

# Configuration pour le déploiement
if not os.path.exists('uploads'):
    os.makedirs('uploads', exist_ok=True)
if not os.path.exists('data'):
    os.makedirs('data', exist_ok=True)

# Configuration du style CSS
st.markdown("""
<style>
    .stButton > button[kind="primary"] {
        background-color: #1E40AF !important;
        border: 1px solid #1E40AF !important;
        color: white !important;
        width: 300px !important;
        margin-left:300px !important;
    }
    
    .stButton > button[kind="primary"]:hover {
        background-color: #1E3A8A !important;
        border: 1px solid #1E3A8A !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialisation de l'état de session
def initialize_session_state():
    if "active_view" not in st.session_state:
        st.session_state.active_view = "chat"
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"type": "answer", "text": """
👋 **Bienvenue dans l'Assistant Document Intelligent RAG !**

Je suis votre expert IA pour analyser et comprendre vos documents avec les technologies les plus avancées.


Je fournirai des réponses intelligentes basées sur la recherche vectorielle avancée.
        """}
    ]
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    if "text_index" not in st.session_state:
        st.session_state.text_index = None
    if "document_chunks" not in st.session_state:
        st.session_state.document_chunks = []
    if "document_texts" not in st.session_state:
        st.session_state.document_texts = {}
    if "show_history" not in st.session_state:
        st.session_state.show_history = False
    if "show_help" not in st.session_state:
        st.session_state.show_help = False

def render_sidebar():
    """Affiche la sidebar avec navigation et informations"""
    with st.sidebar:
        st.markdown("""
        <div style='text-align: center; padding: 1rem 0;'>
            <h1 style='color: black; margin-top: -69px; font-size: 1.8rem;'>🤖 RAG Mistral</h1>
            <p style='color:black; font-size: 0.9rem; margin: 0;'>Assistant document intelligent</p>
            <p style='color: black; font-size: 0.8rem; margin: 0;'>FAISS + LangChain + Mistral</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Navigation principale
        st.markdown("### 📋 Navigation")
        
        if st.button("⚙️ Paramètres & Aide", use_container_width=True):
            st.session_state.show_help = not st.session_state.show_help
            st.session_state.show_history = False

        if st.button("📄 Documents", use_container_width=True):
            st.session_state.active_view = "upload"
            st.session_state.show_help = False
            st.session_state.show_history = False

        if st.button("📊 Historique", use_container_width=True):
            st.session_state.show_history = not st.session_state.show_history
            st.session_state.show_help = False

        if st.button("🔄 Nouvelle Conversation", use_container_width=True):
            st.session_state.messages = [
                {"type": "answer", "text": "🔄 Nouvelle conversation démarrée. Posez vos questions sur les documents !"}
            ]
            st.session_state.show_help = False
            st.session_state.show_history = False
            st.rerun()

        # Menu Aide
        if st.session_state.show_help:
            st.markdown("---")
            st.markdown("### 💡 Technologies & Fonctionnalités")
            
            st.markdown("""
            **🎯 Architecture RAG Complète :**
            - **FAISS** : Indexation vectorielle Facebook AI
            - **LangChain** : Orchestration des composants IA
            - **Sentence Transformers** : Embeddings sémantiques
            - **Mistral AI** : Génération de réponses
            
            **🔍 Recherche Vectorielle :**
            - Compréhension sémantique des requêtes
            - Similarité basée sur le sens des mots
            - Résultats contextuellement pertinents
            - Support multi-documents
            """)

        # Affichage des documents chargés
        if st.session_state.text_index and st.session_state.text_index.ntotal > 0:
            st.markdown("---")
            doc_stats = st.session_state.text_index.get_document_stats()
            total_docs = len(doc_stats)
            total_chunks = st.session_state.text_index.ntotal
            
            st.markdown("### 📊 Index FAISS")
            st.markdown(f"**{total_docs}** document(s) indexé(s)  \n**{total_chunks}** segments vectoriels")
            
            with st.expander("📋 Détails de l'index"):
                for doc_name, chunk_count in doc_stats.items():
                    st.markdown(f"• `{doc_name}`  \n  ({chunk_count} embeddings)")

        # Historique des conversations
        if st.session_state.show_history:
            st.markdown("---")
            st.markdown("### 📝 Historique Récent")
            
            if st.session_state.messages:
                questions = [msg for msg in st.session_state.messages if msg["type"] == "question"]
                if questions:
                    st.markdown("**Dernières questions :**")
                    for i, msg in enumerate(questions[-8:]):
                        words = msg['text'].split()[:4]
                        short_text = ' '.join(words)
                        if len(msg['text'].split()) > 4:
                            short_text += "..."
                        st.markdown(f"**{i+1}.** {short_text}")
                        st.caption(f"🕒 {msg.get('time', 'Récent')}")
                else:
                    st.info("Aucune question posée encore.")
            else:
                st.info("Aucune conversation en cours.")

def render_upload_view():
    """Affiche la vue d'upload des documents"""
    st.header("📄 Importation de Documents")
    
    uploaded_files = st.file_uploader(
        "Choisissez jusqu'à 5 documents PDF ou DOCX",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        help="Formats supportés : PDF, DOCX - Maximum 5 fichiers"
    )
    
    if uploaded_files:
        st.markdown("### 📁 Fichiers Sélectionnés")
        for i, file in enumerate(uploaded_files, 1):
            st.markdown(f"""
            <div style='background: #f8f9fa; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;'>
                <strong>{i}. {file.name}</strong>
                <div style='color: #6c757d; font-size: 0.9rem;'>
                    {file.size // 1024} KB • {file.type}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        if st.button("🚀 Indexer avec FAISS et LangChain", type="primary", use_container_width=True):
            if len(uploaded_files) > 5:
                st.error("❌ Maximum 5 fichiers autorisés")
            else:
                with st.spinner("🔍 Création de l'index FAISS avec LangChain..."):
                    all_chunks, processed_files = process_multiple_files(uploaded_files, st.session_state.document_texts)
                    
                    if not all_chunks:
                        st.error("❌ Aucun document valide n'a pu être analysé.")
                    else:
                        st.session_state.text_index = FAISSIndex()
                        st.session_state.text_index.add(all_chunks)
                        
                        st.success(f"✅ **{len(processed_files)} document(s) indexé(s) avec FAISS !**")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("📚 Documents", len(processed_files))
                        with col2:
                            st.metric("🔢 Segments", len(all_chunks))
                        with col3:
                            st.metric("🔍 Technologie", "FAISS")
                        
                        st.session_state.active_view = "chat"
                        st.rerun()

def render_chat_view():
    """Affiche la vue de chat principale"""
    if not st.session_state.text_index or st.session_state.text_index.ntotal == 0:
        st.warning("📄 **Aucun document chargé**")
        st.info("Veuillez d'abord importer et indexer des documents avec FAISS pour pouvoir poser des questions.")
        
        if st.button("📂 Importer des documents maintenant", use_container_width=True, type="primary"):
            st.session_state.active_view = "upload"
            st.rerun()
        st.stop()

    st.header("💬 Conversation avec Recherche Vectorielle")
    
    if st.session_state.text_index and st.session_state.text_index.ntotal > 0:
        doc_stats = st.session_state.text_index.get_document_stats()
        total_docs = len(doc_stats)
        total_chunks = st.session_state.text_index.ntotal
        st.success(f"📖 **{total_docs} document(s) indexé(s) avec FAISS** - {total_chunks} embeddings vectoriels disponibles")

    chat_container = st.container()
    
    with chat_container:
        for msg in st.session_state.messages:
            if msg["type"] == "question":
                with st.chat_message("user", avatar="👤"):
                    st.write(msg["text"])
                    if "time" in msg:
                        st.caption(f"🕒 {msg['time']}")
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    st.write(msg["text"])
                    if "time" in msg:
                        st.caption(f"🕒 {msg['time']}")

    if prompt := st.chat_input("Posez votre question sur les documents..."):
        current_time = datetime.now().strftime("%H:%M")
        st.session_state.messages.append(
            {"type": "question", "text": prompt, "time": current_time}
        )
        
        with chat_container:
            with st.chat_message("user", avatar="👤"):
                st.write(prompt)
                st.caption(f"🕒 {current_time}")
        
        with st.spinner("🔍 Recherche vectorielle FAISS en cours..."):
            scores, indices, context_chunks_with_metadata = st.session_state.text_index.search(prompt, k=8)
            
            if context_chunks_with_metadata:
                sources = list(set(chunk['source'] for chunk in context_chunks_with_metadata))
                if len(sources) > 0:
                    st.info(f"📖 **Sources trouvées par FAISS :** {', '.join(sources[:3])}" + ("..." if len(sources) > 3 else ""))
        
        with st.spinner("🤖 Analyse avec Mistral AI..."):
            answer = smart_text_analysis_with_mistral(
                context_chunks_with_metadata, 
                prompt, 
                st.session_state.messages
            )
        
        st.session_state.messages.append(
            {"type": "answer", "text": answer, "time": current_time}
        )
        
        with chat_container:
            with st.chat_message("assistant", avatar="🤖"):
                st.write(answer)
                st.caption(f"🕒 {current_time}")
        
        st.rerun()

def main():
    """Fonction principale de l'application"""
    st.set_page_config(
        page_title="RAG Mistral - Assistant Intelligent Multi-Documents",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    initialize_session_state()
    render_sidebar()
    
    st.title("🤖 Assistant Document Intelligent RAG")
    st.markdown("**Recherche Vectorielle FAISS** • **Framework LangChain** • **Modèle Mistral AI**")
    
    if st.session_state.active_view == "upload":
        render_upload_view()
    elif st.session_state.active_view == "chat":
        render_chat_view()

if __name__ == "__main__":
    main()