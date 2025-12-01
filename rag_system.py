import hashlib
from datetime import datetime
import streamlit as st

# Import LangChain pour FAISS
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from utils import extract_text, split_text

class FAISSIndex:
    """Index FAISS avec embeddings HuggingFace pour la recherche vectorielle"""
    def __init__(self):
        self.vector_store = None
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.chunks_with_metadata = []
    
    def add(self, chunks_with_metadata):
        """Ajoute les chunks à l'index FAISS avec LangChain"""
        self.chunks_with_metadata = chunks_with_metadata
        
        # Conversion en documents LangChain
        documents = []
        for chunk in chunks_with_metadata:
            doc = Document(
                page_content=chunk['text'],
                metadata={
                    'source': chunk['source'],
                    'file_hash': chunk['file_hash'],
                    'chunk_size': chunk['chunk_size']
                }
            )
            documents.append(doc)
        
       
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
    
    def search(self, query, k=8):
        """Recherche vectorielle avec FAISS et scores de similarité"""
        if self.vector_store is None:
            return [], [], []
      
        docs_and_scores = self.vector_store.similarity_search_with_score(query, k=k)
        
        scores = []
        indices = []
        top_chunks = []
        
        for i, (doc, score) in enumerate(docs_and_scores):
            scores.append(1 - score)  # Convertir distance en similarité (1 - distance)
            indices.append(i)
            top_chunks.append({
                'text': doc.page_content,
                'source': doc.metadata['source'],
                'file_hash': doc.metadata['file_hash'],
                'chunk_size': doc.metadata.get('chunk_size', 0)
            })
        
        return scores, indices, top_chunks
    
    @property
    def ntotal(self):
        return len(self.chunks_with_metadata)
    
    def get_document_stats(self):
        doc_stats = {}
        for chunk in self.chunks_with_metadata:
            source = chunk['source']
            if source not in doc_stats:
                doc_stats[source] = 0
            doc_stats[source] += 1
        return doc_stats

def process_multiple_files(uploaded_files, document_texts, max_files=5):
    """Traite plusieurs fichiers avec meilleure gestion des métadonnées"""
    if len(uploaded_files) > max_files:
        st.warning(f"⚠️ Maximum {max_files} fichiers autorisés. Seuls les {max_files} premiers seront traités.")
        uploaded_files = uploaded_files[:max_files]
    
    all_chunks = []
    processed_files = []
    
    for uploaded_file in uploaded_files:
        with st.spinner(f"🔍 Analyse approfondie de {uploaded_file.name}..."):
            try:
                # Extraction du texte
                text = extract_text(uploaded_file)
                
                if not text.strip():
                    st.warning(f"📄 {uploaded_file.name} - Document vide ou illisible")
                    continue
                
                # Découpage en chunks avec la nouvelle fonction LangChain
                chunks = split_text(text)
                
                # Stockage des textes par fichier
                file_hash = hashlib.md5(uploaded_file.name.encode()).hexdigest()[:8]
                document_texts[file_hash] = {
                    'name': uploaded_file.name,
                    'text': text,
                    'chunks': chunks,
                    'size': uploaded_file.size,
                    'processed_at': datetime.now().strftime("%H:%M")
                }
                
                # Ajout des chunks avec métadonnées enrichies
                for chunk in chunks:
                    all_chunks.append({
                        'text': chunk,
                        'source': uploaded_file.name,
                        'file_hash': file_hash,
                        'chunk_size': len(chunk)
                    })
                
                processed_files.append(uploaded_file.name)
                
            except Exception as e:
                st.error(f"❌ Erreur critique avec {uploaded_file.name}: {str(e)}")
    
    return all_chunks, processed_files