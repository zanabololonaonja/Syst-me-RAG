from PyPDF2 import PdfReader
import docx
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter

def extract_text(file):
    """Extraction du texte PDF ou DOCX avec meilleure gestion des erreurs"""
    try:
        if file.name.endswith(".pdf"):
            reader = PdfReader(file)
            text = ""
            for page_num, page in enumerate(reader.pages, 1):
                page_text = page.extract_text() or ""
                # Ajout du numéro de page pour référence
                text += f"\n[Page {page_num}]\n{page_text}\n"
            return text.strip()
        elif file.name.endswith(".docx"):
            doc = docx.Document(file)
            full_text = []
            for para in doc.paragraphs:
                if para.text.strip():
                    full_text.append(para.text)
            return "\n".join(full_text)
        else:
            return ""
    except Exception as e:
        st.error(f"❌ Erreur lors de l'extraction du document: {str(e)}")
        return ""

def split_text(text, chunk_size=800, overlap=100):
    """Découpe le texte en segments plus intelligents avec meilleure préservation du contexte"""
    # ✅ UTILISATION DE LANGCHAIN POUR LE DECOUPAGE
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
    )
    
    chunks = text_splitter.split_text(text)
    return chunks