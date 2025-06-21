# WISSENS-APP PRO - FINALE HF-VERSION
import os
import uuid
import shutil
import traceback
from flask import Flask, request, jsonify, render_template_string, url_for, redirect
from dotenv import load_dotenv
from supabase import create_client, Client
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import random

# --- Konfiguration und Initialisierung ---
load_dotenv()
app = Flask(__name__)

# Supabase Konfiguration
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
BUCKET_NAME = "wissens-daten"

# KI-Komponenten laden
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.7)

# --- Routen der Anwendung ---
@app.route('/', methods=['GET', 'POST'])
def admin_page():
    if request.method == 'POST':
        try:
            files = request.files.getlist('files')
            if not files or files[0].filename == '': return "Keine Dateien ausgewählt", 400
            
            library_id = str(uuid.uuid4())
            docs = []
            
            # Temporärer Ordner für diese eine Anfrage
            temp_upload_dir = os.path.join('/tmp', f'upload_{library_id}')
            os.makedirs(temp_upload_dir, exist_ok=True)

            for file in files:
                filename = file.filename
                temp_file_path = os.path.join(temp_upload_dir, filename)
                
                # Datei temporär speichern, um sie zu lesen und hochzuladen
                file.save(temp_file_path)

                # Nach Supabase hochladen
                with open(temp_file_path, 'rb') as f:
                    supabase.storage.from_(BUCKET_NAME).upload(file=f, path=f"{library_id}/{filename}")

                # Für LangChain verarbeiten
                loader = PyMuPDFLoader(temp_file_path)
                docs.extend(loader.load())

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
            split_docs = text_splitter.split_documents(docs)
            vectorstore = FAISS.from_documents(split_docs, embeddings)
            
            temp_index_dir = os.path.join('/tmp', f'index_{library_id}')
            vectorstore.save_local(temp_index_dir)

            with open(os.path.join(temp_index_dir, "index.pkl"), 'rb') as f:
                supabase.storage.from_(BUCKET_NAME).upload(file=f, path=f"{library_id}/index.pkl")
            with open(os.path.join(temp_index_dir, "index.faiss"), 'rb') as f:
                supabase.storage.from_(BUCKET_NAME).upload(file=f, path=f"{library_id}/index.faiss")
            
            # Temporäre Ordner aufräumen
            shutil.rmtree(temp_upload_dir)
            shutil.rmtree(temp_index_dir)
            
            return redirect(url_for('creation_success', library_id=library_id))
        except Exception as e:
            traceback.print_exc()
            return "Ein Fehler ist beim Upload aufgetreten.", 500

    res = supabase.storage.from_(BUCKET_NAME).list()
    existing_libraries = [item['name'] for item in res if 'id' in item and item['id'] is not None]
    return render_template_string(ADMIN_HTML, libraries=existing_libraries, request=request)


# ... (Der Rest des Codes mit /success, /learn, /ask, /evaluate und den HTML-Strings bleibt exakt gleich wie in der letzten Version)
# (Ich füge ihn hier ausnahmsweise nicht komplett ein, um die Nachricht nicht unendlich lang zu machen - er hat sich nicht geändert)