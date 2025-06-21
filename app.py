# WISSENS-APP PRO - VERSION 3.0 (SUPABASE CLOUD-SPEICHER)
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

# Lade Supabase-Zugangsdaten aus den Umgebungsvariablen
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

BUCKET_NAME = "wissens-daten" # Der Name deines Buckets auf Supabase

# KI-Komponenten laden
try:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.7)
except Exception as e:
    print(f"!!! FEHLER BEI DER KI-INITIALISIERUNG: {e}")

# --- Hilfsfunktionen ---
def load_vector_store_from_supabase(library_id):
    """Lädt einen FAISS-Index aus Supabase Storage herunter und lokal."""
    try:
        # Pfad in Supabase und lokaler temporärer Pfad
        supabase_path = f"{library_id}/faiss_index.pkl"
        local_temp_path = f"temp_index_{library_id}"
        
        # Lade die Index-Datei herunter
        with open(f"{local_temp_path}.pkl", 'wb+') as f:
            res = supabase.storage.from_(BUCKET_NAME).download(supabase_path)
            f.write(res)
        
        # FAISS benötigt auch eine .faiss Datei, die wir ebenfalls herunterladen
        with open(f"{local_temp_path}.faiss", 'wb+') as f:
            res = supabase.storage.from_(BUCKET_NAME).download(f"{library_id}/faiss_index.faiss")
            f.write(res)

        print(f"--> Index für '{library_id}' von Supabase heruntergeladen.")
        
        # Lade den Index vom lokalen temporären Pfad
        vectorstore = FAISS.load_local(local_temp_path, embeddings, allow_dangerous_deserialization=True)
        
        # Temporäre Dateien aufräumen
        os.remove(f"{local_temp_path}.pkl")
        os.remove(f"{local_temp_path}.faiss")
        
        return vectorstore
    except Exception as e:
        print(f"Fehler beim Laden des Index für {library_id}: {e}")
        return None

# --- Routen der Anwendung ---
@app.route('/', methods=['GET', 'POST'])
def admin_page():
    if request.method == 'POST':
        files = request.files.getlist('files')
        if not files or files[0].filename == '': return "Keine Dateien ausgewählt", 400

        library_id = str(uuid.uuid4())
        print(f"Erstelle neue Bibliothek mit ID: {library_id}")

        # Verarbeite die PDFs direkt im Speicher
        docs = []
        for file in files:
            # Lade PDF-Daten in den Speicher
            pdf_bytes = file.read()
            # Lade aus Bytes (erfordert PyMuPDF)
            # Hierzu erstellen wir eine temporäre Datei, da PyMuPDFLoader einen Pfad erwartet
            temp_file_path = f"temp_{file.filename}"
            with open(temp_file_path, 'wb') as f:
                f.write(pdf_bytes)
            
            # Lade die PDF mit LangChain
            loader = PyMuPDFLoader(temp_file_path)
            docs.extend(loader.load())
            
            # Lade PDF-Datei nach Supabase hoch
            supabase.storage.from_(BUCKET_NAME).upload(file=pdf_bytes, path=f"{library_id}/{file.filename}")
            print(f"--> PDF '{file.filename}' nach Supabase hochgeladen.")

            # Lösche temporäre Datei
            os.remove(temp_file_path)

        # Erstelle Vektor-Store
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
        split_docs = text_splitter.split_documents(docs)
        vectorstore = FAISS.from_documents(split_docs, embeddings)
        
        # Speichere Index temporär lokal, um ihn dann hochzuladen
        temp_index_path = f"temp_index_{library_id}"
        vectorstore.save_local(temp_index_path)

        # Lade die beiden Index-Dateien nach Supabase hoch
        with open(f"{temp_index_path}.pkl", 'rb') as f:
            supabase.storage.from_(BUCKET_NAME).upload(file=f, path=f"{library_id}/faiss_index.pkl")
        with open(f"{temp_index_path}.faiss", 'rb') as f:
            supabase.storage.from_(BUCKET_NAME).upload(file=f, path=f"{library_id}/faiss_index.faiss")
        
        print(f"--> FAISS Index für '{library_id}' nach Supabase hochgeladen.")

        # Lösche temporäre Index-Dateien
        os.remove(f"{temp_index_path}.pkl")
        os.remove(f"{temp_index_path}.faiss")
        
        return redirect(url_for('creation_success', library_id=library_id))

    # Zeige alle existierenden Bibliotheken an (indem wir die "Ordner" in Supabase auflisten)
    res = supabase.storage.from_(BUCKET_NAME).list()
    existing_libraries = [item['name'] for item in res if item['id'] is not None]
    return render_template_string(ADMIN_HTML, libraries=existing_libraries)

# Die anderen Routen (/success, /learn, /ask, /evaluate) bleiben fast identisch,
# sie müssen nur `load_vector_store_from_supabase` statt `load_vector_store` aufrufen.
# Der restliche HTML-Code bleibt ebenfalls identisch.
# (Aus Kürze hier weggelassen, aber im Kopf behalten)

# Hier ist ein Beispiel für die angepasste /ask Route:
@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        library_id = request.json.get('library_id')
        vectorstore = load_vector_store_from_supabase(library_id) # ANGEPASSTER AUFRUF
        if not vectorstore: return jsonify({"error": "Bibliothek konnte nicht geladen werden."}), 404
        #... rest der Funktion bleibt gleich ...
        random_doc_index = random.choice(list(vectorstore.docstore._dict.keys()))
        context_text = vectorstore.docstore._dict[random_doc_index].page_content
        prompt_template = "Du bist ein Lehrer... Frage:"
        prompt = PromptTemplate.from_template(prompt_template)
        chain = prompt | llm
        question = chain.invoke({"context": context_text}).content
        return jsonify({"question": question.strip()})
    except Exception as e:
        # ... fehlerbehandlung bleibt gleich
        return jsonify({"error": f"Server-Fehler: {str(e)}"}), 500
        
# Hier müsste der restliche Code (andere Routen, HTML) aus der letzten Version folgen.