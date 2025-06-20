# WISSENS-APP PRO - FINALE VERSION 2.1 (RENDER-KOMPATIBEL)
import os
import uuid
import shutil
import traceback
from flask import Flask, request, jsonify, render_template_string, url_for, redirect
from dotenv import load_dotenv
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

# Intelligenter PFAD-BLOCK für Render und lokale Entwicklung
if os.environ.get('RENDER'):
    DATA_PATH = '/var/data'
else:
    DATA_PATH = '.'

UPLOAD_FOLDER = os.path.join(DATA_PATH, 'uploads')
VECTOR_STORE_FOLDER = os.path.join(DATA_PATH, 'vector_stores')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VECTOR_STORE_FOLDER, exist_ok=True)

# KI-Komponenten laden
try:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.7)
except Exception as e:
    print("!!! FEHLER BEI DER KI-INITIALISIERUNG - API-SCHLÜSSEL PRÜFEN !!!")
    print(f"Fehlerdetails: {e}")

# --- Hilfsfunktionen ---
def load_vector_store(library_id):
    path = os.path.join(VECTOR_STORE_FOLDER, library_id)
    if os.path.exists(path):
        return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    return None

# --- Routen der Anwendung ---
@app.route('/', methods=['GET', 'POST'])
def admin_page():
    if request.method == 'POST':
        if 'files' not in request.files: return "Keine Dateien im Request gefunden", 400
        files = request.files.getlist('files')
        if not files or files[0].filename == '': return "Keine Dateien ausgewählt", 400

        library_id = str(uuid.uuid4())
        library_upload_path = os.path.join(UPLOAD_FOLDER, library_id)
        library_vector_path = os.path.join(VECTOR_STORE_FOLDER, library_id)
        os.makedirs(library_upload_path, exist_ok=True)

        for file in files:
            file.save(os.path.join(library_upload_path, file.filename))

        docs = []
        for filename in os.listdir(library_upload_path):
            loader = PyMuPDFLoader(os.path.join(library_upload_path, filename))
            docs.extend(loader.load())
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
        split_docs = text_splitter.split_documents(docs)
        
        vectorstore = FAISS.from_documents(split_docs, embeddings)
        vectorstore.save_local(library_vector_path)
        
        return redirect(url_for('creation_success', library_id=library_id))

    existing_libraries = [d for d in os.listdir(VECTOR_STORE_FOLDER) if os.path.isdir(os.path.join(VECTOR_STORE_FOLDER, d))]
    return render_template_string(ADMIN_HTML, libraries=existing_libraries)

@app.route('/success/<library_id>')
def creation_success(library_id):
    learn_url = request.host_url.replace('http://', 'https://') + f'learn/{library_id}'
    return render_template_string(SUCCESS_HTML, learn_url=learn_url)

@app.route('/learn/<library_id>')
def learn_page(library_id):
    if not os.path.exists(os.path.join(VECTOR_STORE_FOLDER, library_id)):
        return "Bibliothek nicht gefunden!", 404
    return render_template_string(LERNER_HTML, library_id=library_id)

@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        library_id = request.json.get('library_id')
        vectorstore = load_vector_store(library_id)
        if not vectorstore: return jsonify({"error": "Bibliothek konnte nicht geladen werden."}), 404
        if not vectorstore.docstore._dict: return jsonify({"error": "Diese Bibliothek enthält keine Dokumente."}), 500

        random_doc_index = random.choice(list(vectorstore.docstore._dict.keys()))
        context_text = vectorstore.docstore._dict[random_doc_index].page_content

        prompt_template = "Du bist ein Lehrer. Erstelle eine prägnante Frage zum folgenden Textabschnitt... Frage:"
        prompt = PromptTemplate.from_template(prompt_template)
        chain = prompt | llm
        question = chain.invoke({"context": context_text}).content
        return jsonify({"question": question.strip()})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Server-Fehler in /ask: {str(e)}"}), 500

@app.route('/evaluate', methods=['POST'])
def evaluate_answer():
    try:
        data = request.json
        library_id, question, participant_answer = data.get('library_id'), data.get('question'), data.get('answer')
        vectorstore = load_vector_store(library_id)
        if not vectorstore: return jsonify({"error": "Bibliothek konnte nicht geladen werden."}), 404
        
        retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
        relevant_docs = retriever.get_relevant_documents(question + " " + participant_answer)
        context_from_docs = "\n\n".join([doc.page_content for doc in relevant_docs])

        evaluation_template = """Bewerte als strenger Prüfer... Bewertung und Begründung:"""
        prompt = PromptTemplate.from_template(evaluation_template)
        chain = prompt | llm
        evaluation_result = chain.invoke({"context": context_from_docs, "question": question, "answer": participant_answer}).content
        return jsonify({"evaluation": evaluation_result})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Server-Fehler in /evaluate: {str(e)}"}), 500

# Gekürzte HTML-Strings, um die Lesbarkeit zu verbessern
ADMIN_HTML = """<!DOCTYPE html>..."""
SUCCESS_HTML = """<!DOCTYPE html>..."""
LERNER_HTML = """<!DOCTYPE html>... (vollständiger Code wie zuvor)"""

if __name__ == '__main__':
    app.run(debug=True, port=5001)