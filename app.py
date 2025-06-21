# WISSENS-APP PRO - VERSION 3.5 (FINALER BUGFIX)
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

# Intelligenter PFAD-BLOCK für Render und lokale Entwicklung
if os.environ.get('RENDER'):
    DATA_PATH = '/var/data'
else:
    DATA_PATH = '.'

UPLOAD_FOLDER = os.path.join(DATA_PATH, 'uploads')
VECTOR_STORE_FOLDER = os.path.join(DATA_PATH, 'vector_stores')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VECTOR_STORE_FOLDER, exist_ok=True)

# Supabase Konfiguration
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
BUCKET_NAME = "wissens-daten"

# KI-Komponenten laden
try:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.7)
except Exception as e:
    print(f"!!! FEHLER BEI DER KI-INITIALISIERUNG: {e}")

# --- Hilfsfunktionen ---
def load_vector_store_from_supabase(library_id):
    try:
        supabase_path_pkl = f"{library_id}/index.pkl" # Korrekter Dateiname
        supabase_path_faiss = f"{library_id}/index.faiss" # Korrekter Dateiname
        local_temp_folder = f"temp_index_{library_id}"
        os.makedirs(local_temp_folder, exist_ok=True) # Erstelle den temporären Ordner
        
        # Korrekte lokale Pfade
        local_pkl_path = os.path.join(local_temp_folder, "index.pkl")
        local_faiss_path = os.path.join(local_temp_folder, "index.faiss")

        with open(local_pkl_path, 'wb+') as f:
            f.write(supabase.storage.from_(BUCKET_NAME).download(path=supabase_path_pkl))
        with open(local_faiss_path, 'wb+') as f:
            f.write(supabase.storage.from_(BUCKET_NAME).download(path=supabase_path_faiss))
        
        vectorstore = FAISS.load_local(local_temp_folder, embeddings, allow_dangerous_deserialization=True)
        shutil.rmtree(local_temp_folder) # Räume den ganzen Ordner auf
        return vectorstore
    except Exception as e:
        print(f"Fehler beim Laden des Index für {library_id}: {e}")
        return None

# --- Routen der Anwendung ---
@app.route('/', methods=['GET', 'POST'])
def admin_page():
    if request.method == 'POST':
        try:
            files = request.files.getlist('files')
            if not files or files[0].filename == '': return "Keine Dateien ausgewählt", 400
            
            library_id = str(uuid.uuid4())
            docs = []
            for file in files:
                filename = file.filename
                pdf_bytes = file.read()
                
                supabase.storage.from_(BUCKET_NAME).upload(
                    path=f"{library_id}/{filename}", file=pdf_bytes, file_options={"content-type": "application/pdf"}
                )

                temp_file_path = f"temp_{filename}"
                with open(temp_file_path, 'wb') as f: f.write(pdf_bytes)
                loader = PyMuPDFLoader(temp_file_path)
                docs.extend(loader.load())
                os.remove(temp_file_path)

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
            split_docs = text_splitter.split_documents(docs)
            vectorstore = FAISS.from_documents(split_docs, embeddings)
            
            temp_index_folder = f"temp_index_{library_id}"
            vectorstore.save_local(temp_index_folder)

            # === HIER IST DIE KORREKTUR FÜR DEN FileNotFoundError ===
            # Korrekter Pfad zu den Dateien INNERHALB des Ordners
            pkl_path = os.path.join(temp_index_folder, "index.pkl")
            faiss_path = os.path.join(temp_index_folder, "index.faiss")

            with open(pkl_path, 'rb') as f_pkl:
                supabase.storage.from_(BUCKET_NAME).upload(file=f_pkl.read(), path=f"{library_id}/index.pkl")
            with open(faiss_path, 'rb') as f_faiss:
                supabase.storage.from_(BUCKET_NAME).upload(file=f_faiss.read(), path=f"{library_id}/index.faiss")
            
            shutil.rmtree(temp_index_folder)
            # === ENDE DER KORREKTUR ===
            
            return redirect(url_for('creation_success', library_id=library_id))
        except Exception as e:
            traceback.print_exc()
            return "Ein Fehler ist beim Upload aufgetreten.", 500

    res = supabase.storage.from_(BUCKET_NAME).list()
    existing_libraries = [item['name'] for item in res if item['id'] is not None]
    return render_template_string(ADMIN_HTML, libraries=existing_libraries, request=request)

# ... der Rest des Codes (andere Routen und HTML) bleibt exakt gleich ...

@app.route('/success/<library_id>')
def creation_success(library_id):
    learn_url = request.host_url.replace('http://', 'https') + f'learn/{library_id}'
    return render_template_string(SUCCESS_HTML, learn_url=learn_url)

@app.route('/learn/<library_id>')
def learn_page(library_id):
    res = supabase.storage.from_(BUCKET_NAME).list(path=library_id)
    if not res: return "Bibliothek nicht gefunden!", 404
    return render_template_string(LERNER_HTML, library_id=library_id)

@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        library_id = request.json.get('library_id')
        vectorstore = load_vector_store_from_supabase(library_id)
        if not vectorstore: return jsonify({"error": "Bibliothek konnte nicht geladen werden oder ist leer."}), 404
        if not vectorstore.docstore._dict: return jsonify({"error": "Diese Bibliothek enthält keine Dokumente."}), 500
        random_doc_index = random.choice(list(vectorstore.docstore._dict.keys()))
        context_text = vectorstore.docstore._dict[random_doc_index].page_content
        prompt_template = "Du bist ein Lehrer..."
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
        vectorstore = load_vector_store_from_supabase(library_id)
        if not vectorstore: return jsonify({"error": "Bibliothek konnte nicht geladen werden."}), 404
        retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
        relevant_docs = retriever.get_relevant_documents(question + " " + participant_answer)
        context_from_docs = "\n\n".join([doc.page_content for doc in relevant_docs])
        evaluation_template = """Bewerte als strenger Prüfer..."""
        prompt = PromptTemplate.from_template(evaluation_template)
        chain = prompt | llm
        evaluation_result = chain.invoke({"context": context_from_docs, "question": question, "answer": participant_answer}).content
        return jsonify({"evaluation": evaluation_result})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Server-Fehler in /evaluate: {str(e)}"}), 500

ADMIN_HTML = """
<!DOCTYPE html><html lang="de"><head><title>Admin: Wissensbibliotheken</title>
<style>body{font-family:sans-serif;max-width:800px;margin:auto;padding:20px;background:#f0f8ff} h1,h2{color:#005a9c} a{color:#007bff} ul{list-style-type:none;padding:0} li{background:#fff;margin-bottom:10px;padding:15px;border-radius:5px;box-shadow:0 2px 4px rgba(0,0,0,0.1)}</style>
</head><body>
<h1>Admin-Bereich</h1><h2>Neue Bibliothek erstellen</h2>
<form method="post" enctype="multipart/form-data"><p>Wählen Sie die PDFs aus, um eine neue, permanente Bibliothek zu erstellen:</p><input type="file" name="files" multiple required><button type="submit">Bibliothek erstellen</button></form><hr>
<h2>Bestehende Bibliotheken</h2>
{% if libraries %}<ul>{% for lib_id in libraries %}<li><strong>Bibliothek-ID:</strong> {{ lib_id }}<br><strong>Lern-Link:</strong> <a href="/learn/{{ lib_id }}" target="_blank">{{ request.host_url.replace('http://', 'https://') }}learn/{{ lib_id }}</a></li>{% endfor %}</ul>
{% else %}<p>Noch keine Bibliotheken erstellt.</p>{% endif %}
</body></html>
"""
SUCCESS_HTML = """
<!DOCTYPE html><html lang="de"><head><title>Erfolg!</title>
<style>body{font-family:sans-serif;text-align:center;padding-top:50px} div{background:#e0ffe0;border:1px solid green;padding:30px;max-width:700px;margin:auto;border-radius:10px}</style>
</head><body>
<div><h1>Bibliothek erfolgreich erstellt!</h1><p>Teilen Sie diesen permanenten Link mit Ihren Teilnehmern:</p><a href="{{ learn_url }}">{{ learn_url }}</a><br><br><a href="/">Zurück zur Admin-Seite</a></div>
</body></html>
"""
LERNER_HTML = """
<!DOCTYPE html><html lang="de"><head><title>Lern-Modul</title>
<style>body{font-family:sans-serif;max-width:600px;margin:auto;padding:20px;background-color:#f4f7f6}#qa-area{display:none;margin-top:20px}#question{font-weight:bold;margin-bottom:10px;font-size:1.1em}#evaluation{margin-top:20px;padding:15px;border:1px solid #ccc;background:#fff;white-space:pre-wrap;border-radius:5px}button{background-color:#28a745;color:white;padding:10px 15px;border:none;border-radius:5px;cursor:pointer}button:hover{background-color:#218838}button:disabled{background-color:#6c757d}</style>
</head><body><h1>Lernen mit KI</h1><button id="get-question-btn">Neue Frage anfordern</button><div id="qa-area"><p id="question"></p><textarea id="answer" rows="4" style="width:100%;" placeholder="Ihre Antwort hier..."></textarea><button id="submit-answer-btn">Antwort abschicken</button></div><div id="evaluation" style="display:none;"></div>
<script>
    const libraryId = '{{ library_id }}';
    const getQuestionBtn = document.getElementById('get-question-btn');
    const submitAnswerBtn = document.getElementById('submit-answer-btn');
    const qaArea = document.getElementById('qa-area');
    const questionEl = document.getElementById('question');
    const answerEl = document.getElementById('answer');
    const evaluationEl = document.getElementById('evaluation');
    async function getQuestion() {
        getQuestionBtn.disabled = true;
        getQuestionBtn.textContent = 'Generiere Frage...';
        const response = await fetch('/ask', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ library_id: libraryId })});
        const data = await response.json();
        if (response.ok) {
            questionEl.textContent = data.question;
            qaArea.style.display = 'block';
            evaluationEl.style.display = 'none';
            getQuestionBtn.style.display = 'none';
        } else {
            alert('Fehler beim Fragen holen: ' + data.error);
            getQuestionBtn.disabled = false;
            getQuestionBtn.textContent = 'Neue Frage anfordern';
        }
    }
    async function evaluateAnswer() {
        const question = questionEl.textContent;
        const answer = answerEl.value;
        if (!answer.trim()) { alert('Bitte geben Sie eine Antwort ein.'); return; }
        submitAnswerBtn.disabled = true;
        submitAnswerBtn.textContent = 'Bewerte...';
        const response = await fetch('/evaluate', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ library_id: libraryId, question: question, answer: answer })});
        const data = await response.json();
        submitAnswerBtn.disabled = false;
        submitAnswerBtn.textContent = 'Antwort abschicken';
        if (response.ok) {
            evaluationEl.innerText = data.evaluation;
            evaluationEl.style.display = 'block';
            qaArea.style.display = 'none';
            getQuestionBtn.style.display = 'block';
            getQuestionBtn.disabled = false;
            getQuestionBtn.textContent = 'Nächste Frage anfordern';
            answerEl.value = '';
        } else {
            alert('Fehler bei der Bewertung: ' + data.error);
        }
    }
    getQuestionBtn.addEventListener('click', getQuestion);
    submitAnswerBtn.addEventListener('click', evaluateAnswer);
</script>
</body></html>
"""

if __name__ == '__main__':
    app.run(debug=True, port=5001)