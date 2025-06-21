# MINIMALE TEST-VERSION
from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World! Der Basis-Server funktioniert.'

# Kein Supabase, kein Google, kein LangChain, nichts Komplexes.