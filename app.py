from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from openai import OpenAI
import numpy as np
import os
from vetor import get_model_index_textos
from responder import fallback_palavras_chave, obter_embedding_openai

# Inicializa o app Flask
app = Flask(__name__)
CORS(app)

# Inicializa o cliente da nova API da OpenAI
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Função principal com busca vetorial + fallback seguro
def responder_pergunta(pergunta):
    _, index, textos = get_model_index_textos()
    vetor_pergunta = [obter_embedding_openai(pergunta)]
    vetor_pergunta = np.array(vetor_pergunta).astype("float32")
    D, I = index.search(vetor_pergunta, k=8)
    indices_validos = [i for i in I[0] if i != -1]

    if not indices_validos or max(D[0]) > 2.5:
        return fallback_palavras_chave(pergunta)

    trechos_mais_relevantes = [textos[i] for i in indices_validos]
    contexto = "\n".join(trechos_mais_relevantes)

    prompt = f"""
Você é Lyka, uma inteligência artificial treinada exclusivamente para auxiliar em casos de desaparecimentos aéreos.

⚠️ REGRAS:
1. Você **não tem conhecimento próprio** dos casos. Sua função é apenas auxiliar com informações e estratégias baseadas no conteúdo fornecido.
2. Responda **apenas** com base no conteúdo abaixo.
3. Não invente, deduza ou complemente informações.
4. Se a resposta **não estiver no conteúdo abaixo**, responda **exatamente**:
"Essa informação não está disponível no momento. Não possuo conhecimento sobre casos ainda."
5. Seja curta, objetiva e direta.

📚 CONTEÚDO:
{contexto}

🗨️ PERGUNTA:
{pergunta}

💬 RESPOSTA:
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1000
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"[ERRO] Falha na API OpenAI: {e}")
        return "[Erro temporário na IA. Respondendo com base reduzida:]\n\n" + fallback_palavras_chave(pergunta)

# Rota da API
@app.route("/responder", methods=["POST"])
def chat():
    dados = request.json
    pergunta = dados.get("mensagem", "")
    resposta = responder_pergunta(pergunta)
    return jsonify({"resposta": resposta})

# Rota raiz para teste no navegador
@app.route("/", methods=["GET"])
def home():
    return send_file("index.html")

# Inicializa o servidor Flask
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
