import faiss
import numpy as np
import openai
import os  # ✅ Para acessar a variável de ambiente com a chave

# ✅ Usa variável de ambiente para segurança
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Carrega o índice e textos como antes
index = faiss.read_index("indice_desaparecidos.index")
textos = np.load("textos_desaparecidos.npy", allow_pickle=True)

def obter_embedding_openai(texto):
    response = openai.Embedding.create(
        input=texto,
        model="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']

# Função utilitária que retorna o "modelo" (função para gerar embedding), índice e textos
def get_model_index_textos():
    # Aqui model será a função para gerar embedding, não um SentenceTransformer
    return obter_embedding_openai, index, textos
