import openai
import numpy as np
import faiss
import os  # ✅ Para acessar variáveis de ambiente

# ✅ Use variável de ambiente em vez de chave fixa
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Função para obter embeddings pela API OpenAI
def obter_embedding(texto):
    response = openai.Embedding.create(
        input=texto,
        model="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']

# 1. Base de conhecimento simples
corpus = [
    "O avião desapareceu durante o voo noturno.",
    "O resgate foi enviado para a área montanhosa.",
    "As investigações apontam falha técnica.",
    "O clima estava severo no momento do desaparecimento.",
    "O último contato foi feito próximo ao radar."
]

# 2. Gera embeddings para todo o corpus usando API OpenAI
corpus_embeddings = [obter_embedding(texto) for texto in corpus]
corpus_embeddings = np.array(corpus_embeddings).astype('float32')

# 3. Cria o índice FAISS
dimension = len(corpus_embeddings[0])
index = faiss.IndexFlatL2(dimension)
index.add(corpus_embeddings)

# 4. Entrada do usuário
consulta = input("Digite sua pergunta: ")

# 5. Gera embedding da consulta
consulta_embedding = np.array([obter_embedding(consulta)]).astype('float32')

# 6. Busca os k=5 textos mais próximos
k = 5
distancias, indices = index.search(consulta_embedding, k)

# 7. Exibe os resultados
print("\nResultados mais semelhantes:")
for i, idx in enumerate(indices[0]):
    print(f"{i+1}. {corpus[idx]} (distância: {distancias[0][i]:.4f})")
