import openai
import faiss
import numpy as np
import json
import os
from tqdm import tqdm

# ✅ Use variável de ambiente segura para a chave
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Lista para armazenar os blocos de texto
textos = []

# Carrega a base de conhecimento (JSON enriquecido)
with open("conhecimento_lyka.json", "r", encoding="utf-8") as f:
    base_conhecimento = json.load(f)

# Processamento por categoria (cada item de lista vira um vetor separado)
for categoria, dados in base_conhecimento.items():
    if isinstance(dados, list):
        for item in dados:
            if isinstance(item, dict) and "perguntas" in item and "resposta" in item:
                bloco = f"=== {categoria.upper().replace('_', ' ')} ===\n\nPerguntas possíveis:\n" + \
                        "\n".join(item["perguntas"]) + "\n\nResposta:\n" + item["resposta"]
                textos.append(bloco.lower())
            else:
                bloco = f"=== {categoria.upper().replace('_', ' ')} ===\n\n{item}"
                textos.append(bloco)

# Carrega a memória incremental (se houver)
if os.path.exists("memoria_incremental.json"):
    with open("memoria_incremental.json", "r", encoding="utf-8") as f:
        memoria = json.load(f)
        for item in memoria.get("memoria", []):
            textos.append(f"=== MEMORIA INCREMENTAL ===\n\n{item}")

# Função para obter embeddings da OpenAI
def get_openai_embedding(text):
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response['data'][0]['embedding']

# Gera vetores
print("🔄 Gerando embeddings com OpenAI...")
vetores = np.array([get_openai_embedding(text) for text in tqdm(textos)])

# Cria o índice FAISS
dim = len(vetores[0])
index = faiss.IndexFlatL2(dim)
index.add(vetores)

# Salva o índice FAISS e os textos associados
faiss.write_index(index, "indice_desaparecidos.index")
np.save("textos_desaparecidos.npy", np.array(textos))

print(f"Base vetorial criada com {len(textos)} blocos usando embeddings da OpenAI!")
