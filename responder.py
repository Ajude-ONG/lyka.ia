from vetor import get_model_index_textos  # Assumo que retorna model, index, textos
import openai
import numpy as np
import os
from memoria import salvar_na_memoria

# Configurar chave OpenAI via variável de ambiente
openai.api_key = os.environ.get("OPENAI_API_KEY")

def obter_embedding_openai(texto):
    response = openai.Embedding.create(
        input=texto.lower(),
        model="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']

def responder_lyka(pergunta):
    model, index, textos = get_model_index_textos()
    vetor_pergunta = [obter_embedding_openai(pergunta)]
    vetor_pergunta = np.array(vetor_pergunta).astype('float32')

    D, I = index.search(vetor_pergunta, k=20)
    indices_validos = [i for i in I[0] if i != -1]

    if not indices_validos or max(D[0]) > 5.0:
        resposta = fallback_palavras_chave(pergunta)
        salvar_na_memoria(pergunta, resposta)
        return resposta

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
        response = openai.ChatCompletion.create(
            model="gpt-4o",  # Pode mudar para outro se desejar
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1000,
        )
        resposta = response.choices[0].message.content.strip()
        salvar_na_memoria(pergunta, resposta)
        return resposta
    except Exception as e:
        resposta_erro = f"[Erro ao gerar resposta]: {str(e)}"
        salvar_na_memoria(pergunta, resposta_erro)
        return resposta_erro

def fallback_palavras_chave(pergunta):
    p = pergunta.lower()
    if "procedimento" in p or "buscar" in p:
        return "Em buscas aéreas, verifique rotas, pontos de impacto, sinais visuais e relatos locais."
    elif "elt" in p:
        return "ELT é um transmissor de emergência que pode ajudar a localizar aeronaves desaparecidas."
    elif "tático" in p or "resgate" in p:
        return "Utilize análise visual, olfato, sons e medidas de segurança em campo."
    else:
        return "Essa informação não está disponível no momento."

# Interface de teste local
if __name__ == "__main__":
    print("🧠 Lyka — Fallback + Vetor com OpenAI Integrado")
    while True:
        pergunta = input("\nAgente: ")
        if pergunta.lower() in ["sair", "exit"]:
            print("Encerrando Lyka. Até logo.")
            break
        resposta = responder_lyka(pergunta)
        print("\nLyka:\n" + resposta)
