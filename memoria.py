import sqlite3
from datetime import datetime

# Inicializa ou conecta ao banco de dados SQLite local
conn = sqlite3.connect("memoria_lyka.db")
cursor = conn.cursor()

# Cria a tabela de memória, se ainda não existir
cursor.execute("""
CREATE TABLE IF NOT EXISTS memoria (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pergunta TEXT NOT NULL,
    resposta TEXT NOT NULL,
    data_hora TEXT NOT NULL
)
""")
conn.commit()

def salvar_na_memoria(pergunta, resposta):
    data_hora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
    INSERT INTO memoria (pergunta, resposta, data_hora)
    VALUES (?, ?, ?)
    """, (pergunta, resposta, data_hora))
    conn.commit()

def recuperar_memoria(limit=20):
    cursor.execute("""
    SELECT pergunta, resposta, data_hora FROM memoria
    ORDER BY id DESC
    LIMIT ?
    """, (limit,))
    return cursor.fetchall()

def limpar_memoria():
    cursor.execute("DELETE FROM memoria")
    conn.commit()

def fechar_memoria():
    conn.close()
