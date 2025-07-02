import json
with open("base_desaparecimentos_aereos.json", "r", encoding="utf-8") as f:
    conhecimento = json.load(f)
print("Casos carregados:", len(conhecimento["casos_registrados"]))
