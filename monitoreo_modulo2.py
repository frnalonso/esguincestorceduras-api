"""
App de ejemplo: Neo4j + LangChain + Ollama
Subdominio: Monitoreo y Frecuencia en lesiones musculoesqueléticas

Flujo:
  1) Pregunta en lenguaje natural
  2) LLM genera SOLO la consulta Cypher (NL -> Cypher)
  3) Ejecutamos la consulta en Neo4j (Cypher -> datos)
  4) Devolvemos resultados en JSON

Sin GraphCypherQAChain: control total del prompt y del Cypher generado.

Requisitos:
  pip install "langchain>=0.3" langchain-community neo4j fastapi uvicorn "langchain-ollama>=0.1.0"
  ollama pull llama3.1   # o el modelo que prefieras

Variables de entorno:
  export NEO4J_URI="bolt://localhost:7687"        # o la URI de Aura
  export NEO4J_USER="neo4j"
  export NEO4J_PASSWORD="tu_clave"
  export OLLAMA_MODEL="llama3.1"                  # opcional
"""

import os
from typing import Optional

from fastapi import FastAPI
from langchain_community.graphs import Neo4jGraph
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

# =============================
# Conexión Neo4j
# =============================

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    username=os.getenv("NEO4J_USER", "neo4j"),
    password=os.getenv("NEO4J_PASSWORD", "password"),
)

# =============================
# Esquema documental (opcional, solo para imprimir)
# =============================

DOMAIN_SCHEMA = """
Labels:
  Paciente(obra_social, contacto, domicilio, observaciones, id, sexo, fecha_ingreso, nombre, edad)
  Sintoma(descripcion, lado, lugar_afectado, fecha_registro, intensidad, notas, id, nombre)
  Indicador(valor_num, valor, fecha_observacion, id, nombre)
  TipoLesion(descripcion, zona, gravedad_aparente, lado, causa, fecha_evento, id, nombre)
  Diagnostico(descripcion, grado, estado, fecha_diagnostico, id, comentarios)
  Monitoreo(fecha, tipo, estado, observaciones, resultado_general, id)
  Frecuencia(id, cada, unidad, duracion_total, observaciones)

Relationships:
  (Paciente)-[:PRESENTA_SINTOMA]->(Sintoma)
  (Sintoma)-[:TIENE_INDICADOR]->(Indicador)
  (Sintoma)-[:ASOCIA_A]->(TipoLesion)
  (TipoLesion)-[:ESPECIFICA_EN]->(Diagnostico)
  (TipoLesion)-[:REQUIERE_MONITOREO]->(Monitoreo)
  (Monitoreo)-[:OCURRE_CADA]->(Frecuencia)
  (Monitoreo)-[:ACTUALIZA]->(Diagnostico)
"""

print(">>> ESQUEMA DOCUMENTAL (referencial):")
print(DOMAIN_SCHEMA)
print(">>> FIN ESQUEMA\n")

# =============================
# Prompt para generar Cypher (NL -> Cypher)
# =============================

CYHER_PROMPT_TMPL = """
Eres un asistente experto en Cypher para Neo4j.
Tu tarea es, dado el esquema y una pregunta en lenguaje natural,
generar UNA sola consulta Cypher correcta.

Usa EXCLUSIVAMENTE las Labels, propiedades y Relationships del esquema.

Esquema detectado (graph.schema):
{schema}

ACLARACIONES CRÍTICAS SOBRE LAS PROPIEDADES:

- La etiqueta Monitoreo solo tiene las propiedades:
  - fecha
  - tipo
  - estado
  - observaciones
  - resultado_general
  - id

- La propiedad 'detalle' NO EXISTE en la etiqueta Monitoreo.
  NUNCA uses 'm.detalle' ni 'detalle' en ninguna consulta.
- La propiedad 'nombre' TAMPOCO existe en Monitoreo. No la uses con 'm.nombre'.

Ejemplos de consultas correctas (few-shot, NO inventar labels/propiedades):

// Pregunta: "¿Qué monitoreos se recomiendan para una torcedura?"
MATCH (t:TipoLesion {nombre:"Torcedura"})-[:REQUIERE_MONITOREO]->(m:Monitoreo)
OPTIONAL MATCH (m)-[:OCURRE_CADA]->(f:Frecuencia)
RETURN
  t.nombre           AS tipo_lesion,
  m.tipo             AS tipo_monitoreo,
  m.estado           AS estado_monitoreo,
  f.cada             AS cada,
  f.unidad           AS unidad;

// Pregunta: "¿Cada cuánto se realiza el monitoreo de control funcional?"
MATCH (m:Monitoreo {tipo:"Control funcional"})-[:OCURRE_CADA]->(f:Frecuencia)
RETURN
  m.tipo            AS tipo_monitoreo,
  f.cada            AS cada,
  f.unidad          AS unidad,
  f.duracion_total  AS duracion_total,
  f.observaciones   AS observaciones;

// Pregunta: "¿Qué tipos de monitoreos existen?"
MATCH (m:Monitoreo)
RETURN DISTINCT m.tipo AS tipo_monitoreo;

// Pregunta: "¿Cuántos síntomas existen?"
MATCH (s:Sintoma)
RETURN count(s) AS total_sintomas;

Instrucciones IMPORTANTES:
- No inventes labels ni propiedades que no aparezcan en el esquema.
- Para Monitoreo usa solamente: fecha, tipo, estado, observaciones, resultado_general, id.
- NO uses nunca la propiedad 'detalle' en ningún nodo.
- Devuelve SOLO la consulta Cypher, sin explicación adicional, sin comentarios,
  sin texto extra, sin comillas, sin envoltorios tipo ```cypher.
- No incluyas la palabra 'Cypher' en la salida, solo la consulta.

Pregunta: {question}
"""

cypher_prompt = PromptTemplate(
    input_variables=["schema", "question"],
    template=CYHER_PROMPT_TMPL,
)

# =============================
# LLM (Ollama) para NL -> Cypher
# =============================

llm = ChatOllama(
    model=os.getenv("OLLAMA_MODEL", "llama3.1"),
    temperature=0.0,
)


def clean_cypher_output(text: str) -> str:
    """
    Limpia la salida del LLM por si viene envuelta en ```...``` o con basura extra.
    """
    cypher = text.strip()

    # Quitar bloques ```cypher ... ``` o ``` ...
    if cypher.startswith("```"):
        # eliminar primera línea ```
        lines = cypher.splitlines()
        # sacar primera y última si terminan en ```
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cypher = "\n".join(lines).strip()

    # Por si el modelo mete algo tipo "Consulta:" al inicio (muy básico)
    prefixes = ["Consulta:", "Query:", "Cypher:"]
    for p in prefixes:
        if cypher.lower().startswith(p.lower()):
            cypher = cypher[len(p):].strip()

    return cypher


def generate_cypher(question: str) -> str:
    """
    Genera una consulta Cypher a partir de la pregunta en lenguaje natural.
    """
    schema_text = graph.schema  # esquema real detectado por Neo4j
    prompt = cypher_prompt.format(schema=schema_text, question=question)
    response = llm.invoke(prompt)

    # ChatOllama devuelve un mensaje con .content
    raw_text = getattr(response, "content", str(response))
    cypher = clean_cypher_output(raw_text)

    print(">>> PROMPT ENVIADO AL LLM:")
    print(prompt)
    print(">>> RESPUESTA BRUTA DEL LLM:")
    print(raw_text)
    print(">>> CYPHER LIMPIO:")
    print(cypher)
    print(">>> FIN GENERACIÓN CYPHER\n")

    return cypher


def run_cypher_query(cypher: str):
    """
    Ejecuta la consulta Cypher en Neo4j y devuelve los resultados.
    """
    print(">>> EJECUTANDO CYPHER EN NEO4J:")
    print(cypher)
    print(">>> ------------------------------")
    result = graph.query(cypher)
    print(">>> RESULTADO NEO4J:")
    print(result)
    print(">>> FIN EJECUCIÓN\n")
    return result


# =============================
# FastAPI
# =============================

app = FastAPI(title="Lesiones (Monitoreo/Frecuencia): NL -> Cypher -> Neo4j (sin GraphCypherQAChain)")


@app.get("/query")
def query(question: str, explain: Optional[bool] = False):
    """
    Endpoint principal:
      - question: pregunta en lenguaje natural
      - explain=True para ver también la consulta Cypher generada
    """
    cypher = generate_cypher(question)

    try:
        result = run_cypher_query(cypher)
    except Exception as e:
        # Si hay error en Neo4j (por ejemplo, vuelve a aparecer 'detalle')
        return {
            "error": str(e),
            "cypher_generado": cypher,
        }

    if explain:
        return {
            "question": question,
            "cypher": cypher,
            "result": result,
        }

    # Modo simple: solo datos y la consulta para referencia
    return {
        "cypher": cypher,
        "result": result,
    }


# =============================
# Launcher
# =============================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
