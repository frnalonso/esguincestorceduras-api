"""
App de ejemplo: Neo4j + LangChain + Ollama
Dominio: Lesiones esguinces y torceduras

- NL -> Cypher -> Respuesta, usando GraphCypherQAChain
- Prompt guiado por esquema (schema-guided)
- Endpoint /query para pruebas
- Endpoint /fuzzy para pertenencia gaussiana (lógica difusa) sobre Indicadores

Requisitos:
  pip install "langchain>=0.3" langchain-community neo4j fastapi uvicorn gradio "langchain-ollama>=0.1.0"
  ollama pull llama3.1   # o el modelo que prefieras

Variables de entorno:
  export NEO4J_URI="bolt://localhost:7687"
  export NEO4J_USER="neo4j"
  export NEO4J_PASSWORD="tu_clave"
  export OLLAMA_MODEL="llama3.1"   # opcional
"""

import os
import math
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, Body
from pydantic import BaseModel

from langchain_community.graphs import Neo4jGraph
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain

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
# Esquema de dominio (frames/slots simplificado)
# =============================
# Nodos: Paciente, Sintoma, Indicador, TipoLesion, Diagnostico, Tratamiento, Monitoreo, Frecuencia, ProfesionalSalud
# Relaciones principales:
#   (:Paciente)-[:PRESENTA_SINTOMA]->(:Sintoma)
#   (:Sintoma)-[:TIENE_INDICADOR]->(:Indicador)
#   (:Indicador)-[:ASOCIA_A]->(:TipoLesion)
#   (:Diagnostico)-[:ESPECIFICA_EN]->(:TipoLesion)
#   (:Diagnostico)-[:SUGIERE_TRATAMIENTO]->(:Tratamiento)
#   (:ProfesionalSalud)-[:DERIVA_A]->(:ProfesionalSalud)
#   (:Tratamiento)-[:PROGRAMA]->(:Monitoreo)
#   (:Monitoreo)-[:OCURRE_CADA]->(:Frecuencia)
#   (:ProfesionalSalud)-[:ACTUALIZA]->(:Indicador)
# Atributos típicos:
#   Paciente{dni, nombre, edad}
#   Sintoma{nombre}
#   Indicador{nombre, valor, unidad}
#   TipoLesion{nombre, grado}
#   Diagnostico{fecha, grado, probabilidad}
#   Tratamiento{nombre, descripcion}
#   Monitoreo{detalle}
#   Frecuencia{unidad, cantidad}
#   ProfesionalSalud{nombre, matricula, especialidad}

DOMAIN_SCHEMA = """
Labels:
  Paciente(dni, nombre, edad)
  Sintoma(nombre)
  Indicador(nombre, valor, unidad)
  TipoLesion(nombre, grado)
  Diagnostico(fecha, grado, probabilidad)
  Tratamiento(nombre, descripcion)
  Monitoreo(detalle)
  Frecuencia(unidad, cantidad)
  ProfesionalSalud(nombre, matricula, especialidad)

Relationships:
  (Paciente)-[:PRESENTA_SINTOMA]->(Sintoma)
  (Sintoma)-[:TIENE_INDICADOR]->(Indicador)
  (Indicador)-[:ASOCIA_A]->(TipoLesion)
  (Diagnostico)-[:ESPECIFICA_EN]->(TipoLesion)
  (Diagnostico)-[:SUGIERE_TRATAMIENTO]->(Tratamiento)
  (ProfesionalSalud)-[:DERIVA_A]->(ProfesionalSalud)
  (Tratamiento)-[:PROGRAMA]->(Monitoreo)
  (Monitoreo)-[:OCURRE_CADA]->(Frecuencia)
  (ProfesionalSalud)-[:ACTUALIZA]->(Indicador)

Constraints/Índices sugeridos:
  CREATE CONSTRAINT pac_dni IF NOT EXISTS FOR (p:Paciente) REQUIRE p.dni IS UNIQUE;
  CREATE INDEX indic_idx IF NOT EXISTS FOR (i:Indicador) ON (i.nombre);
  CREATE INDEX sint_idx IF NOT EXISTS FOR (s:Sintoma) ON (s.nombre);
  CREATE INDEX tipo_idx IF NOT EXISTS FOR (t:TipoLesion) ON (t.nombre);
"""

# =============================
# Few-shot de Cypher (dominio)
# =============================

FEWSHOT = """
Ejemplos válidos de Cypher en este dominio (NO inventar labels/propiedades):

# 1) Indicadores de un síntoma que presenta un paciente
// Pregunta: "¿Qué indicadores se registraron para el síntoma dolor de tobillo del paciente DNI 30111222?"
MATCH (p:Paciente{dni:30111222})-[:PRESENTA_SINTOMA]->(s:Sintoma{nombre:"dolor de tobillo"})-[:TIENE_INDICADOR]->(i:Indicador)
RETURN i.nombre AS indicador, i.valor AS valor, i.unidad AS unidad;

# 2) Tipo de lesión asociado por indicadores
// Pregunta: "¿Qué tipo de lesión se asocia a edema grado 2?"
MATCH (:Indicador{nombre:"edema", valor:2})-[:ASOCIA_A]->(t:TipoLesion)
RETURN t.nombre AS tipo, t.grado AS grado;

# 3) Tratamientos sugeridos por diagnóstico
// Pregunta: "Tratamiento sugerido para esguince de tobillo grado I"
MATCH (d:Diagnostico)-[:ESPECIFICA_EN]->(tles:TipoLesion{nombre:"esguince de tobillo", grado:"I"})
MATCH (d)-[:SUGIERE_TRATAMIENTO]->(t:Tratamiento)
RETURN t.nombre AS tratamiento, t.descripcion AS descripcion;

# 4) Rutina de monitoreo y su frecuencia
// Pregunta: "¿Con qué frecuencia se monitorea el plan de hielo y elevación?"
MATCH (tr:Tratamiento{nombre:"hielo y elevación"})-[:PROGRAMA]->(m:Monitoreo)-[:OCURRE_CADA]->(f:Frecuencia)
RETURN f.cantidad AS cada, f.unidad AS unidad;
"""

# =============================
# Prompt para generar Cypher (schema-guided)
# =============================

CYHER_PROMPT_TMPL = (
    "Eres un asistente experto en Cypher para Neo4j.\n"
    "Usa EXCLUSIVAMENTE las Labels y Relationships del esquema.\n"
    "Esquema:\n{schema}\n\n"
    "Ejemplos de consultas correctas (few-shot):\n{fewshot}\n\n"
    "Instrucciones:\n"
    "- No inventes labels o propiedades.\n"
    "- Prefiere MATCH y propiedades exactas cuando se proveen (e.g., dni, nombre).\n"
    "- Devuelve SOLO la consulta Cypher, sin explicación adicional.\n\n"
    "Pregunta: {question}\n"
)

cypher_prompt = PromptTemplate(
    input_variables=["schema", "fewshot", "question"],
    template=CYHER_PROMPT_TMPL,
)

# =============================
# LLM (Ollama)
# =============================

llm = ChatOllama(
    model=os.getenv("OLLAMA_MODEL", "llama3.1"),
    temperature=0.0,
)

# =============================
# Cadena NL -> Cypher -> Resultado
# =============================

chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    cypher_prompt=cypher_prompt,
    verbose=True,
    return_intermediate_steps=True,
    allow_dangerous_requests=True,
)

# =============================
# Lógica difusa (gaussiana) para Indicadores
# =============================

class FuzzyQuery(BaseModel):
    indicador_nombre: str
    # Si no se provee valor, se intenta resolver desde el grafo (por dni + síntoma indicados)
    valor: Optional[float] = None
    media: float
    sigma: float
    # Opcional para lookup del valor desde grafo:
    dni: Optional[int] = None
    sintoma: Optional[str] = None


def gaussian_membership(x: float, mu: float, sigma: float) -> float:
    if sigma <= 0:
        return 0.0
    return math.exp(-((x - mu) ** 2) / (2 * (sigma ** 2)))


def fetch_indicator_value(dni: int, sintoma: str, indicador_nombre: str) -> Optional[float]:
    query = (
        "MATCH (p:Paciente{dni:$dni})-[:PRESENTA_SINTOMA]->(s:Sintoma{nombre:$sint})"
        "-[:TIENE_INDICADOR]->(i:Indicador{nombre:$iname}) RETURN i.valor AS valor LIMIT 1"
    )
    rows = graph.query(query, params={"dni": dni, "sint": sintoma, "iname": indicador_nombre})
    if rows:
        return rows[0].get("valor")
    return None

# =============================
# FastAPI
# =============================

app = FastAPI(title="Lesiones: Neo4j + LangChain + Ollama")


@app.get("/query")
def query(question: str, explain: Optional[bool] = False):
    result = chain.invoke({
        "query": question,
        "schema": DOMAIN_SCHEMA,
        "fewshot": FEWSHOT,
        "question": question,
    })
    return result if explain else {"answer": result.get("result")}


@app.post("/fuzzy")
def fuzzy(q: FuzzyQuery):
    x = q.valor
    if x is None and q.dni and q.sintoma:
        x = fetch_indicator_value(dni=q.dni, sintoma=q.sintoma, indicador_nombre=q.indicador_nombre)
    if x is None:
        return {"error": "No hay valor del indicador. Proporcione 'valor' o (paciente+sintoma)"}
    mu = gaussian_membership(x, q.media, q.sigma)
    return {
        "indicador": q.indicador_nombre,
        "valor": x,
        "gauss_media": q.media,
        "gauss_sigma": q.sigma,
        "pertenencia": round(mu, 4)
    }


# =============================
# Launcher
# =============================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
