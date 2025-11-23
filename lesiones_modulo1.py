"""
App de ejemplo: Neo4j + LangChain + Ollama
Dominio: Lesiones musculoesqueléticas (esguinces y torceduras)

- NL -> Cypher -> Respuesta, usando GraphCypherQAChain
- Prompt guiado por esquema (schema-guided)
- Endpoint /query para pruebas

Requisitos:
  pip install "langchain>=0.3" langchain-community neo4j fastapi uvicorn "langchain-ollama>=0.1.0"
  ollama pull llama3.1   # o el modelo que prefieras

Variables de entorno:
  export NEO4J_URI="bolt://localhost:7687"
  export NEO4J_USER="neo4j"
  export NEO4J_PASSWORD="tu_clave"
  export OLLAMA_MODEL="llama3.1"   # opcional
"""

import os
from typing import Optional

from fastapi import FastAPI
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
# Esquema de dominio (subdominio clínico)
# =============================
DOMAIN_SCHEMA = """
Labels:
  Paciente(obra_social, contacto, domicilio, observaciones, id, sexo, fecha_ingreso, nombre, edad)
  Sintoma(descripcion, lado, lugar_afectado, fecha_registro, intensidad, notas, id, nombre)
  Indicador(valor_num, valor, fecha_observacion, id, nombre)
  TipoLesion(descripcion, zona, gravedad_aparente, lado, causa, fecha_evento, id, nombre)
  Diagnostico(descripcion, grado, estado, fecha_diagnostico, id, comentarios)
  Tratamiento(objetivo, estado, lugar, observaciones, inicio, id, nombre)
  ProfesionalSalud(tipo, fecha_alta, institucion, id, nombre, especialidad)

Relationships:
  (Paciente)-[:PRESENTA_SINTOMA]->(Sintoma)
  (Sintoma)-[:TIENE_INDICADOR]->(Indicador)
  (Sintoma)-[:ASOCIA_A]->(TipoLesion)
  (TipoLesion)-[:ESPECIFICA_EN]->(Diagnostico)
  (Diagnostico)-[:SUGIERE_TRATAMIENTO]->(Tratamiento)
  (Tratamiento)-[:DERIVA_A]->(ProfesionalSalud)

Constraints/Índices sugeridos:
  // Unicidad de id por entidad
  CREATE CONSTRAINT paciente_id IF NOT EXISTS
  FOR (p:Paciente) REQUIRE p.id IS UNIQUE;

  CREATE CONSTRAINT sintoma_id IF NOT EXISTS
  FOR (s:Sintoma) REQUIRE s.id IS UNIQUE;

  CREATE CONSTRAINT indicador_id IF NOT EXISTS
  FOR (i:Indicador) REQUIRE i.id IS UNIQUE;

  CREATE CONSTRAINT tipolesion_id IF NOT EXISTS
  FOR (t:TipoLesion) REQUIRE t.id IS UNIQUE;

  CREATE CONSTRAINT diagnostico_id IF NOT EXISTS
  FOR (d:Diagnostico) REQUIRE d.id IS UNIQUE;

  CREATE CONSTRAINT tratamiento_id IF NOT EXISTS
  FOR (tr:Tratamiento) REQUIRE tr.id IS UNIQUE;

  CREATE CONSTRAINT profesional_id IF NOT EXISTS
  FOR (pr:ProfesionalSalud) REQUIRE pr.id IS UNIQUE;
"""

# =============================
# Few-shot de Cypher (subdominio clínico)
# =============================

FEWSHOT = """
Ejemplos válidos de Cypher en este dominio (NO inventar labels/propiedades):

# 1) Síntomas de un paciente
// Pregunta: "¿Qué síntomas presenta el paciente con el nombre Carlos López?"
MATCH (p:Paciente {nombre:"Carlos López"})-[:PRESENTA_SINTOMA]->(s:Sintoma)
RETURN s.nombre AS sintoma;


# 2) Indicadores de un síntoma que presenta un paciente
//Pregunta: "¿Qué indicadores se registraron para el síntoma Dolor del paciente con nombre Carlos López?"
MATCH (p:Paciente {nombre:"Carlos López"})-[:PRESENTA_SINTOMA]->(s:Sintoma {nombre:"Dolor"})
MATCH (s)-[:TIENE_INDICADOR]->(i:Indicador)
RETURN
  i.nombre            AS indicador,
  i.valor_num         AS valor_numerico,
  i.valor             AS valor_descriptivo,
  i.fecha_observacion AS fecha_observacion;

# 3) Tipo de lesión asociado por indicadores
// Pregunta: "¿Qué tipo de lesión se asocia a una Inflamación moderada?"
MATCH (s:Sintoma {nombre:"Inflamación"})-[:TIENE_INDICADOR]->(i:Indicador {valor:"moderada"})
MATCH (s)-[:ASOCIA_A]->(t:TipoLesion)
RETURN
  t.nombre            AS tipo_lesion,
  t.gravedad_aparente AS gravedad_aparente,
  t.zona              AS zona_afectada,
  t.lado              AS lado_comprometido,
  t.descripcion        AS descripcion;

"""

print(">>> FEWSHOT ACTUAL EN EJECUCIÓN:\n")
print(FEWSHOT)
print(">>> FIN FEWSHOT\n")

# =============================
# Prompt para generar Cypher (schema-guided)
# =============================

CYHER_PROMPT_TMPL = (
    "Eres un asistente experto en Cypher para Neo4j.\n"
    "Usa EXCLUSIVAMENTE las Labels, propiedades y Relationships del esquema.\n"
    "Esquema:\n{schema}\n\n"
    "Ejemplos de consultas correctas (few-shot):\n{fewshot}\n\n"
    "Instrucciones IMPORTANTES:\n"
    "- No inventes labels o propiedades.\n"
    "- En este dominio, la entidad Paciente se identifica SIEMPRE con la propiedad `id`.\n"
    "- La propiedad `dni` NO EXISTE en el esquema y NO DEBE usarse en ningún MATCH.\n"
    "- Cuando en la pregunta se mencione 'id', debes usar la propiedad `id` en el MATCH.\n"
    "- Prefiere MATCH y propiedades exactas cuando se proveen (por ejemplo: id, nombre).\n"
    "- Instrucciones IMPORTANTES:\n"
    "- La relación ASOCIA_A SIEMPRE va de Sintoma a TipoLesion, nunca desde Indicador.\n"
    "- Para el caso de inflamación moderada, debes usar Sintoma(nombre) + Indicador(valor).\n"
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
# FastAPI
# =============================

app = FastAPI(title="Lesiones (Subdominio clínico): Neo4j + LangChain + Ollama")


@app.get("/query")
def query(question: str, explain: Optional[bool] = False):
    """
    Endpoint principal:
      - question: pregunta en lenguaje natural
      - explain=True para ver también pasos intermedios (Cypher, resultados crudos)
    """
    result = chain.invoke({
        "query": question,
        "schema": DOMAIN_SCHEMA,
        "fewshot": FEWSHOT,
        "question": question,
    })
    return result if explain else {"answer": result.get("result")}


# =============================
# Launcher
# =============================
if __name__ == "__main__":

    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
   

