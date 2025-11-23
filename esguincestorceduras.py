"""
App de ejemplo: Neo4j + LangChain + Ollama
Dominio: Lesiones musculoesqueléticas (esguinces y torceduras)

Incluye dos submódulos:

1) Subdominio clínico general (Paciente, Síntoma, Indicador, TipoLesion, Diagnostico, Tratamiento, ProfesionalSalud)
   - Flujo: NL -> Cypher -> Respuesta usando GraphCypherQAChain
   - Endpoint: /query_clinico

2) Subdominio Monitoreo y Frecuencia
   - Flujo: NL -> Cypher -> Respuesta usando GraphCypherQAChain
   - Endpoint: /query_monitoreo

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
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

from fastapi import FastAPI
from langchain_community.graphs import Neo4jGraph
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate


# ============================================================
# Conexión Neo4j (compartida por ambos submódulos)
# ============================================================

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    username=os.getenv("NEO4J_USER", "neo4j"),
    password=os.getenv("NEO4J_PASSWORD", "password"),
)

# LLM (compartido)
llm = ChatOllama(
    model=os.getenv("OLLAMA_MODEL", "llama3.1"),
    temperature=0.0,
)

# ============================================================
# SUBMÓDULO 1: Dominio clínico general (GraphCypherQAChain)
# ============================================================

DOMAIN_SCHEMA_CLINICO = """
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

FEWSHOT_CLINICO = """
Ejemplos válidos de Cypher en este dominio (NO inventar labels/propiedades):

# Pregunta: “Estoy ayudando a una persona que se torció el tobillo y desde entonces tiene dolor, algo de inflamación y siente cierta inestabilidad al caminar. ¿Qué podría tener y qué tipo de tratamiento inicial se suele recomendar en estos casos?”

MATCH (s1:Sintoma {nombre:"Dolor"})
MATCH (s2:Sintoma {nombre:"Inflamación"})
MATCH (s3:Sintoma {nombre:"Inestabilidad"})
MATCH (s1)-[:ASOCIA_A]->(t:TipoLesion)
MATCH (t)-[:ESPECIFICA_EN]->(d:Diagnostico)
MATCH (d)-[:SUGIERE_TRATAMIENTO]->(tr:Tratamiento)
RETURN
  t.nombre        AS posible_lesion,
  d.descripcion   AS diagnostico,
  tr.nombre       AS tratamiento_inicial,
  tr.objetivo     AS objetivo;

# 1) Síntomas de un paciente
// Pregunta: "¿Qué síntomas presenta el paciente con el nombre Carlos López?"
MATCH (p:Paciente {nombre:"Carlos López"})-[:PRESENTA_SINTOMA]->(s:Sintoma)
RETURN s.nombre AS sintoma;

# 2) Indicadores de un síntoma que presenta un paciente
// Pregunta: "¿Qué indicadores se registraron para el síntoma Dolor del paciente con nombre Carlos López?"
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
  t.descripcion       AS descripcion;

# 4) Tratamiento sugerido por un diagnóstico
// Pregunta: "¿Qué tratamiento se sugiere para el diagnóstico 'Esguince de tobillo derecho con inflamacion grado II'?"
MATCH (d:Diagnostico {descripcion:"Esguince de tobillo derecho con inflamación"})
      -[:SUGIERE_TRATAMIENTO]->(t:Tratamiento)
RETURN
  t.nombre    AS tratamiento,
  t.objetivo  AS objetivo,
  t.lugar     AS lugar;

"""

print(">>> FEWSHOT CLÍNICO EN EJECUCIÓN:\n")
print(FEWSHOT_CLINICO)
print(">>> FIN FEWSHOT CLÍNICO\n")

CYHER_PROMPT_TMPL_CLINICO = (
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
    "- La relación ASOCIA_A SIEMPRE va de Sintoma a TipoLesion, nunca desde Indicador.\n"
    "- Para el caso de inflamación moderada, debes usar Sintoma(nombre) + Indicador(valor).\n"
    "- En TipoLesion NUNCA uses la propiedad 'descripcion' para identificar el tipo. "
    "Para identificar una lesión usa SIEMPRE la propiedad 'nombre'.\n"
    "-Para obtener tratamientos sugeridos, usa la relación (Diagnostico)-[:SUGIERE_TRATAMIENTO]->(Tratamiento).\n"
    "- Devuelve SOLO la consulta Cypher, sin explicación adicional.\n\n"
    "Pregunta: {question}\n"
)

cypher_prompt_clinico = PromptTemplate(
    input_variables=["schema", "fewshot", "question"],
    template=CYHER_PROMPT_TMPL_CLINICO,
)

# Cadena NL -> Cypher -> Resultado (clínico)
chain_clinico = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    cypher_prompt=cypher_prompt_clinico,
    verbose=True,
    return_intermediate_steps=True,
    allow_dangerous_requests=True,
)

# ============================================================
# SUBMÓDULO 2: Monitoreo y Frecuencia (GraphCypherQAChain)
# ============================================================

# Regla lógica simple (tipo demonio):
# IF intensidad == 'severa' AND síntoma == 'Hematoma'
# THEN crear alerta de derivación médica inmediata

CYpher_REGLA_HEMATOMA_SEVERO = """
MATCH (p:Paciente)-[:PRESENTA_SINTOMA]->(s:Sintoma)
WHERE s.intensidad = 'Intenso'
  AND toLower(s.nombre) = 'dolor'
MERGE (a:Alerta {
    codigo: 'DERIVACION_INMEDIATA_DOLOR'
})
ON CREATE SET
    a.descripcion = 'Dolor intenso: se recomienda derivación médica inmediata',
    a.created_at  = datetime(),
    a.resuelta    = false
MERGE (p)-[:REQUIERE_ATENCION]->(a)
RETURN
  p.id        AS paciente_id,
  p.nombre    AS nombre_paciente,
  a.codigo    AS codigo_alerta,
  a.descripcion AS motivo;
"""

DOMAIN_SCHEMA_MONITOREO = """
Labels:
  Paciente(obra_social, contacto, domicilio, observaciones, id, sexo, fecha_ingreso, nombre, edad)
  Sintoma(descripcion, lado, lugar_afectado, fecha_registro, intensidad, notas, id, nombre)
  Indicador(valor_num, valor, fecha_observacion, id, nombre)
  TipoLesion(descripcion, zona, gravedad_aparente, lado, causa, fecha_evento, id, nombre)
  Diagnostico(descripcion, grado, estado, fecha_diagnostico, id, comentarios)
  Monitoreo(fecha, detalle, estado, observaciones, resultado_general, id)
  Frecuencia(id, cada, unidad, cantidad, observaciones)

Relationships:
  (Paciente)-[:PRESENTA_SINTOMA]->(Sintoma)
  (Sintoma)-[:TIENE_INDICADOR]->(Indicador)
  (Sintoma)-[:ASOCIA_A]->(TipoLesion)
  (TipoLesion)-[:ESPECIFICA_EN]->(Diagnostico)
  (TipoLesion)-[:REQUIERE_MONITOREO]->(Monitoreo)
  (Monitoreo)-[:OCURRE_CADA]->(Frecuencia)
  (Monitoreo)-[:ACTUALIZA]->(Diagnostico)
"""

print(">>> ESQUEMA MONITOREO/FRECUENCIA (referencial):")
print(DOMAIN_SCHEMA_MONITOREO)
print(">>> FIN ESQUEMA MONITOREO/FRECUENCIA\n")

FEWSHOT_MONITOREO = """
Ejemplos válidos de Cypher en este dominio (NO inventar labels/propiedades):

# 1) Monitoreos recomendados para una torcedura
// Pregunta: "¿Qué monitoreos se recomiendan para una torcedura y cuál es el detalle de cada uno?"
MATCH (t:TipoLesion {nombre:"Torcedura"})-[:REQUIERE_MONITOREO]->(m:Monitoreo)
OPTIONAL MATCH (m)-[:OCURRE_CADA]->(f:Frecuencia)
RETURN
  t.nombre  AS tipo_lesion,
  m.detalle AS detalle_monitoreo;

# 2) Frecuencia de un monitoreo específico
// Pregunta: "¿Cada cuánto se realiza el monitoreo de control funcional?"
MATCH (m:Monitoreo {detalle:"Control funcional"})-[:OCURRE_CADA]->(f:Frecuencia)
RETURN
  m.detalle       AS detalle_monitoreo,
  f.cada          AS cada,
  f.unidad        AS unidad,
  f.cantidad      AS cantidad,
  f.observaciones AS observaciones;

# 3) Tipos de monitoreo existentes
// Pregunta: "¿Qué tipos de monitoreos existen?"
MATCH (m:Monitoreo)
RETURN DISTINCT m.detalle AS tipo_monitoreo;

# 4) Frecuencias existentes
// Pregunta: "¿Qué frecuencias existen?"
MATCH (f:Frecuencia)
RETURN DISTINCT
  f.cantidad AS cantidad,
  f.unidad   AS unidad;

# 5) Resultado general de un monitoreo
// Pregunta: "¿Qué resultado general se registró para el monitoreo 'Control inicial'?"
MATCH (m:Monitoreo {detalle:"Control inicial"})-[:ACTUALIZA]->(d:Diagnostico)
RETURN
  m.detalle           AS detalle_monitoreo,
  m.resultado_general AS resultado_general;

# 6) Controles y frecuencia para un esguince de tobillo
// Pregunta: "Pasaron unos días y esta persona sigue con dolor, el tobillo se siente rígido y todavía tiene algo de limitación para moverlo. 
// Con la información que manejás, ¿qué tipo de controles o seguimientos suelen hacerse en estos casos y cada cuánto conviene realizarlos
// para asegurarse de que la recuperación va bien?"
MATCH (t:TipoLesion {nombre:"Esguince de tobillo"})
MATCH (t)-[:REQUIERE_MONITOREO]->(m:Monitoreo)
OPTIONAL MATCH (m)-[:OCURRE_CADA]->(f:Frecuencia)
RETURN
  t.nombre        AS tipo_lesion,
  m.detalle       AS tipo_monitoreo,
  f.cantidad      AS cantidad,
  f.unidad        AS unidad,
  f.cada          AS cada,
  f.observaciones AS observaciones;

# 7) Inferencia desde síntomas para obtener monitoreos y frecuencia
// Pregunta: "Pasaron unos días y la persona sigue con dolor y rigidez en el tobillo.
// Todavía le cuesta moverlo. ¿Cada cuánto habría que controlarlo y qué seguimiento se recomienda?"
MATCH (s1:Sintoma {nombre:"Dolor"})-[:ASOCIA_A]->(t:TipoLesion)
MATCH (s2:Sintoma {nombre:"Rigidez"})-[:ASOCIA_A]->(t)
MATCH (t)-[:REQUIERE_MONITOREO]->(m:Monitoreo)
OPTIONAL MATCH (m)-[:OCURRE_CADA]->(f:Frecuencia)
RETURN
  t.nombre        AS tipo_lesion,
  m.detalle       AS tipo_monitoreo,
  f.cantidad      AS cantidad,
  f.unidad        AS unidad,
  f.cada          AS cada,
  f.observaciones AS observaciones;
"""


CYHER_PROMPT_TMPL_MONITOREO = (
    "Eres un asistente experto en Cypher para Neo4j y debes responder SIEMPRE en español.\n"
    "Tu tarea es, dado el esquema y una pregunta en lenguaje natural,\n"
    "generar UNA sola consulta Cypher correcta.\n\n"
    "Usa EXCLUSIVAMENTE las Labels, propiedades y Relationships del esquema.\n\n"
    "Esquema detectado (graph.schema):\n{schema}\n\n"
    "Ejemplos de consultas correctas (few-shot, NO inventar labels/propiedades):\n"
    "{fewshot}\n\n"
    "ACLARACIONES CRÍTICAS SOBRE LAS PROPIEDADES:\n"
    "- La etiqueta Monitoreo solo tiene las propiedades:\n"
    "  - fecha\n"
    "  - detalle\n"
    "  - estado\n"
    "  - observaciones\n"
    "  - resultado_general\n"
    "  - id\n\n"
    "- La etiqueta Frecuencia tiene las propiedades:\n"
    "  - id\n"
    "  - cada\n"
    "  - unidad\n"
    "  - cantidad\n"
    "  - observaciones\n\n"
    "- La propiedad 'tipo' NO EXISTE en la etiqueta Monitoreo.\n"
    "- Debes usar ÚNICAMENTE la propiedad 'detalle' para identificar un monitoreo.\n\n"
    "Instrucciones IMPORTANTES:\n"
    "- No inventes labels ni propiedades que no aparezcan en el esquema.\n"
    "- Para Monitoreo usa solamente: fecha, detalle, estado, observaciones, resultado_general, id.\n"
    "- Para Frecuencia usa solamente: cada, unidad, cantidad, observaciones, id.\n"
    "- No uses nunca la propiedad 'tipo'.\n"
    "- Devuelve SOLO la consulta Cypher, sin explicación adicional.\n\n"
    "Pregunta: {question}\n"
)

cypher_prompt_monitoreo = PromptTemplate(
    input_variables=["schema", "fewshot", "question"],
    template=CYHER_PROMPT_TMPL_MONITOREO,
)

# Cadena NL -> Cypher -> Resultado (Monitoreo)
chain_monitoreo = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    cypher_prompt=cypher_prompt_monitoreo,
    verbose=True,
    return_intermediate_steps=True,
    allow_dangerous_requests=True,
)

# ============================================================
# FastAPI – App unificada
# ============================================================

app = FastAPI(
    title="Lesiones musculoesqueléticas"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # para pruebas locales; luego podés restringir
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/query_clinico")
def query_clinico(question: str, explain: Optional[bool] = False):
    """
    Endpoint del subdominio clínico general:
      - question: pregunta en lenguaje natural
      - explain=True para ver también pasos intermedios (Cypher, resultados crudos)
    """
    result = chain_clinico.invoke(
        {
            "query": question,
            "schema": DOMAIN_SCHEMA_CLINICO,
            "fewshot": FEWSHOT_CLINICO,
            "question": question,
        }
    )
    return result if explain else {"answer": result.get("result")}



@app.get("/regla_hematoma")

def ejecutar_regla_hematoma_severo():
  """ Ejecuta la regla: IF intensidad == 'severa' AND síntoma == 'Dolor' THEN crear/actualizar una alerta de derivación inmediata. """ 
  print("[REGLA] Ejecutando regla de dolor severo...") 
  resultado = graph.query(CYpher_REGLA_HEMATOMA_SEVERO) 
  print("[REGLA] Resultado regla dolor severo:", resultado) 
  return resultado

def regla_hematoma():
    """
    Ejecuta la regla lógica para detectar hematomas severos
    y devolver las alertas generadas.
    """
    try:
        resultado = ejecutar_regla_hematoma_severo()
        return {"alerts": resultado}
    except Exception as e:
        return {"error": str(e)}


@app.get("/query_monitoreo")
def query_monitoreo(question: str, explain: Optional[bool] = False):
    """
    Endpoint del subdominio Monitoreo/Frecuencia:
      - question: pregunta en lenguaje natural
      - explain=True para ver también pasos intermedios (Cypher, resultados crudos)
    """
    if not question.strip():
        return {
            "error": "La pregunta 'question' no puede estar vacía. "
                     "Por favor, ingresá una consulta en lenguaje natural."
        }

    result = chain_monitoreo.invoke(
        {
            "query": question,
            "schema": DOMAIN_SCHEMA_MONITOREO,
            "fewshot": FEWSHOT_MONITOREO,
            "question": question,
        }
    )
    return result if explain else {"answer": result.get("result")}


# ============================================================
# Launcher
# ============================================================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("esguincestorceduras:app", host="0.0.0.0", port=8000, reload=True)
