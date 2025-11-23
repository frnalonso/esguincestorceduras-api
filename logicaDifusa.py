# -*- coding: utf-8 -*-
"""
Frames Difusos en Neo4j – Dominio: Esguinces y Torceduras

Este script muestra cómo:
1) Definir una semilla mínima de frames, slots e instancias.
2) Asignar parámetros gaussianos a un slot con imprecisión (dolor_actual).
3) Fuzificar un valor crisp usando una gaussiana.
4) Calcular entropía difusa de ese slot.
5) Definir una regla/demonio operativo.
6) Consultar todas las relaciones con pertenencia > 0.6.

"""
import neo4j
from neo4j import GraphDatabase

# ============================================================
# 0) CONFIGURACIÓN DEL DRIVER
# ============================================================

# COMPLETAR con tus datos de Neo4j Aura/Local:
URI = "neo4j+s://cc11dd0d.databases.neo4j.io"   # ej: "neo4j+s://xxxx.databases.neo4j.io"
USER = "neo4j"
PASSWORD = "CDBQ-VtbJP6ihFHkPOttTfURo7ylsa_Uf2bTojEdWsE"

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))


def close_driver():
    driver.close()


def run(query: str, **params):
    """Ejecuta una consulta Cypher y devuelve una lista de registros."""
    with driver.session() as session:
        result = session.run(query, **params)
        return list(result)


def run_one(query: str, **params):
    """Ejecuta una consulta Cypher y devuelve un único registro (o None)."""
    with driver.session() as session:
        result = session.run(query, **params)
        return result.single()


def run_many(script: str):
    """
    Ejecuta muchas sentencias Cypher separadas por ';'.
    Ignora líneas vacías y comentarios que comienzan con //.
    """
    # separar por ';'
    statements = []
    current = []

    for line in script.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # quitar comentarios de línea
        if stripped.startswith("//"):
            continue
        # conservar el resto de la línea
        current.append(line)
        if ";" in line:
            stmt = "\n".join(current)
            # cortar en ';' para evitar ejecutar cadenas vacías
            parts = stmt.split(";")
            for p in parts[:-1]:
                if p.strip():
                    statements.append(p.strip())
            current = [parts[-1]] if parts[-1].strip() else []

    # por si quedó algo sin ';' al final
    tail = "\n".join(current).strip()
    if tail:
        statements.append(tail)

    results = []
    for stmt in statements:
        with driver.session() as session:
            session.run(stmt)
        results.append(stmt)
    return results


# ============================================================
# 1) SEMILLA MÍNIMA: FRAMES, SLOTS E INSTANCIA
# ============================================================

seed_esguinces = """
// === Clases base del dominio ===
MERGE (:Frame {name:'Paciente'});
MERGE (:Frame {name:'Sintoma'});
MERGE (:Frame {name:'TipoLesion'});
MERGE (:Frame {name:'Diagnostico'});
MERGE (:Frame {name:'Monitoreo'});
MERGE (:Frame {name:'Frecuencia'});

// === Paciente e instancia de caso clínico ===
MERGE (p:Paciente {id:'P-001', nombre:'Paciente Demo'});

MERGE (caso:FrameInstance {id:'Caso-001'})
MERGE (caso)-[:INSTANCE_OF]->(:Frame {name:'Diagnostico'});

// === Slots difusos principales ===
// dolor_actual (0–10)
MERGE (s_dolor:Slot:Dolor {slot_name:'dolor_actual'})
  ON CREATE SET s_dolor.valor_crisp = 7.5;

// gravedad_aparente (0–10)
MERGE (s_grav:Slot:Gravedad {slot_name:'gravedad_aparente'})
  ON CREATE SET s_grav.valor_crisp = 6.0;

// mejora_observada (0–100%)
MERGE (s_mej:Slot:Mejora {slot_name:'mejora_observada'})
  ON CREATE SET
      s_mej.valor_crisp = 40.0,
      // para simplificar, fijamos μ manualmente
      s_mej.mu_baja   = 0.7,
      s_mej.mu_media  = 0.3,
      s_mej.mu_alta   = 0.0;

// intervalo_controles (días)
MERGE (s_freq:Slot:Frecuencia {slot_name:'intervalo_controles'})
  ON CREATE SET s_freq.valor_crisp = 7;

// Vincular slots al caso clínico
MERGE (caso)-[:TIENE_SLOT]->(s_dolor);
MERGE (caso)-[:TIENE_SLOT]->(s_grav);
MERGE (caso)-[:TIENE_SLOT]->(s_mej);
MERGE (caso)-[:TIENE_SLOT]->(s_freq);

// === Algunos nodos de concepto para las relaciones difusas ===
MERGE (si:Sintoma {tipo:'Dolor tobillo'});
MERGE (tl:TipoLesion {nombre:'Esguince tobillo'});
MERGE (dg:Diagnostico {codigo:'D-001'});
MERGE (mon:Monitoreo  {id:'M-001'});
MERGE (frec:Frecuencia {descripcion:'Controles clínicos'});

// Relación Paciente–Síntoma con pertenencia difusa
MERGE (p)-[r1:PRESENTA_SINTOMA]->(si)
ON CREATE SET
  r1.etiqueta = 'Severo',
  r1.mu       = 0.8;

// Relación Síntoma–TipoLesion con pertenencia difusa
MERGE (si)-[r2:ASOCIA_A]->(tl)
ON CREATE SET
  r2.etiqueta = 'Muy probable',
  r2.mu       = 0.7;

// Relación TipoLesion–Diagnostico
MERGE (tl)-[r3:ESPECIFICA_EN]->(dg)
ON CREATE SET
  r3.etiqueta = 'II',
  r3.mu       = 0.6;

// Relación Diagnostico–Monitoreo
MERGE (dg)-[r4:PROGRAMA]->(mon)
ON CREATE SET
  r4.etiqueta = 'Corta',
  r4.mu       = 0.7;

// Relación Monitoreo–Frecuencia
MERGE (mon)-[r5:OCURRE_CADA]->(frec)
ON CREATE SET
  r5.etiqueta = 'Media',
  r5.mu       = 0.6;
"""


# ============================================================
# 2) PARÁMETROS GAUSSIANOS PARA EL SLOT CON IMPRECISIÓN
#    (dolor_actual)
# ============================================================

gaussian_params_dolor = """
MATCH (t:Slot:Dolor {slot_name:'dolor_actual'})
SET t.funcion_pertenencia = 'gaussiana',
    // a = centro, b = desviación estándar
    t.a_leve     = 2.0,  t.b_leve     = 1.5,
    t.a_moderado = 5.0,  t.b_moderado = 1.5,
    t.a_severo   = 8.0,  t.b_severo   = 1.2;
"""


# ============================================================
# 3) FUZIFICACIÓN GAUSSIANA DE UN VALOR INGRESADO
# ============================================================

fuzz_gauss_dolor = """
MATCH (s:Slot:Dolor {slot_name:'dolor_actual'})
WHERE s.valor_crisp IS NOT NULL
  AND s.funcion_pertenencia = 'gaussiana'

WITH s, s.valor_crisp AS x,
     s.a_leve     AS al, coalesce(s.b_leve,1e-6)     AS bl,
     s.a_moderado AS am, coalesce(s.b_moderado,1e-6) AS bm,
     s.a_severo   AS as, coalesce(s.b_severo,1e-6)   AS bs

SET s.mu_leve     = exp( - ((x - al) * (x - al)) / (2.0 * (bl * bl)) ),
    s.mu_moderado = exp( - ((x - am) * (x - am)) / (2.0 * (bm * bm)) ),
    s.mu_severo   = exp( - ((x - as) * (x - as)) / (2.0 * (bs * bs)) ),
    s.fuzzy_last_updated = datetime()

RETURN s.slot_name AS slot, x AS valor,
       round(s.mu_leve,3)     AS mu_leve,
       round(s.mu_moderado,3) AS mu_moderado,
       round(s.mu_severo,3)   AS mu_severo;
"""


# ============================================================
# 4) INCERTIDUMBRE CON ENTROPÍA DIFUSA
# ============================================================

entropy_dolor = """
MATCH (s:Slot:Dolor {slot_name:'dolor_actual'})
WITH s,
     coalesce(s.mu_leve,0.0)     AS u1,
     coalesce(s.mu_moderado,0.0) AS u2,
     coalesce(s.mu_severo,0.0)   AS u3
WITH s, [u1,u2,u3] AS U

WITH s,
     reduce(
        h = 0.0,
        m IN U |
        h + CASE WHEN m > 0 THEN -(m * log(m)) ELSE 0.0 END
     ) / log(3.0) AS Hnorm

SET s.entropia = round(Hnorm,4),
    s.entropia_updated_at = datetime()

RETURN s.slot_name AS slot,
       s.valor_crisp AS dolor_actual,
       s.entropia    AS entropia;
"""


# ============================================================
# 5) REGLAS / DEMONIOS OPERATIVOS
#    (riesgo alto si dolor severo y mejora baja)
# ============================================================

rule_alerta_riesgo_alto = """
WITH 0.7 AS umbral_dolor, 0.6 AS umbral_mejora
MATCH (sd:Slot:Dolor   {slot_name:'dolor_actual'}),
      (sm:Slot:Mejora  {slot_name:'mejora_observada'})
WHERE coalesce(sd.mu_severo,0.0) >= umbral_dolor
  AND coalesce(sm.mu_baja,0.0)   >= umbral_mejora

MERGE (al:Alerta {codigo:'RIESGO_ALTO_ESGUINCE'})
SET al.ts          = datetime(),
    al.severidad   = 'alta',
    al.descripcion = 'Dolor severo y baja mejoría en esguince/torcedura',
    al.resuelta    = false

RETURN al.codigo, al.severidad, al.descripcion, al.ts;
"""


# ============================================================
# 6) CONSULTA DE TODAS LAS RELACIONES CON μ > 0.6
# ============================================================

query_rel_mu_gt_06 = """
MATCH (a)-[r]->(b)
WHERE r.mu IS NOT NULL AND r.mu > 0.6
RETURN a, type(r) AS relacion, r.etiqueta, r.mu, b;
"""



# ============================================================
# EJEMPLO DE USO (opcional)
# ============================================================

if __name__ == "__main__":
    # 1) Semilla mínima
    print("Semilla mínima...")
    run_many(seed_esguinces)

    # 2) Parámetros gaussianos
    print("Parámetros gaussianos dolor_actual...")
    run_many(gaussian_params_dolor)

    # 3) Fuzificación gaussiana
    print("Fuzificación dolor_actual...")
    res_fuzz = run(fuzz_gauss_dolor)
    for r in res_fuzz:
        print(r)

    # 4) Entropía
    print("Entropía difusa de dolor_actual...")
    res_ent = run(entropy_dolor)
    for r in res_ent:
        print(r)

    # 5) Regla / demonio
    print("Ejecutando regla de alerta...")
    res_rule = run(rule_alerta_riesgo_alto)
    for r in res_rule:
        print(r)

    # 6) Relaciones con μ > 0.6
    print("Relaciones con mu > 0.6...")
    res_rel = run(query_rel_mu_gt_06)
    for r in res_rel:
        print(r)

    close_driver()
