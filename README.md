# esguincestorceduras-api

> **Prototipo de integración** entre Neo4j + Ollama + LangChain para la materia “Inteligencia Artificial” de la Universidad Tecnológica Nacional – Facultad Regional Mendoza (UTN FRM).

## Descripción

Este proyecto es un prototipo API/web para el análisis de **esguinces y torceduras** (u otras lesiones musculoesqueléticas) mediante una combinación de:

- Base de datos grafos con Neo4j  
- Lógica de lenguaje con LangChain  
- Posible integración con Ollama (modelo local)  
- Frontend o endpoint para exponer resultados via `esguincestorceduras.py`

## Contenido del repositorio

- `esguincestorceduras.py` → archivo principal que ejecuta la aplicación.  
- `app.py`, `app_nl_cypher.py`, `lesiones_modulo1.py`, `logicaDifusa.py`, `monitoreo_modulo2.py` → scripts de módulos, lógica de negocio, monitoreo, etc.  
- Carpeta `neo4j/` → scripts o configuraciones relativas a la base de datos Neo4j.  
- `index.html` → página de interfaz básica.  
- `.gitignore` → exclusiones de archivos que no deben subirse.  
- `enviroment.env` → archivo de variables de entorno **(no debería incluirse en el repo, moverlo a `.gitignore`)**.

## Requisitos

Asegurate de tener instalado en tu entorno:

- Python 3.x  
- Librerías del proyecto (puede que tengas un `requirements.txt`, si no lo creás vos)  
- Neo4j corriendo o acceso remoto a una instancia.  
- (Opcional) Ollama y/o modelo local para procesamiento de lenguaje.

## Instalación y ejecución

1. Cloná el repositorio:  
   ```bash
   git clone https://github.com/frnalonso/esguincestorceduras-api.git
   cd esguincestorceduras-api
