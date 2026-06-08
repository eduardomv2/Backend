"""
BrújulaTec ITSM — Backend FastAPI v2
======================================
Endpoints:
  POST /predict        ← test vocacional → top 3 + guarda resultado
  GET  /health         ← status del servidor
  GET  /dashboard      ← KPIs para el dashboard (requiere API key)
  GET  /dashboard/raw  ← todos los registros (requiere API key)

Archivos necesarios en la misma carpeta:
  model_bayesnet_swipeonly.pkl
  feature_cols_swipeonly.json

Los resultados se guardan en: resultados.json
"""

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ─────────────────────────────────────────
# INICIALIZACIÓN
# ─────────────────────────────────────────
app = FastAPI(title="BrújulaTec API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = joblib.load("model_bayesnet_swipeonly.pkl")
with open("feature_cols_swipeonly.json") as f:
    FEATURE_COLS = json.load(f)

# Archivo donde se guardan los resultados
RESULTADOS_FILE = Path("resultados.json")
# Contraseña para el dashboard (cámbiala en producción)
ADMIN_KEY = os.getenv("ADMIN_KEY", "admin2026")

CARRERAS_LABEL = {
    "INF":  "Ingeniería en Informática",
    "IND":  "Ingeniería Industrial",
    "ELEC": "Ingeniería en Electrónica",
    "MEC":  "Ingeniería en Mecánica",
    "ENRV": "Ingeniería en Energías Renovables",
    "GE":   "Ingeniería en Gestión Empresarial",
}

BACHILLERATOS = [
    "CBTis 36", "CETis 46", "COBAC 24", "FIME UADEC",
    "COBAC Castaños", "Telebachillerato", "ICC Monclova",
    "La Salle", "Bachillerato Incorporado", "Otro",
]

RAZONAMIENTOS = {
    "INF":  "Tu perfil muestra alta afinidad con lógica computacional y desarrollo de software.",
    "IND":  "Tu interés en procesos, calidad y eficiencia se alinea con Ingeniería Industrial.",
    "ELEC": "Tu afinidad con circuitos, automatización y sistemas de control encaja con Electrónica.",
    "MEC":  "Tu perfil técnico orientado a maquinaria y diseño se alinea con Mecánica.",
    "ENRV": "Tu interés en sostenibilidad y sistemas energéticos apunta a Energías Renovables.",
    "GE":   "Tu perfil de liderazgo y visión de negocios se alinea con Gestión Empresarial.",
}

KEYWORDS = {
    "INF":  ["programar","codigo","algoritmo","app","software","sistema","base de datos",
             "red","web","movil","backend","frontend","python","javascript","nube","api"],
    "IND":  ["produccion","planta","proceso","calidad","lean","six sigma","iso","logistica",
             "manufactura","taller","supervisar","kanban","inventario","cadena"],
    "ELEC": ["circuito","sensor","arduino","plc","automatizacion","voltaje","electronico",
             "actuador","tablero","modbus","raspberry","firmware","cableado"],
    "MEC":  ["cad","solidworks","autocad","pieza","maquina","cnc","torno","soldadura",
             "resistencia","materiales","fresadora","maquinado","prototipo"],
    "ENRV": ["solar","panel","fotovoltaico","eolico","energia limpia","renovable",
             "auditoria energetica","cfe","emisiones","co2","semarnat","kwh"],
    "GE":   ["empresa","negocio","ventas","marketing","finanzas","presupuesto","estrategia",
             "recursos humanos","kpi","rentabilidad","canvas","gerente","administrar"],
}


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def cargar_resultados() -> list:
    if not RESULTADOS_FILE.exists():
        return []
    try:
        with open(RESULTADOS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def guardar_resultado(registro: dict):
    datos = cargar_resultados()
    datos.append(registro)
    with open(RESULTADOS_FILE, "w", encoding="utf-8") as f:
        json.dump(datos, f, ensure_ascii=False, indent=2)


def analizar_texto(texto: str) -> dict:
    if not texto or len(texto.strip()) < 20:
        return {"carrera_texto": None, "keywords_detectadas": []}
    t = texto.lower()
    for a, b in {"á":"a","é":"e","í":"i","ó":"o","ú":"u","ñ":"n"}.items():
        t = t.replace(a, b)
    scores = {c: sum(1 for kw in kws if kw in t) / len(kws) for c, kws in KEYWORDS.items()}
    mejor = max(scores, key=scores.get)
    if scores[mejor] < 0.04:
        return {"carrera_texto": None, "keywords_detectadas": []}
    hits = [kw for kw in KEYWORDS[mejor] if kw in t][:5]
    return {"carrera_texto": mejor, "keywords_detectadas": hits}


def suavizar(probas: np.ndarray, T: float = 3.0) -> np.ndarray:
    log_p = np.log(np.clip(probas, 1e-10, 1.0))
    s = log_p / T
    e = np.exp(s - s.max())
    return e / e.sum()


def verificar_admin(x_admin_key: Optional[str]):
    if x_admin_key != ADMIN_KEY:
        raise HTTPException(status_code=401, detail="No autorizado")


# ─────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────
class DatosDemo(BaseModel):
    bachillerato: str
    municipio: str
    edad: int

class TestInput(BaseModel):
    swipe_responses: dict
    texto_libre: Optional[str] = ""
    datos_demo: Optional[DatosDemo] = None   # bachillerato, municipio, edad


# ─────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────
@app.get("/health")
def health():
    total = len(cargar_resultados())
    return {"status": "ok", "modelo": "BayesNet_swipeonly", "tests_guardados": total}


@app.get("/bachilleratos")
def get_bachilleratos():
    """Lista de bachilleratos para el selector del frontend."""
    return {"bachilleratos": BACHILLERATOS}


@app.post("/predict")
def predict(body: TestInput):
    # 1. Construir vector
    vector = {col: body.swipe_responses.get(col, 0) for col in FEATURE_COLS}
    X = pd.DataFrame([vector])[FEATURE_COLS].values.astype(float)

    # 2. Predicción
    probas_raw = MODEL.predict_proba(X)[0]
    clases = MODEL.classes_
    probas_display = suavizar(probas_raw)
    ranking = sorted(zip(clases, probas_raw, probas_display), key=lambda x: x[1], reverse=True)

    # 3. Texto libre
    texto_info = analizar_texto(body.texto_libre or "")
    texto_extra = None
    if texto_info["carrera_texto"]:
        kws = ", ".join(texto_info["keywords_detectadas"])
        texto_extra = (
            f"Tu descripción mencionó términos de "
            f"{CARRERAS_LABEL.get(texto_info['carrera_texto'], '')} ({kws}), "
            f"lo que refuerza tu perfil."
        )

    # 4. Top 3
    top3 = []
    for carrera_id, prob_real, prob_display in ranking[:3]:
        razon = RAZONAMIENTOS.get(carrera_id, "")
        if texto_info["carrera_texto"] == carrera_id and texto_extra:
            razon += " " + texto_extra
            texto_extra = None
        top3.append({
            "carrera_id": carrera_id,
            "nombre_carrera": CARRERAS_LABEL.get(carrera_id, carrera_id),
            "porcentaje_match": round(float(prob_display) * 100, 1),
            "razonamiento": razon,
        })

    # 5. Guardar en JSON
    registro = {
        "id": str(uuid.uuid4())[:8],
        "fecha": datetime.now().isoformat(),
        "bachillerato": body.datos_demo.bachillerato if body.datos_demo else "No especificado",
        "municipio": body.datos_demo.municipio if body.datos_demo else "No especificado",
        "edad": body.datos_demo.edad if body.datos_demo else 0,
        "carrera_1": top3[0]["carrera_id"],
        "carrera_2": top3[1]["carrera_id"],
        "carrera_3": top3[2]["carrera_id"],
        "pct_1": top3[0]["porcentaje_match"],
        "texto_libre": body.texto_libre or "",
        "confianza": round(float(max(probas_raw)), 3),
    }
    guardar_resultado(registro)

    return {
        "top3": top3,
        "texto_analisis": texto_extra,
        "confianza": registro["confianza"],
    }


@app.get("/stats/bachilleratos")
def stats_bachilleratos_publicos():
    """
    Top 3 bachilleratos por volumen de tests — público, sin auth.
    Solo devuelve: nombre, tests, carrera más común y porcentaje.
    """
    datos = cargar_resultados()
    if not datos:
        return {"bachilleratos": [], "total": 0}

    por_bach: dict = {}
    for d in datos:
        b = d.get("bachillerato", "Otro")
        c = d.get("carrera_1", "?")
        if b not in por_bach:
            por_bach[b] = {"total": 0, "carreras": {}}
        por_bach[b]["total"] += 1
        por_bach[b]["carreras"][c] = por_bach[b]["carreras"].get(c, 0) + 1

    resumen = []
    for b, info in sorted(por_bach.items(), key=lambda x: -x[1]["total"]):
        top_c = max(info["carreras"], key=info["carreras"].get)
        top_pct = round(info["carreras"][top_c] / info["total"] * 100)
        resumen.append({
            "nombre": b,
            "tests": info["total"],
            "top_carrera": top_c,
            "top_pct": top_pct,
        })

    return {"bachilleratos": resumen[:3], "total": len(datos)}


@app.get("/dashboard")
def dashboard(x_admin_key: Optional[str] = Header(None)):
    """KPIs agregados — requiere header X-Admin-Key."""
    verificar_admin(x_admin_key)
    datos = cargar_resultados()
    if not datos:
        return {"total": 0, "por_carrera": {}, "por_bachillerato": {}, "por_municipio": {}}

    total = len(datos)
    hoy = datetime.now().date().isoformat()
    hoy_count = sum(1 for d in datos if d.get("fecha", "").startswith(hoy))

    # Por carrera (carrera_1 = resultado principal)
    por_carrera: dict = {}
    for d in datos:
        c = d.get("carrera_1", "?")
        por_carrera[c] = por_carrera.get(c, 0) + 1

    # Por bachillerato → top carrera
    por_bach: dict = {}
    for d in datos:
        b = d.get("bachillerato", "Otro")
        c = d.get("carrera_1", "?")
        if b not in por_bach:
            por_bach[b] = {"total": 0, "carreras": {}}
        por_bach[b]["total"] += 1
        por_bach[b]["carreras"][c] = por_bach[b]["carreras"].get(c, 0) + 1

    # Calcular top carrera por bachillerato
    resumen_bach = []
    for b, info in sorted(por_bach.items(), key=lambda x: -x[1]["total"]):
        top_c = max(info["carreras"], key=info["carreras"].get)
        top_pct = round(info["carreras"][top_c] / info["total"] * 100)
        resumen_bach.append({
            "nombre": b,
            "tests": info["total"],
            "top_carrera": top_c,
            "top_pct": top_pct,
        })

    # Por municipio
    por_municipio: dict = {}
    for d in datos:
        m = d.get("municipio", "Otro")
        por_municipio[m] = por_municipio.get(m, 0) + 1

    # Últimos 10
    ultimos = sorted(datos, key=lambda x: x.get("fecha", ""), reverse=True)[:10]

    return {
        "total": total,
        "hoy": hoy_count,
        "por_carrera": por_carrera,
        "por_bachillerato": resumen_bach,
        "por_municipio": por_municipio,
        "ultimos": [
            {
                "id": u["id"],
                "bachillerato": u["bachillerato"],
                "carrera_1": u["carrera_1"],
                "fecha": u["fecha"],
            }
            for u in ultimos
        ],
    }


@app.get("/dashboard/raw")
def dashboard_raw(x_admin_key: Optional[str] = Header(None)):
    """Todos los registros completos — solo admin."""
    verificar_admin(x_admin_key)
    return {"registros": cargar_resultados()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)