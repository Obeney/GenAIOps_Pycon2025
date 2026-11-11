# app/run_eval.py
"""
Evaluación del RAG con mlflow.genai.evaluate y jueces LLM (LLM-as-a-judge).

Cada pregunta del dataset genera una traza; los jueces la califican y dejan su
razonamiento (rationale) adjunto a la traza. El resultado agregado queda en un
run del experimento "rag_evaluation", enlazado a la versión de la app
(proveedor + modelo + prompt + k) y a la versión del prompt.

Uso:
    python app/run_eval.py
    PROMPT_VERSION=prompts:/asistente_coberturas_ideam@v2_resumido_directo python app/run_eval.py
    RETRIEVER_K=6 python app/run_eval.py
    EVAL_SCORERS=correctness,retrieval_groundedness python app/run_eval.py   # solo algunos jueces
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import re

import mlflow
from dotenv import load_dotenv
from mlflow.entities import Feedback, SpanType
from mlflow.genai.judges import make_judge
from mlflow.genai.scorers import (
    Correctness,
    Guidelines,
    RelevanceToQuery,
    RetrievalGroundedness,
    RetrievalRelevance,
    Safety,
    scorer,
)

from app.llm_provider import describe, get_judge_model_uri
from app.prompt_registry import DEFAULT_PROMPT
from app.rag_pipeline import build_chain, load_vectorstore_from_disk
from app.tracking import EVAL_EXPERIMENT, app_config, init_tracking

load_dotenv()

# Configuración
PROMPT_VERSION = os.getenv("PROMPT_VERSION") or DEFAULT_PROMPT
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE") or 512)        # deben coincidir con el índice construido
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP") or 50)
K = int(os.getenv("RETRIEVER_K") or 4)
DATASET_PATH = "tests/eval_dataset.json"

# Paralelismo de MLflow. Por defecto evalúa ~10 filas x 10 jueces a la vez, lo que
# supera los límites gratuitos de Groq (30 solicitudes y 8K tokens por minuto).
# Con Groq usamos un modo pausado; cualquier valor puesto en .env tiene prioridad.
if os.getenv("LLM_PROVIDER", "openai").strip().lower() == "groq":
    os.environ.setdefault("MLFLOW_GENAI_EVAL_MAX_WORKERS", "1")         # una pregunta a la vez
    os.environ.setdefault("MLFLOW_GENAI_EVAL_MAX_SCORER_WORKERS", "2")  # 2 jueces en paralelo
    os.environ.setdefault("MLFLOW_GENAI_EVAL_SCORER_RATE_LIMIT", "0.2") # ~12 llamadas/min
    os.environ.setdefault("MLFLOW_GENAI_EVAL_MAX_RETRIES", "8")         # más reintentos ante 429


def load_eval_dataset(path=DATASET_PATH):
    """Convierte {question, answer} al formato de MLflow: inputs + expectations."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return [
        {
            "inputs": {"question": item["question"]},
            "expectations": {"expected_response": item["answer"]},
        }
        for item in raw
    ]


def _numeros(texto: str) -> set[str]:
    """Cifras de 2 o más dígitos: '1:100.000' → {'100000'}; '85%' → {'85'}. Los dígitos sueltos
    se ignoran porque suelen ser viñetas de lista o fragmentos de escalas cortadas por el chunking."""
    cifras = {re.sub(r"[.,]", "", n) for n in re.findall(r"\d+(?:[.,]\d+)*", texto or "")}
    return {c for c in cifras if len(c) > 1}


@scorer
def numeros_respaldados(inputs: dict, outputs: str, trace) -> Feedback:
    """¿Todas las cifras de la respuesta aparecen en los fragmentos recuperados o en la pregunta?"""
    contexto = inputs.get("question", "")
    for span in trace.search_spans(span_type=SpanType.RETRIEVER):
        for doc in span.outputs or []:
            contexto += " " + (doc.get("page_content") or "")
    sin_respaldo = sorted(_numeros(outputs) - _numeros(contexto))
    return Feedback(
        value=not sin_respaldo,
        rationale=f"Cifras sin respaldo en el contexto: {sin_respaldo}" if sin_respaldo
        else "Todas las cifras de la respuesta están en el contexto recuperado o en la pregunta.",
    )


def get_scorers(judge_model: str):
    """
    Jueces predefinidos de MLflow + guías propias en lenguaje natural.
    👉 Desafío: agrega aquí tus propios scorers (ver README).
    Con EVAL_SCORERS=correctness,safety se ejecutan solo los indicados.
    """
    scorers = [
        Correctness(model=judge_model),            # ¿coincide con la respuesta esperada?
        RelevanceToQuery(model=judge_model),       # ¿responde a lo que se preguntó?
        RetrievalGroundedness(model=judge_model),  # ¿está sustentada en los documentos? (alucinación)
        RetrievalRelevance(model=judge_model),     # ¿los fragmentos recuperados son pertinentes?
        Safety(model=judge_model),                 # ¿contenido dañino u ofensivo?
        Guidelines(
            name="idioma_espanol",
            guidelines="La respuesta debe estar escrita en español.",
            model=judge_model,
        ),
        Guidelines(
            name="tono_profesional",
            guidelines=(
                "La respuesta debe tener un tono formal, técnico y profesional, apropiado "
                "para un asistente sobre la metodología CORINE Land Cover del IDEAM, "
                "y no debe inventar procedimientos, datos ni cifras."
            ),
            model=judge_model,
        ),
        Guidelines(
            name="no_inventa_leyenda",
            guidelines=(
                "La respuesta no debe inventar categorías, códigos o umbrales de la leyenda "
                "CORINE Land Cover que no estén en los documentos oficiales del IDEAM."
            ),
            model=judge_model,
        ),
        make_judge(
            name="claridad_tecnica",
            instructions=(
                "Evalúa si la respuesta {{ outputs }} a la pregunta {{ inputs }} es clara y "
                "comprensible para un intérprete de coberturas de la tierra con formación media: "
                "usa la terminología técnica correctamente, la explica cuando hace falta y está "
                "bien organizada. Responde true si lo es y false si no."
            ),
            feedback_value_type=bool,
            model=judge_model,
        ),
        numeros_respaldados,
    ]
    selected = [n.strip() for n in os.getenv("EVAL_SCORERS", "").split(",") if n.strip()]
    if selected:
        unknown = set(selected) - {sc.name for sc in scorers}
        if unknown:
            raise ValueError(f"EVAL_SCORERS desconocidos: {unknown}. "
                             f"Disponibles: {[sc.name for sc in scorers]}")
        scorers = [sc for sc in scorers if sc.name in selected]
    return scorers


def main():
    vectordb = load_vectorstore_from_disk()
    chain = build_chain(vectordb, prompt_version=PROMPT_VERSION, k=K)
    prompt_uri = chain.combine_docs_chain.llm_chain.prompt.metadata["prompt_uri"]

    model_id = init_tracking(EVAL_EXPERIMENT, prompt_uri, K)
    judge_model = get_judge_model_uri()

    print(f"📊 Experimento MLflow: {EVAL_EXPERIMENT}")
    print(f"⚙️  App: {describe()} | prompt={prompt_uri} | k={K}")
    scorers = get_scorers(judge_model)
    print(f"⚖️  Juez: {judge_model} | {len(scorers)} jueces: {[sc.name for sc in scorers]}")
    if os.getenv("MLFLOW_GENAI_EVAL_MAX_WORKERS") == "1":
        print("🐢 Modo pausado para respetar los límites del proveedor (puede tardar varios minutos)")

    def predict_fn(question: str) -> str:
        result = chain.invoke({"question": question, "chat_history": []})
        return result["answer"]

    run_name = f"{describe()['llm_provider']}|{prompt_uri.replace('prompts:/', '')}|k{K}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            **app_config(prompt_uri, K),
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "judge_model": judge_model,
            "dataset": DATASET_PATH,
        })
        results = mlflow.genai.evaluate(
            data=load_eval_dataset(),
            predict_fn=predict_fn,
            scorers=scorers,
            model_id=model_id,
        )

    print("\n📈 Resultados agregados:")
    for name, value in sorted(results.metrics.items()):
        print(f"   {name:40s} {value:.2f}" if isinstance(value, (int, float)) else f"   {name}: {value}")
    print(f"\n🔎 Detalle por pregunta (trazas + razonamiento de cada juez): mlflow ui → "
          f"experimento '{EVAL_EXPERIMENT}' → run '{run_name}'")
    return results


if __name__ == "__main__":
    main()
