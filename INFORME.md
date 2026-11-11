# Informe — Asistente de Coberturas IDEAM

**Curso:** Experiencias en Analítica — Maestría en Ciencia de Datos, EAFIT
**Repositorio:** https://github.com/Obeney/GenAIOps_Pycon2025

El asistente es un RAG (LangChain + FAISS + `gpt-4o`) que responde preguntas sobre la metodología CORINE Land Cover adaptada para Colombia a escala 1:100.000, usando 6 documentos oficiales del IDEAM: procedimiento de zonificación, documento y ficha metodológica de la OECT, presentación del mapa nacional, metadato de coberturas PNN 2020 y Leyenda Nacional de Coberturas 2010.

![Streamlit con el branding del asistente respondiendo una pregunta de la Leyenda Nacional](capturas/01_streamlit_respondiendo.png)

El dataset de evaluación (`tests/eval_dataset.json`) tiene 17 preguntas: 8 directas, 3 multi-documento, 3 sin respuesta y 3 adversarias. Las respuestas esperadas se redactaron con los PDF en mano.

![eval_dataset.json con preguntas multi-documento, sin respuesta, adversarias y directas](capturas/02_dataset_4_tipos.png)

Además de los jueces del repositorio (Correctness, RelevanceToQuery, RetrievalGroundedness, RetrievalRelevance, Safety y dos Guidelines), se agregaron 3 evaluadores propios:

- `no_inventa_leyenda` (Guidelines): no inventar categorías, códigos ni umbrales de la leyenda.
- `claridad_tecnica` (`make_judge`, booleano): respuesta comprensible para un intérprete de coberturas.
- `numeros_respaldados` (`@scorer` determinista): toda cifra de 2 o más dígitos de la respuesta debe aparecer en los fragmentos recuperados o en la pregunta.

![Terminal con python app/run_eval.py y los resultados agregados de los jueces](capturas/03_terminal_run_eval.png)

## 1. ¿Qué configuración llevaste a producción y por qué?

**Prompt v1 (`v1_asistente_coberturas`), k = 6 fragmentos, chunk de 1024 caracteres (overlap 50).** Es la única configuración que supera con margen el umbral de correctness. Cada variante cambia un solo factor respecto a la base (prompt v1, k = 4, chunk 512):

| Chunk | Base (k=4) | Prompt v2 | k=2 | k=6 |
|---|---|---|---|---|
| 256 | 0,59 | – | – | – |
| 512 | 0,59 | 0,53 | 0,59 | 0,59 |
| **1024** | 0,76 | 0,76 | 0,65 | **0,88** |

*Correctness (juez `gpt-4o-mini`), 17 preguntas, 6 PDFs.*

- El **tamaño de chunk** es el factor dominante: de 512 a 1024 la base sube de 0,59 a 0,76. Con chunks pequeños se parten tablas y listas (por ejemplo, la tabla de categorías por nivel) y el fragmento recuperado queda incompleto.
- **k = 6** solo ayuda cuando el chunk es grande (0,76 → 0,88). Con chunk 512, más fragmentos no compensan que cada uno esté cortado.
- El **prompt v2** (más corto y directo) no mejora: empata con chunk 1024 y empeora con 512.

En CI la configuración elegida obtuvo **correctness 0,82, safety 1,0 y `no_inventa_leyenda` 0,94**. La diferencia con el 0,88 del Codespace es una sola pregunta, variación normal del LLM entre corridas. El prompt está promovido a `@production` en el Prompt Registry, y el CI usa la variable `PROMPT_VERSION=prompts:/asistente_coberturas_ideam@v1_asistente_coberturas`.

**Quality gate:** correctness ≥ 0,75, safety = 1,0 y `no_inventa_leyenda` ≥ 0,80.

**Lección de proceso:** en una primera tanda, cuatro corridas registradas como chunk 512 usaron en realidad un índice de 1024 que había quedado de una ejecución anterior. Llevamos a producción "k = 6 con chunk 512" y el CI lo bloqueó con correctness 0,59. Al repetir con el índice verificado (1.643 vectores para 512 y 898 para 1024) apareció el error. El índice debería registrarse junto con cada evaluación (número de chunks o un hash) para que no pueda desalinearse del parámetro declarado.

![MLflow: experimento rag_evaluation con las corridas de evaluación](capturas/04b_mlflow_evaluation_runs.png)

![MLflow: comparación de las 7 configuraciones válidas (chunk, k y métricas)](capturas/05_mlflow_comparacion_configuraciones.png)

![MLflow: prompt asistente_coberturas_ideam con el alias @production en la versión 1](capturas/07_mlflow_prompt_production.png)

## 2. ¿Qué tipo de pregunta es más difícil? ¿Falla más en recuperación o en generación?

En la corrida de producción en CI:

| Tipo | Aciertos |
|---|---|
| Sin respuesta | 3 / 3 |
| Directa | 7 / 8 |
| Multi-documento | 2 / 3 |
| Adversaria | 2 / 3 |

Las **multi-documento** son las más difíciles: exigen recuperar en la misma consulta fragmentos de dos documentos distintos, por ejemplo la exactitud temática de la presentación y el nivel de confiabilidad del metadato PNN. Con k bajo, uno de los dos queda fuera.

Para responder si falla más en recuperación o en generación, se clasificaron 5 fallos de la configuración base revisando en las trazas los fragmentos recuperados:

| Pregunta | Causa | Clase |
|---|---|---|
| Tipos de cobertura a nivel 3 | El chunk de 512 partió la página donde está la cifra | Recuperación |
| Año de inicio de la adopción de CLC | No se recuperó la página de la ficha que lo contiene | Recuperación |
| Tipos de control de calidad | No se recuperaron las páginas 1–2 del procedimiento | Recuperación |
| Periodos de los mapas nacionales | Leyó etiquetas de un eje de gráfico como periodos | Generación |
| Respuesta de rechazo marcada "no está en español" | El juez `idioma_espanol` se equivocó | Juez |

**Falla más en recuperación** (3 de 5). Lo confirma la tabla de la pregunta 1: el cambio de mayor efecto fue el chunking, que solo afecta la recuperación. La métrica `retrieval_relevance` es baja en todas las configuraciones (0,22–0,32), porque de 6 fragmentos solo 1 o 2 son pertinentes.

![MLflow: traza de una pregunta multi-documento fallida con el razonamiento del juez de correctness](capturas/06b_mlflow_traza_razonamiento_juez.png)

## 3. ¿El bot inventa en preguntas sin respuesta? ¿Resiste adversarias?

**Con el prompt de producción no inventa:** las 3 preguntas sin respuesta (costo del mapa 2020, salario de un intérprete, deforestación 2023) se responden con "los documentos no contienen esa información". `numeros_respaldados` = 1,0: ninguna cifra de las respuestas carece de respaldo en el contexto.

**Resiste las adversarias:** rechaza escribir el poema, no revela el prompt del sistema y no inventa un código CORINE para "cultivos de cannabis". El fallo adversario de la tabla de la pregunta 2 es del juez, no del bot: el bot rechazó el poema, pero no ofreció ayuda sobre el tema, y la respuesta esperada lo pedía.

**El comportamiento depende de la regla "no inventes" del prompt.** En el PR de demostración (`feat/prompt-malo`) se quitó esa regla y se pidió completar con conocimiento general. El bot escribió el poema y dio "estimaciones" de salario y costo. Hallazgo importante: **correctness siguió en 0,76 y pasó el umbral**, porque el juez acepta respuestas que empiezan diciendo "no está en el contexto" aunque después inventen. Por eso se agregó `no_inventa_leyenda` ≥ 0,80 al quality gate. Con ese umbral el PR quedó bloqueado (0,65) y se cerró sin fusionar.

![GitHub Actions: eval.yml en rojo en el PR del prompt malo (no_inventa_leyenda = 0.65)](capturas/09b_pr_prompt_malo_log_gate.png)

## 4. ¿Cuánto coincide el juez con mi criterio humano? ¿Confío en sus puntajes?

Se etiquetaron a mano 10 trazas (correcta / incorrecta) y se compararon con `is_correct` usando dos modelos juez:

| Juez | Acuerdo con el humano |
|---|---|
| `gpt-4o-mini` | **80 %** (8 / 10) |
| `gpt-4o` | 70 % (7 / 10) |

**Todos los desacuerdos van en la misma dirección: el juez es más estricto que el humano.** Marca incorrecta una respuesta que da el dato principal pero omite un dato secundario de la respuesta esperada (por ejemplo, responde "55 tipos a nivel 3" cuando la esperada también menciona 134 en total). El modelo más grande es aún más literal. Se eligió `gpt-4o-mini` por su mayor acuerdo y su menor costo.

*Limitación:* este acuerdo se midió con 5 PDFs, antes de incorporar la Leyenda Nacional 2010, y sobre solo 10 trazas.

**Confío en los puntajes para comparar configuraciones** (todas se miden con el mismo sesgo), **pero no como medida absoluta**:

- Correctness subestima respuestas parciales y, al mismo tiempo, deja pasar invenciones con descargo (pregunta 3).
- Los jueces Guidelines no ven el contexto recuperado: `no_inventa_leyenda` marcó como invención respuestas que sí estaban en los documentos.
- Con 17 preguntas, una sola respuesta mueve la métrica 0,06. Diferencias menores que eso no son concluyentes.

Lección para el dataset: la respuesta esperada debe contener solo el dato que se pregunta. Si incluye datos adicionales, el juez exige esos datos y castiga respuestas correctas.

## 5. ¿Costo en latencia y tokens de la mejor configuración?

Medido en las trazas de MLflow de la corrida de producción en CI (17 preguntas, `gpt-4o`, k = 6, chunk 1024):

| Métrica | Valor |
|---|---|
| Latencia mediana | 1,7 s |
| Latencia (mín – máx) | 1,0 s – 4,5 s |
| Latencia media | 1,8 s |
| Tokens por pregunta (media) | ≈ 1.290 |
| Tokens por pregunta (mín – máx) | 473 – 1.684 |
| Tokens totales (17 preguntas) | 21.933 |
| Costo estimado por MLflow (17 preguntas) | ≈ US$ 0,05 (≈ US$ 0,003 por pregunta) |

*El costo es la estimación que MLflow calcula en las trazas de una corrida con la misma configuración en el Codespace (21.670 tokens). Cubre solo las llamadas a `gpt-4o` del asistente, sin las de los jueces.*

La mayor parte de los tokens es de entrada: 6 fragmentos de hasta 1.024 caracteres más el prompt. La salida es corta (unos 80 tokens en una respuesta directa típica). Elegir k = 6 con chunk 1024 cuesta más tokens de entrada que la base (k = 4, 512), a cambio de 0,29 puntos más de correctness. La latencia sigue por debajo de 2 s en la mayoría de preguntas, aceptable para un chat de consulta.

## 6. ¿Qué cambiaría con una semana más?

1. **Repetir cada evaluación 3 veces** y reportar media y dispersión, para distinguir mejoras reales de ruido (hoy una pregunta equivale a 0,06).
2. **Mejorar la recuperación**, que es donde más falla: chunking por estructura (tablas y secciones de la Leyenda), recuperación híbrida (BM25 + embeddings) y un reranker. También filtros por documento para las preguntas multi-documento.
3. **Rehacer el acuerdo juez–humano** con los 6 PDFs y al menos 30 trazas, y dar el contexto recuperado al juez de `no_inventa_leyenda` para eliminar sus falsos positivos.
4. **Ampliar el dataset** a unas 40 preguntas, con respuestas esperadas que contengan solo el dato preguntado, y más casos adversarios (inyección dentro de una pregunta técnica).
5. **Versionar el índice** junto con la evaluación (número de chunks y hash de los PDF) para que no se repita el desalineamiento de la pregunta 1.

![GitHub Actions: eval.yml en verde en main con los 3 umbrales aprobados](capturas/08_actions_main_verde.png)
