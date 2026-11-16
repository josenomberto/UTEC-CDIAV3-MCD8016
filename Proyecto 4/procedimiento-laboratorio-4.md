# Guía de Procedimiento y Arquitectura Técnica: Asistente de Compras por Voz (Laboratorio 4)

Este documento establece la especificación técnica, el marco metodológico y el procedimiento secuencial detallado para la implementación, evaluación y entrega del **Laboratorio 4: Asistente de compras por voz**, correspondiente al curso **MCD8016 / CS8086 – Deep Learning** de la Maestría en Ciencia de Datos e Inteligencia Artificial (UTEC). El desarrollo se encuentra diseñado y optimizado para su ejecución integral dentro de **Google Colab** (compatible tanto con GPU estándar Tesla T4 de 15 GB como con GPUs L4/A100 en Colab Pro), bajo la restricción estricta de procesamiento local sin APIs externas de inferencia.

---

## 1. Marco Operativo, Alineación con el Sílabo y Decisiones Arquitectónicas

La construcción de un asistente de compras multimodal gobernado por voz exige orquestar cinco subsistemas computacionales independientes: reconocimiento automático del habla (ASR), recuperación de información sobre un catálogo denso (RAG), razonamiento y control transaccional mediante un modelo de lenguaje (LLM) con uso de herramientas, backend de comercio electrónico reproducible (tienda simulada) y síntesis de habla con clonación de voz (TTS).

### 1.1. Alineación con los Contenidos del Curso (MCD8016 – Deep Learning)

La arquitectura propuesta integra de manera directa las unidades temáticas del sílabo del curso:
* **Tema 4 (Modelos Fundamentales y Arquitectura Transformer):** Uso de arquitecturas Encoder-only (Bi-Encoders para búsqueda vectorial), Encoder-Decoder (Whisper para ASR) y Decoder-only autorregresivos (Qwen2.5-7B-Instruct para control del agente).
* **Tema 5 (Self-Supervised Learning y Foundation Models):** Aplicación de aprendizaje contrastivo (*Contrastive Learning*, Tema 5.1) mediante `multilingual-e5-base` para alinear consultas asimétricas y descripciones de productos en un espacio latente compartido, así como modelos fundacionales de audio como **Whisper** (Tema 5.4) y **XTTS-v2**.
* **Tema 7 (Large Language Models):** Diseño de *prompting* estructurado y razonamiento guiado (*Chain-of-Thought* / ReAct, Tema 7.2), generación sustentada en recuperación (**RAG** y **Agentes**, Tema 7.4) e instrumentación local con el ecosistema **Hugging Face** (Tema 7.5).
* **Tema 8 y Tema 9 (MLOps, Ética y Sesgos en Deep Learning):** Reproducibilidad determinista mediante control de semillas, perfilado desglosado de latencias y gestión de VRAM (Tema 8); mitigación de alucinaciones tratando todo atributo no verificable como `"Desconocido"` y uso ético de clonación de voz bajo consentimiento informado con disyunción estricta respecto al texto de referencia (Tema 9).

### 1.2. Selección Justificada de Modelos y Presupuesto de Memoria (VRAM)

Para garantizar la ejecución fluida dentro de Google Colab sin depender de APIs externas ni enfrentar bloqueos por permisos, se seleccionan y justifican los siguientes modelos preentrenados:

| Subsistema | Modelo Seleccionado | Precisión / Cuantización | VRAM Estimada | Justificación Técnica y Comparativa |
| :--- | :--- | :--- | :---: | :--- |
| **ASR (Voz a Texto)** | `openai/whisper-small` (`faster-whisper` / HF pipeline) | FP16 | ~1.2 GB | Modelo fundacional Encoder-Decoder (244M parámetros). Supera ampliamente a `whisper-tiny`/`base` en la transcripción de marcas comerciales, cifras decimales y presupuestos en español, manteniendo una latencia 3x menor y un consumo de memoria muy inferior a `whisper-large-v3` (~6 GB). |
| **RAG (Embeddings Densos)** | `intfloat/multilingual-e5-base` | FP16 | ~1.1 GB | Entrenado con aprendizaje contrastivo asimétrico mediante prefijos `query:` y `passage:`. A diferencia de `paraphrase-multilingual-MiniLM-L12-v2` o `mpnet` (diseñados para similitud simétrica de oraciones de longitud similar), `multilingual-e5-base` está optimizado para recuperación de información donde una consulta corta de 3–6 palabras debe emparejarse con fichas extensas de producto, elevando sustancialmente el $nDCG@10$. |
| **Agente / LLM** | `Qwen/Qwen2.5-7B-Instruct` | 4-bit NF4 (`bitsandbytes`, `double_quant`) | ~5.2 GB | Modelo Decoder-only de 7.6B parámetros con soporte nativo para *tool calling* estructurado en JSON y excelente dominio del español. Se elige frente a `meta-llama/Llama-3.1-8B-Instruct` porque **no es un repositorio restringido (*ungated*)**, permitiendo al evaluador ejecutar el notebook en Colab sin requerir tokens privados ni aprobación previa de licencia de Meta, además de presentar menor tasa de errores sintácticos al invocar herramientas. |
| **TTS (Clonación de Voz)** | `coqui/XTTS-v2` | FP16 | ~3.2 GB | Modelo generativo multilingüe con clonación de voz *zero-shot* a partir de un audio de referencia de 6–15 segundos. Preserva timbre y prosodia en español sin necesidad de entrenamiento adicional (*fine-tuning*). |
| **Evaluación Acústica (SECS)** | `speechbrain/spkrec-ecapa-voxceleb` | FP32 (CPU/GPU) | ~0.2 GB | Codificador ECAPA-TDNN empleado exclusivamente en la fase de evaluación para medir cuantitativamente la **semejanza de voz** (similitud coseno entre embeddings del audio de referencia y del audio sintetizado). |
| **Total en Pico** | **Todos los modelos residentes** | **Mixta (NF4 + FP16)** | **~10.9 GB** | **Cabe íntegramente en una GPU estándar de Google Colab (Tesla T4 de 15 GB de VRAM)**, dejando >3.5 GB de margen para el contexto CUDA y activaciones intermedias, y opera con holgura en GPUs L4 (24 GB) o A100 (40 GB). |

### 1.3. Decisiones Arquitectónicas Clave

1. **Gestión Disciplinada de Memoria en GPU:** Se implementa una función de monitoreo de VRAM (`report_vram_usage()`), inferencia bajo `torch.inference_mode()`, cuantización NF4 de 4 bits con doble cuantización para el LLM y liberación explícita de tensores temporales (`gc.collect()` y `torch.cuda.empty_cache()`).
2. **Determinismo y Separación Estricta de Responsabilidades:** El modelo de lenguaje nunca realiza cálculos aritméticos libres sobre precios ni decide la disponibilidad de inventario por cuenta propia. Las validaciones de cantidades, presupuesto máximo, existencias, detección de atributos desconocidos, control de atomicidad y **prevención de pedidos duplicados (idempotencia)** se ejecutan mediante código determinista en Python dentro de `SimulatedStore`.
3. **Uso Riguroso de las Particiones `train` (Desarrollo) y `test` (Evaluación):** En estricto cumplimiento de las reglas del laboratorio, las consultas de desarrollo (`train`) se utilizan para ajustar hiperparámetros del recuperador (como la longitud de truncamiento de descripciones y el peso $\alpha$ de combinación híbrida léxico-densa), mientras que las consultas de evaluación (`test`) permanecen completamente aisladas hasta la corrida final de métricas y ablación.

### 1.4. Planificación Temporal del Proyecto

| Periodo | Fase de Trabajo | Entregables y Metas Clave |
| :--- | :--- | :--- |
| **14 – 16 Nov** | **Fase 1 & Fase 2:** Datos, Tienda y RAG | Descarga/filtrado de $\ge 5\,000$ productos (incluyendo `product_bullet_point`) y 300 consultas (100 `train` para ajuste + 200 `test` para evaluación); clase `SimulatedStore` con idempotencia; línea base BM25, embeddings `multilingual-e5-base`, ajuste en `train` y ablación en `test` ($nDCG@10$). |
| **17 – 18 Nov** | **Fase 3:** Agente y Lógica Transaccional | Carga de `Qwen2.5-7B-Instruct` en 4 bits; definición de herramientas (`tools`); estado conversacional persistente; tratamiento explícito de atributos desconocidos y validaciones deterministas. |
| **19 Nov** | **Fase 4:** Pipeline de Voz y Batería de Pruebas | Integración de `whisper-small` y `XTTS-v2` con voz de referencia autorizada; perfilado de latencias; ejecución de los 20 escenarios operativos (incluyendo los 6 casos obligatorios por audio). |
| **20 – 21 Nov** | **Fase 5 & Fase 6:** Evaluación, Video y Entrega | Comparación oracular (ASR vs. transcripción correcta); evaluación de inteligibilidad (WER/CER inverso) y semejanza de voz (ECAPA-TDNN); análisis de 5 aciertos y 5 fallos; grabación del video (máx. 20 min). |

---

## 2. Fase 1: Curación del Catálogo ESCI y Tienda Simulada

La fase inicial construye el catálogo local en español y el entorno transaccional reproducible, garantizando integridad referencial absoluta entre las consultas y los productos indexados.

### 2.1. Ingesta y Filtrado del Dataset ESCI

Se utiliza el **Amazon Shopping Queries Dataset (ESCI)** a partir de sus archivos oficiales `shopping_queries_dataset_products.parquet` y `shopping_queries_dataset_examples.parquet` (con descarga automática integrada desde el repositorio oficial de Amazon Science en AWS S3 si no se encuentran en el entorno local de Colab):

* **Filtrado Idiomático y de Tarea:** Se seleccionan únicamente los registros en español (`product_locale == 'es'`) pertenecientes a la versión reducida para ranking (`small_version == 1`).
* **Verificación Previa de Integridad Referencial:** Se filtran las consultas cuyos productos anotados existen al 100% en la tabla `df_products_es`, evitando referencias huérfanas.
* **Selección Estratificada de Consultas ($\ge 200$ consultas de evaluación):** Para satisfacer de manera inequívoca el requisito de *"al menos 200 consultas de evaluación, con todos sus productos anotados incluidos en el catálogo. Separarán las consultas por query_id, utilizando train para desarrollo y test para evaluación"*, se muestrean con semilla fija (`RANDOM_SEED = 42`):
  * **100 consultas de desarrollo (`split == 'train'`):** Utilizadas para calibrar el sistema de recuperación (pesos de fusión, preprocesamiento y prompt del agente).
  * **200 consultas de evaluación (`split == 'test'`):** Reservadas exclusivamente para reportar el $nDCG@10$ final y el experimento de ablación, sin emplearse jamás para ajustar el sistema.
* **Construcción del Catálogo ($\ge 5\,000$ productos con todos sus atributos):** Se incluyen primero todos los productos anotados en las 300 consultas seleccionadas (aprox. 4 300 – 4 800 productos únicos) y se completa mediante muestreo aleatorio determinista hasta alcanzar un mínimo de **5 000 productos únicos** ($\max(5000, |\mathcal{P}_{\text{anotados}}|)$). Se conservan todos los campos textuales y metadatos disponibles en ESCI:
  * `product_id`, `product_title`, `product_description`, `product_bullet_point`, `product_brand`, `product_color`.
  * **Justificación de `product_bullet_point`:** En el dataset ESCI en español, más del 35% de los productos carecen de texto en `product_description` pero almacenan sus especificaciones técnicas (dimensiones, material, compatibilidad, potencia) en `product_bullet_point`. Conservar este atributo es indispensable tanto para la calidad del RAG como para responder consultas específicas del usuario sin declarar falsos desconocidos.

### 2.2. Aislamiento Estricto de Etiquetas ESCI

Las etiquetas de relevancia (`Exact`, `Substitute`, `Complement`, `Irrelevant`) se separan en el dataframe `df_eval_ground_truth` para evaluación offline. El dataframe `df_catalog` entregado al índice vectorial y a la tienda simulada **elimina cualquier columna vinculada a `esci_label`, `query` o `gain`**, verificándose mediante una aserción explícita en código que el agente no tenga acceso a los juicios de relevancia.

### 2.3. Arquitectura de la Tienda Simulada Determinista (`SimulatedStore`)

La clase `SimulatedStore` administra el catálogo verificable, los precios, las existencias, los carritos por sesión, los presupuestos máximos y el historial de pedidos:

1. **Inicialización Determinista (`random_seed = 42`) y Restauración (`reset_store`):** Genera precios entre `5.00` y `150.00` y existencias iniciales entre `0` y `25` unidades (con un 8% de productos inicialmente agotados para evaluar escenarios de falta de stock). El método `reset_store()` restaura de forma exacta el inventario inicial y limpia carritos, presupuestos y pedidos.
2. **Atributos No Verificables Tratados como Desconocidos:** Cuando se consulta la ficha de un producto mediante `get_product_info(product_id)`, cualquier atributo (`brand`, `color`, `description`, `bullet_point`) que esté vacío en el catálogo se devuelve explícitamente con el valor `"Desconocido"`, impidiendo que el LLM invente características no sustentadas.
3. **Validación Determinista de Cantidades, Presupuesto y Existencias:**
   * `update_cart` valida que la cantidad sea un número entero y verifica si supera el stock actual para emitir advertencias inmediatas.
   * `set_budget` y `calculate_quote` validan mediante código si el subtotal acumulado del carrito respeta el presupuesto máximo de la sesión (`within_budget: bool`, `budget_difference: float`).
4. **Doble Verificación Atómica e Idempotencia (Prevención de Pedidos Duplicados):**
   * Al invocar `register_order`, se recalcula la cotización y se comprueban nuevamente el presupuesto y las existencias actuales de todos los ítems antes de descontar una sola unidad (evitando cambios parciales si un producto se agotó entre la cotización y la confirmación).
   * Se registra una firma transaccional (`order_signature`) por sesión. Si el usuario o el agente repite la confirmación del mismo pedido, `register_order` detecta la firma previa y retorna `status: "duplicate_order"` con la referencia del pedido ya creado, **sin duplicar el registro ni descontar doblemente el inventario**.
5. **Simulación Reproducible de Contingencias (`simulate_stock_change`):** Permite alterar programáticamente el stock de un producto entre el turno de cotización y el turno de confirmación para evaluar el sexto escenario obligatorio de prueba.

---

## 3. Fase 2: Motor de Recuperación (RAG), Línea Base Léxica, Ajuste en Desarrollo y Ablación

### 3.1. Métrica de Evaluación: $nDCG@10$ y Modalidades de Evaluación

Siguiendo la especificación del laboratorio, se asignan los siguientes valores de relevancia ($r_i$) a los candidatos anotados:
$$r_i = \begin{cases} 
1.0 & \text{si } \text{esci\_label} = \textbf{Exact} \\ 
0.1 & \text{si } \text{esci\_label} = \textbf{Substitute} \\ 
0.01 & \text{si } \text{esci\_label} = \textbf{Complement} \\ 
0.0 & \text{si } \text{esci\_label} = \textbf{Irrelevant o producto no anotado} 
\end{cases}$$

La ganancia acumulada descontada en corte $k=10$ se calcula directamente sobre los valores de relevancia especificados:
$$DCG@10 = \sum_{i=1}^{\min(10, |\mathcal{R}|)} \frac{r_i}{\log_2(i + 1)}, \qquad nDCG@10 = \frac{DCG@10}{IDCG@10}$$

Para ofrecer máxima transparencia experimental, el notebook calcula y reporta el $nDCG@10$ bajo **dos protocolos complementarios**:
1. **$nDCG@10$ sobre Candidatos Anotados (Re-ranking ESCI):** Evalúa la capacidad de cada modelo para ordenar correctamente el conjunto de candidatos anotados para cada consulta (métrica oficial del *Task 1: Query-Product Ranking* de ESCI).
2. **$nDCG@10$ sobre Catálogo Completo ($5\,000$ productos):** Evalúa la recuperación *end-to-end* buscando directamente los 10 mejores productos entre los 5 000 ítems del catálogo.

### 3.2. Línea Base de Búsqueda Léxica (BM25)

Se implementa `BM25Okapi` con un tokenizador normalizado en español (conversión a minúsculas, eliminación de signos de puntuación y filtrado de tokens vacíos). Se construyen índices léxicos tanto sobre títulos como sobre texto enriquecido para establecer una línea base sólida basada en coincidencia exacta de términos.

### 3.3. Búsqueda Semántica Densa y Experimento de Ablación

Utilizando el codificador denso multilingüe `intfloat/multilingual-e5-base` (con prefijo `"query: "` para consultas y `"passage: "` para documentos, indexado en `faiss.IndexFlatIP` con vectores normalizados L2), se realiza un **experimento de ablación formal sobre la representación documental del catálogo**:

* **Variante A (Ablación – Solo Título):** El embedding de cada producto se construye únicamente con `product_title`.
* **Variante B (Enriquecida – Título + Marca + Color + Viñetas + Descripción):** El embedding integra todos los campos verificables del catálogo:
  $$\text{Passage} = \text{"passage: Título: "} + \text{title} + \text{" | Marca: "} + \text{brand} + \text{" | Color: "} + \text{color} + \text{" | Características: "} + \text{bullets} + \text{" | Descripción: "} + \text{description}$$
* **Variante C (Fusión Híbrida Léxico-Densa ajustada en `train`):** Combina las puntuaciones normalizadas de BM25 y del modelo denso enriquecido ($\text{Score}_{\text{hyb}} = \alpha \cdot \hat{S}_{\text{denso}} + (1 - \alpha) \cdot \hat{S}_{\text{BM25}}$), donde el hiperparámetro $\alpha \in [0, 1]$ se selecciona maximizando el $nDCG@10$ **exclusivamente sobre las 100 consultas de desarrollo (`train`)**, evaluándose luego sin modificaciones sobre las 200 consultas de prueba (`test`).

---

## 4. Fase 3: Agente Conversacional, Estado, Herramientas y Validaciones

### 4.1. Arquitectura del Agente y Estado de Conversación (`ConversationState`)

Se implementa un controlador de agente basado en el paradigma **ReAct (Reasoning + Acting)** sobre `Qwen/Qwen2.5-7B-Instruct` (4-bit), acoplado a un objeto explícito de estado conversacional (`ConversationState`) por sesión que conserva de forma estructurada:
* `max_budget`: Presupuesto máximo vigente indicado por el usuario (actualizable en cualquier turno).
* `preferences`: Diccionario de restricciones y preferencias activas (color, marca, categoría, atributos deseados).
* `candidate_products`: Últimos productos recuperados y presentados al usuario con sus identificadores verificados.
* `pending_confirmation`: Bandera booleana que indica si existe una cotización vigente esperando confirmación expresa del comprador.
* `turn_history`: Historial compacto de turnos y trazas de operaciones ejecutadas (`tool_calls_log`).

### 4.2. Herramientas Deterministas del Agente (`Tools`)

El agente dispone de cinco herramientas deterministas cuyas salidas incluyen validaciones de código:

1. `buscar_productos(query: str, top_k: int = 5) -> dict`: Ejecuta la recuperación híbrida/densa sobre el catálogo, cruza cada resultado con `SimulatedStore` para adjuntar precio y stock actual, y etiqueta explícitamente como `"Desconocido"` cualquier atributo ausente en el catálogo.
2. `consultar_detalle_producto(product_id: str) -> dict`: Recupera la ficha completa verificada de un producto específico junto con su precio y existencias en tiempo real.
3. `actualizar_carrito(product_id: str, cantidad: int) -> dict`: Añade, modifica la cantidad (`cantidad > 0`) o elimina (`cantidad == 0`) un ítem del carrito activo, validando en código que el `product_id` exista y que la cantidad solicitada no exceda el stock disponible, y recalculando automáticamente el estado frente al presupuesto de la sesión.
4. `preparar_cotizacion(presupuesto_maximo: Optional[float] = None) -> dict`: Registra o actualiza el presupuesto máximo en el estado de la sesión, calcula subtotales e importe total mediante código, verifica existencias de cada línea y determina programáticamente si la cotización cumple todas las restricciones (`cumple_presupuesto` y `cumple_stock`).
5. `confirmar_pedido(confirmacion_usuario: bool) -> dict`: Requiere que `confirmacion_usuario == True`. Ejecuta `store.register_order()`, realizando la **doble verificación atómica** de stock y presupuesto y el chequeo de **idempotencia** contra pedidos duplicados. Si un producto cambió de stock a 0 antes de confirmar, devuelve el fallo detallado e invoca automáticamente sugerencias de productos sustitutos disponibles que entren en el presupuesto.

### 4.3. Reglas de Control para Ambigüedad y Atributos Desconocidos

* **Detección de Ambigüedad:** Si la consulta inicial es genérica (p. ej., *"quiero comprar una mochila"* o *"busco audífonos"* sin especificar presupuesto, uso o preferencia entre varias opciones de precio muy distinto) o si el usuario pide *"agregar al carrito"* sin haber elegido qué opción prefiere entre las alternativas mostradas, el controlador bloquea cualquier llamada a `confirmar_pedido` y solicita una aclaración puntual.
* **Prohibición de Alucinación de Atributos:** El prompt del sistema y el formateador de herramientas obligan al agente a responder *"Ese atributo figura como desconocido en el catálogo"* cuando un campo vale `"Desconocido"`, prohibiendo inferir características a partir del nombre del producto.

---

## 5. Fase 4: Pipeline Multimodal de Voz (ASR y TTS con Clonación) y Métricas Acústicas

### 5.1. Reconocimiento Automático del Habla (ASR) en Español

* Se utiliza `openai/whisper-small` en precisión FP16 forzado a `language="es"` y `task="transcribe"`.
* Los audios de entrada se normalizan a 16 kHz mono.
* Para evaluar la sensibilidad del sistema a errores del ASR, cada audio cuenta con su transcripción real (*ground truth*), permitiendo cuantificar los **errores de transcripción específicos en cantidades, importes (presupuestos/precios) y nombres de productos**, además del *Word Error Rate* (WER).

### 5.2. Síntesis de Habla con Clonación de Voz (TTS) y Verificación de Disyunción

* Se utiliza `coqui/XTTS-v2` empleando un archivo de grabación de referencia (`reference_voice.wav`, 10–15 s, 22.05 kHz mono) con voz propia o autorizada por un integrante del grupo.
* **Verificación Programática de Disyunción Textual:** Antes de sintetizar cada respuesta, una función de validación comprueba que el texto de salida del asistente sea distinto del texto pronunciado en la grabación de referencia (calculando el solapamiento de $n$-gramas y exigiendo igualdad nula).

### 5.3. Evaluación de Inteligibilidad y Semejanza con la Voz de Referencia

Para cumplir rigurosamente con el requerimiento *"Evaluarán también la inteligibilidad y semejanza con la voz de referencia"*, se implementan dos métricas objetivas reproducibles en Colab complementadas con evaluación subjetiva:

1. **Inteligibilidad del Audio Sintetizado (Round-Trip ASR WER / CER):**
   * El archivo `.wav` generado por `XTTS-v2` se pasa nuevamente por `Whisper` para obtener una transcripción inversa ($\hat{y}_{\text{TTS}\to\text{ASR}}$) que se compara contra el texto original generado por el agente ($y_{\text{agente}}$).
   * Un **CER (Character Error Rate) $< 5\%$** y **WER $< 10\%$** demuestran alta inteligibilidad acústica y ausencia de artefactos severos de pronunciación.
2. **Semejanza con la Voz de Referencia (Speaker Embedding Cosine Similarity – SECS):**
   * Se extraen los vectores de identidad vocal (*speaker embeddings* de 192 dimensiones) tanto del audio de referencia original ($\mathbf{e}_{\text{ref}}$) como de cada respuesta generada por TTS ($\mathbf{e}_{\text{gen}}$) utilizando el modelo preentrenado `speechbrain/spkrec-ecapa-voxceleb`.
   * Se calcula la similitud coseno:
     $$\text{SECS} = \frac{\mathbf{e}_{\text{ref}} \cdot \mathbf{e}_{\text{gen}}}{\|\mathbf{e}_{\text{ref}}\|_2 \|\mathbf{e}_{\text{gen}}\|_2}$$
   * Valores de $\text{SECS} \ge 0.75$ confirman una alta fidelidad tímbrica respecto al locutor de referencia.

### 5.4. Medición Desglosada de Latencias

Se separa explícitamente el **tiempo de carga inicial en frío de los modelos ($T_{\text{load}}$)** del tiempo de respuesta por turno de conversación, desglosando cada turno desde que finaliza la grabación de entrada hasta que el audio de respuesta está listo para reproducirse:
$$T_{\text{total}} = T_{\text{ASR}} + T_{\text{RAG}} + T_{\text{Agente/Tools}} + T_{\text{TTS}}$$

---

## 6. Fase 5: Pruebas de Funcionamiento (20 Escenarios y 6 Pruebas por Voz)

### 6.1. Batería de 20 Escenarios Operativos Reproducibles

Se diseñan y ejecutan 20 escenarios con estado inicial reproducible (`store.reset_store()` antes de cada escenario), registrando el estado inicial, las herramientas invocadas, el cumplimiento de restricciones (presupuesto, cantidades, stock) y el estado final de la tienda:
* **Escenarios 1 a 6 (Ejecutados por Voz + Réplica Oracular por Texto):** Cubren las 6 situaciones obligatorias exigidas por la guía del laboratorio, con al menos 4 de ellos en modalidad multi-turno (superando el mínimo de 3 pruebas multi-turno por voz).
* **Escenarios 7 a 20 (Ejecutados sobre el Agente):** Evalúan compras multiproducto con límite de presupuesto ajustado, consultas sobre atributos desconocidos (verificando que no haya alucinación), comparación de alternativas Exact vs. Substitute, carritos con múltiples modificaciones de cantidad, presupuestos imposibles, rechazo de compra sin confirmación explícita e intentos de inyección de precios falsos por parte del usuario.

### 6.2. Detalle de los 6 Escenarios Obligatorios por Voz

| ID | Situación Obligatoria Cubierta | Modalidad | Descripción Operativa y Verificación |
| :---: | :--- | :---: | :--- |
| **V1** | Compra que cumple restricciones y termina con pedido confirmado | Multi-turno (3 turnos) | El usuario busca un producto con presupuesto máximo de 100€, solicita cotización para 2 unidades y confirma la compra. Se verifica descuento exacto de inventario y creación de `order_id = 1`. |
| **V2** | Solicitud ambigua que requiere una aclaración | Multi-turno (2 turnos) | El usuario pide *"Quiero comprar un cargador rápido, agrégalo al carrito"* sin especificar modelo, marca ni presupuesto entre múltiples opciones. El agente solicita aclaración antes de modificar el carrito. |
| **V3** | Cambio de cantidad o preferencia conservando las demás restricciones | Multi-turno (3 turnos) | El usuario fija presupuesto de 80€ y elige un producto (3 unidades); al ver que supera el presupuesto o cambiar de opinión, reduce la cantidad a 1 unidad manteniendo el límite de 80€ y confirma. |
| **V4** | Producto agotado que requiere buscar alternativa o comunicar sin solución | Turno único / Multi-turno | El usuario solicita un producto específico cuyo stock inicial en `SimulatedStore` es `0`. El agente detecta `stock == 0`, informa el agotamiento y propone un sustituto con stock disponible. |
| **V5** | Corrección de una transcripción antes de ejecutar una compra | Multi-turno (3 turnos) | El audio/ASR introduce un error en la cantidad o importe (p. ej. transcribe *5 unidades* en vez de *2 unidades*). Al escuchar la cotización del agente, el usuario corrige verbalmente la cantidad antes de confirmar. |
| **V6** | Cambio de stock entre cotización y confirmación + repetición de confirmación sin duplicar pedido | Multi-turno (3 turnos) | Turno 1: Se cotiza un producto con stock disponible. Entre el Turno 1 y 2 se ejecuta `simulate_stock_change(pid, 0)` (o cambio a alternativa disponible). Al confirmar y repetir el mensaje de confirmación, se verifica que el sistema detecte la caída de stock, pase a la alternativa autorizada y, ante la doble confirmación, bloquee el duplicado (`duplicate_order`). |

### 6.3. Evaluación Oracular y Métricas Consolidadas a Reportar

Los 6 escenarios de voz se ejecutan dos veces desde el mismo estado inicial: (1) usando la salida real de Whisper ASR y (2) usando la transcripción manual exacta (*modo oracular*). A partir de los 20 escenarios y las pruebas de recuperación se genera el reporte consolidado exigido:
1. **$nDCG@10$** (en candidatos anotados y en catálogo completo) comparando BM25, Variante A (Solo Título), Variante B (Enriquecido) y Variante C (Híbrido).
2. **Porcentaje de escenarios completados correctamente** ($\% \text{ Éxito en los 20 escenarios}$ y comparativa Voz ASR vs. Oráculo).
3. **Conteo de incumplimientos de restricciones** (presupuesto excedido, venta sin stock, modificación parcial o pedido duplicado).
4. **Errores de transcripción del ASR desglosados por entidad:** tasa de error en **cantidades**, **importes monetarios** y **nombres de productos/marcas**.
5. **Métricas de calidad de voz TTS:** Inteligibilidad (**WER / CER inverso**) y Semejanza con la voz de referencia (**SECS**).
6. **Tiempos de ejecución desglosados por etapa** ($T_{\text{load}}$ inicial vs. $T_{\text{ASR}}, T_{\text{RAG}}, T_{\text{Agente}}, T_{\text{TTS}}, T_{\text{total}}$ en consultas posteriores).
7. **Análisis detallado de al menos 5 aciertos y 5 fallos**, identificando causas raíz y limitaciones de la solución.

---

## 7. Fase 6: Estructura de Entrega y Video Explicativo (Máximo 20 Minutos)

### 7.1. Contenido del Código Fuente (Jupyter Notebook en Colab)

El archivo principal `Laboratorio 4 - Asistente de compras por voz.ipynb` incluye instrucciones de ejecución en Google Colab, versiones exactas de librerías, configuración de hardware/semillas y todas las fases estructuradas secuencialmente:
1. Preparación de datos ESCI, construcción de índices (BM25 y FAISS) y tienda simulada con restauración.
2. Evaluación del modelo base léxico, ajuste en `train` y experimento de ablación en `test`.
3. Integración de voz (Whisper ASR + XTTS-v2), agente (`Qwen2.5-7B-Instruct`), memoria conversacional, herramientas y validaciones deterministas.
4. Ejecución de los 20 escenarios operativos, pruebas por voz vs. transcripción oracular, métricas de voz, tiempos por etapa y análisis de 5 aciertos y 5 fallos.

### 7.2. Guión Cronometrado del Video Explicativo (Máximo 20 Minutos)

* **Bloque 1 – Metodología (4 minutos):** Diseño general del sistema, fundamentación teórica según el sílabo de Deep Learning, justificación de modelos seleccionados (`whisper-small`, `multilingual-e5-base`, `Qwen2.5-7B-Instruct` 4-bit, `XTTS-v2`) y formulación del experimento de ablación.
* **Bloque 2 – Implementación (4 minutos):** Explicación del código de recuperación, aislamiento de etiquetas ESCI, control del agente y estado conversacional, validaciones deterministas en `SimulatedStore` (doble chequeo de stock e idempotencia) y gestión de memoria VRAM en Colab.
* **Bloque 3 – Demostración (8 minutos):** Ejecución con entrada y respuesta por voz de (a) una compra completa que termina con pedido confirmado y (b) un caso que requiere aclaración o recuperación ante falta de stock. **Al menos una interacción se muestra íntegra en tiempo real, con su espera real sin cortes de edición.**
* **Bloque 4 – Resultados (4 minutos):** Comparación cuantitativa con el modelo base léxico (BM25), resultados de la ablación ($nDCG@10$), latencias por etapa, métricas de voz (inteligibilidad y semejanza), análisis de los 5 aciertos y 5 fallos, errores del ASR y limitaciones del sistema.
