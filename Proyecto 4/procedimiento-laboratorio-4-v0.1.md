# Guía de Procedimiento y Arquitectura Técnica: Asistente de Compras por Voz (Laboratorio 4)

Este documento establece la especificación técnica, el marco metodológico y el procedimiento secuencial detallado para la implementación, evaluación y entrega del **Laboratorio 4: Asistente de compras por voz**. El desarrollo se encuentra optimizado para su ejecución integral en **Google Colab Pro** bajo la restricción estricta de procesamiento local sin dependencias de interfaces de programación de aplicaciones (APIs) externas de inferencia, con fecha límite de entrega fijada para el **sábado 21 de noviembre de 2026**.

---

## 1. Marco Operativo y Decisiones Arquitectónicas

La construcción de un asistente de compras multimodal gobernado por voz exige orquestar cinco subsistemas computacionales independientes: captura y reconocimiento del habla (ASR), recuperación semántica sobre un catálogo denso (RAG), razonamiento y control transaccional (modelo de lenguaje y agente), backend de comercio electrónico reproducible (tienda simulada) y síntesis del habla con clonación de voz (TTS). 

Para garantizar la estabilidad del entorno y la estricta reproducibilidad en Google Colab Pro, la arquitectura se sustenta en tres decisiones de diseño fundamentales:

1. **Gestión Disciplinada de Memoria en GPU Unificada:** Aunque Colab Pro provee acceso a tarjetas gráficas de alto rendimiento como NVIDIA A100 o L4, la coexistencia de modelos generativos de audio, lenguaje y representaciones vectoriales densas exige un presupuesto de memoria de video (VRAM) rigurosamente planificado. Se adopta la cuantización en 4 bits (GGUF o BitsAndBytes) para el modelo de lenguaje principal, asignando aproximadamente 5.5 GB de VRAM a su inferencia; modelos compactos y precisos para ASR como Whisper en precisión media de punto flotante (FP16), requiriendo 1.5 GB de VRAM; transformadores de codificación de oraciones para búsqueda semántica con un consumo de 0.8 GB de VRAM; y módulos de clonación zero-shot como Coqui XTTS-v2 con 3.5 GB de VRAM.
2. **Determinismo y Separación Estricta de Responsabilidades:** El modelo de lenguaje nunca toma decisiones contables ni calcula inventarios de forma libre. Las restricciones de presupuesto, existencias, reservas de ítems y cotizaciones se validan exclusivamente en código determinista mediante funciones de orden superior, evitando de raíz cualquier alucinación financiera o inconsistencia en inventario.
3. **Variable de Ablación Aislada en Recuperación:** Se selecciona como variable de ablación la **Estrategia de Formulación del Documento a Indexar (Solo Título vs. Título Enriquecido con Descripción y Metadatos)**. Esta selección aísla de forma directa el impacto de la riqueza contextual frente a la especificidad léxica, evaluándose formalmente mediante la métrica normalizada de ganancia acumulada descontada en las diez primeras posiciones ($nDCG@10$) bajo las ponderaciones oficiales del benchmark de consultas de compras de Amazon.

La distribución temporal de trabajo se planifica de manera escalonada a lo largo del periodo asignado para permitir iteraciones continuas y mitigación temprana de cuellos de botella:

| Periodo | Fase de Trabajo | Entregables y Metas Clave |
| :--- | :--- | :--- |
| **14 – 16 Nov** | **Fase 1 & Fase 2:** Datos, Tienda y RAG | Filtrado de 5 000 productos y 200 consultas; implementación de la clase `SimulatedStore`; índices léxico y denso; cálculo de $nDCG@10$ y reporte de ablación. |
| **17 – 18 Nov** | **Fase 3:** Agente y Lógica Transaccional | Carga del LLM local en Colab Pro; definición e instrumentación de herramientas (`tools`); validaciones deterministas de código, doble chequeo de existencias y control de memoria conversacional. |
| **19 Nov** | **Fase 4:** Pipeline de Voz y Batería de Pruebas | Integración de Whisper y XTTS-v2 con el clip de voz autorizado; registro de latencias; ejecución de los 20 escenarios operativos (incluyendo los 6 casos mandatorios de audio). |
| **20 – 21 Nov** | **Fase 5 & Fase 6:** Análisis, Video y Entrega | Comparación oracular (ASR vs. Ground Truth); documentación analítica de 5 aciertos y 5 fallos; grabación y edición del video de 20 minutos; verificación final del notebook reproducible. |

---

## 2. Fase 1: Curación del Catálogo ESCI y Tienda Simulada

La fase inicial establece la base de datos de productos y el entorno de simulación transaccional. Debe garantizarse la coherencia referencial absoluta entre las consultas de prueba y el catálogo disponible.

### 2.1. Ingesta y Filtrado del Dataset ESCI

El conjunto de datos oficial de consultas de compras de Amazon comprende archivos estructurados en formato parquet. Para los requerimientos del proyecto se procesan los archivos correspondientes a ejemplos anotados y descripciones de producto.

* **Filtrado Idiomático y de Tarea:** Se restringen los registros a aquellos cuyo identificador de localización de producto corresponde a español (`product_locale == 'es'`) y pertenecen a la versión reducida de ordenamiento (`small_version == 1`), correspondiente a la tarea de clasificación y ranking del benchmark.
* **Selección Estratificada de Consultas:** Se seleccionan al menos **200 consultas únicas** (`query_id`), divididas equitativamente respetando su partición original en:
  * **100 consultas de desarrollo (`split == 'train'`):** Empleadas para el ajuste de instrucciones del agente y la validación preliminar del motor de búsqueda.
  * **100 consultas de evaluación (`split == 'test'`):** Reservadas exclusivamente para la evaluación final de ranking, prohibiéndose cualquier ajuste o sintonización de hiperparámetros con este subconjunto.
* **Integridad Referencial del Catálogo ($\ge 5\,000$ productos):** Se extraen todos los identificadores de producto (`product_id`) asociados a las 200 consultas seleccionadas. Posteriormente, se incorporan productos adicionales del catálogo en español hasta completar al menos 5 000 productos únicos. Se verifica mediante validaciones automatizadas en código que el conjunto de productos anotados sea un subconjunto estricto del catálogo final:
$$\mathcal{P}_{\text{anotados}} \subseteq \mathcal{P}_{\text{catálogo}}, \quad |\mathcal{P}_{\text{catálogo}}| \ge 5000$$

### 2.2. Aislamiento Estricto de Etiquetas ESCI

Las etiquetas de juicio de relevancia emitidas por anotadores humanos (`Exact`, `Substitute`, `Complement`, `Irrelevant`) se desacoplan y almacenan en un archivo independiente reservado para la evaluación offline. El catálogo expuesto al motor de búsqueda semántica y a las herramientas del agente conversacional únicamente contiene información descriptiva factual: identificador, título, descripción, marca, color y viñetas informativas.

### 2.3. Arquitectura de la Tienda Simulada Determinista

La tienda simulada gestiona de manera centralizada los inventarios, carritos activos por sesión y órdenes procesadas. Su inicialización parte de una semilla pseudoaleatoria fija (`random_seed = 42`) para garantizar que existencias y precios sean idénticos en cada ejecución:

* **Precios Ficticios:** Asignados de forma determinista entre 5.00€ y 150.00€.
* **Existencias en Almacén:** Asignadas entre 0 y 25 unidades. Una proporción del 8% del catálogo se configura de inicio en 0 existencias para posibilitar los escenarios obligatorios de desabastecimiento y sugerencia de alternativas.
* **Método de Restauración (`reset_store`):** Restaura el inventario a su estado inicial y purga todas las sesiones y pedidos, permitiendo evaluar la batería de escenarios desde un punto de partida neutral y reproducible.
* **Atomicidad y Doble Chequeo:** La tienda no descuenta existencias al cotizar ni al modificar el carrito; la reserva y decremento de inventario ocurren exclusivamente durante la ejecución del método `register_order`, momento en el cual se valida nuevamente la disponibilidad antes de emitir la confirmación.

```python
# ==============================================================================
# Definición de la Tienda Simulada Determinista
# ==============================================================================
import copy
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

class SimulatedStore:
    def __init__(self, catalog_df: pd.DataFrame, random_seed: int = 42):
        self.catalog_df = catalog_df
        self.seed = random_seed
        self.initial_inventory: Dict[str, Dict] = {}
        self._initialize_store()
        self.reset_store()

    def _initialize_store(self):
        rng = np.random.RandomState(self.seed)
        for _, row in self.catalog_df.iterrows():
            pid = str(row["product_id"])
            price = round(float(rng.uniform(5.0, 150.0)), 2)
            stock = int(rng.choice([0, rng.randint(1, 25)], p=[0.08, 0.92]))
            self.initial_inventory[pid] = {
                "product_id": pid,
                "price": price,
                "stock": stock
            }

    def reset_store(self):
        self.inventory: Dict[str, Dict] = copy.deepcopy(self.initial_inventory)
        self.carts: Dict[str, Dict[str, int]] = {}
        self.orders: List[Dict] = []

    def get_product_info(self, product_id: str) -> Optional[Dict]:
        return self.inventory.get(str(product_id), None)

    def update_cart(self, session_id: str, product_id: str, quantity: int) -> bool:
        pid = str(product_id)
        if pid not in self.inventory:
            return False
        if session_id not in self.carts:
            self.carts[session_id] = {}
        if quantity <= 0:
            self.carts[session_id].pop(pid, None)
        else:
            self.carts[session_id][pid] = quantity
        return True

    def calculate_quote(self, session_id: str) -> Dict:
        cart = self.carts.get(session_id, {})
        items = []
        total = 0.0
        for pid, qty in cart.items():
            info = self.inventory[pid]
            subtotal = round(info["price"] * qty, 2)
            total += subtotal
            items.append({
                "product_id": pid,
                "quantity": qty,
                "unit_price": info["price"],
                "subtotal": subtotal,
                "available_stock": info["stock"]
            })
        return {"items": items, "total": round(total, 2)}

    def register_order(self, session_id: str, max_budget: Optional[float] = None) -> Dict:
        quote = self.calculate_quote(session_id)
        if not quote["items"]:
            return {"status": "error", "message": "El carrito está vacío."}
        if max_budget is not None and quote["total"] > max_budget:
            return {
                "status": "error",
                "message": f"El total ({quote['total']}€) supera el presupuesto asignado ({max_budget}€)."
            }
        for item in quote["items"]:
            current_stock = self.inventory[item["product_id"]]["stock"]
            if item["quantity"] > current_stock:
                return {
                    "status": "out_of_stock",
                    "product_id": item["product_id"],
                    "requested": item["quantity"],
                    "available": current_stock,
                    "message": f"Existencias insuficientes para el producto {item['product_id']}."
                }
        for item in quote["items"]:
            self.inventory[item["product_id"]]["stock"] -= item["quantity"]

        order = {
            "order_id": len(self.orders) + 1,
            "session_id": session_id,
            "items": quote["items"],
            "total": quote["total"]
        }
        self.orders.append(order)
        self.carts[session_id] = {}
        return {"status": "success", "order": order}
```

---

## 3. Fase 2: Motor de Recuperación (RAG), Línea Base Léxica y Ablación

El sistema de recuperación alimenta al agente con los productos más pertinentes a la consulta del comprador. Es imperativo que todo atributo comunicado al usuario se encuentre debidamente sustentado en el contexto recuperado; cualquier propiedad no verificable documentalmente debe catalogarse formalmente como desconocida.

### 3.1. Métrica de Evaluación: nDCG@10

La evaluación cuantitativa del ordenamiento se realiza calculando la ganancia acumulada descontada normalizada sobre los diez primeros elementos ($nDCG@10$). Las ponderaciones asignadas a cada etiqueta en la escala de relevancia reflejan la utilidad real para el usuario final en un escenario de comercio electrónico:

$$r_i = \begin{cases} 
1.0 & \text{si el juicio de relevancia es } \textbf{Exact} \\ 
0.1 & \text{si el juicio de relevancia es } \textbf{Substitute} \\ 
0.01 & \text{si el juicio de relevancia es } \textbf{Complement} \\ 
0.0 & \text{si el juicio de relevancia es } \textbf{Irrelevant o ítem omitido} 
\end{cases}$$

El valor de ganancia descontada para una lista de productos clasificados hasta la posición diez se formula como:

$$DCG@10 = \sum_{i=1}^{\min(10, |\mathcal{R}|)} \frac{r_i}{\log_2(i + 1)}$$

$$nDCG@10 = \frac{DCG@10}{IDCG@10}$$

donde $IDCG@10$ representa la ganancia acumulada descontada ideal que resultaría de posicionar de forma descendente todas las anotaciones de relevancia perfecta disponibles para la consulta evaluada.

### 3.2. Implementación de la Línea Base Léxica (BM25)

Se emplea el algoritmo BM25Okapi sobre las representaciones tokenizadas de los títulos y descripciones de los productos. Esta línea base suministra la referencia tradicional basada en coincidencia exacta de términos y frecuencia inversa de documentos, permitiendo medir la ventaja comparativa de los modelos densos.

### 3.3. Experimento de Ablación: Estrategia de Formulación del Documento

Para aislar la influencia de la semántica descriptiva sin variar la arquitectura del codificador denso (empleando un modelo multilingüe local como `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`), se evalúan dos estrategias de preparación del texto a vectorizar:

* **Variante A (Ablación / Representación Restringida):** Se genera el vector denso considerando únicamente la cadena correspondiente al título del producto (`product_title`).
* **Variante B (Sistema Propuesto / Representación Enriquecida):** Se genera el vector denso concatenando título, marca, color y descripción completa:
  $$\text{Documento} = \text{"Título: "} + \text{title} + \text{" | Marca: "} + \text{brand} + \text{" | Color: "} + \text{color} + \text{" | Detalle: "} + \text{description}$$

Ambas variantes se evalúan sobre las 100 consultas de prueba. Esta comparación experimental ilustra si la inclusión de texto descriptivo extenso favorece la recuperación de consultas complejas o si introduce ruido semántico que degrada el posicionamiento de productos exactos.

---

## 4. Fase 3: Agente Conversacional, Herramientas y Lógica Transaccional

El agente actúa como el motor de orquestación lógica y diálogo. No se precisa una arquitectura multi-agente; un único agente instruido con llamadas a funciones ejecutado localmente mediante un modelo como Llama-3.1-8B-Instruct cuantizado en 4 bits provee la capacidad requerida para razonar, invocar herramientas y sintetizar respuestas.

### 4.1. Definición Formal de Herramientas (Tools)

El modelo interactúa con el entorno mediante un esquema formal de invocación de funciones deterministas en Python:

* `buscar_catalogo(query: str, max_results: int = 5) -> str`: Realiza la búsqueda por similitud de cosenos sobre el índice vectorial del RAG y devuelve una lista estructurada con identificadores, títulos y atributos constatables.
* `consultar_producto(product_id: str) -> str`: Inspecciona el precio unitario y las existencias exactas en la tienda simulada.
* `gestionar_carrito(session_id: str, product_id: str, cantidad: int) -> str`: Inserta, modifica la cantidad o retira un producto del carrito de la sesión activa.
* `obtener_cotizacion(session_id: str) -> str`: Calcula subtotales acumulados, verifica existencias previas y valida el importe total contra el presupuesto disponible.
* `confirmar_pedido(session_id: str) -> str`: Ejecuta la transacción definitiva en la tienda simulada con control de atomicidad.

### 4.2. Control de Estado Conversacional y Manejo de Ambigüedad

La memoria conversacional retiene de forma persistente a lo largo de los turnos de diálogo:
* El presupuesto monetario máximo estipulado por el comprador.
* Preferencias acumuladas explícitas (marcas solicitadas, colores, categorías).
* Identificadores de productos examinados en turnos previos.

Cuando la solicitud del usuario carece de especificidad (por ejemplo, si el comprador indica *"quiero unos auriculares inalámbricos"* sin indicar presupuesto, tecnología de cancelación de ruido o preferencias de formato), el agente tiene expresamente prohibido agregar ítems al carrito basándose en suposiciones. En su lugar, emite una pregunta de aclaración directa y breve antes de proceder con cualquier modificación de estado.

### 4.3. Doble Verificación Determinista y Prevención de Duplicados

La integridad transaccional se garantiza en la capa de código:
1. Durante la cotización preliminar se informa al usuario el monto total y la disponibilidad estimada.
2. Al recibir la orden explícita de confirmación, la herramienta `confirmar_pedido` vuelve a consultar el inventario físico en tiempo real. Si entre la cotización y la confirmación otro evento agotó el producto, la compra se aborta de forma atómica, impidiendo órdenes parciales o cargos indebidos.
3. El sistema comunica inmediatamente la indisponibilidad al usuario y propone alternativas recuperadas mediante el RAG que cumplan con el presupuesto y preferencias originales.

---

## 5. Fase 4: Pipeline Multimodal de Voz (ASR y TTS con Clonación)

El pipeline de voz constituye la interfaz de entrada y salida del asistente. La interacción se inicia a partir de un flujo de audio hablado y concluye con una respuesta acústica sintetizada que replica el timbre de una voz de referencia autorizada.

### 5.1. Reconocimiento Automático del Habla (ASR)

Se despliega localmente Whisper en su versión pequeña o mediana, configurando la decodificación para transcripción en idioma español. El módulo recibe archivos de audio con tasa de muestreo de 16 kHz y emite la transcripción textual normalizada que alimenta la ventana de contexto del agente.

### 5.2. Síntesis del Habla con Clonación de Voz (TTS)

Se incorpora un modelo generativo de síntesis de habla con capacidad de clonación rápida como Coqui XTTS-v2:
* **Audio de Referencia:** Un archivo de audio de 10 a 20 segundos de duración, grabado en un entorno silencioso con la voz de un miembro del equipo que haya otorgado su consentimiento explícito documentado.
* **Disyunción Textual Obligatoria:** Se verifica en el código del pipeline que el texto redactado por el agente para la respuesta no contenga frases ni estructuras presentes en el texto del audio de referencia, salvaguardando la generalización acústica del modelo.

### 5.3. Perfilado y Desglose de Latencias

El sistema instrumenta la medición temporal de cada etapa con precisión de milisegundos para diagnosticar cuellos de botella:
* **Tiempo de Carga Inicial ($T_{\text{init}}$):** Latencia requerida para descargar, montar en VRAM e inicializar todos los modelos al arrancar la sesión de Colab Pro.
* **Latencias Desglosadas por Turno:**
  * $T_{\text{ASR}}$: Tiempo transcurrido en el procesamiento del audio y emisión del texto transcripto.
  * $T_{\text{RAG}}$: Tiempo de inferencia del codificador de oraciones y consulta top-k en el índice FAISS.
  * $T_{\text{LLM}}$: Tiempo consumido por el modelo de lenguaje en razonamiento e invocación de herramientas.
  * $T_{\text{TTS}}$: Tiempo de codificación acústica y generación del archivo de audio resultante.
  * $T_{\text{Total}}$: Lapso acumulado desde que finaliza la grabación de entrada hasta que el archivo de audio de salida está disponible para su reproducción.

---

## 6. Fase 5: Batería de Pruebas Operativas y Evaluación Experimental

El desempeño global del asistente se mide a través de un conjunto representativo de interacciones que abarcan flujos nominales, errores acústicos y contingencias de inventario.

### 6.1. Protocolo de los 20 Escenarios Operativos

Se estructuran 20 casos de prueba que cubren compras simples, pedidos multiproducto, reconfiguraciones de presupuesto y cancelaciones. Previo al inicio de cada escenario, se ejecuta `store.reset_store()` para restituir los inventarios a los valores exactos derivados de la semilla de inicialización.

### 6.2. Los Seis Escenarios Obligatorios por Voz

Se definen y ejecutan seis pruebas conducidas estrictamente mediante archivos de audio, garantizando que al menos tres de ellas incorporen interacción multi-turno:

1. **Compra Nominal Exitosa:** El usuario solicita productos que satisfacen presupuesto y existencias, confirmando la orden de manera satisfactoria.
2. **Aclaración por Solicitud Ambigua:** El usuario plantea una petición imprecisa que activa la detección de ambigüedad en el agente, solicitando especificaciones antes de actuar.
3. **Modificación de Restricciones en Diálogo Multi-Turno:** El comprador altera cantidades o preferencias en un segundo turno, debiendo el agente actualizar el carrito preservando el límite presupuestario fijado al inicio.
4. **Manejo de Producto Agotado:** El usuario solicita un ítem con inventario en cero; el agente identifica la restricción y propone una alternativa análoga recuperada del catálogo o declara la ausencia de solución compatible.
5. **Corrección de Error Acústico del ASR:** El transcriptor interpreta incorrectamente un número o nombre; el comprador detecta la equivocación en la respuesta del asistente y la rectifica verbalmente antes de validar la compra.
6. **Manejo de Concurrencia y Stock Agotado:** Se fuerza programáticamente el agotamiento de existencias entre la fase de cotización y la confirmación; el agente gestiona el fallo devuelto por la tienda, notifica la situación y reintenta el proceso sin duplicar pedidos ni corromper el carrito.

### 6.3. Evaluación Oracular para Aislamiento de Errores

Para discernir con exactitud si los fallos en las interacciones de voz obedecen a imperfecciones del transcriptor acústico o a inconsistencias lógicas del modelo de lenguaje, **los seis escenarios de audio se replican en modo oracular**. En esta variante se alimenta al agente con la transcripción manual exacta de los audios, comparando los árboles de ejecución resultantes y aislando la tasa de error introducida por el transcriptor en cantidades, precios y denominaciones de producto.

### 6.4. Análisis de Aciertos y Fallos

Se documenta el análisis sistemático de **cinco aciertos relevantes y cinco fallos característicos**, detallando para cada uno:
* Archivo de audio de entrada y transcripción obtenida.
* Herramientas invocadas y parámetros transmitidos.
* Causa raíz identificada (e.g., colisión fonética en ASR, baja discriminación semántica en el vector RAG o truncamiento de memoria en el LLM).
* Limitación del sistema evidenciada y estrategia de mitigación técnica.

---

## 7. Fase 6: Empaquetado del Notebook y Guión Técnico del Video (20 min)

La entrega académica consiste en el código fuente completamente ejecutable en Google Colab Pro y un video explicativo de duración máxima de 20 minutos con una estructura de bloques rigurosamente pautada.

### 7.1. Estructura y Reproducibilidad del Jupyter Notebook

El cuaderno de trabajo debe mantener un flujo secuencial limpio que permita su ejecución integral de inicio a fin:

* **Módulo de Preparación del Entorno:** Descarga de dependencias locales, verificación de aceleración por GPU y carga de pesos cuantizados.
* **Módulo de Datos y Simulación:** Filtrado del subconjunto ESCI en español, muestreo estratificado de consultas (train/test), verificación de integridad del catálogo y construcción de la tienda determinista.
* **Módulo de Recuperación y Ablación:** Implementación de BM25, embeddings multilingües, cálculo de $nDCG@10$ y reporte comparativo entre la representación reducida y la enriquecida.
* **Módulo de Agente y Herramientas:** Definición de la memoria conversacional, funciones de validación transaccional y prompt rector del asistente.
* **Módulo de Voz:** Inicialización de Whisper y XTTS-v2 con la carga de la pista de voz autorizada.
* **Módulo de Pruebas y Resultados:** Automatización de los 20 escenarios, comparativa oracular, tablas consolidadas de métricas y gráficos de latencia.

### 7.2. Pauta Temporal del Video Explicativo (Máximo 20 Minutos)

El contenido audiovisual debe distribuirse de manera estricta entre los cuatro segmentos temáticos establecidos:

* **Segmento 1: Metodología y Decisiones de Diseño (Minutos 0:00 a 4:00 - 4 min):** Exposición de la arquitectura integral, justificación de los modelos locales seleccionados, estrategia de cuantización para Colab Pro y planteamiento formal del experimento de ablación en RAG.
* **Segmento 2: Implementación Técnica y Gestión de Recursos (Minutos 4:00 a 8:00 - 4 min):** Demostración del funcionamiento de la tienda simulada, aislamiento de etiquetas de relevancia, código de las herramientas del agente, mecanismos de validación determinista y gestión de VRAM.
* **Segmento 3: Demostración en Vivo por Voz (Minutos 8:00 a 16:00 - 8 min):** Demostración interactiva de una compra exitosa multi-turno y un caso complejo que requiera aclaración por ambigüedad o recuperación ante existencias agotadas. Es requisito mandatorio mostrar al menos una interacción completa con su tiempo de espera real sin cortes de edición.
* **Segmento 4: Resultados, Análisis Crítico y Conclusiones (Minutos 16:00 a 20:00 - 4 min):** Presentación de los resultados de $nDCG@10$ comparando BM25 frente a las variantes de ablación, métricas de cumplimiento de restricciones, desglose de latencias por subsistema, examen de los aciertos y fallos representativos y conclusiones técnicas.

---

## 8. Verificación Integral Previa a la Entrega

Antes de consolidar el envío a través del formulario oficial antes del cierre del sábado 21 de noviembre de 2026, el equipo debe verificar metódicamente cada uno de los siguientes puntos de control:

1. **Garantía del Catálogo y Consultas:** El catálogo final dispone de al menos 5 000 productos en español y 200 consultas de prueba (100 en train y 100 en test), habiéndose comprobado por aserción que el 100% de los productos anotados en dichas consultas se encuentran presentes en el catálogo.
2. **Aislamiento Estricto de Juicios de Relevancia:** Las etiquetas de juicio de relevancia de ESCI residen en estructuras aisladas y no son accesibles para el catálogo del RAG ni para las herramientas del agente.
3. **Determinismo del Backend:** La tienda simulada opera con semilla fija (`random_seed = 42`) y el método `reset_store` restaura el estado inicial del inventario de forma reproducible en todas las corridas.
4. **Evaluación Formal de Ablación:** Se cuenta con la tabla comparativa de $nDCG@10$ calculada sobre el conjunto de test con las ponderaciones oficiales, contrastando BM25 frente a la variante basada únicamente en títulos y la variante enriquecida con descripciones completas.
5. **Validación Determinista en Código:** Las reglas de presupuesto y verificación de existencias son procesadas en código nativo de Python, incluyendo la doble verificación en el momento exacto de la confirmación del pedido.
6. **Autenticidad de la Voz Clonada:** El audio de referencia cuenta con el consentimiento explícito documentado del integrante del grupo y el texto sintetizado por el asistente difiere en su totalidad de la referencia acústica.
7. **Batería Experimental Completa:** Se ejecutaron y registraron los 20 escenarios operativos, incluyendo los 6 casos conducidos por voz (con al menos tres en modalidad multi-turno) y su contrastación oracular mediante transcripción correcta.
8. **Métricas de Rendimiento y Latencia:** Se dispone del registro del tiempo de inicialización de modelos y del desglose por etapas ($T_{\text{ASR}}, T_{\text{RAG}}, T_{\text{LLM}}, T_{\text{TTS}}, T_{\text{Total}}$) para cada interacción.
9. **Cumplimiento de Pauta Audiovisual:** El video final respeta el límite máximo de 20 minutos y la distribución cronometrada exacta de 4, 4, 8 y 4 minutos entre sus cuatro bloques reglamentarios.
