# Laboratorio 4 · Asistente de compras por voz

La entrega combina un notebook de Colab, dos módulos Python, los resultados obtenidos al ejecutar el laboratorio y un video de hasta 20 minutos. El enunciado de la carpeta principal indica el 25 de noviembre como fecha de entrega.

## Ejecución en Colab

1. Carga `Laboratorio 4 - Asistente de compras por voz.ipynb` y selecciona un entorno con GPU. Sube `asistente.py` y `evaluacion_operativa.py` a `/content`, o clona la rama que contenga los tres archivos.
2. Ejecuta las celdas en orden. Los Parquet de ESCI se descargan del repositorio de Amazon Science. Las versiones de las bibliotecas principales están fijadas en la celda de instalación. La salida de configuración registra Python, CUDA y las versiones efectivamente cargadas.
3. Proporciona un WAV de 10 a 15 segundos con voz propia o autorizada y su transcripción exacta. Lee los términos de XTTS-v2 antes de cargar el modelo. La referencia se incorpora al ZIP de evidencia cuando se ejecuta la evaluación por voz.
4. En la evaluación, usa el modo `microfono`. Autoriza el acceso al micrófono, lee cada intervención y pulsa **Detener grabación** al terminar. Corrige la transcripción manual solicitada si tu grabación difiere del guion. Colab guarda las grabaciones, las respuestas y sus trazas.
5. El modo `archivos` repite los casos a partir de un manifiesto generado por la grabación anterior. Cada caso del 01 al 06 contiene una lista de objetos con las claves `audio` (ruta relativa al manifiesto) y `transcripcion` (texto manual de lo que se dijo). El modo `texto` ejecuta una comprobación preliminar de los 20 escenarios. Para este último puedes omitir las celdas de voz y demostración.

## Diseño y evaluación

- ESCI filtra español y la versión reducida de ranking. El catálogo conserva al menos 5.000 productos y cubre los candidatos anotados de las 100 consultas de desarrollo y las 200 de evaluación. Las etiquetas permanecen en la tabla de evaluación. Los resultados incluyen nDCG@10 sobre los candidatos anotados y sobre el catálogo completo.
- El recuperador combina BM25 y E5. El peso se fija en desarrollo para la evaluación final. La ablación compara representaciones densas con títulos frente a fichas enriquecidas. El notebook conserva resultados agregados y por consulta para estudiar errores.
- Qwen2.5-3B-Instruct usa cuantización NF4. Propone búsquedas, consultas de ficha, cambios de carrito y cotizaciones. Los datos de producto y los importes salen de la tienda. El controlador exige una cotización válida y una confirmación explícita para registrar pedidos. La tienda revalida stock y presupuesto e impide que una confirmación repetida cree otro pedido.
- Whisper-small transcribe en español. XTTS-v2 sintetiza en español con el WAV autorizado. E5 pasa a CPU durante la interacción y ECAPA-TDNN se ejecuta en CPU para limitar el uso de VRAM. El notebook registra las cargas de modelos y la duración de ASR, recuperación, agente y TTS. La latencia desde el fin de la grabación se mide en el navegador.
- El catálogo operativo de cuatro fichas permite reproducir las condiciones de stock, presupuesto y precio de los 20 casos. La demostración usa el catálogo ESCI. Los seis casos por voz se repiten con la transcripción manual, desde el mismo estado inicial. El caso de corrección usa un guion reproducible y las discrepancias reales de Whisper se cuentan con las demás métricas.
- La evaluación registra WER del ASR, errores por entidad anotada, WER y CER de retrotranscripción de TTS y similitud coseno de embeddings de voz. Revisa ejemplos de audio porque la retrotranscripción depende del reconocedor y la coincidencia de entidades con variantes léxicas puede ocultar errores de contexto.

## Análisis de resultados

Revisa el borrador generado en `analisis-resultados.md`. Abre `resumen.json`, `escenarios.json`, `recuperacion.csv` y `consultas-por-metodo.csv` del ZIP. Compara BM25, E5 e híbrido, y explica el efecto de quitar los atributos enriquecidos. Elige al menos cinco aciertos y cinco fallos observados. Para cada uno indica el caso, turno o consulta, el resultado esperado, el resultado obtenido, la causa según la traza y la limitación que revela. Si hay menos de cinco escenarios fallidos, examina errores por turno o consultas donde el primer resultado híbrido tiene una ganancia inferior a la mejor anotada. Contrasta voz y transcripción manual para localizar el origen de los fallos. Conserva las cifras originales y guarda el análisis junto al notebook antes de la entrega.

## Video de entrega

| Bloque | Duración | Evidencia |
|---|---:|---|
| Metodología | 4 min | Diseño, elección de modelos y protocolo de evaluación |
| Implementación | 4 min | Recuperación, agente, estado, validaciones y gestión de memoria |
| Demostración | 8 min | Compra confirmada y caso de aclaración o recuperación, con entrada y salida por voz. Muestra al menos una interacción completa con su espera real. |
| Resultados | 4 min | nDCG@10, ablación, métricas de voz y operación, errores y limitaciones |

Para la demostración, activa `EJECUTAR_DEMO` y vuelve a ejecutar esa celda para cada turno. Usa los códigos que el asistente muestre y confirma después de oír la cotización. Los archivos de voz, los resultados y el video deben provenir de la ejecución del grupo.

## Referencias técnicas

- [Datos oficiales de ESCI](https://github.com/amazon-science/esci-data)
- [Instalación de Coqui TTS](https://coqui-tts.readthedocs.io/en/latest/installation.html)
- [Compatibilidad de PyTorch y TorchCodec](https://github.com/meta-pytorch/torchcodec#compatibility-with-torch-versions)
- [Licencia de XTTS-v2](https://huggingface.co/coqui/XTTS-v2/blob/main/LICENSE.txt)
