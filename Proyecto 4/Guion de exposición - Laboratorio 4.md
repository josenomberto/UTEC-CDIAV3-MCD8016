# Guion de exposición del Laboratorio 4

**Presentan David, José Carlos y Alejandro, en ese orden.** El reloj corre desde el inicio del video. Cada diapositiva tiene un solo expositor. Las indicaciones entre corchetes son acciones de producción.

| Segmento | Tiempo | Diapositivas | Expositor |
|---|---|---|---|
| 1. Metodología y decisiones de diseño | 0:00–4:00 | 1–4 | David |
| 2. Implementación técnica y gestión de recursos | 4:00–8:00 | 5–7 | David (5) y José Carlos (6–7) |
| 3. Demostración en vivo por voz | 8:00–16:00 | 8–9 | José Carlos (8) y Alejandro (9) |
| 4. Resultados, análisis crítico y conclusiones | 16:00–20:00 | 10–14 | Alejandro |

## Segmento 1 · Metodología y decisiones de diseño · 0:00–4:00

### Diapositiva 1 · David · 0:00–0:25  
**Asistente de compras por voz**

Buenos días. Somos David, José Carlos y Alejandro. Presentamos un asistente que entiende una solicitud de compra en español, recomienda productos y registra un pedido después de que el usuario lo confirma. Vamos a explicar cómo lo diseñamos, mostrarlo en vivo y revisar qué pasó en las pruebas.

### Diapositiva 2 · David · 0:25–1:35  
**Qué ocurre en un turno de voz**

Imaginemos que alguien pide dos auriculares negros por menos de cien euros. El primer paso es convertir el audio en texto con Whisper. Ese texto activa la búsqueda de fichas que podrían responder a la solicitud. El agente mira esas fichas y lo que ya se habló para decidir si debe mostrar alternativas, preguntar algo o preparar una cotización. También consulta a la tienda para saber los precios y las existencias. Al final, XTTS lee la respuesta en voz alta.

[Recorrer el diagrama de izquierda a derecha. Señalar dónde se cruza la información del catálogo con la de la tienda.]

Si el usuario cambia la cantidad en el siguiente turno, el asistente tiene que recordar el color y el presupuesto, y actualizar solo la cantidad.

### Diapositiva 3 · David · 1:35–2:45  
**Por qué elegimos modelos locales**

Usamos Whisper-small para transcribir español y reducir el costo de inferencia frente a variantes más grandes. E5 nos permite comparar significados, incluso si la consulta y la ficha usan términos distintos. Qwen2.5-3B-Instruct se encarga de interpretar instrucciones y decidir qué herramientas llamar. XTTS-v2 genera la respuesta en español con una grabación de voz propia o autorizada.

Para Colab Pro elegimos cargar los pesos de Qwen en **4 bits con NF4**, usando FP16 para el cómputo. Eso reduce la memoria que ocupan los pesos y deja espacio para compartir la GPU con los modelos de voz. Toda la inferencia se hace en Colab. La disponibilidad de GPU depende de la sesión.

### Diapositiva 4 · David · 2:45–4:00  
**Cómo evaluamos las fichas enriquecidas**

BM25 sirve de referencia porque puntúa coincidencias de palabras. E5 ayuda cuando una consulta dice «auriculares sin cable» y la ficha usa «audífonos inalámbricos». Para medir el aporte de los atributos, comparamos E5 con solo el título frente a E5 con el título, marca, color, viñetas y descripción. El modelo, las consultas y la métrica se mantienen iguales entre ambas variantes. Así aislamos el efecto del texto disponible en la ficha.

También evaluamos una combinación de BM25 y E5. Elegimos el peso del híbrido en las 100 consultas de desarrollo y lo congelamos para evaluar con nDCG@10 en las 200 consultas de prueba.

## Segmento 2 · Implementación técnica y gestión de recursos · 4:00–8:00

### Diapositiva 5 · David · 4:00–6:00  
**Cómo aislamos las etiquetas del catálogo**

[Mostrar en el notebook la preparación de ESCI y las columnas que se usan para construir el índice.]

El catálogo tiene 7.908 productos e incluye los candidatos anotados de las consultas de prueba. Separamos las consultas por identificador. Las 100 de desarrollo se usan para elegir el peso de la recuperación y las 200 de prueba se reservan para evaluar.

Aquí conviene distinguir dos conjuntos de datos. El índice contiene los textos y atributos que el agente puede usar para buscar y explicar los productos. Las etiquetas de relevancia se guardan en la tabla de evaluación y sirven para comparar el orden recuperado con los juicios de ESCI. El agente consulta las fichas y los datos de la tienda. Las etiquetas permanecen fuera de sus herramientas.

[Mostrar una ficha del catálogo y la tabla de relevancia por separado. Luego ceder la palabra a José Carlos.]

### Diapositiva 6 · José Carlos · 6:00–7:20  
**La tienda verifica cada pedido**

[Mostrar en el notebook las funciones de herramientas y el estado de la tienda.]

La tienda simulada genera un estado reproducible con la semilla 42. En el código, el agente puede llamar a las funciones de búsqueda, consulta de producto, actualización de carrito, cotización y confirmación. Si hay varias alternativas y alguien dice «agrégalo», el agente pide aclaración antes de tocar el carrito.

La cotización calcula el total a partir de los precios y las cantidades de la tienda. Al confirmar, el código vuelve a validar stock y presupuesto. Si la compra es viable, actualiza pedido e inventario juntos. También detecta una confirmación repetida para que el mismo pedido quede registrado una sola vez.

### Diapositiva 7 · José Carlos · 7:20–8:00  
**Cómo administramos la VRAM**

[Mostrar la celda de cuantización y el traslado de E5 a CPU.]

En una sesión con GPU, Qwen usa NF4 y el reconocimiento de voz aprovecha FP16. E5 pasa a CPU durante la interacción y la comparación de voz con ECAPA también se ejecuta en CPU. Así la GPU queda disponible para el agente y los modelos de voz. Medimos por separado cuánto tarda cargar los modelos y cuánto tarda un turno cuando ya están cargados.

## Segmento 3 · Demostración en vivo por voz · 8:00–16:00

### Diapositiva 8 · José Carlos · 8:00–12:30  
**Caso 01 · Compra exitosa en varios turnos**

[Antes de grabar, ejecutar las celdas de modelos y del entorno controlado. Crear el agente del caso 01 con `VoiceShoppingAgent(tienda=tienda_pruebas, recuperar=recuperar_pruebas, planificar=planificador_local, sesion="01")` tras restaurar la tienda y usarlo en la celda de demostración en lugar de `asistente_catalogo`. Mantener el mismo agente entre turnos. Comprobar el micrófono.]

[En el video, mostrar un turno completo en una toma continua, desde que se habla hasta que termina de reproducirse la respuesta. Mantener visible la espera real desde que se detiene la grabación, sin acelerar ni cortar.]

Voy a buscar auriculares, agregar dos unidades y confirmar después de escuchar la cotización. En cada turno podremos contrastar lo que oye el asistente con lo que hace la tienda.

1. [Grabar y reproducir.] «Busca auriculares para comparar las opciones».
2. [Grabar. Mostrar la transcripción, la espera completa y la respuesta de audio.] «Agrega dos unidades del producto A1 al carrito».
3. [Grabar y reproducir.] «Confirmo el pedido».

En el ejemplo guardado, dos unidades a 20 euros dieron una cotización de 40 euros. El pedido se registró y el stock de A1 bajó de diez a ocho.

[Mostrar el total y el inventario reales de la ejecución en vivo. Los botones de la diapositiva reproducen la entrada y la respuesta guardadas del turno 01-02 como evidencia complementaria. Ceder la palabra a Alejandro.]

### Diapositiva 9 · Alejandro · 12:30–16:00  
**Caso 06 · Recuperación tras un cambio de stock**

[Restaurar la tienda y crear un agente nuevo con la sesión `"06"`. Buscar auriculares y pedir por voz dos A1. Tras la cotización, ejecutar `tienda_pruebas.simulate_stock_change("A1", 0)` y mostrar el cambio.]

Aquí la tienda ya había cotizado dos unidades de A1 por 40 euros. Vamos a simular que se agotan justo antes de la confirmación.

1. [Grabar y reproducir.] «Confirmo el pedido». Mostrar el bloqueo y la alternativa A2.
2. [Grabar y reproducir.] «Quita el producto A1 y agrega dos unidades del producto A2 al carrito». Mostrar la nueva cotización.
3. [Grabar, reproducir y repetir.] «Confirmo el pedido». Comprobar que al final hay un solo pedido.

En la evidencia, el pedido de dos A2 quedó en 50 euros. La primera confirmación de A1 se rechazó por falta de stock. La última confirmó que el pedido de A2 ya estaba registrado.

[Los botones reproducen la entrada y la respuesta guardadas del rechazo por stock, turno 06-03. Pasar a resultados al minuto 16:00.]

## Segmento 4 · Resultados, análisis crítico y conclusiones · 16:00–20:00

### Diapositiva 10 · Alejandro · 16:00–16:50  
**Qué método ordenó mejor los productos**

nDCG@10 premia que los resultados más relevantes aparezcan primero. Entre los candidatos anotados, BM25 con títulos obtuvo **0,6667**. E5 con títulos obtuvo **0,7130** y con fichas enriquecidas subió a **0,7256**. El híbrido logró **0,7210** entre los anotados y lideró la búsqueda en el catálogo completo con **0,6265**. La mejora de E5 en la ablación indica cuánto aportó añadir atributos en este conjunto.

### Diapositiva 11 · Alejandro · 16:50–17:30  
**El asistente resolvió 19 de 20 escenarios**

La carpeta de evidencias registra **19 de 20 escenarios resueltos**, un **95 %**, y cero incumplimientos de las restricciones verificadas. Los seis casos por voz abarcaron 21 turnos. Los seis se completaron con audio y también al repetirlos con la transcripción manual. Esa repetición permite comprobar cuánto afectó la transcripción al resultado final.

### Diapositiva 12 · Alejandro · 17:30–18:15  
**La síntesis de voz domina el tiempo de respuesta**

Con los modelos ya cargados, un turno tomó en promedio **12,06 segundos**. La transcripción tomó **0,46**, la recuperación **0,02**, el agente **2,76** y la síntesis **8,81 segundos**. La síntesis fue la etapa más lenta. El error de transcripción de la entrada fue **4,83 %** y hubo un error de producto en nueve menciones evaluadas. Estas cifras corresponden al tiempo de procesamiento. La espera desde el fin de la grabación se muestra en la demostración.

### Diapositiva 13 · Alejandro · 18:15–19:20  
**Qué explican los aciertos y los fallos**

Entre los aciertos, el caso 03 conservó el presupuesto al cambiar la cantidad. En el caso 05, Whisper confundió el código A1 con «aún no». El agente pidió aclaración y el usuario corrigió la solicitud. El único fallo operativo fue el caso 18. Con entrada de texto, se pidieron tres unidades de A1. Apareció un intento sobre A3 y el carrito quedó en dos unidades de A1.

El informe también registra cinco consultas en las que el primer resultado fue irrelevante aunque existía una coincidencia exacta. Por ejemplo, «movil infantil 8 años» tuvo nDCG@10 de **0,1017**. Es un problema del orden de búsqueda que puede aparecer incluso cuando el resto de la compra funciona.

### Diapositiva 14 · Alejandro · 19:20–20:00  
**Conclusiones técnicas**

Las fichas enriquecidas mejoraron E5 entre los candidatos anotados, y las validaciones permitieron detener una compra cuando cambió el stock y evitar un duplicado. Los resultados dejan tres frentes de mejora: interpretar cambios de carrito con varias instrucciones, ordenar mejor las consultas difíciles y reducir el tiempo de síntesis. El notebook y las evidencias permiten reproducir los resultados. Gracias.
