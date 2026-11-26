# Guion de exposición del Laboratorio 4

**Orden de exposición**  
David, José Carlos y Alejandro

| Expositor | Tiempo | Diapositivas | Contenido |
|---|---:|---|---|
| David | 0:00–6:00 | 1–5 | Metodología, arquitectura y recuperación |
| José Carlos | 6:00–13:00 | 6–7 | Validaciones y demostración de una compra |
| Alejandro | 13:00–20:00 | 7–12 | Cambio de stock, resultados y cierre |

El reparto conserva los cuatro bloques del video. La metodología ocupa 4 minutos, la implementación 4, la demostración 8 y los resultados 4. Las indicaciones entre corchetes son acciones para el expositor.

## 1. David · Metodología, arquitectura y recuperación

### 0:00–4:00 · Del ejemplo de compra al catálogo (diapositivas 1 a 3)

Buenos días. Somos David, José Carlos y Alejandro. Para el Laboratorio 4 desarrollamos un asistente que recibe una solicitud de compra por voz, busca productos y puede registrar un pedido en una tienda simulada después de que el usuario lo confirme.

[Diapositiva 1. Señalar el resultado de la evaluación en la portada.]

En la ejecución que vamos a presentar, el asistente resolvió 19 de 20 escenarios, incluidos los seis casos por voz. Primero veremos cómo funciona. Luego escucharemos dos ejemplos y revisaremos qué salió bien y dónde falló.

Pensemos en una solicitud como «Busca dos auriculares negros por menos de cien euros». Para responder, el asistente tiene que entender el producto, la cantidad, el color y el presupuesto. Luego debe revisar qué opciones ofrece el catálogo y consultar cuáles tienen stock y a qué precio. Si el usuario después dice «Mejor quiero solo uno», el sistema debe cambiar la cantidad y conservar el resto de la solicitud.

[Señalar los tres pasos de la diapositiva 2.]

El cambio de cantidad afecta la cotización. El asistente recalcula el total, presenta el nuevo importe y espera que el usuario confirme. Si los auriculares se agotan en ese intervalo, la tienda bloquea el pedido y el asistente puede buscar otra opción.

Ese ejemplo resume las dos fuentes de información que usamos. El catálogo contiene las fichas de los productos, con datos como título, descripción, marca o color cuando están disponibles. La tienda simulada contiene los precios, el inventario, los carritos y los pedidos. Así, el asistente puede recomendar un producto por sus características y consultar por separado si la compra es viable. Cuando la ficha carece de un atributo, el asistente debe tratarlo como desconocido.

Para el catálogo usamos Amazon Shopping Queries, también conocido como ESCI. Tomamos la versión reducida para ranking y filtramos los registros en español. El catálogo quedó con **7.908 productos**, por encima del mínimo de 5.000 del enunciado, e incluye todos los productos anotados para las consultas seleccionadas.

Separamos las consultas por identificador. Usamos 100 de desarrollo para ajustar el recuperador y dejamos 200 de prueba para la evaluación final. Las etiquetas ESCI indican si un producto coincide exactamente con la consulta, si puede sustituirlo, si lo complementa o si es irrelevante. Esas etiquetas nos permiten evaluar el orden de los resultados y se mantienen fuera de las herramientas que consulta el agente.

Para repetir las pruebas bajo las mismas condiciones, usamos la semilla 42. Podemos restaurar el estado inicial de la tienda y comprobar qué cambia cuando modificamos la entrada o el método de recuperación.

### 4:00–6:00 · Qué ocurre en un turno de voz (diapositivas 4 y 5)

Sigamos la consulta de los auriculares por el diagrama. Whisper-small convierte lo que decimos en texto. El recuperador busca fichas que puedan responder a esa solicitud. Qwen usa esas fichas y lo que ya se dijo en la conversación para decidir si corresponde ofrecer productos, hacer una pregunta o preparar una cotización. Por último, XTTS-v2 lee la respuesta en voz alta a partir de una grabación propia o autorizada. El texto de la respuesta es nuevo.

La inferencia se ejecuta en Google Colab. Para administrar la memoria de la GPU, usamos Qwen cuantizado a cuatro bits y pasamos E5 a CPU durante la interacción.

En la diapositiva 5 se ve por qué usamos dos métodos. **BM25** da peso a las palabras que coinciden, lo que sirve para encontrar una marca o un modelo. Si alguien busca «auriculares sin cable» y la ficha dice «audífonos inalámbricos», **E5** ayuda a reconocer la similitud y **FAISS** localiza las fichas más cercanas. Ponemos ambos puntajes en una escala común y damos un peso de 20 % a BM25 y 80 % a E5. Elegimos esos pesos con las consultas de desarrollo y los mantuvimos fijos en la prueba.

También hicimos una ablación para medir cuánto aporta la información adicional de la ficha. Comparamos E5 usando solo el título frente a E5 con el título, la marca, el color, las viñetas y la descripción. Las dos variantes se evaluaron sobre las mismas 200 consultas.

**Pase a José Carlos**  
Hasta aquí vimos cómo encontramos los productos. José Carlos va a explicar cómo el asistente convierte esos resultados en una cotización y qué comprueba antes de registrar el pedido.

## 2. José Carlos · Validaciones y compra por voz

### 6:00–8:00 · Qué verifica la tienda antes de comprar (diapositiva 6)

Gracias, David. El agente puede buscar productos, consultar sus fichas, modificar el carrito, cotizar y registrar pedidos. La operación depende de lo que el usuario pida y de la información que devuelve la tienda en cada turno.

Por ejemplo, si el usuario dice «agrega dos» y hay varias opciones en la conversación, el asistente debe aclarar a qué producto se refiere. Cuando el producto ya está identificado, el código verifica que la cantidad sea válida y que haya suficientes unidades. También calcula el total y comprueba el presupuesto.

[Recorrer la tabla desde «Interpretar» hasta «Actualizar».]

La cotización y la confirmación son dos momentos distintos. Al cotizar, el asistente comunica los productos, las cantidades y el importe. Cuando el usuario confirma, la tienda vuelve a revisar las condiciones vigentes. Si todo sigue siendo válido, registra el pedido y descuenta las unidades en una sola operación. Si alguna condición falla, bloquea la compra. Y si el usuario repite una confirmación ya procesada, el sistema conserva el mismo pedido.

### 8:00–13:00 · Compra válida (diapositiva 7 y notebook)

En el caso 01 empezamos buscando auriculares. Elegimos dos unidades de los auriculares azules, que tienen código A1 y cuestan 20 euros por unidad. Vamos a escuchar el turno en el que se agrega el producto al carrito.

[En la diapositiva 7, reproducir «01 · Entrada» y «01 · Respuesta». Si el reproductor no está disponible, abrir `01-02-entrada.wav` y `01-02-respuesta.wav` en la carpeta `evidencias_exposicion_asistente_voz/audios`.]

La respuesta cotiza dos unidades por 40 euros y espera la confirmación. En el siguiente turno, la tienda registró el pedido número 1 y las existencias de A1 bajaron de diez a ocho. Podemos comprobarlo en el estado inicial y final de la traza.

[Para el video, pasar a Colab y mostrar al menos un turno de principio a fin con el micrófono. En el catálogo controlado del caso 01, pedir «Agrega dos unidades del producto A1 al carrito» después de buscar los auriculares. Desde que termina la grabación, dejar visible la espera real hasta que la respuesta esté lista y reproducirla. Continuar con «Confirmo el pedido» y mostrar el estado final.]

**Pase a Alejandro**  
Ahora Alejandro va a repetir la parte de la confirmación bajo otra condición. El producto tendrá stock al cotizar, pero se agotará antes de que el usuario confirme.

## 3. Alejandro · Recuperación ante un cambio de stock y resultados

### 13:00–16:00 · Cambio de stock (diapositiva 7 y notebook)

Gracias, José Carlos. En el caso 06 se cotizaron los mismos dos auriculares A1 por 40 euros. Antes de que el usuario confirmara, el stock de A1 bajó a cero. Escuchemos qué ocurrió al intentar comprar.

[Reproducir «06 · Entrada» y «06 · Respuesta» desde la diapositiva 7. Si hace falta, abrir `06-03-entrada.wav` y `06-03-respuesta.wav` en la carpeta de evidencias.]

La tienda volvió a consultar el inventario y bloqueó el registro. El asistente explicó que el stock había cambiado y ofreció A2. El usuario retiró A1 y eligió dos unidades de A2, que se volvieron a cotizar por 50 euros. La primera confirmación válida creó un pedido. La siguiente devolvió el aviso de que ese pedido ya estaba registrado.

[Mostrar en la traza el rechazo por falta de stock, la nueva cotización y el estado final con un solo pedido de dos A2.]

### 16:00–20:00 · Cómo leer los resultados y qué entregar (diapositivas 8 a 12)

[Diapositiva 8. Señalar primero la columna de candidatos anotados.]

Para leer esta tabla, pensemos en los diez primeros resultados. **nDCG@10** sube cuando los productos más relevantes aparecen arriba. Cuanto más se acerca a uno, mejor es el orden. Entre los candidatos anotados, BM25 con solo título obtuvo **0,6667**. E5 con títulos obtuvo **0,7130**, y al añadir los atributos y la descripción subió a **0,7256**.

El híbrido obtuvo **0,7210** entre los anotados. Al buscar en todo el catálogo, obtuvo el mayor puntaje de esa columna, **0,6265**, con un tiempo medio de **33,32 milisegundos** por consulta.

[Diapositiva 9. Señalar los resultados con audio y con transcripción manual.]

En la ejecución de la carpeta de evidencias se completaron **19 de 20 escenarios**, es decir, el **95 %**, sin incumplimientos registrados. Los seis casos por voz abarcaron 21 turnos, cada uno con un audio de entrada y uno de respuesta. Se completaron los seis con audio y los seis al repetirlos con la transcripción manual.

[Diapositiva 10. Señalar la barra de síntesis de voz.]

Con los modelos ya cargados, el procesamiento tomó en promedio **12,06 segundos** por turno. **8,81 segundos** se usaron para generar la voz, casi tres cuartos del total. El error de transcripción de la entrada fue de **4,83 %**. En los datos clave detectamos un error de producto en nueve menciones evaluadas. La carpeta registra el tiempo de cómputo. La espera completa desde el fin de la grabación se muestra aparte en Colab.

[Diapositiva 11. Señalar los casos 03, 05 y 18 y la consulta de ejemplo.]

El caso 03 muestra cómo el asistente pasó de dos unidades a una y mantuvo el límite de 70 euros. En el caso 05, Whisper interpretó el código A1 como «aún no». El asistente pidió aclaración y el usuario corrigió el pedido antes de confirmarlo. Ambos casos se resolvieron.

El único fallo operativo fue el caso 18. El usuario pidió tres unidades de A1 por texto, la traza registra una operación sobre A3 y el carrito quedó con dos A1. La evidencia también documenta cinco fallos de búsqueda. Por ejemplo, «movil infantil 8 años» tuvo un primer resultado irrelevante aunque existía una coincidencia exacta. Su nDCG@10 fue 0,1017.

[Diapositiva 12. Cerrar con la evidencia entregable.]

El notebook reúne la configuración y los pasos para reproducir los casos. La carpeta de evidencias permite revisar los audios, las transcripciones, las operaciones y el estado final de la tienda. Los resultados dejan dos puntos para mejorar: la actualización del carrito en solicitudes compuestas y el orden de los resultados en consultas difíciles.

Gracias.
