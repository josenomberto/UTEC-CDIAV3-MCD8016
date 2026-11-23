"""Escenarios reproducibles, evaluación de voz y exportación de evidencia."""

import json
import math
import re
import shutil
from pathlib import Path
from time import perf_counter

from asistente import Asistente, normalizar


BUSCAR = "Busca auriculares para comparar las opciones."
AGREGAR = "Agrega dos unidades del producto A1 al carrito."
CONFIRMAR = "Confirmo el pedido."

# Cada caso declara el resultado esperado antes de ejecutar el modelo.
# Los seis primeros se ejecutan con voz y se repiten con su transcripción manual.
CASOS = [
    ("01", "Compra válida", [BUSCAR, AGREGAR, CONFIRMAR], {"A1": 2}, (), None),
    ("02", "Aclaración", [BUSCAR, "Agrégalo al carrito."], None, ("aclarar",), {}),
    ("03", "Cambio de cantidad con presupuesto", [
        "Busca auriculares con presupuesto máximo de 70 euros.", AGREGAR,
        "Cambia la cantidad del producto A1 a una unidad.", CONFIRMAR,
    ], {"A1": 1}, (), None),
    ("04", "Agotado al agregar", [BUSCAR, "Agrega una unidad del producto A3 al carrito."],
     None, ("insufficient_stock", "buscar"), {}),
    ("05", "Corrección de transcripción", [
        BUSCAR, "Agrega cinco unidades del producto A1 al carrito.",
        "Corrijo la transcripción anterior: dije dos unidades del producto A1, no cinco.", CONFIRMAR,
    ], {"A1": 2}, (), None),
    ("06", "Caída de stock y doble confirmación", [
        BUSCAR, AGREGAR, CONFIRMAR,
        "Quita el producto A1 y agrega dos unidades del producto A2 al carrito.", CONFIRMAR, CONFIRMAR,
    ], {"A2": 2}, ("out_of_stock", "buscar", "duplicate_order"), None),
    ("07", "Pedido multiproducto", [
        BUSCAR, "Agrega una unidad del producto A1 y dos del producto A2.", CONFIRMAR,
    ], {"A1": 1, "A2": 2}, (), None),
    ("08", "Presupuesto insuficiente", [
        "Busca auriculares. Tengo un presupuesto máximo de 10 euros.", AGREGAR, CONFIRMAR,
    ], None, ("cotizar",), {"A1": 2}),
    ("09", "Cantidad negativa", [BUSCAR, "Agrega menos dos unidades del producto A1."],
     None, ("error",), {}),
    ("10", "Detalle verificable", [BUSCAR, "Consulta el detalle del producto A1."], None, ("detalle",), {}),
    ("11", "Atributo desconocido", [BUSCAR, "Consulta la marca del producto A1."],
     None, ("detalle",), {}),
    ("12", "Eliminación", [BUSCAR, AGREGAR, "Quita el producto A1 del carrito.", CONFIRMAR],
     None, ("removed",), {}),
    ("13", "Confirmación sin carrito", [CONFIRMAR], None, ("error",), {}),
    ("14", "Negación de confirmación", [BUSCAR, AGREGAR, "No confirmo el pedido.", CONFIRMAR],
     None, ("cancelar_confirmacion",), {"A1": 2}),
    ("15", "Cantidad superior al stock", [BUSCAR, "Agrega 99 unidades del producto A1."],
     None, ("insufficient_stock",), {}),
    ("16", "Comparación sin modificar carrito", [BUSCAR], None, ("buscar",), {}),
    ("17", "Reducción de presupuesto", [
        BUSCAR, AGREGAR, "Cambia mi presupuesto máximo a 5 euros y vuelve a cotizar.", CONFIRMAR,
    ], None, ("presupuesto", "cotizar"), {"A1": 2}),
    ("18", "Preferencia y cambio de cantidad", [
        "Busca auriculares azules; mi preferencia es el color azul.", AGREGAR,
        "Cambia la cantidad del producto A1 a tres unidades, mantén mi preferencia.",
    ], None, ("cotizar",), {"A1": 3}),
    ("19", "Rechazo de código inexistente", [BUSCAR, "Agrega dos unidades del producto Z999."],
     None, (), {}),
    ("20", "Sin permiso de compra", [BUSCAR, AGREGAR, "No hagas la compra todavía."],
     None, (), {"A1": 2}),
]


# Entidades manuales del guion. Las equivalencias aceptan dígitos y palabras.
ENTIDADES = {
    "01": {1: {"cantidad": ["dos|2"], "producto": ["a1|a uno|a un|a 1"]}},
    "03": {
        0: {"importe": ["70|setenta"]}, 1: {"cantidad": ["dos|2"], "producto": ["a1|a uno|a 1"]},
        2: {"cantidad": ["una|1"], "producto": ["a1|a uno|a 1"]},
    },
    "04": {1: {"cantidad": ["una|1"], "producto": ["a3|a tres|a 3"]}},
    "05": {
        1: {"cantidad": ["cinco|5"], "producto": ["a1|a uno|a 1"]},
        2: {"cantidad": ["dos|2", "cinco|5"], "producto": ["a1|a uno|a 1"]},
    },
    "06": {
        1: {"cantidad": ["dos|2"], "producto": ["a1|a uno|a 1"]},
        3: {"cantidad": ["dos|2"], "producto": ["a1|a uno|a 1", "a2|a dos|a 2"]},
    },
}


def evaluar_entidades(caso, indice, texto, exacto):
    texto = normalizar(texto)
    exacto = normalizar(exacto)
    errores = {}
    for tipo, patrones in ENTIDADES.get(caso, {}).get(indice, {}).items():
        errores[tipo] = []
        for patron in patrones:
            patron_completo = rf"\b(?:{patron})\b"
            if not re.search(patron_completo, exacto):
                raise ValueError(f"La transcripción manual {caso}/{indice + 1} no contiene {patron}.")
            errores[tipo].append({"variantes": patron, "error": not bool(re.search(patron_completo, texto))})
    return errores


def construir_tienda_pruebas(store_class):
    import pandas as pd

    catalogo = pd.DataFrame([
        {"product_id": "A1", "product_title": "Auriculares azules", "product_color": "azul"},
        {"product_id": "A2", "product_title": "Auriculares negros", "product_color": "negro"},
        {"product_id": "A3", "product_title": "Auriculares verdes", "product_color": "verde"},
        {"product_id": "C1", "product_title": "Cargador USB-C", "product_color": "blanco"},
    ]).fillna("")
    tienda = store_class(catalogo, random_seed=42)
    for pid, precio, stock in [("A1", 20.0, 10), ("A2", 25.0, 8), ("A3", 18.0, 0), ("C1", 15.0, 7)]:
        tienda.initial_inventory[pid].update(price=precio, stock=stock)
    tienda.reset_store(verbose=False)
    return tienda, catalogo


def _estado(tienda, sesion):
    return {
        "inventario": {pid: p["stock"] for pid, p in tienda.inventory.items()},
        "carrito": dict(tienda.carts.get(sesion, {})),
        "presupuesto": tienda.session_budgets.get(sesion),
        "pedidos": list(tienda.orders),
    }


def _verificar(caso, inicial, final, turnos, preferencias):
    cid, _, _, pedido_esperado, eventos_esperados, carrito_esperado = caso
    operaciones = [op for turno in turnos for op in turno["operaciones"]]
    eventos = {op["accion"] for op in operaciones} | {op.get("status") for op in operaciones}
    pedidos = final["pedidos"]
    vendido = {item["product_id"]: item["quantity"] for p in pedidos for item in p["items"]}
    infracciones = []
    if len(pedidos) > 1:
        infracciones.append("pedido duplicado")
    hubo_confirmacion = any(op["accion"] == "confirmar" and op.get("status") == "success" for op in operaciones)
    if pedidos and not hubo_confirmacion:
        infracciones.append("pedido sin confirmación")
    if any(p["max_budget"] is not None and p["total"] > p["max_budget"] for p in pedidos):
        infracciones.append("presupuesto excedido")
    inventario_esperado = dict(inicial["inventario"])
    if cid == "06":
        inventario_esperado["A1"] = 0
    for pedido in pedidos:
        for linea in pedido["items"]:
            inventario_esperado[linea["product_id"]] -= linea["quantity"]
    if min(inventario_esperado.values()) < 0:
        infracciones.append("venta sin stock")
    if final["inventario"] != inventario_esperado:
        infracciones.append("stock inconsistente o cambio parcial")
    diferencias = list(infracciones)
    if vendido != (pedido_esperado or {}):
        diferencias.append(f"Pedido esperado {pedido_esperado}, obtenido {vendido}.")
    faltantes = set(eventos_esperados) - eventos
    if faltantes:
        diferencias.append(f"Operaciones esperadas ausentes: {', '.join(sorted(faltantes))}.")
    if carrito_esperado is not None and final["carrito"] != carrito_esperado:
        diferencias.append(f'Carrito esperado {carrito_esperado}, obtenido {final["carrito"]}.')
    if cid in {"03", "08", "17"}:
        esperado = {"03": 70, "08": 10, "17": 5}[cid]
        if final["presupuesto"] != esperado:
            diferencias.append(f"Presupuesto esperado: {esperado}.")
    if cid == "11" and not any("marca desconocido" in normalizar(t["respuesta"]) for t in turnos):
        diferencias.append("La respuesta no declara desconocida la marca.")
    if cid == "18" and "azul" not in normalizar(preferencias):
        diferencias.append("No conservó la preferencia por azul.")
    return diferencias, infracciones


def ejecutar(caso, tienda, recuperar, planificador, directorio, modo="texto", voz=None, manifiesto=None):
    cid, nombre, frases, *_ = caso
    if modo != "texto" and manifiesto is None:
        raise ValueError(f"Falta el manifiesto de audio para {cid}.")
    tienda.reset_store(verbose=False)
    asistente = Asistente(tienda, recuperar, planificador, sesion=cid)
    inicial = _estado(tienda, cid)
    turnos = []
    for indice, frase in enumerate(frases):
        if cid == "06" and indice == 2:
            tienda.simulate_stock_change("A1", 0)
        info = manifiesto[indice] if manifiesto is not None else {}
        correcta = info.get("transcripcion", frase)
        if modo == "voz":
            entrada = Path(info["audio"])
            copia = directorio / "audios" / f"{cid}-{indice + 1:02}-entrada{entrada.suffix}"
            copia.parent.mkdir(parents=True, exist_ok=True)
            if entrada.resolve() != copia.resolve():
                shutil.copy2(entrada, copia)
            salida = copia.with_name(f"{cid}-{indice + 1:02}-respuesta.wav")
            traza, _ = voz.turno(asistente, copia, salida)
            traza["entidades"] = evaluar_entidades(cid, indice, traza["entrada"], correcta)
        else:
            traza = asistente.turno(correcta)
        traza["transcripcion_correcta"] = correcta
        turnos.append(traza)
    final = _estado(tienda, cid)
    diferencias, infracciones = _verificar(caso, inicial, final, turnos, asistente.estado.preferencias)
    return {
        "id": cid, "nombre": nombre, "modo": modo, "exito": not diferencias,
        "diferencias": diferencias, "infracciones": infracciones, "inicial": inicial,
        "final": final, "turnos": turnos, "preferencias_finales": asistente.estado.preferencias,
        "eventos_externos": [{"antes_del_turno": 3, "producto": "A1", "stock": 0}] if cid == "06" else [],
    }


def cargar_manifiesto(ruta):
    ruta = Path(ruta)
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    for cid, _, frases, *_ in CASOS[:6]:
        filas = datos.get(cid, [])
        if len(filas) != len(frases):
            raise ValueError(f"Caso {cid}: se requieren {len(frases)} grabaciones y transcripciones manuales.")
        for fila in filas:
            audio = ruta.parent / fila["audio"]
            if not audio.is_file() or not fila.get("transcripcion", "").strip():
                raise ValueError(f"Caso {cid}: falta audio o transcripción en {audio}.")
            fila["audio"] = str(audio)
    return datos


def grabar_colab(ruta):
    """Graba un turno real y devuelve el reloj del navegador al detenerlo."""
    import base64
    from google.colab.output import eval_js

    datos = eval_js("""
    (async () => {
      const stream = await navigator.mediaDevices.getUserMedia({audio: true});
      const boton = document.createElement('button');
      boton.textContent = 'Detener grabación';
      document.body.appendChild(boton);
      const recorder = new MediaRecorder(stream);
      const partes = [];
      recorder.ondataavailable = (event) => partes.push(event.data);
      let fin;
      try {
        const audio = await new Promise((resolve, reject) => {
          recorder.onerror = reject;
          recorder.onstop = () => {
            const lector = new FileReader();
            lector.onerror = reject;
            lector.onload = () => resolve(lector.result.split(',')[1]);
            lector.readAsDataURL(new Blob(partes, {type: recorder.mimeType}));
          };
          boton.onclick = () => { fin = performance.now(); recorder.stop(); };
          recorder.start();
        });
        return {audio, fin, tipo: recorder.mimeType};
      } finally {
        stream.getTracks().forEach(track => track.stop());
        boton.remove();
      }
    })()
    """)
    extension = ".ogg" if "ogg" in datos["tipo"] else ".mp4" if "mp4" in datos["tipo"] else ".webm"
    ruta = Path(ruta).with_suffix(extension)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_bytes(base64.b64decode(datos["audio"]))
    return ruta, datos["fin"]


def grabar_casos(tienda, recuperar, planificador, voz, directorio):
    """Graba y ejecuta los seis casos con medición desde el botón de parada."""
    from IPython.display import display, Markdown
    from google.colab.output import eval_js

    resultados = []
    manifiesto = {}
    for caso in CASOS[:6]:
        cid, nombre, frases, *_ = caso
        tienda.reset_store(verbose=False)
        asistente = Asistente(tienda, recuperar, planificador, sesion=cid)
        inicial = _estado(tienda, cid)
        turnos = []
        manifiesto[cid] = []
        for indice, frase in enumerate(frases):
            if cid == "06" and indice == 2:
                tienda.simulate_stock_change("A1", 0)
            display(Markdown(f"**{cid} · {nombre} · turno {indice + 1}**\n\nLee: «{frase}»"))
            entrada, fin = grabar_colab(directorio / "audios" / f"{cid}-{indice + 1:02}-entrada")
            salida = entrada.with_name(f"{cid}-{indice + 1:02}-respuesta.wav")
            traza, audio = voz.turno(asistente, entrada, salida)
            display(audio)
            traza["desde_fin_grabacion_s"] = max(0, (eval_js("performance.now()") - fin) / 1000)
            print(traza["respuesta"])
            correcta = input("Transcripción manual. Pulsa Enter si leíste la frase tal como aparece: ").strip() or frase
            traza["transcripcion_correcta"] = correcta
            traza["entidades"] = evaluar_entidades(cid, indice, traza["entrada"], correcta)
            turnos.append(traza)
            manifiesto[cid].append({"audio": str(entrada.relative_to(directorio)), "transcripcion": correcta})
        final = _estado(tienda, cid)
        diferencias, infracciones = _verificar(caso, inicial, final, turnos, asistente.estado.preferencias)
        resultados.append({
            "id": cid, "nombre": nombre, "modo": "voz", "exito": not diferencias,
            "diferencias": diferencias, "infracciones": infracciones, "inicial": inicial,
            "final": final, "turnos": turnos, "preferencias_finales": asistente.estado.preferencias,
            "eventos_externos": [{"antes_del_turno": 3, "producto": "A1", "stock": 0}] if cid == "06" else [],
        })
        (directorio / "manifiesto-grabaciones.json").write_text(
            json.dumps(manifiesto, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (directorio / "resultados-voz-parciales.json").write_text(
            json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return resultados, manifiesto


def evaluar_audio(resultados, voz):
    import torch
    import torchaudio
    from jiwer import cer, wer
    from speechbrain.inference.speaker import EncoderClassifier

    inicio = perf_counter()
    modelo = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", savedir="modelos/ecapa",
        run_opts={"device": "cpu"},
    )
    carga_s = perf_counter() - inicio

    def vector(ruta):
        wav, frecuencia = torchaudio.load(ruta)
        mono = torchaudio.functional.resample(wav.mean(0), frecuencia, 16000)
        with torch.inference_mode():
            return modelo.encode_batch(mono.unsqueeze(0)).flatten()

    referencia = vector(voz.referencia)
    for resultado in resultados:
        if resultado["modo"] != "voz":
            continue
        for turno in resultado["turnos"]:
            audio = turno["audio_salida"]
            retro = voz.transcribir(audio)
            salida_esperada = normalizar(turno["respuesta"])
            turno["calidad_voz"] = {
                "retrotranscripcion": retro,
                "wer_tts": wer(salida_esperada, normalizar(retro)),
                "cer_tts": cer(salida_esperada, normalizar(retro)),
                "secs": torch.nn.functional.cosine_similarity(referencia, vector(audio), dim=0).item(),
                "wer_asr": wer(normalizar(turno["transcripcion_correcta"]), normalizar(turno["entrada"])),
            }
    return carga_s


def resumir(resultados, cargas):
    def media(valores):
        datos = [v for v in valores if isinstance(v, (int, float)) and math.isfinite(v)]
        return sum(datos) / len(datos) if datos else None

    principales = [r for r in resultados if r["modo"] != "oracle"]
    voces = [r for r in resultados if r["modo"] == "voz"]
    oracle = [r for r in resultados if r["modo"] == "oracle"]
    turnos = [t for r in voces for t in r["turnos"]]
    entidades = {}
    for tipo in ("cantidad", "importe", "producto"):
        errores = [e["error"] for t in turnos for e in t.get("entidades", {}).get(tipo, [])]
        entidades[tipo] = {"errores": sum(errores), "evaluadas": len(errores), "tasa": media(errores)}
    return {
        "escenarios": len(principales),
        "exito_pct": 100 * sum(r["exito"] for r in principales) / len(principales),
        "incumplimientos": sum(len(r["infracciones"]) for r in principales),
        "exito_voz_pct": 100 * media([r["exito"] for r in voces]) if voces else None,
        "exito_oracle_pct": 100 * media([r["exito"] for r in oracle]) if oracle else None,
        "errores_asr_entidades": entidades,
        "voz": {k: media(t.get("calidad_voz", {}).get(k) for t in turnos)
                for k in ("wer_tts", "cer_tts", "secs", "wer_asr")},
        "carga_modelos_s": cargas,
        "latencias_medias_s": {k: media(t.get(k) for t in turnos)
                              for k in ("asr_s", "rag_s", "agente_s", "tts_s", "total_s", "desde_fin_grabacion_s")},
        "aciertos": [r["id"] for r in principales if r["exito"]],
        "fallos": [{"id": r["id"], "evidencia": r["diferencias"]} for r in principales if not r["exito"]],
    }


def escribir_analisis(ruta, resultados, consultas):
    """Extrae casos observados y explica qué downstream comparte cada medición."""
    principales = [r for r in resultados if r["modo"] != "oracle"]
    oraculo = {r["id"]: r for r in resultados if r["modo"] == "oracle"}
    lineas = [
        "# Análisis de resultados",
        "",
        "Los casos operativos reproducen condiciones con cuatro fichas. Las consultas de ESCI "
        "miden recuperación con candidatos anotados. La evaluación acústica depende del "
        "reconocedor usado para retranscribir la respuesta.",
        "",
    ]
    for exito, titulo in ((True, "Aciertos"), (False, "Fallos")):
        lineas.extend((f"## {titulo}", ""))
        cuenta = 0
        for r in (r for r in principales if r["exito"] == exito):
            diferencia = " ".join(r["diferencias"])
            if exito:
                operaciones = [op for t in r["turnos"] for op in t["operaciones"]]
                ultimo = operaciones[-1] if operaciones else {}
                diferencia = (
                    f'Cumplió el estado y las operaciones esperadas. Última operación: '
                    f'{ultimo.get("accion", "sin operación")} {ultimo.get("status", "")}. '
                    f'Pedidos: {len(r["final"]["pedidos"])}. Carrito final: {r["final"]["carrito"]}.'
                )
            elif r["id"] in oraculo:
                diferencia += (
                    " El caso pasó con la transcripción manual, lo que apunta a un efecto del ASR."
                    if oraculo[r["id"]]["exito"] else
                    " El caso también falló con la transcripción manual. Revisar decisión, herramientas y estado."
                )
            lineas.append(f'- Caso {r["id"]}, {r["nombre"]} ({r["modo"]}). {diferencia}')
            cuenta += 1
            if cuenta == 5:
                break
        for q in consultas:
            if cuenta == 5:
                break
            ganancia = q["ganancia_primer_resultado_hibrido"]
            mejor = q["mejor_ganancia_anotada"]
            if (ganancia == mejor) != exito:
                continue
            causa = (
                "La puntuación híbrida puso primero un producto con la mejor ganancia anotada."
                if exito else
                "La puntuación híbrida priorizó un producto con menor relevancia anotada."
            )
            lineas.append(
                f'- Consulta ESCI {q["query_id"]}. El producto {q["primer_pid_hibrido"]} '
                f'obtuvo ganancia {ganancia:g}, frente al máximo anotado {mejor:g}. {causa} '
                f'nDCG@10 en candidatos anotados: {q["ndcg10_anotados"]:.3f}.'
            )
            cuenta += 1
        if cuenta < 5:
            lineas.append(f'- Se observaron {cuenta} ejemplos de este tipo en los datos inspeccionados.')
        lineas.append("")
    ruta.write_text("\n".join(lineas), encoding="utf-8")
