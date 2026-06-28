import logging
import os
import time
import traceback
import threading
from typing import Optional

import numpy as np
import pandas as pd
import psutil
import psycopg2
from psycopg2 import sql
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    explained_variance_score,
    r2_score,
)
from sklearn.model_selection import train_test_split

try:
    import pynvml
    pynvml.nvmlInit()
    GPU_DISPONIBLE = True
except (ImportError, pynvml.NVMLError):
    GPU_DISPONIBLE = False

# Asegúrate de que openml_descargador.py y result.py estén en el path
from openml_descargador import OpenMLDescargador

# ----------------------------------------------------------------------
# Importaciones opcionales de las herramientas AutoML
try:
    from auto_sklearn2 import AutoSklearnClassifier, AutoSklearnRegressor
    AUTOSKLEARN_DISPONIBLE = True
except ImportError:
    AUTOSKLEARN_DISPONIBLE = False
    print("auto-sklearn no está instalado. Se omitirán sus ejecuciones.")

# ----------------------------------------------------------------------
# Configuración de logging
logging.basicConfig(
    # filename="auto_ml_evaluacion.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("evaluacion")

from dotenv import load_dotenv
load_dotenv()  # Carga variables de entorno desde .env

# ----------------------------------------------------------------------
# Conexión a PostgreSQL
def obtener_conexion():
    """Devuelve una conexión a la base de datos usando variables de entorno."""
    try:
        conn = psycopg2.connect(
            host=os.environ.get("DB_HOST"),
            port=os.environ.get("DB_PORT"),
            dbname=os.environ.get("DB_NAME"),
            user=os.environ.get("DB_USER"),
            password=os.environ.get("DB_PASSWORD"),
        )
        conn.autocommit = True
        return conn
    except Exception as e:
        logger.error(f"No se pudo conectar a PostgreSQL: {e}")
        raise

def asegurar_conexion(conn):
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
    except:
        conn = obtener_conexion()
    return conn

# ----------------------------------------------------------------------
# Inserción de resultados
INSERT_QUERY = sql.SQL("""
    INSERT INTO otros_automl (
        nombre_automl, task_id, nombre_dataset, fuente, tiempo,
        f1, accuracy, "precision", recall,
        mae, mse, rmse, medae, ev, r2,
        silhouette, calinski, davies,
        cpu_porcentaje, gpu_porcentaje, ram_mb
    ) VALUES (
        %s, %s, %s, %s, %s,
        %s, %s, %s, %s,
        %s, %s, %s, %s, %s, %s,
        NULL, NULL, NULL,
        %s, %s, %s
    )
""")

def guardar_resultado(conn, registro: dict):
    """Inserta un registro en la tabla otros_automl."""
    with conn.cursor() as cur:
        cur.execute(INSERT_QUERY, (
            registro["nombre_automl"],
            registro["task_id"],
            registro["nombre_dataset"],
            registro["fuente"],
            registro["tiempo"],
            registro["f1"],
            registro["accuracy"],
            registro["precision"],
            registro["recall"],
            registro["mae"],
            registro["mse"],
            registro["rmse"],
            registro["medae"],
            registro["ev"],
            registro["r2"],
            registro["cpu_porcentaje"],
            registro["gpu_porcentaje"],
            registro["ram_mb"],
        ))

# ----------------------------------------------------------------------
# Preparación de los datos
def cargar_y_dividir(task_id: int, tipo: str):
    """
    Descarga dataset, divide en train/test.
    Retorna (X_train, X_test, y_train, y_test, nombre_dataset) o lanza excepción.
    """
    descargador = OpenMLDescargador()
    resultado = descargador.obtener_datos_tarea(task_id)
    if not resultado.is_success:
        raise RuntimeError(f"Fallo en descarga: {resultado.get_error()}")

    nombre, _, X, y = resultado.unwrap()

    # Dividir (20% test)
    stratify = y if tipo == "clasificacion" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=912, stratify=stratify
    )
    return X_train, X_test, y_train, y_test, nombre

# ----------------------------------------------------------------------
# Monitor de recursos
class ResourceMonitor:
    """Monitorea CPU, RAM y GPU en un hilo de fondo durante la ejecución."""

    def __init__(self, intervalo=1.0):
        self.intervalo = intervalo
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        self._cpu_samples = []
        self._ram_samples = []
        self._gpu_samples = []

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _monitor(self):
        while self._running:
            with self._lock:
                self._cpu_samples.append(psutil.cpu_percent(interval=None))
                self._ram_samples.append(psutil.virtual_memory().used / (1024 * 1024))
                if GPU_DISPONIBLE:
                    try:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        self._gpu_samples.append(util.gpu)
                    except pynvml.NVMLError:
                        self._gpu_samples.append(0)
            time.sleep(self.intervalo)

    def get_peak_usage(self):
        with self._lock:
            cpu_peak = max(self._cpu_samples) if self._cpu_samples else 0
            ram_peak = max(self._ram_samples) if self._ram_samples else 0
            gpu_peak = max(self._gpu_samples) if self._gpu_samples else 0
        return {
            "cpu_porcentaje": round(cpu_peak, 2),
            "ram_mb": round(ram_peak, 2),
            "gpu_porcentaje": round(gpu_peak, 2),
        }

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


# ----------------------------------------------------------------------
# Funciones de evaluación por herramienta

def evaluar_autosklearn(tipo: str, X_train, y_train, X_test, y_test):
    """Entrena y evalúa auto-sklearn 2. Retorna (métricas_dict, tiempo, recursos_dict)."""
    # Definir tipos de features
    feat_types = []
    for col in X_train.columns:
        if X_train[col].dtype.name in ("object", "category"):
            feat_types.append("Categorical")
        else:
            feat_types.append("Numerical")

    if tipo == "clasificacion":
        automl = AutoSklearnClassifier(
            random_state=912,
            time_limit=1200
        )
    else:
        automl = AutoSklearnRegressor(
            random_state=912,
            time_limit=1200
        )

    inicio = time.perf_counter()
    with ResourceMonitor(intervalo=0.5) as monitor:
        automl.fit(X_train.copy(), y_train.copy())
        y_pred = automl.predict(X_test)
    tiempo = time.perf_counter() - inicio
    recursos = monitor.get_peak_usage()

    metricas = calcular_metricas(tipo, y_test, y_pred)
    return metricas, tiempo, recursos

# ----------------------------------------------------------------------
# Cálculo de métricas
def calcular_metricas(tipo: str, y_true, y_pred):
    """Calcula todas las métricas necesarias. Las no aplicables se ponen en None."""
    met = {
        "f1": None, "accuracy": None, "precision": None, "recall": None,
        "mae": None, "mse": None, "rmse": None, "medae": None,
        "ev": None, "r2": None,
    }
    if tipo == "clasificacion":
        # Weighted para soportar múltiples clases
        met["accuracy"] = float(accuracy_score(y_true, y_pred))
        met["f1"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        met["precision"] = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
        met["recall"] = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    else:  # regresion
        met["mae"] = float(mean_absolute_error(y_true, y_pred))
        met["mse"] = float(mean_squared_error(y_true, y_pred))
        met["rmse"] = float(np.sqrt(met["mse"]))
        met["medae"] = float(median_absolute_error(y_true, y_pred))
        met["ev"] = float(explained_variance_score(y_true, y_pred))
        met["r2"] = float(r2_score(y_true, y_pred))
    return met

def get_metricas_error():
    """Retorna un diccionario de métricas con valor -1111 para indicar error."""
    return {
        "f1": -1111,
        "accuracy": -1111,
        "precision": -1111,
        "recall": -1111,
        "mae": -1111,
        "mse": -1111,
        "rmse": -1111,
        "medae": -1111,
        "ev": -1111,
        "r2": -1111,
        "cpu_porcentaje": -1111,
        "gpu_porcentaje": -1111,
        "ram_mb": -1111,
    }

# ----------------------------------------------------------------------
# Procesamiento de un archivo
def procesar_archivo(ruta: str, tipo: str, conn):
    """
    Lee task_ids de 'ruta' y ejecuta las dos herramientas.
    """
    fuente = os.path.splitext(os.path.basename(ruta))[0]
    with open(ruta, "r") as f:
        lineas = [line.strip() for line in f if line.strip()]

    for task_id_str in lineas:
        try:
            task_id = int(task_id_str)
        except ValueError:
            logger.warning(f"Línea no numérica ignorada en {ruta}: {task_id_str}")
            continue

        logger.info(f"Iniciando task_id={task_id} ({tipo})")
        X_train, X_test, y_train, y_test, nombre_dataset = None, None, None, None, ""

        try:
            X_train, X_test, y_train, y_test, nombre_dataset = cargar_y_dividir(task_id, tipo)
        except Exception as e:
            logger.error(f"Error al cargar datos del task {task_id}: {e}\n{traceback.format_exc()}")
            continue


        # Probar auto-sklearn 2
        if AUTOSKLEARN_DISPONIBLE:
            try:
                metricas, tiempo, recursos = evaluar_autosklearn(tipo, X_train, y_train, X_test, y_test)
                registro = {
                    "nombre_automl": "auto_sklearn2",
                    "task_id": task_id,
                    "nombre_dataset": nombre_dataset,
                    "fuente": fuente,
                    "tiempo": round(tiempo, 2),
                    **metricas,
                    **recursos,
                }
                try:
                    conn = asegurar_conexion(conn)
                    guardar_resultado(conn, registro)
                    logger.info(f"Guardado OK: auto_sklearn2, task {task_id}")
                except Exception as e:
                    logger.error(f"Error al insertar en DB (auto_sklearn2, task {task_id}): {e}")
            except Exception as e:
                logger.error(f"Fallo en auto-sklearn para task {task_id}: {e}\n{traceback.format_exc()}")
                # Registrar en BD con métricas de error (-1111)
                metricas_error = get_metricas_error()
                registro_error = {
                    "nombre_automl": "auto_sklearn2",
                    "task_id": task_id,
                    "nombre_dataset": nombre_dataset,
                    "fuente": fuente,
                    "tiempo": 0,  # Sin tiempo ya que falló
                    **metricas_error,
                }
                try:
                    conn = asegurar_conexion(conn)
                    guardar_resultado(conn, registro_error)
                    logger.info(f"Guardado REGISTRO DE ERROR: auto_sklearn2, task {task_id} con métricas=-1111")
                except Exception as db_error:
                    logger.error(f"Error al insertar registro de error en DB (auto_sklearn2, task {task_id}): {db_error}")
        else:
            logger.info(f"auto-sklearn no disponible, se omite task {task_id}")

# ----------------------------------------------------------------------
# Punto de entrada
def main():
    archivos = [
        ("data/openml-cc18.txt", "clasificacion"),
        ("data/openml-ctr23.txt", "regresion"),
    ]

    conn = None
    try:
        conn = obtener_conexion()
        logger.info("Conexión a base de datos establecida.")
    except Exception as e:
        print(f"Error al conectar a la base de datos: {e}")
        logger.critical("No se puede continuar sin base de datos.")
        return

    try:
        for archivo, tipo in archivos:
            if not os.path.exists(archivo):
                logger.warning(f"Archivo no encontrado: {archivo}, se omite.")
                continue
            logger.info(f"Procesando archivo {archivo} ({tipo})")
            try:
                procesar_archivo(archivo, tipo, conn)
            except Exception as e:
                logger.critical(f"Error inesperado en archivo {archivo}: {e}")
    finally:
        if GPU_DISPONIBLE:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
        if conn:
            conn.close()
            logger.info("Conexión a base de datos cerrada.")

if __name__ == "__main__":
    main()
