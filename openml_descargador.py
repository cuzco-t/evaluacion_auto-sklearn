import openml
import numpy as np
import pandas as pd

from scipy import sparse
from result import Result

class OpenMLDescargador:
    """Carga datasets de OpenML a partir de un task_id."""

    def obtener_datos_tarea(self, task_id: int) -> Result[tuple[str, str, pd.DataFrame, pd.Series], str]:
        """
        Descarga el dataset asociado a una tarea de OpenML.

        Args:
            task_id: Identificador de la tarea en OpenML.

        Returns:
            Result ok con (nombre_dataset, descripcion, X, y) o fail con mensaje de error.
        """
        try:
            task = openml.tasks.get_task(task_id)
            dataset = task.get_dataset()
            target = dataset.default_target_attribute
            X, y, _, _ = dataset.get_data(target=target)
        except openml.exceptions.OpenMLServerException:
            return Result.fail("Problemas con el servidor de OpenML")
        except openml.exceptions.OpenMLPrivateDatasetError:
            return Result.fail("El dataset es privado")
        except Exception:
            return Result.fail("Error desconocido al obtener datos")

        # Convertir a DataFrame/Series si es necesario
        X = self._a_dataframe(X)
        y = self._a_serie(y)
        if X is None or y is None:
            return Result.fail("No se pudo convertir X o y a formato pandas")

        return Result.ok((dataset.name, dataset.description, X, y))

    def _a_dataframe(self, X):
        if isinstance(X, pd.DataFrame):
            return X
        if sparse.issparse(X):
            return pd.DataFrame.sparse.from_spmatrix(X)
        if isinstance(X, np.ndarray):
            return pd.DataFrame(X)
        return None

    def _a_serie(self, y):
        if y is None or isinstance(y, pd.Series):
            return y
        if isinstance(y, np.ndarray):
            return pd.Series(y)
        return None