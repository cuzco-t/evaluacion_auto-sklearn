from typing import Generic, TypeVar, Optional

V = TypeVar("V")  # Tipo del valor exitoso
E = TypeVar("E")  # Tipo del error


class Result(Generic[V, E]):
    __slots__ = ("_is_success", "_value", "_error")

    def __init__(self, is_success, value=None, error=None):
        self._is_success = is_success
        self._value = value
        self._error = error

    @staticmethod
    def ok(value=None):
        return Result(True, value=value)

    @staticmethod
    def fail(error=None):
        return Result(False, error=error)

    @property
    def is_success(self):
        return self._is_success

    @property
    def is_failure(self):
        return not self._is_success

    def get_value(self):
        if not self._is_success:
            raise ValueError(
                "Operación inválida: no se puede obtener el valor de un resultado fallido"
            )
        return self._value

    def get_error(self):
        if self._is_success:
            raise ValueError(
                "Operación inválida: no se puede obtener el error de un resultado exitoso"
            )
        return self._error

    def __repr__(self):
        if self._is_success:
            return f"Result.ok({self._value})"
        return f"Result.fail({self._error})"

    def unwrap(self):
        if not self._is_success:
            raise ValueError(self.get_error())
        return self._value

