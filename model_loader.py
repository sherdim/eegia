"""
Реестр моделей для временных рядов (sktime).

Использование:
    from model_registry import get_model, list_models
    model = get_model("tsf")
"""

from typing import Dict, Any, Callable
from warnings import warn
from inspect import signature

# Фильтр параметров
def _filter_kwargs(cls, params: dict):
    """Удаляет неподдерживаемые параметры по сигнатуре конструктора"""
    sig = signature(cls.__init__)
    valid_keys = set(sig.parameters.keys())
    return {k: v for k, v in params.items() if k in valid_keys}

# Определяем безопасное число потоков
def _get_safe_n_jobs():
    """Возвращает безопасное значение n_jobs для sktime"""
    try:
        from multiprocessing import cpu_count
        # Используем не больше 8 потоков
        return min(8, max(1, cpu_count()))
    except Exception:
        return 1
    
def _get_model_parameters_list(cls):
    """Получить список доступных параметров модели"""
    if cls is None:
        return []
    
    try:
        sig = signature(cls.__init__)
        parameters = []
        
        for name, param in sig.parameters.items():
            if name == 'self':
                continue
                
            param_info = name
            if param.default != param.empty:
                param_info += f" (по умолчанию: {param.default})"
            else:
                param_info += " [обязательный]"
                
            parameters.append(param_info)
        
        return parameters
    
    except Exception:
        return []
    
# ИМПОРТ МОДЕЛЕЙ
def _import_rocket():
    try:
        from sktime.classification.kernel_based import RocketClassifier
        return RocketClassifier
    except ImportError:
        warn("ROCKET недоступен")
        return None
       
def _import_tsf():
    try:
        from sktime.classification.interval_based import TimeSeriesForestClassifier
        return TimeSeriesForestClassifier
    except ImportError:
        warn("TSF недоступен")
        return None

def _import_weasel():
    try:
        from sktime.classification.dictionary_based import MUSE
        return MUSE
    except ImportError:
        warn("MUSE недоступен")
        return None

def _import_stc():
    try:
        from sktime.classification.shapelet_based import ShapeletTransformClassifier
        return ShapeletTransformClassifier
    except ImportError:
        warn("STC недоступен")
        return None


# РЕЕСТР МОДЕЛЕЙ
# Ключ: имя модели (для использования в коде)
# Значение: функция, возвращающая экземпляр модели

_MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "ROCKET": {
        "class": _import_rocket(), 
        "parameters": _get_model_parameters_list(_import_rocket()),
        "description": "1"
    },
    "TSF": {
        "class": _import_tsf(), 
        "parameters": _get_model_parameters_list(_import_tsf()),
        "description": "2"
    },
    "MUSE": {
        "class": _import_weasel(), 
        "parameters": _get_model_parameters_list(_import_weasel()),
        "description": "3"
    },
    "STC": {
        "class": _import_stc(), 
        "parameters": _get_model_parameters_list(_import_stc()),
        "description": "4"
    },
}

# Удаляем недоступные модели
_MODEL_REGISTRY = {
    name: meta for name, meta in _MODEL_REGISTRY.items()
    if meta["class"] is not None
}


def get_model(model_name: str, *args, **kwargs):
    """
    Возвращает экземпляр модели по её имени (без учёта регистра).
    Автоматически подставляет random_state и n_jobs, если они не указаны пользователем.
    """
    model_name = model_name.upper()
    if model_name not in _MODEL_REGISTRY:
        available = ", ".join(_MODEL_REGISTRY.keys())
        raise ValueError(f"Модель '{model_name}' не найдена. Доступные: {available}")
    
    cls = _MODEL_REGISTRY[model_name]["class"]
    
    # дефолтные параметры
    defaults = {"random_state": 42, "n_jobs": _get_safe_n_jobs()}

    # пользовательские имеют приоритет
    defaults.update(kwargs)

    # фильтруем параметры
    params = _filter_kwargs(cls, defaults)
    print(params)
    return cls(*args, **params)


def list_models() -> list:
    """Возвращает список доступных моделей"""
    return list(_MODEL_REGISTRY.keys())

def model_description(model_name: str) -> str:
    params_list = _MODEL_REGISTRY[model_name]["parameters"]
    description = _MODEL_REGISTRY[model_name]["description"]
    parameters = "\n".join(f"- {param}" for param in params_list)
    # return  description + "\n" + parameters
    return parameters