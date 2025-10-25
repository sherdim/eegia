import os
import pickle
import traceback
import json
from datetime import datetime
import gc

import dash
from dash import dcc, html, Input, Output, State, DiskcacheManager
import dash_bootstrap_components as dbc
import diskcache

import plotly.graph_objects as go
import pandas as pd
import numpy as np

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sktime.transformations.panel.compose import ColumnConcatenator

# Твои модули
from dataset_loader import load_dataset, datasets_list as dtl
from model_loader import get_model, list_models, model_description

dash.register_page(__name__, path="/training", name="Обучение моделей")

# === Настройки ===
SAVE_FOLDER = "saved_annotations"
MAX_SEGMENT_LENGTH_DEFAULT = 500
TEST_SIZE_DEFAULT = 0.3
APPLY_FILTER_DEFAULT = True
APPLY_CONCAT_DEFAULT = False  # по умолчанию выключено

# === DiskCache для фоновых задач ===
cache = diskcache.Cache("./cache")
background_callback_manager = DiskcacheManager(cache)

# === Вспомогательные функции ===

def get_available_annotation_files(dataset_name: str):
    """Получить список доступных файлов аннотаций для датасета"""
    if not os.path.exists(SAVE_FOLDER):
        return []
    
    annotation_files = []
    for fname in os.listdir(SAVE_FOLDER):
        if fname.startswith(f"{dataset_name}_annotations_"):
            annotation_files.append(fname)
    
    return sorted(annotation_files)

def load_annotations_from_file(filename: str):
    """Загрузить аннотации из конкретного файла"""
    if not filename:
        raise ValueError("Не указан файл аннотаций")
    
    fpath = os.path.join(SAVE_FOLDER, filename)
    annotations = []
    
    if filename.endswith('.json'):
        with open(fpath, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                if isinstance(data, dict) and 'annotations' in data:
                    data = data['annotations']
                if isinstance(data, list):
                    annotations.extend(data)
            except Exception as e:
                raise ValueError(f"Ошибка чтения JSON файла: {e}")
    elif filename.endswith('.csv'):
        try:
            df = pd.read_csv(fpath)
            cols = [c.lower() for c in df.columns]
            df.columns = cols
            required = {'onset', 'duration', 'description'}
            if not required.issubset(set(cols)):
                raise ValueError("CSV файл должен содержать колонки: onset, duration, description")
            for _, r in df.iterrows():
                annotations.append({
                    'onset': float(r['onset']),
                    'duration': float(r['duration']),
                    'description': str(r['description']).strip()
                })
        except Exception as e:
            raise ValueError(f"Ошибка чтения CSV файла: {e}")
    else:
        raise ValueError(f"Неподдерживаемый формат файла: {filename}")
    
    if not annotations:
        raise ValueError("Файл не содержит аннотаций")
    
    # Удаление дубликатов
    seen = set()
    unique = []
    for a in annotations:
        key = (round(float(a['onset']), 3), round(float(a['duration']), 3), a['description'])
        if key not in seen:
            seen.add(key)
            unique.append({
                'onset': float(a['onset']),
                'duration': float(a['duration']),
                'description': a['description']
            })
    
    return unique

def extract_segments(raw, annotations, max_length=None, apply_filter=False):
    """Извлечение сегментов ЭЭГ на основе аннотаций"""
    if apply_filter:
        try:
            raw = raw.copy().filter(1, 40, fir_design='firwin')
        except Exception:
            pass

    sfreq = raw.info['sfreq']
    X, y = [], []
    for ann in annotations:
        onset = float(ann['onset'])
        duration = float(ann['duration'])
        label = ann['description']
        if duration <= 0:
            continue
        start_samp = int(onset * sfreq)
        end_samp = int((onset + duration) * sfreq)
        if end_samp <= start_samp:
            continue
        try:
            data, _ = raw[:, start_samp:end_samp]
        except Exception:
            continue
        if max_length:
            if data.shape[1] > max_length:
                data = data[:, :max_length]
            elif data.shape[1] < max_length:
                pad = np.zeros((data.shape[0], max_length - data.shape[1]))
                data = np.hstack([data, pad])
        df = pd.DataFrame(data.T, columns=raw.ch_names)
        X.append(df)
        y.append(label)
    return X, y

def parse_kwargs_string(kwargs_text: str):
    """Парсинг параметров модели из текстового формата"""
    if not kwargs_text or not kwargs_text.strip():
        return {}
    
    kwargs = {}
    lines = kwargs_text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
            
        if '=' in line:
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()
            
            # Парсинг значений
            if value.lower() == 'true':
                value = True
            elif value.lower() == 'false':
                value = False
            elif value.lower() == 'none':
                value = None
            elif value.isdigit():
                value = int(value)
            elif value.replace('.', '').isdigit():
                value = float(value)
            elif value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            elif value.startswith("'") and value.endswith("'"):
                value = value[1:-1]
                
            kwargs[key] = value
    
    return kwargs



available_datasets = list(dtl.keys())
available_models = list_models()

layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H3("Обучение моделей на размеченных ЭЭГ"), width=8)
    ], align="center", className="my-2"),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Конфигурация"),
                dbc.CardBody([
                    dbc.Label("Выберите датасет"),
                    dcc.Dropdown(
                        id="dataset-select",
                        options=[{"label": f"{k} — {v}", "value": k} for k, v in dtl.items()],
                        value=available_datasets[0] if available_datasets else None,
                        clearable=False
                    ),
                    html.Br(),
                    dbc.Label("Выберите файл аннотаций"),
                    dcc.Dropdown(
                        id="annotation-file-select",
                        options=[],
                        placeholder="Выберите файл аннотаций...",
                        clearable=False
                    ),
                    html.Br(),
                    dbc.Label("Выберите модель"),
                    dcc.Dropdown(
                        id="model-select",
                        options=[{"label": m, "value": m} for m in available_models],
                        value=available_models[0] if available_models else None,
                        clearable=False
                    ),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Макс. длина сегмента"),
                            html.Br(),
                            dcc.Input(id="max-length", type="number", min=1, value=MAX_SEGMENT_LENGTH_DEFAULT)
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Доля тестовой выборки"),
                            html.Br(),
                            dcc.Input(id="test-size", type="number", min=0.01, max=0.5, step=0.01, value=TEST_SIZE_DEFAULT)
                        ], width=6)
                    ]),
                    html.Br(),
                    dbc.Checklist(
                        options=[{"label": "Применить фильтр 1–40 Гц", "value": "filter"}],
                        value=["filter"] if APPLY_FILTER_DEFAULT else [],
                        id="apply-filter"
                    ),
                    html.Br(),
                    dbc.Checklist(
                        options=[{"label": "Применить конкатенацию каналов", "value": "concat"}],
                        value=["concat"] if APPLY_CONCAT_DEFAULT else [],
                        id="apply-concat"
                    ),
                    html.Br(),
                    dbc.Button("Запустить обучение", id="run-btn", color="primary", className="me-2"),
                    dbc.Button("Скачать модель", id="download-model-btn", color="secondary"),
                    dcc.Download(id="download-model")
                ])
            ])
        ], width=4),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Параметры модели"),
                dbc.CardBody([
                    dbc.Label("Описание модели"),
                    dcc.Markdown(
                        id="model-description",
                        style={
                            "background": "#f8f9fa",
                            "padding": "10px",
                            "borderRadius": "5px",
                            "fontSize": "12px",
                            "height": "200px",
                            "overflowY": "auto"
                        }
                    ),
                    html.Br(),
                    dbc.Label("Дополнительные параметры модели"),
                    dcc.Textarea(
                        id="model-kwargs",
                        style={
                            "background": "#f8f9fa",
                            "padding": "10px",
                            "borderRadius": "5px",
                            "fontSize": "12px",
                            "width": "100%",
                            "height": "200px"},
                        value=""
                    ),
                    html.Div([
                        html.Small("Формат: параметр=значение, по одному на строку", 
                                 style={"color": "#6c757d"})
                    ], style={"marginTop": "5px"}),
                    html.Div(id="kwargs-warning", style={"color": "red", "marginTop": "6px"})
                ])
            ])
        ], width=4),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Лог выполнения"),
                dbc.CardBody([
                    html.Div(id="status-box", style={
                        "whiteSpace": "pre-wrap",
                        "height": "300px",
                        "overflow": "auto",
                        "background": "#f8f9fa",
                        "padding": "8px",
                        "fontFamily": "monospace",
                        "fontSize": "13px"
                    })
                ])
            ])
        ], width=4),
    ], className="g-3"),

    dbc.Row([
        dbc.Col([dbc.Card([dbc.CardHeader("Отчёт по классификации"), dbc.CardBody(html.Div(id="report-table"))])], width=6),
        dbc.Col([dbc.Card([dbc.CardHeader("Матрица ошибок"), dbc.CardBody(dcc.Graph(id="confmat-fig"))])], width=6),
    ], className="g-3 mt-3"),

    dcc.Store(id="trained-model-store", data=None)
], fluid=True)

# === Callback: обновление списка файлов аннотаций ===
@app.callback(
    Output("annotation-file-select", "options"),
    Output("annotation-file-select", "value"),
    Input("dataset-select", "value")
)
def update_annotation_files(dataset_name):
    if not dataset_name:
        return [], None
    
    files = get_available_annotation_files(dataset_name)
    options = [{"label": f, "value": f} for f in files]
    
    if files:
        return options, files[0]
    else:
        return options, None

# === Callback: обновление описания модели ===
@app.callback(
    Output("model-description", "children"),
    Input("model-select", "value")
)
def update_model_description(model_name):
    if not model_name:
        return "Выберите модель для просмотра описания"
    
    try:
        description = model_description(model_name)
        return description
    except Exception as e:
        return f"Не удалось загрузить описание модели: {e}"

# === Callback: обучение модели ===
@app.callback(
    output=[
        Output("status-box", "children"),
        Output("report-table", "children"),
        Output("confmat-fig", "figure"),
        Output("trained-model-store", "data"),
        Output("kwargs-warning", "children"),
    ],
    inputs=[
        Input("run-btn", "n_clicks"),
        State("dataset-select", "value"),
        State("annotation-file-select", "value"),
        State("model-select", "value"),
        State("max-length", "value"),
        State("test-size", "value"),
        State("apply-filter", "value"),
        State("apply-concat", "value"),
        State("model-kwargs", "value"),
    ],
    background=True,
    running=[
        (Output("run-btn", "disabled"), True, False),
        (Output("run-btn", "children"), "Выполняется...", "Запустить обучение"),
    ],
    progress=[Output("status-box", "children")],
    progress_default="",
    prevent_initial_call=True,
)
def on_run(set_progress, n_clicks, dataset_name, annotation_file, model_name, max_length, test_size, apply_filter_list, apply_concat_list, kwargs_text):
    logs = ""
    def log(msg):
        nonlocal logs
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logs += f"[{ts}] {msg}\n"
        set_progress(logs)

    try:
        log(f"Запуск обучения: датасет={dataset_name}, модель={model_name}")

        if not dataset_name or not model_name:
            raise ValueError("Пожалуйста, выберите датасет и модель")
        
        if not annotation_file:
            raise ValueError("Пожалуйста, выберите файл аннотаций")

        # Парсинг параметров
        user_kwargs = parse_kwargs_string(kwargs_text)
        if user_kwargs:
            log(f"Загружены параметры модели: {user_kwargs}")

        # Загрузка
        log("Загрузка данных...")
        raw = load_dataset(dataset_name)
        annotations = load_annotations_from_file(annotation_file)
        log(f"Загружено аннотаций: {len(annotations)}")

        # Извлечение сегментов
        apply_filter = "filter" in (apply_filter_list or [])
        X, y = extract_segments(raw, annotations, max_length=max_length, apply_filter=apply_filter)
        log(f"Извлечено сегментов: {len(X)}; классы: {sorted(set(y))}")

        if not X:
            raise ValueError("Нет сегментов для обучения")

        # Преобразование y в правильный формат для sktime
        y = pd.Series(y)

        # Разбиение
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=float(test_size), stratify=y, random_state=42
        )
        log(f"Train: {len(X_train)} / Test: {len(X_test)}")

        # Преобразование (конкатенация, если нужно)
        use_concat = "concat" in (apply_concat_list or [])
        if use_concat:
            log("Применяется конкатенация каналов...")
            concat = ColumnConcatenator()
            X_train_proc = concat.fit_transform(X_train)
            X_test_proc = concat.transform(X_test)
        else:
            log("Используются мультиканальные данные (без конкатенации)")
            X_train_proc = X_train
            X_test_proc = X_test

        # Обучение
        log("Создание и обучение модели...")
        model = get_model(model_name, **user_kwargs)
        
        # Преобразование данных в формат, совместимый с sktime
        # Убедимся, что X_train_proc и X_test_proc являются списками DataFrame
        if not isinstance(X_train_proc, list):
            if isinstance(X_train_proc, pd.DataFrame):
                X_train_proc = [X_train_proc]
            else:
                X_train_proc = list(X_train_proc)
        
        if not isinstance(X_test_proc, list):
            if isinstance(X_test_proc, pd.DataFrame):
                X_test_proc = [X_test_proc]
            else:
                X_test_proc = list(X_test_proc)
        
        model.fit(X_train_proc, y_train)

        # Оценка
        log("Оценка на тестовой выборке...")
        y_pred = model.predict(X_test_proc)
        cr = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        cr_df = pd.DataFrame(cr).T.round(4)
        report_html = dbc.Table.from_dataframe(
            cr_df.reset_index().rename(columns={"index": "Класс"}),
            striped=True, bordered=True, hover=True, size="sm"
        )

        # Матрица ошибок
        labels = sorted(set(y_test) | set(y_pred))
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        fig = go.Figure(go.Heatmap(z=cm, x=labels, y=labels, showscale=True, colorscale="Blues"))
        fig.update_layout(
            xaxis_title="Предсказано",
            yaxis_title="Истина",
            margin=dict(t=40, b=40, l=40, r=20)
        )

        # Сохранение модели
        try:
            trained_model_pickle = pickle.dumps(model).hex()
            log("Модель успешно сериализована.")
        except Exception as e:
            trained_model_pickle = None
            log(f"⚠️ Не удалось сохранить модель: {e}")

        log("✅ Обучение завершено!")

        return logs, report_html, fig, trained_model_pickle, ""

    except Exception as e:
        tb = traceback.format_exc()
        log(f"❌ Ошибка: {e}\n{tb}")
        return logs, html.Div(f"Ошибка: {e}", style={"color": "red"}), go.Figure(), None, "Ошибка", ""
    finally:
        del X, y, X_train, X_test, X_train_proc, X_test_proc, model
        gc.collect()


# === Callback: скачивание модели ===
@app.callback(
    Output("download-model", "data"),
    Input("download-model-btn", "n_clicks"),
    State("trained-model-store", "data"),
    prevent_initial_call=True
)
def download_model(_, model_hex):
    if not model_hex:
        return dcc.send_string("Модель не обучена или не может быть загружена.", filename="model_error.txt")
    try:
        data = bytes.fromhex(model_hex)
        return dcc.send_bytes(data, filename=f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
    except Exception as e:
        return dcc.send_string(f"Ошибка при подготовке файла: {e}", filename="download_error.txt")
