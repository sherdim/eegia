import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

# Создаём основное приложение
app = dash.Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)
server = app.server

# Главный layout с навигацией
app.layout = dbc.Container([
    dbc.NavbarSimple(
        brand="EEG Tools",
        color="primary",
        dark=True,
        children=[
            dbc.NavItem(dbc.NavLink("Разметка ЭЭГ", href="/annotation")),
            dbc.NavItem(dbc.NavLink("Обучение моделей", href="/training")),
        ],
    ),
    html.Div(dash.page_container, style={"padding": "20px"})
], fluid=True)

if __name__ == "__main__":
    app.run_server(debug=True, port=8050)
