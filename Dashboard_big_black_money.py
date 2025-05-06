import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc
from datetime import datetime

# Cargar el dataset
df = pd.read_csv('Big_Black_Money_Dataset.csv')



# Preprocesamiento de datos
def convert_to_datetime(date_str):
    formats = ["%d/%m/%Y", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            pass
    return None 

df['Transaction Date'] = df['Date of Transaction'].apply(convert_to_datetime)
df = df.dropna(subset=['Transaction Date'])
df['Month'] = df['Transaction Date'].dt.month
df['Semester'] = (df['Transaction Date'].dt.month - 1) // 6 + 1
df['Year'] = df['Transaction Date'].dt.year


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Diseño del Dashboard
app.layout = dbc.Container([
    html.H1("Análisis de Transacciones Big Black Money", className="mb-4 text-center"),
    dbc.Row([
        dbc.Col([
            html.H2("Monto de Transacciones a lo largo del Tiempo", className="mb-3 text-center"),
            dcc.Graph(id='time-series-chart'),
            dcc.RangeSlider(
                id='date-range-slider',
                min=df['Transaction Date'].min().timestamp(),
                max=df['Transaction Date'].max().timestamp(),
                value=[df['Transaction Date'].min().timestamp(), df['Transaction Date'].max().timestamp()],
                marks={ts: {'label': date.strftime('%Y-%m')} for date, ts in zip(df['Transaction Date'], df['Transaction Date'].map(lambda x: x.timestamp()))},
                step=86400 * 30,
            ),
            dcc.Dropdown(
                id='time-aggregation-dropdown',
                options=[
                    {'label': 'Mes', 'value': 'Month'},
                    {'label': 'Semestre', 'value': 'Semester'},
                    {'label': 'Año', 'value': 'Year'},
                ],
                value='Month',
                clearable=False,
                className="mt-2"
            )
        ], md=6),
        dbc.Col([
            html.H2("Total de Transacciones por País", className="mb-3 text-center"),
            dcc.Graph(id='country-bar-chart'),
            dbc.Button("Cambiar color de barras", id='change-bar-color-button', n_clicks=0, className="mt-2"),
        ], md=6),
    ]),
    dbc.Row([
        dbc.Col([
            html.H2("Distribución de Transacciones por Industria", className="mb-3 text-center"),
            dcc.Graph(id='industry-pie-chart'),
            dcc.Dropdown(
                id='industry-dropdown',
                options=[{'label': 'Todas', 'value': 'all'}] + [{'label': industry, 'value': industry} for industry in df['Industry'].unique()],
                value='all',  # Establecer 'all' como el valor inicial
                multi=True,  # Habilitar la selección múltiple
                clearable=False,
                className="mt-2"
            )
        ], md=6),
        dbc.Col([
            html.H2("Distribución del Monto de Transacción por Industria", className="mb-3 text-center"),
            dcc.Graph(id='amount-vs-id-scatter'),
        ], md=6),
    ]),
], fluid=True)

# Callbacks para la interactividad
# 1. Gráfico de Líneas con Selector de Rango
@app.callback(
    Output('time-series-chart', 'figure'),
    Input('date-range-slider', 'value'),
    Input('time-aggregation-dropdown', 'value')
)
def update_time_series_chart(date_range, time_aggregation):
    start_date = datetime.fromtimestamp(date_range[0])
    end_date = datetime.fromtimestamp(date_range[1])
    filtered_df = df[(df['Transaction Date'] >= start_date) & (df['Transaction Date'] <= end_date)]

    if time_aggregation == 'Month':
        aggregated_df = filtered_df.groupby('Month')['Amount (USD)'].mean().reset_index()
        # Para mostrar el nombre del mes en lugar del número
        aggregated_df['Month Name'] = aggregated_df['Month'].apply(lambda x: datetime(2025, x, 1).strftime('%B'))
        x_label = 'Month Name'
        y_label = 'Promedio del Monto de Transacción (USD)'
    elif time_aggregation == 'Semester':
        aggregated_df = filtered_df.groupby('Semester')['Amount (USD)'].mean().reset_index()
        x_label = 'Semester'
        y_label = 'Promedio del Monto de Transacción (USD)'
    elif time_aggregation == 'Year':
        aggregated_df = filtered_df.groupby('Year')['Amount (USD)'].mean().reset_index()
        x_label = 'Year'
        y_label = 'Promedio del Monto de Transacción (USD)'

    fig = px.line(aggregated_df, x=x_label, y='Amount (USD)', title=f'Promedio del Monto de Transacciones por {time_aggregation}')
    fig.update_layout(xaxis_title=x_label, yaxis_title=y_label)
    return fig

# 2. Gráfico de Barras con Botón de Color
@app.callback(
    Output('country-bar-chart', 'figure'),
    Input('change-bar-color-button', 'n_clicks')
)
def update_country_bar_chart(n_clicks):
    colors = ['blue', 'green', 'purple', 'orange', 'teal']
    color_index = n_clicks % len(colors)  # Ciclo a través de los colores
    fig = px.bar(df, x='Country', y='Amount (USD)', title='Total de Transacciones por País', color_discrete_sequence=[colors[color_index]])
    fig.update_layout(xaxis_title='País', yaxis_title='Monto de Transacción (USD)')
    return fig

# 3. Gráfico de Pastel por Industria
@app.callback(
    Output('industry-pie-chart', 'figure'),
    Input('industry-dropdown', 'value')
)
def update_industry_pie_chart(selected_industries):
    if 'all' in selected_industries or not selected_industries:
        filtered_df = df.copy()
        title_text = 'Distribución de Transacciones por Industria (Todas)'
    else:
        filtered_df = df[df['Industry'].isin(selected_industries)]
        title_text = f'Distribución de Transacciones para las Industrias Seleccionadas'
    industry_totals = filtered_df.groupby('Industry')['Amount (USD)'].sum().reset_index()
    if not industry_totals.empty:
        industry_totals['Percentage'] = industry_totals['Amount (USD)'] / industry_totals['Amount (USD)'].sum()
    else:
        return px.pie(title='No hay datos para las industrias seleccionadas')

    fig = px.pie(industry_totals,
                 names='Industry',
                 values='Amount (USD)',
                 title=title_text,
                 hover_data=['Percentage'],
                 )
    fig.update_traces(textposition='inside', textinfo='percent+label')

    return fig


# 4. Gráfico de Caja y Bigotes por Industria
@app.callback(
    Output('amount-vs-id-scatter', 'figure'),
    Input('industry-dropdown', 'value')
)
def update_amount_vs_industry_boxplot(selected_industries):
    plot_df = df.copy()
    title = 'Distribución del Monto de Transacción por Industria'
    x_variable = 'Industry'
    y_variable = 'Amount (USD)'

    if 'all' in selected_industries or not selected_industries:
        fig = px.box(plot_df, x=x_variable, y=y_variable, title=title)
    else:
        filtered_df = plot_df[plot_df['Industry'].isin(selected_industries)]
        fig = px.box(filtered_df, x=x_variable, y=y_variable, title=title)

    fig.update_layout(xaxis_title=x_variable, yaxis_title=y_variable)
    return fig

if __name__ == '__main__':
    app.run(debug=True)
