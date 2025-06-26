#!pip install hmmlearn

import pandas as pd
import pickle
import requests
from io import StringIO
from hmmlearn import hmm
import numpy as np
import plotly.graph_objects as go

TICKER1 = "WIN1!"
TAXA = 0.03

# === Carregar base de dados diretamente do GitHub ===
url_csv = "https://raw.githubusercontent.com/HoffNathan/AlphaRise_Win_model_pb/main/dados_win1.csv"
response = requests.get(url_csv)
df = pd.read_csv(StringIO(response.text), index_col=0, parse_dates=True)
df.index = pd.to_datetime(df.index)
df = df.loc['2007-01-01':'2050-12-31'].copy()

# === Selecionar variáveis de entrada para o modelo ===
col_inicio = 8
col_fim = 37
x_hmm = df.iloc[:, col_inicio:col_fim]

# === Carregar modelo HMM diretamente do GitHub público ===
url_modelo = "https://raw.githubusercontent.com/HoffNathan/AlphaRise_Win_model_pb/main/hmm_win_model.sav"
response_modelo = requests.get(url_modelo)
hmm_model = pickle.loads(response_modelo.content)

# === Fazer a previsão dos regimes ===
df["hmm_result"] = hmm_model.predict(x_hmm).astype(str)

#== backtest ===
df["hmm_result"] = df["hmm_result"].astype(str)

df["Retorno_short"] = np.where((df["hmm_result"] == '0'), -df[f"{TICKER1}_target($)"] -TAXA, 0)
df["Retorno_long"] = np.where((df["hmm_result"] == '1'), df[f"{TICKER1}_target($)"] -TAXA, 0)
df["Retorno_both"] = np.where((df["hmm_result"] == '1'), df[f"{TICKER1}_target($)"] -TAXA, df["Retorno_short"])

df["Retorno_short"] = df["Retorno_short"].astype(float)
df["Retorno_long"] = df["Retorno_long"].astype(float)
df["Retorno_both"] = df["Retorno_both"].astype(float)

df["Retorno_both_acm"] = df["Retorno_both"].cumsum()
df["Retorno_short_acm"] = df["Retorno_short"].cumsum()
df["Retorno_long_acm"] = df["Retorno_long"].cumsum()

# Calcula o retorno acumulado para a estratégia buy and hold
df["Retorno_buy_hold_acm"] = (df[f"{TICKER1}_close"] - df[f"{TICKER1}_close"].iloc[0])

# Criação do gráfico interativo
fig = go.Figure()

# Adiciona as linhas de retorno acumulado (both, short, long)
fig.add_trace(go.Scatter(x=df.index, y=df['Retorno_both_acm'], mode='lines', name='Retorno Acumulado (both)'))
fig.add_trace(go.Scatter(x=df.index, y=df['Retorno_short_acm'], mode='lines', name='Retorno Acumulado (short)'))
fig.add_trace(go.Scatter(x=df.index, y=df['Retorno_long_acm'], mode='lines', name='Retorno Acumulado (long)'))
fig.add_trace(go.Scatter(x=df.index, y=df["Retorno_buy_hold_acm"], mode='lines', name='Retorno Acumulado (buy n hold)'))


# Títulos e rótulos
fig.update_layout(
    title='Estratégia - WIN',
    xaxis_title='Data',
    yaxis_title='Retorno Acumulado',
    template="plotly_dark",  # Opcional: altere o tema
    xaxis_rangeslider_visible=False,  # Remove o range slider
    plot_bgcolor='rgba(0, 0, 0, 0)',  # Altera o fundo do gráfico
    width=900,  # Ajusta a largura do gráfico
    height=500,  # Ajusta a altura do gráfico
)

# Exibe o gráfico
fig.show()

#== tabela de backtest ===
df[[f"{TICKER1}_open",f"{TICKER1}_close","hmm_result", "Retorno_both"]].tail(10)

#== Acurácia mercado vs Modelo ===
df["WIN1!_target(bi)"].describe()

#== Acurácia mercado vs Modelo ===

df["hmm_result"].astype(str)
df["WIN1!_target(bi)"] = df["WIN1!_target(bi)"].astype(str)
df["Acc_aux"] = 1
df["Acc_model(%)"] = np.where(df["hmm_result"] == df["WIN1!_target(bi)"], 1, 0)
df["Acc_model(%)"] = (df["Acc_model(%)"].rolling(360).sum() / df["Acc_aux"].rolling(360).sum()) * 100
df["Acc_model(%)"].tail(10)
