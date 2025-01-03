# Importando bibliotecas necessárias
import pandas as pd
from pycaret.time_series import TSForecastingExperiment
import streamlit as st

# Configuração da página do Streamlit
st.set_page_config(
    page_title='Forecast Licenciamentos Automóveis',
    layout='wide',
    initial_sidebar_state='auto'
)

# Layout principal
col1, col2 = st.columns([1, 1], gap='large')

with col1:
    st.header('***Forecast Licenciamentos Caminhões por meio do PyCaret***', divider='green')
    st.markdown('''
O segmento de licenciamentos de caminhões é uma parte crucial do setor automotivo no Brasil, sendo monitorado pela Anfavea (Associação Nacional dos Fabricantes de Veículos Automotores). A Anfavea coleta e divulga dados estatísticos sobre a produção, licenciamento e exportação de veículos, incluindo caminhões. Esses dados são fundamentais para compreender o desempenho do mercado, identificar tendências e planejar estratégias de negócios.

O módulo **PyCaret Time Series** é uma ferramenta avançada para a análise e previsão de dados de séries temporais, utilizando aprendizado de máquina e técnicas estatísticas clássicas</span>. Esse módulo permite que os usuários realizem tarefas complexas de previsão de séries temporais de forma simplificada, automatizando todo o processo, desde a preparação dos dados até a implantação do modelo.

O módulo **PyCaret Time Series Forecasting** suporta uma ampla gama de métodos de previsão, como ARIMA, Prophet e LSTM. Ele também oferece diversos recursos para lidar com valores ausentes, realizar decomposição de séries temporais e criar visualizações informativas dos dados.
''')

with col2:
    st.image(
        'https://th.bing.com/th/id/OIP.aPSCItq2ardc51c8JfjcXgHaEo?rs=1&pid=ImgDetMain',
        use_container_width=True
    )

# Carregando o arquivo Excel local
try:
    data = pd.read_excel(r"C:\Tablets\Caminhoes.xlsx")
    data['Mês'] = pd.to_datetime(data['Mês'])  # Ajustar o nome da coluna de data, se necessário
    data.set_index('Mês', inplace=True)  # Definir a coluna de data como índice
    st.success("Base de dados carregada com sucesso.")
except Exception as e:
    st.error(f"Erro ao carregar a base de dados: {e}")
    st.stop()

# Visualizar a base de dados no Streamlit
col1, col2 = st.columns([1, 1], gap='large')

with col1:
    st.write('***Base de Dados Anfavea***')
    st.dataframe(data, height=500)


with col2:
    # Configuração inicial do experimento
    s = TSForecastingExperiment()
    s.setup(data=data, target='CAMINHÕES', session_id=123)
    st.write("**Configuração inicial do PyCaret concluída.**")

    # Comparar modelos
    best = s.compare_models()

    # Obter a tabela de comparação
    comparison_df = s.pull()
    st.write("### Comparação de Modelos")
    st.dataframe(comparison_df)

    # Botão para download da tabela de comparação
    csv = comparison_df.to_csv(index=False).encode('utf-8')
    st.download_button("Baixar Comparação", data=csv, file_name="model_comparison.csv", mime='text/csv')

col1, col2, col3= st.columns([1, 1, 1], gap='large')

with col1:
    # Plotar previsões
    st.write("**Previsão com horizonte de 36 períodos:**")
    s.plot_model(best, plot='forecast', data_kwargs={'fh': 36})
    st.image(r"C:\Tablets\caminhãofor.png", use_container_width=True)

with col2:
    # Finalizar o modelo
    final_best = s.finalize_model(best)
    st.write("**Modelo finalizado:**")
    st.write(final_best)

with col3:
    st.markdown('''O Orthogonal Matching Pursuit (OMP) pode ser adaptado para problemas de séries temporais como uma técnica para selecionar as variáveis mais relevantes (lags ou características derivadas) que ajudam a prever os valores futuros da série. Isso é particularmente útil em séries temporais de alta dimensão ou quando há múltiplos fatores (preditores) e queremos encontrar um subconjunto esparso de variáveis relevantes.
________________________________________
Como o OMP é aplicado a séries temporais
1.	Definição do problema de previsão:
o	Para uma série temporal yty_tyt, o objetivo é prever valores futuros yt+hy_{t+h}yt+h (onde hhh é o horizonte de previsão) com base em lags passados yt−1,yt−2,…,yt−py_{t-1}, y_{t-2}, \dots, y_{t-p}yt−1,yt−2,…,yt−p ou outras características relacionadas, como sazonais, tendências ou variáveis externas.
2.	Formulação como um problema de regressão esparsa:
o	A série temporal é convertida em uma matriz de preditores (X\mathbf{X}X) e uma variável alvo (y\mathbf{y}y).
	Exemplo: Para prever yt+1y_{t+1}yt+1, os lags yt,yt−1,yt−2,…y_t, y_{t-1}, y_{t-2}, \dotsyt,yt−1,yt−2,… formam as colunas de X\mathbf{X}X.
o	O problema se torna: yt+1=w1yt+w2yt−1+⋯+ϵy_{t+1} = w_1 y_t + w_2 y_{t-1} + \dots + \epsilonyt+1=w1yt+w2yt−1+⋯+ϵ
o	Aqui, wiw_iwi são os coeficientes associados aos lags. O OMP busca encontrar quais lags (yt,yt−1,…y_t, y_{t-1}, \dotsyt,yt−1,…) têm os coeficientes mais relevantes (não nulos).
3.	Passos do algoritmo:
o	O OMP seleciona iterativamente os lags ou características mais relevantes com base na correlação entre os preditores (lags) e o vetor de resíduos (erro restante na previsão).
o	Após cada iteração, ele ajusta os coeficientes do modelo e reduz o residual até atingir um critério de parada (exemplo: número máximo de lags ou um erro residual baixo).
4.	Previsão:
o	Após identificar os lags relevantes, o modelo final é usado para prever os valores futuros da série.
________________________________________
Vantagens do OMP em séries temporais
1.	Esparsidade e interpretabilidade:
o	OMP identifica apenas os lags ou características mais importantes, resultando em um modelo mais simples e interpretável.
o	Em séries temporais longas ou multidimensionais, isso é crucial para evitar o sobreajuste e reduzir a complexidade.
2.	Eficiência computacional:
o	Como o algoritmo é iterativo e adiciona apenas um preditor por vez, ele é computacionalmente eficiente em comparação com métodos que ajustam todos os coeficientes simultaneamente.
3.	Robustez em alta dimensionalidade:
o	Quando há muitas possíveis variáveis explicativas (como múltiplas séries temporais ou muitos lags), o OMP ajuda a identificar rapidamente o subconjunto mais relevante.
''', unsafe_allow_html=True)
   
col1, col2=st.columns([1,1], gap='large')

with col1:
    st.markdown('''Esses indicadores são métricas comuns usadas para avaliar a qualidade de modelos de previsão. Aqui está a explicação de cada uma delas, com base nos valores apresentados:

### 1. **Modelo**  
   - `omp_cds_dt` refere-se ao modelo **Orthogonal Matching Pursuit (OMP)** com **Condicional de Deseasonalização e Detrending** (CDS e DT).  
   - Esse método combina uma abordagem de seleção de atributos com a remoção de componentes sazonais e tendências dos dados.

---

### 2. **Métricas de Avaliação**

#### **MASE (Mean Absolute Scaled Error)**  
   - Métrica de erro absoluto escalado.  
   - **Interpretação**:  
     - O valor é comparado com um modelo de referência simples, como o Naïve.  
     - **Valor reportado: 0.1664.**  
     - Um valor menor que 1 indica que o modelo é melhor que o modelo de referência.

---

#### **RMSSE (Root Mean Squared Scaled Error)**  
   - Similar ao MASE, mas usa a raiz quadrada do erro quadrático médio escalado.  
   - **Valor reportado: 0.1270.**  
   - Menor valor indica melhor desempenho, com penalização para grandes erros.

---

#### **MAE (Mean Absolute Error)**  
   - Média dos erros absolutos.  
   - **Valor reportado: 269.0386.**  
   - Representa, em média, a diferença absoluta entre as previsões do modelo e os valores reais.

---

#### **RMSE (Root Mean Squared Error)**  
   - Raiz quadrada da média dos erros quadráticos.  
   - **Valor reportado: 269.0386.**  
   - Penaliza mais fortemente grandes erros em comparação ao MAE. Como o MAE e o RMSE são iguais aqui, isso pode indicar que os erros são uniformes.

---

#### **MAPE (Mean Absolute Percentage Error)**  
   - Erro absoluto médio em porcentagem dos valores reais.  
   - **Valor reportado: 0.0242 (2.42%).**  
   - Uma métrica padronizada, útil para interpretar o erro relativo em diferentes escalas de dados.

---

#### **SMAPE (Symmetric Mean Absolute Percentage Error)**  
   - Versão simétrica do MAPE, útil para evitar viés em valores extremos.  
   - **Valor reportado: 0.0245 (2.45%).**  
   - Normaliza o erro em relação à média dos valores reais e previstos.

---

#### **TT (Sec)**  
   - Tempo total para treinar e avaliar o modelo, em segundos.  
   - **Valor reportado: 0.0633.**  
   - Indica que o modelo é eficiente em termos de tempo computacional.

---

### **Resumo da Avaliação**  
O modelo `omp_cds_dt` apresentou excelente desempenho em todas as métricas:  
- **Erro absoluto e percentual baixos** (MAE, MAPE, SMAPE).  
- **Escalabilidade e precisão melhores que um modelo simples** (MASE, RMSSE < 1).  
- **Rápido em execução** (TT = 0.0633 segundos).

Esses resultados indicam que é um modelo eficaz e eficiente para o conjunto de dados avaliado.
''')
    
with col2:

    # Realizar previsões
    predictions = s.predict_model(final_best, fh=36)
    st.write("**Previsões:**")
    st.dataframe(predictions, height=800)
# Botão para download da tabela de comparação
    csv = predictions.to_csv(index=False).encode('utf-8')
    st.download_button("Baixar Previsão", data=csv, file_name="predictions.csv", mime='text/csv')
    

    