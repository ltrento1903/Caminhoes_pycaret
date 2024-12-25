# Importando bibliotecas necessárias
import pandas as pd
from pycaret.time_series import TSForecastingExperiment
import streamlit as st

# Configurar a página do Streamlit
st.set_page_config(page_title='Forecast Caminhões', layout='wide')

# Verificar se o arquivo existe antes de carregar
file_path = r"C:\Tablets\caminhão.xlsx"
if not file_path:
    st.error(f"Arquivo não encontrado: {file_path}")
    st.stop()

# Carregar o arquivo Excel
try:
    data = pd.read_excel(file_path)
except Exception as e:
    st.error(f"Erro ao carregar o arquivo Excel: {e}")
    st.stop()

# Validar a presença das colunas necessárias
if 'Mês' not in data.columns or 'Caminhões' not in data.columns:
    st.error("As colunas 'Mês' e/ou 'Caminhões' não foram encontradas no arquivo.")
    st.stop()

# Processar os dados
try:
    data['Mês'] = pd.to_datetime(data['Mês'])  # Converter a coluna 'Mês' para datetime
    data.set_index('Mês', inplace=True)  # Definir a coluna 'Mês' como índice
except Exception as e:
    st.error(f"Erro ao processar os dados: {e}")
    st.stop()

# Visualizar os dados no console para verificação
print(data.head())

# Inicializar a configuração do experimento de séries temporais
try:
    s = TSForecastingExperiment()
    s.setup(data=data, target='Caminhões', session_id=123)
except Exception as e:
    st.error(f"Erro ao configurar o experimento: {e}")
    st.stop()

# Comparar modelos para encontrar o melhor
try:
    best = s.compare_models()
except Exception as e:
    st.error(f"Erro ao comparar os modelos: {e}")
    st.stop()

# Visualizar previsão com o modelo selecionado
try:
    forecast_plot = s.plot_model(best, plot='forecast', data_kwargs={'fh': 36})
    diagnostics_plot = s.plot_model(best, plot='diagnostics')
    insample_plot = s.plot_model(best, plot='insample')
except Exception as e:
    st.error(f"Erro ao gerar os gráficos: {e}")
    st.stop()

# Finalizar o modelo
try:
    final_best = s.finalize_model(best)
    predictions = s.predict_model(final_best, fh=36)
except Exception as e:
    st.error(f"Erro ao finalizar ou fazer previsões com o modelo: {e}")
    st.stop()

# Obter a tabela de comparação
try:
    comparison_df = s.pull()
except Exception as e:
    st.error(f"Erro ao obter a tabela de comparação: {e}")
    st.stop()

st.header('***Forecast Caminhões por Meio do PyCaret***')

# Interface do Streamlit
col1, col2 = st.columns([1, 1], gap='large')

with col1:
    st.write('Forecast Licenciamentos Caminhões por meio PyCaret')
    st.markdown('''
        PyCaret é uma biblioteca de machine learning de código aberto em Python que simplifica o processo de criação de modelos preditivos. 
        Ela oferece uma interface de baixo código, permitindo que usuários com pouca experiência em programação possam construir, treinar e 
        implementar modelos de machine learning de forma rápida e eficiente.

        - **Facilidade de Uso**: Com apenas algumas linhas de código, você pode realizar tarefas complexas de machine learning, como preparação de dados, 
          seleção de modelos, tuning de hiperparâmetros e avaliação de desempenho.
        - **Versatilidade**: PyCaret suporta várias tarefas de machine learning, incluindo classificação, regressão, clustering, detecção de anomalias e séries temporais.
        - **Automação**: PyCaret automatiza muitas etapas do processo de machine learning, economizando tempo e esforço dos usuários.
    ''')

with col2:
    st.image(
        'https://th.bing.com/th/id/R.4be5e98a0acc50b7a64de559e2ecb947?rik=tTJvyfn%2ffUsYeg&pid=ImgRaw&r=0', width=800)

col1, col2, col3=st.columns([3,1,1], gap='large')

with col1:

    st.write('Gráfico Forecast')
    st.image(r"C:\Tablets\gráfico_cam.png", width=1000)


with col2:
    
# Fazer previsões e exibir resultados
    predictions = s.predict_model(best, fh=36)
    st.write("Previsões:")
    st.dataframe(predictions, height=800)

# Botão para baixar previsões
    predictions_csv = predictions.to_csv(index=False).encode('utf-8')
    st.download_button("Baixar Previsões", data=predictions_csv, file_name="predictions.csv", mime='text/csv')

with col3:
    st.write('Acuracidade do Forecast')
    st.metric(label="Acuracidade do Forecast Novembro", value="7.4%", delta="5.2%")

st.markdown('''**Decision Tree w/ Cond. Deseasonalize & Detrending** é uma abordagem usada em séries temporais para melhorar a precisão das previsões. Aqui está uma explicação detalhada:

1. **Decision Tree**: Uma árvore de decisão é um modelo de aprendizado de máquina que divide os dados em subconjuntos baseados em valores de atributos, criando uma estrutura em forma de árvore. Cada nó interno representa uma "decisão" baseada em um atributo, cada ramo representa o resultado dessa decisão, e cada folha representa uma previsão ou resultado final.

2. **Deseasonalize**: Deseasonalizar significa remover os efeitos sazonais dos dados. A sazonalidade é um padrão que se repete em intervalos regulares ao longo do tempo, como aumentos de vendas durante o Natal. Ao remover a sazonalidade, podemos analisar a tendência subjacente e outros padrões mais claramente.

3. **Detrending**: Detrending é o processo de remover a tendência dos dados. A tendência é o movimento geral dos dados ao longo do tempo, como um aumento constante nas vendas ao longo dos anos. Remover a tendência ajuda a focar em variações de curto prazo e padrões cíclicos.

4. **Cond. (Conditional)**: O termo "condicional" aqui se refere ao fato de que a deseasonalização e o detrending são aplicados de forma condicional, ou seja, dependendo de certas condições ou critérios específicos dos dados.

Ao combinar essas técnicas, a árvore de decisão pode fazer previsões mais precisas, pois está trabalhando com dados que foram ajustados para remover padrões sazonais e tendências de longo prazo. Isso é especialmente útil em séries temporais, onde esses padrões podem obscurecer a verdadeira relação entre as variáveis.


''')