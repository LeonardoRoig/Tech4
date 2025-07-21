import streamlit as st
import joblib
import pandas as pd
# Removed matplotlib and seaborn imports as no graphs will be plotted
# import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np # Import numpy if needed for model prediction

# Load cross-validation scores
try:
    mean_cv_accuracy = joblib.load('mean_cv_accuracy.joblib')
    std_cv_accuracy = joblib.load('std_cv_accuracy.joblib')
except FileNotFoundError:
    mean_cv_accuracy = None
    std_cv_accuracy = None
except Exception as e:
    mean_cv_accuracy = None
    std_cv_accuracy = None

# Load test set accuracy
try:
    test_accuracy = joblib.load('test_accuracy.joblib')
except FileNotFoundError:
    test_accuracy = None
except Exception as e:
    test_accuracy = None

# Function for the main prediction page
def main_page(model, mean_cv_accuracy, std_cv_accuracy, test_accuracy):
    st.title('Previsão de Obesidade')

    # Initialize session state for prediction status
    if 'prediction_made' not in st.session_state:
        st.session_state.prediction_made = False

    st.header("Insira seus dados para a previsão de Obesidade")

    # Define the questions and widgets for input
    perguntas_amigaveis_widgets = {
        'vegetais': {"pergunta": "Com que frequência você come vegetais? (1 a 3): ", "tipo": "number_input", "min_value": 1, "max_value": 3, "step": 1},
        'ref_principais': {"pergunta": "Quantas refeições principais você faz por dia? (1 a 4): ", "tipo": "number_input", "min_value": 1, "max_value": 4, "step": 1},
        'agua': {"pergunta": "Quantos litros de água você bebe por dia? (1 a 3): ", "tipo": "number_input", "min_value": 1, "max_value": 3, "step": 1},
        'atv_fisica': {"pergunta": "Com que frequência você pratica atividade física? (0 a 3): ", "tipo": "number_input", "min_value": 0, "max_value": 3, "step": 1},
        'atv_eletronica': {"pergunta": "Com que frequência você usa dispositivos eletrônicos para lazer? (0 a 2): ", "tipo": "number_input", "min_value": 0, "max_value": 2, "step": 1},
        'idade': {"pergunta": "Qual a sua idade? (inteiro): ", "tipo": "number_input", "min_value": 18, "step": 1}, # Changed min_value to 18
        'historico': {"pergunta": "Você tem histórico familiar de obesidade? ", "tipo": "radio", "opcoes": {0: 'Não', 1: 'Sim'}},
        'al_calorico': {"pergunta": "Você consome frequentemente alimentos calóricos? ", "tipo": "radio", "opcoes": {0: 'Não', 1: 'Sim'}},
        'ctrl_caloria': {"pergunta": "Você monitora a ingestão de calorias? ", "tipo": "radio", "opcoes": {0: 'Não', 1: 'Sim'}},
        'entre_ref': {"pergunta": "Você come entre as refeições principais? ", "tipo": "radio", "opcoes": {0: 'Não', 1: 'Sim'}},
        'fumante': {"pergunta": "Você é fumante? ", "tipo": "radio", "opcoes": {0: 'Não', 1: 'Sim'}},
        'alcool': {"pergunta": "Você consome álcool? ", "tipo": "radio", "opcoes": {0: 'Não', 1: 'Sim'}},
        'transporte': {"pergunta": "Seu meio de transporte principal envolve caminhada ou bicicleta? ", "tipo": "radio", "opcoes": {0: 'Sim', 1: 'Não'}},
    }

    # Define the order of features expected by the model
    # This list MUST match the order and names of features the model was trained on
    # Based on the notebook, the training features were:
    # X = obesidade_df.drop(['obesidade', 'peso', 'altura'], axis=1)
    # And after feature engineering (like one-hot encoding Gender), the columns became:
    # 'vegetais', 'ref_principais', 'agua', 'atv_fisica', 'atv_eletronica',
    # 'idade', 'historico', 'al_calorico', 'fumante', 'ctrl_caloria',
    # 'entre_ref', 'alcool', 'transporte', 'feminino', 'masculino'
    colunas_features = [
        'vegetais', 'ref_principais', 'agua', 'atv_fisica', 'atv_eletronica',
        'idade', 'historico', 'al_calorico', 'fumante', 'ctrl_caloria',
        'entre_ref', 'alcool', 'transporte', 'feminino', 'masculino'
    ]


    dados_entrada = {}

    # Add a single input for Gender
    genero_selecionado = st.radio("Qual o seu gênero?", ['Feminino', 'Masculino'], key='genero_input_pred')

    # Map the single gender input back to the 'feminino' and 'masculino' columns
    if genero_selecionado == 'Feminino':
        dados_entrada['feminino'] = 1
        dados_entrada['masculino'] = 0
    else:
        dados_entrada['feminino'] = 0
        dados_entrada['masculino'] = 1

    # Add input widgets for all features except 'feminino' and 'masculino'
    for coluna in colunas_features:
        # Skip 'feminino' and 'masculino' as they are handled by the single gender input
        if coluna in ['feminino', 'masculino']:
            continue

        if coluna in perguntas_amigaveis_widgets:
            widget_info = perguntas_amigaveis_widgets[coluna]
            pergunta = widget_info["pergunta"]
            tipo_widget = widget_info["tipo"]

            if tipo_widget == "number_input":
                min_value = widget_info.get("min_value")
                max_value = widget_info.get("max_value")
                step = widget_info.get("step")
                format_str = widget_info.get("format")
                # Use a consistent key for each widget
                dados_entrada[coluna] = st.number_input(pergunta, min_value=min_value, max_value=max_value, step=step, format=format_str, key=f'{coluna}_input')
            elif tipo_widget == "radio":
                opcoes = list(widget_info["opcoes"].keys())
                opcoes_labels = list(widget_info["opcoes"].values())
                # Use a consistent key for each widget
                selected_label = st.radio(pergunta, opcoes_labels, key=f'{coluna}_input')
                dados_entrada[coluna] = opcoes[opcoes_labels.index(selected_label)]
        else:
            # Handle columns not in the friendly questions map, if any (shouldn't happen if colunas_features is correct)
            st.warning(f"Widget not defined for column: {coluna}. Using text input.")
            dados_entrada[coluna] = st.text_input(f"Enter value for '{coluna}': ", key=f'{coluna}_input_fallback')


    # Display test set accuracy below the input fields
    if test_accuracy is not None:
        st.subheader('Performance do Modelo no Conjunto de Teste:')
        st.write(f"Acurácia no Teste: **{test_accuracy:.2f}**")
        st.info("Este valor indica a performance do modelo em dados que ele não viu durante o treinamento.")


    # Add a button to trigger the prediction
    if st.button('Prever Obesidade', key='predict_button'):
        if model is not None:
            # Create a DataFrame with the input data
            # Ensure the order of columns matches the training data
            # Create the DataFrame directly from the dictionary
            novo_dado_df = pd.DataFrame([dados_entrada])
            # Reindex the columns to ensure they are in the correct order
            novo_dado_df = novo_dado_df.reindex(columns=colunas_features, fill_value=0)


            # Make the prediction
            previsao = model.predict(novo_dado_df)
            previsao_proba = model.predict_proba(novo_dado_df)[:, 1]

            # Display the prediction result
            st.subheader('Resultado da Previsão:')
            if previsao[0] == 1:
                st.write(f"A previsão é: **Obeso**")
            else:
                st.write(f"A previsão é: **Não Obeso**")

            st.write(f"Probabilidade de ser Obeso: **{previsao_proba[0]:.2f}**")

            # Display overall model performance metrics below the prediction
            st.subheader('Performance Geral do Modelo (Cross-Validation):')
            if mean_cv_accuracy is not None and std_cv_accuracy is not None:
                 st.write(f"Acurácia Média: {mean_cv_accuracy:.2f} (+/- {std_cv_accuracy*2:.2f})")
                 st.info("Estes valores indicam a performance geral do modelo em diferentes subconjuntos dos dados, não a confiança desta previsão específica.")

            # Set session state to indicate a prediction has been made
            st.session_state.prediction_made = True

        else:
            st.error("Model not loaded. Cannot make prediction.")

# # Removed the entire comparison_graphs_page function and its content

# Load the trained model
try:
    model = joblib.load('obesity_model.joblib')
except FileNotFoundError:
    st.error("Model file 'obesity_model.joblib' not found. Please ensure the trained model is saved in the same directory as app.py")
    model = None

# Load cross-validation scores
try:
    mean_cv_accuracy = joblib.load('mean_cv_accuracy.joblib')
    std_cv_accuracy = joblib.load('std_cv_accuracy.joblib')
except FileNotFoundError:
    mean_cv_accuracy = None
    std_cv_accuracy = None
except Exception as e:
    mean_cv_accuracy = None
    std_cv_accuracy = None

# Load test set accuracy
try:
    test_accuracy = joblib.load('test_accuracy.joblib')
except FileNotFoundError:
    test_accuracy = None
except Exception as e:
    test_accuracy = None

# Display the main page content directly
main_page(model, mean_cv_accuracy, std_cv_accuracy, test_accuracy)
