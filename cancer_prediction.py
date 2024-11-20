import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier  # Importando o RandomForest
from sklearn.metrics import accuracy_score

# 1. Carregar o dataset
file_path = "survey lung cancer.csv"  # Substitua pelo caminho correto do arquivo
df = pd.read_csv(file_path)

# Remover espaços em branco nos nomes das colunas para evitar erros
df.columns = df.columns.str.strip()

# 2. Pré-processamento
print("Colunas disponíveis no dataset:", df.columns)

# Verificar se todas as colunas necessárias estão presentes no dataset
categorical_columns = [
    "GENDER", "SMOKING", "YELLOW_FINGERS", "ANXIETY", "PEER_PRESSURE", "CHRONIC DISEASE",
    "FATIGUE", "ALLERGY", "WHEEZING", "ALCOHOL CONSUMING", "COUGHING", "SHORTNESS OF BREATH",
    "SWALLOWING DIFFICULTY", "CHEST PAIN", "LUNG_CANCER"
]

for column in categorical_columns:
    if column not in df.columns:
        raise ValueError(f"Coluna ausente no dataset: {column}")

# Converter colunas categóricas para valores numéricos usando LabelEncoder
label_encoders = {}
for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Separar as features (X) e o alvo (y)
X = df.drop(columns=["LUNG_CANCER"])  # Remover a coluna de destino
y = df["LUNG_CANCER"]  # A coluna alvo

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Treinar o modelo Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)
nb_accuracy = accuracy_score(y_test, y_pred_nb)  # Avaliar o modelo

# 4. Treinar o modelo de Árvore de Decisão
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, y_pred_dt)  # Avaliar o modelo

# 5. Treinar o modelo Random Forest (substituindo o Risk Score)
rf_model = RandomForestClassifier(random_state=42)  # Usando RandomForest em vez de risk_score
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)  # Avaliar o modelo

# 6. Coletar dados do usuário
def ask_questions():
    """
    Solicita informações do usuário e retorna os dados formatados como um DataFrame.
    """
    user_data = {}
    questions = {
        "GENDER": "Gênero (0 = Feminino, 1 = Masculino): ",
        "AGE": "Idade (número inteiro): ",
        "SMOKING": "Fuma? (0 = Não, 1 = Sim): ",
        "YELLOW_FINGERS": "Dedos amarelados? (0 = Não, 1 = Sim): ",
        "ANXIETY": "Ansiedade? (0 = Não, 1 = Sim): ",
        "PEER_PRESSURE": "Pressão social? (0 = Não, 1 = Sim): ",
        "CHRONIC DISEASE": "Doença crônica? (0 = Não, 1 = Sim): ",
        "FATIGUE": "Fadiga? (0 = Não, 1 = Sim): ",
        "ALLERGY": "Alergia? (0 = Não, 1 = Sim): ",
        "WHEEZING": "Chiado no peito? (0 = Não, 1 = Sim): ",
        "ALCOHOL CONSUMING": "Consumo de álcool? (0 = Não, 1 = Sim): ",
        "COUGHING": "Tosse? (0 = Não, 1 = Sim): ",
        "SHORTNESS OF BREATH": "Falta de ar? (0 = Não, 1 = Sim): ",
        "SWALLOWING DIFFICULTY": "Dificuldade para engolir? (0 = Não, 1 = Sim): ",
        "CHEST PAIN": "Dor no peito? (0 = Não, 1 = Sim): ",
    }
    for column, question in questions.items():
        while True:
            try:
                value = int(input(question))
                if column == "AGE" and value >= 0:  # Idade deve ser positiva
                    user_data[column] = value
                    break
                elif column != "AGE" and value in {0, 1}:  # Binário
                    user_data[column] = value
                    break
                else:
                    print("Por favor, insira um valor válido.")
            except ValueError:
                print("Entrada inválida. Por favor, insira um número válido.")
    return pd.DataFrame([user_data])

# Coletar respostas do usuário
print("\nResponda as seguintes perguntas:")
user_input = ask_questions()

# 7. Predições para o usuário com Random Forest, Naive Bayes e Decision Tree
rf_prediction = rf_model.predict(user_input)
nb_prediction = nb_model.predict(user_input)
dt_prediction = dt_model.predict(user_input)

# Obter as probabilidades de cada modelo
rf_proba = rf_model.predict_proba(user_input)[0][1]  # Probabilidade de risco alto (classe 1)
nb_proba = nb_model.predict_proba(user_input)[0][1]  # Probabilidade de risco alto (classe 1)
dt_proba = dt_model.predict_proba(user_input)[0][1]  # Probabilidade de risco alto (classe 1)

# Exibir os resultados
print("\nResultados das Predições:")
print(f"Random Forest: {'Risco Alto' if rf_prediction[0] == 1 else 'Risco Baixo'}, Confiança: {rf_proba*100:.2f}%")
print(f"Naive Bayes: {'Risco Alto' if nb_prediction[0] == 1 else 'Risco Baixo'}, Confiança: {nb_proba*100:.2f}%")
print(f"Árvore de Decisão: {'Risco Alto' if dt_prediction[0] == 1 else 'Risco Baixo'}, Confiança: {dt_proba*100:.2f}%")

# Avaliação dos modelos
print("\nAcurácia dos Modelos:")
print(f"Random Forest: {rf_accuracy:.2f}")
print(f"Naive Bayes: {nb_accuracy:.2f}")
print(f"Árvore de Decisão: {dt_accuracy:.2f}")

# 8. Plotando o gráfico para comparar acurácia e confiabilidade
models = ['Random Forest', 'Naive Bayes', 'Decision Tree']
accuracies = [rf_accuracy, nb_accuracy, dt_accuracy]
confidences = [rf_proba, nb_proba, dt_proba]  # Confiança disponível para todos os modelos agora

# Criando o gráfico
fig, ax = plt.subplots(figsize=(10, 6))

# Plotando acurácia
ax.bar(models, accuracies, color='skyblue', label='Acurácia', alpha=0.7)

# Adicionando as linhas de confiança (para todos os modelos)
for i, conf in enumerate(confidences):  
    ax.text(i, accuracies[i] + 0.02, f"{conf*100:.2f}%", ha='center', va='bottom', fontsize=12)

# Títulos e rótulos
ax.set_title('Comparação de Acurácia e Confiabilidade dos Modelos', fontsize=16)
ax.set_ylabel('Acurácia (%)', fontsize=12)
ax.set_ylim(0, 1.1)
plt.show()
