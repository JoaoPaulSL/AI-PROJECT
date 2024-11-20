import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1. Carregar o dataset
file_path = "survey lung cancer.csv"
df = pd.read_csv(file_path)

# Remover espaços em branco nos nomes das colunas
df.columns = df.columns.str.strip()

# 2. Pré-processamento
print("Colunas disponíveis no dataset:", df.columns)

# Verificar se todas as colunas necessárias estão no dataset
categorical_columns = ["GENDER", "SMOKING", "YELLOW_FINGERS", "ANXIETY", "PEER_PRESSURE", "CHRONIC DISEASE",
                       "FATIGUE", "ALLERGY", "WHEEZING", "ALCOHOL CONSUMING", "COUGHING", "SHORTNESS OF BREATH",
                       "SWALLOWING DIFFICULTY", "CHEST PAIN", "LUNG_CANCER"]

for column in categorical_columns:
    if column not in df.columns:
        raise ValueError(f"Coluna ausente no dataset: {column}")

# Converter colunas categóricas em valores numéricos
label_encoders = {}
for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Separar as features (X) e o alvo (y)
X = df.drop(columns=["LUNG_CANCER"])
y = df["LUNG_CANCER"]

# Dividir os dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Pontuação de Risco
def risk_score_prediction(data, weights):
    scores = np.zeros(len(data))
    for column, weight in weights.items():
        scores += data[column] * weight
    return (scores >= np.mean(scores)).astype(int)

# Definir pesos para Pontuação de Risco
risk_weights = {
    "GENDER": 0.1, "AGE": 0.3, "SMOKING": 0.8, "YELLOW_FINGERS": 0.5, "ANXIETY": 0.2,
    "PEER_PRESSURE": 0.1, "CHRONIC DISEASE": 0.7, "FATIGUE": 0.4, "ALLERGY": 0.1,
    "WHEEZING": 0.6, "ALCOHOL CONSUMING": 0.3, "COUGHING": 0.7, "SHORTNESS OF BREATH": 0.8,
    "SWALLOWING DIFFICULTY": 0.5, "CHEST PAIN": 0.6,
}

# Predição com Pontuação de Risco
y_pred_risk = risk_score_prediction(X_test, risk_weights)
risk_accuracy = accuracy_score(y_test, y_pred_risk)

# 4. Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)
nb_accuracy = accuracy_score(y_test, y_pred_nb)

# 5. Árvore de Decisão
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, y_pred_dt)

# 6. Fazer perguntas ao usuário
def ask_questions():
    user_data = {}
    questions = {
        "GENDER": "Gênero (0 = Feminino, 1 = Masculino): ",
        "AGE": "Idade: ",
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
                user_data[column] = int(input(question))
                break
            except ValueError:
                print("Por favor, insira um valor válido (0 ou 1 para opções binárias).")
    return pd.DataFrame([user_data])

# Coletar as respostas do usuário
print("Responda as seguintes perguntas:")
user_input = ask_questions()

# 7. Predições para o usuário
risk_prediction = risk_score_prediction(user_input, risk_weights)
nb_prediction = nb_model.predict(user_input)
dt_prediction = dt_model.predict(user_input)

# Obter as probabilidades de cada modelo
nb_proba = nb_model.predict_proba(user_input)[0][1]  # Probabilidade de câncer (classe 1)
dt_proba = dt_model.predict_proba(user_input)[0][1]  # Probabilidade de câncer (classe 1)

# Resultados
print("\nResultados das Predições:")
print(f"Pontuação de Risco: {'Risco Alto' if risk_prediction[0] == 1 else 'Risco Baixo'}")
print(f"Naive Bayes: {'Risco Alto' if nb_prediction[0] == 1 else 'Risco Baixo'}, Confiança: {nb_proba*100:.2f}%")
print(f"Árvore de Decisão: {'Risco Alto' if dt_prediction[0] == 1 else 'Risco Baixo'}, Confiança: {dt_proba*100:.2f}%")

# Avaliação dos modelos
print("\nAcurácia dos Modelos:")
print(f"Pontuação de Risco: {risk_accuracy:.2f}")
print(f"Naive Bayes: {nb_accuracy:.2f}")
print(f"Árvore de Decisão: {dt_accuracy:.2f}")
