import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

url_tsv = f"https://docs.google.com/spreadsheets/d/1cgLuproMGnR8bs6oq1QGLTU07qV1WhlU4QrxTPgijQ4/export?format=tsv&gid=0"
df = pd.read_csv(url_tsv, sep="\t", decimal=",")

#colunas de entrada e o alvo da predição
FEATURES = ['Quantidade', 'Preço_Unitário']
TARGET = 'Categoria'

# 80% treino, 20% teste
X_train, X_test, y_train, y_test = train_test_split(
    df[FEATURES], df[TARGET], test_size=0.20, random_state=42
)

model_flow = make_pipeline(
    StandardScaler(), 
    KNeighborsClassifier(n_neighbors=5)
)

model_flow.fit(X_train, y_train)

y_pred = model_flow.predict(X_test)

print(f"Desempenho Geral (Acurácia): {accuracy_score(y_test, y_pred):.2%}")
print("\nMétricas Detalhadas por Classe:")
print(classification_report(y_test, y_pred))

print("\nResumo do Dataframe:")
print(f"Colunas disponíveis: {list(df.columns)}")
display(df.head(5))
