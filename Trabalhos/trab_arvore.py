import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score

url_tsv = f"https://docs.google.com/spreadsheets/d/1cgLuproMGnR8bs6oq1QGLTU07qV1WhlU4QrxTPgijQ4/export?format=tsv&gid=0"
df = pd.read_csv(url_tsv, sep="\t")

FEATURES = ["Quantidade", "Preço_Unitário", "Valor_Total"]
TARGET = "Metodo_Pagamento"

df = df.dropna(subset=FEATURES + [TARGET])
df[TARGET] = df[TARGET].str.strip().str.lower()

X_train, X_test, y_train, y_test = train_test_split(
    df[FEATURES], df[TARGET], test_size=0.20, random_state=42
)

model = DecisionTreeClassifier(max_depth=4, random_state=42)  #O melhor resultado foi com 4 (0.6)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"=== Árvore de Decisão ===\nAcurácia: {accuracy_score(y_test, y_pred):.2%}")
print("\nRelatório:\n", classification_report(y_test, y_pred))

plt.figure(figsize=(15, 8))
plot_tree(model, feature_names=FEATURES, class_names=model.classes_, filled=True, rounded=True)
plt.show()
