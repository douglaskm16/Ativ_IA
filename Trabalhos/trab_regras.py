import pandas as pd

url_tsv = "https://docs.google.com/spreadsheets/d/1cgLuproMGnR8bs6oq1QGLTU07qV1WhlU4QrxTPgijQ4/export?format=tsv&gid=0"
df = pd.read_csv(url_tsv, sep="\t")

def classificar_pedido(valor):
    if valor < 500:
        return "Baixo"
    elif valor < 2000:
        return "Médio"
    else:
        return "Alto"

df["Classe_Valor"] = df["Valor_Total"].apply(classificar_pedido)

print(df[["Produto", "Valor_Total", "Classe_Valor"]].head())

print("\nResumo:")
print(df["Classe_Valor"].value_counts())