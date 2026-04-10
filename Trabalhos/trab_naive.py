import pandas as pd

# Trabalho da aula 4
data = [
    ['infantil','miopia','não','reduzida','nenhuma'],
    ['infantil','miopia','sim','normal','gelatinosa'],
    ['infantil','hipermetropia','não','normal','gelatinosa'],
    ['infantil','hipermetropia','sim','normal','dura'],
    ['adolescente','miopia','não','reduzida','gelatinosa'],
    ['adolescente','miopia','sim','reduzida','nenhuma'],
    ['adolescente','miopia','não','normal','dura'],
    ['adolescente','hipermetropia','não','reduzida','gelatinosa'],
    ['adolescente','hipermetropia','sim','normal','dura'],
    ['adulto','miopia','não','normal','gelatinosa'],
    ['adulto','miopia','sim','normal','dura'],
    ['adulto','miopia','sim','normal','gelatinosa'],
    ['adulto','hipermetropia','não','reduzida','nenhuma'],
    ['adulto','hipermetropia','sim','normal','gelatinosa'],
    ['adulto','hipermetropia','não','normal','gelatinosa']
]

colunas = ["Idade","Diagnostico","Astigmatismo","Taxa","Lente"]
df = pd.DataFrame(data, columns=colunas)

alpha = 1
classes = df["Lente"].unique()
total_registros = len(df)
cont_classes = df["Lente"].value_counts()

priors = {}
for c in classes:
    priors[c] = (cont_classes[c] + alpha) / (total_registros + alpha * len(classes))

paciente = {
    "Idade": "adulto",
    "Diagnostico": "miopia",
    "Astigmatismo": "sim",
    "Taxa": "normal"
}

resultados = {}

for c in classes:
    prob_final = priors[c]
    
    for atributo, valor in paciente.items():
        favoraveis = len(df[(df[atributo] == valor) & (df["Lente"] == c)])
        total_da_classe = cont_classes[c]
        num_categorias = df[atributo].nunique()
        
        prob_condicional = (favoraveis + alpha) / (total_da_classe + alpha * num_categorias)
        prob_final *= prob_condicional
    
    resultados[c] = prob_final

soma_total = sum(resultados.values())

print("Probabilidades normalizadas:")
for c, p in resultados.items():
    prob_normalizada = p / soma_total
    print(f"{c} : {round(prob_normalizada, 4)}")

predicao = max(resultados, key=resultados.get)
print("\nClasse prevista:", predicao)
