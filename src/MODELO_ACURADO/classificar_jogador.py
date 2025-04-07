import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Carrega os dados
df = pd.read_csv("classificacao_jogadores_valorant.csv")

# Limpeza das colunas percentuais
for col in ['HSP', 'CSP']:
    if col in df.columns and df[col].dtype == 'object':
        df[col] = df[col].str.replace('%', '', regex=False).astype(float)

# Seleção de atributos
features = ['KD', 'KAST', 'ACS']
target = 'Performance Category'

# Tratamento de dados
df_clean = df[features + [target]].dropna()
le = LabelEncoder()
df_clean[target] = le.fit_transform(df_clean[target])

X = df_clean[features]
y = df_clean[target]

# Treinamento do modelo
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X, y)

# Entrada manual do jogador
print("\nClassificação de jogador personalizado")
nome = input("Digite o nome do jogador: ")
kd = float(input("Informe o KD: "))
kast = float(input("Informe o KAST (%): "))
acs = float(input("Informe o ACS: "))

entrada = pd.DataFrame([[kd, kast, acs]], columns=features)
classe = clf.predict(entrada)
classe_nome = le.inverse_transform(classe)

print(f"\nO jogador {nome} foi classificado como: {classe_nome[0]}")
