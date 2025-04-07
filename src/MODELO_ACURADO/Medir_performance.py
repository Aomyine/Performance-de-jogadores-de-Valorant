import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Criação da pasta caso não exista
output_dir = os.path.join("Images", "graficos")
os.makedirs(output_dir, exist_ok=True)

# Carrega os dados
df = pd.read_csv("classificacao_jogadores_valorant.csv")

# Limpeza das colunas percentuais, se houver
for col in ['HSP', 'CSP']:
    if col in df.columns and df[col].dtype == 'object':
        df[col] = df[col].str.replace('%', '', regex=False).astype(float)

# Define variáveis
features = ['KD', 'KAST', 'ACS']
target = 'Performance Category'
variaveis_eda = features + ['Rating']

# Remove nulos e prepara dados
df_clean = df[features + [target]].dropna()
le = LabelEncoder()
df_clean[target] = le.fit_transform(df_clean[target])

X = df_clean[features]
y = df_clean[target]

# Treina o modelo de Árvore de Decisão
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X, y)

# Estatísticas descritivas
print("Estatísticas descritivas:\n")
print(df[variaveis_eda].describe())

# Histogramas
for var in variaveis_eda:
    plt.figure(figsize=(8, 5))
    sns.histplot(df[var].dropna(), kde=True, bins=30)
    plt.title(f'Histograma de {var}')
    plt.xlabel(var)
    plt.ylabel('Frequência')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'Histograma_{var}.png'))
    plt.close()

# Boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[variaveis_eda])
plt.title('Boxplot das Variáveis')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Boxplots_variaveis.png'))
plt.close()

# Scatterplots
sns.pairplot(df[variaveis_eda].dropna(), diag_kind='kde')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Scatterplots_variaveis.png'))
plt.close()

# Input do usuário
print("\nClassificação de jogador personalizado")
nome = input("Digite o nome do jogador: ")
kd = float(input("Informe o KD: "))
kast = float(input("Informe o KAST (%): "))
acs = float(input("Informe o ACS: "))

# Previsão com nomes de colunas 
entrada = pd.DataFrame([[kd, kast, acs]], columns=features)
classe = clf.predict(entrada)
classe_nome = le.inverse_transform(classe)

print(f"\nO jogador {nome} foi classificado como: {classe_nome[0]}")
