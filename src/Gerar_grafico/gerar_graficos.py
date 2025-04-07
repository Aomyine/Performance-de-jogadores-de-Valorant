import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Criação da pasta para salvar imagens
output_dir = os.path.join("Images", "graficos")
os.makedirs(output_dir, exist_ok=True)

# Carrega os dados
df = pd.read_csv("classificacao_jogadores_valorant.csv")

# Limpeza das colunas percentuais
for col in ['HSP', 'CSP']:
    if col in df.columns and df[col].dtype == 'object':
        df[col] = df[col].str.replace('%', '', regex=False).astype(float)

# Variáveis para análise
variaveis_eda = ['KD', 'KAST', 'ACS', 'Rating']

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
