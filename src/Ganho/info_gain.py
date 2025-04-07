import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

# Carregar dados da raiz do projeto
df = pd.read_csv("classificacao_jogadores_valorant.csv")

# Coluna alvo
target = 'Performance Category'

# Copiar dataframe para limpeza
df_clean = df.copy()

# Remover colunas não numéricas ou irrelevantes para a predição
df_clean = df_clean.drop(columns=['Player ID', 'Player', 'Team', 'CL', 'Kills Max'])

# Tratar colunas que podem conter porcentagens como string
for col in ['KAST', 'HSP', 'CSP']:
    if df_clean[col].dtype == 'object':
        df_clean[col] = df_clean[col].str.replace('%', '', regex=False).astype(float)

# Remover linhas com valores ausentes
df_clean = df_clean.dropna()

# Codificar a variável alvo (Alta Performance, Média, Baixa)
le = LabelEncoder()
df_clean[target] = le.fit_transform(df_clean[target])

# Separar atributos e alvo
X = df_clean.drop(columns=[target])
y = df_clean[target]

# Calcular ganho de informação
info_gain = mutual_info_classif(X, y, discrete_features='auto')
info_gain_df = pd.DataFrame({'Atributo': X.columns, 'Ganho de Informação': info_gain})
info_gain_df = info_gain_df.sort_values(by='Ganho de Informação', ascending=False)

# Exibir os resultados
print("\nGanho de Informação por atributo:\n")
print(info_gain_df.to_string(index=False))
