import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# Carregar dados com classificação atualizada
df = pd.read_csv("classificacao_jogadores_valorant.csv")

# Corrigir colunas com porcentagens
for col in ['HSP', 'CSP']:
    if df[col].dtype == 'object':
        df[col] = df[col].str.replace('%', '', regex=False).astype(float)

# Selecionar atributos para o modelo
features = ['KD', 'KAST', 'ACS']
target = 'Performance Category'

# Pré-processamento
df_clean = df[features + [target]].copy()
df_clean = df_clean.dropna()

# Codificar a variável alvo
le = LabelEncoder()
df_clean[target] = le.fit_transform(df_clean[target])

# Separar X e y
X = df_clean[features]
y = df_clean[target]

# Criar modelo KNN com K=7
knn = KNeighborsClassifier(n_neighbors=7)

# Validação cruzada estratificada (garante proporções entre folds)
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
scores = cross_val_score(knn, X, y, cv=skf)

# Mostrar resultados
print("\nAcurácia em cada fold:", scores)
print("Acurácia média:", scores.mean())
