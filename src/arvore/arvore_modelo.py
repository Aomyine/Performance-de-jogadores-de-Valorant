import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# Carregar dados
df = pd.read_csv("classificacao_jogadores_valorant.csv")

# Limpeza das colunas de porcentagem
for col in ['KAST', 'HSP', 'CSP']:
    if df[col].dtype == 'object':
        df[col] = df[col].str.replace('%', '', regex=False).astype(float)

# Selecionar atributos
features = ['KD', 'KAST', 'ACS']
target = 'Performance Category'

# Preparar dados
df_clean = df[features + [target]].dropna()
le = LabelEncoder()
df_clean[target] = le.fit_transform(df_clean[target])

X = df_clean[features]
y = df_clean[target]

# Modelo de árvore de decisão
clf = DecisionTreeClassifier(random_state=42)
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
scores = cross_val_score(clf, X, y, cv=skf)

print("Acurácia em cada fold:", scores)
print("Acurácia média:", scores.mean())
