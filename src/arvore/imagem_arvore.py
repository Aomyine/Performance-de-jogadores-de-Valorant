import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Carregar os dados (caminho relativo ao projeto)
df = pd.read_csv("classificacao_jogadores_valorant.csv")

# Corrigir possíveis colunas com strings de porcentagem
for col in ['KAST', 'HSP', 'CSP']:
    if df[col].dtype == 'object':
        df[col] = df[col].str.replace('%', '', regex=False).astype(float)

# Selecionar atributos
features = ['KD', 'KAST', 'ACS']
target = 'Performance Category'

# Pré-processamento
df_clean = df[features + [target]].dropna()
le = LabelEncoder()
df_clean[target] = le.fit_transform(df_clean[target])

X = df_clean[features]
y = df_clean[target]

# Treinar o modelo da árvore de decisão
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X, y)

# Plotar a árvore
plt.figure(figsize=(14, 8))
plot_tree(clf, feature_names=features, class_names=le.classes_, filled=True, rounded=True)
plt.title("Árvore de Decisão - Classificação de Jogadores")
plt.tight_layout()

# Salvar a imagem na raiz do projeto
image_path = "arvore_classificacao_valorant.png"
plt.savefig(image_path)

print(f"Árvore salva em: {image_path}")
