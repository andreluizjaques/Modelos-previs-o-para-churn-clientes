#importando as bibliotecas necessárias
import numpy as np # linear algebra
import pandas as pd # data processing
import random
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

#carregando os dados do dataframe
df = pd.read_csv(arquivo)

#verificando a estrutura do dataframe
df.head()

#informações do dataframe
df.info()

#removendo as colunas que não irei precisar
df.drop(columns=['RowNumber', 'CustomerId', 'Surname'], inplace=True)

#veficando como ficou o dataframr do dataframe
df.shape

#descrevendo os dados estatístico dos sados
df.describe()

# Definindo a semente aleatória
random_seed = 42

random.seed(random_seed)
np.random.seed(random_seed)


#separando o target do dataframe
X = df.drop('Exited', axis=1)
y = df['Exited']


#dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#identificando as colunas numéricas e categóricas
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns


#criando as transformações para colunas numéricas e categóricas
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop='first')

#criando o pré-processamento com ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])


#definindo os modelos para teste
models = {
    'Logistic Regression': LogisticRegression(random_state=random_seed),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Support Vector Machine': SVC(probability=True, random_state=random_seed),
    'Decision Tree': DecisionTreeClassifier(random_state=random_seed),
    'Random Forest': RandomForestClassifier(random_state=random_seed),
    'Gradient Boosting': GradientBoostingClassifier(random_state=random_seed),
    'XGBoost': XGBClassifier(random_state=random_seed, use_label_encoder=False, eval_metric='logloss'),
    'LightGBM': LGBMClassifier(random_state=random_seed)
}

#avaliacao dos modelos
results = {}
for name, model in models.items():
    # Criar pipeline com pré-processamento e modelo
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    #treinamento do modelo
    pipeline.fit(X_train, y_train)
    
    #realizando as previsões
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    #cakculando as métricas
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    results[name] = {
        'confusion_matrix': cm,
        'classification_report': report,
        'roc_auc': roc_auc
    }

#conparacao dos resultados
for name, result in results.items():
    print(f"Modelo: {name}")
    print("Matriz de Confusão:")
    print(result['confusion_matrix'])
    print("Relatório de Classificação:")
    print(result['classification_report'])
    print("ROC AUC Score:", result['roc_auc'])
    print("\n")

#plotando as matrizes de confusão para comparação visual
fig, axes = plt.subplots(4, 2, figsize=(15, 20))
axes = axes.ravel()

for idx, (name, result) in enumerate(results.items()):
    sns.heatmap(result['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=axes[idx])
    axes[idx].set_title(name)
    axes[idx].set_xlabel('Predição')
    axes[idx].set_ylabel('Verdadeiro')

plt.tight_layout()
plt.show()






