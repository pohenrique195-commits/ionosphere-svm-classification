import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA

ionosphere = fetch_ucirepo(id=52)
X = ionosphere.data.features.copy()
y = ionosphere.data.targets.iloc[:, 0].copy()

y = y.astype(str).str.strip().str.lower()
y = y.map({"g": 1, "good": 1, "1": 1, "b": 0, "bad": 0, "0": 0})

if y.isna().any():
    raise ValueError("Não foi possível reconhecer todos os rótulos da classe.")

df = X.copy()
df["classe"] = y

print("\nTabela da base:")
print(df.head())
print("\nDistribuição das classes:")
print(df["classe"].value_counts().rename(index={1: "good", 0: "bad"}))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=67, stratify=y
)

modelos = [
    (
        "Modelo 1 - Linear, C=0.1",
        Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="linear", C=0.1))
        ])
    ),
    (
        "Modelo 2 - Linear, C=10",
        Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="linear", C=10))
        ])
    ),
    (
        "Modelo 3 - RBF, C=10",
        Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="rbf", C=10, gamma="scale"))
        ])
    )
]

resultados = []
predicoes = {}

for nome, modelo in modelos:
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    acuracia = accuracy_score(y_test, y_pred)

    resultados.append({
        "Modelo": nome,
        "Acurácia": acuracia
    })
    predicoes[nome] = y_pred

resultados_df = pd.DataFrame(resultados).sort_values(
    by="Acurácia", ascending=False
)

print("\nAcurácias:")
print(resultados_df.to_string(
    index=False,
    formatters={"Acurácia": "{:.2%}".format}
))

for nome, modelo in modelos:
    matriz = confusion_matrix(y_test, predicoes[nome], labels=[0, 1])
    exibicao = ConfusionMatrixDisplay(
        confusion_matrix=matriz,
        display_labels=["bad (0)", "good (1)"]
    )
    exibicao.plot(cmap="Blues", values_format="d")
    plt.title(f"Matriz de confusão - {nome}")
    plt.tight_layout()
    plt.show()

scaler_visual = StandardScaler()
X_train_scaled = scaler_visual.fit_transform(X_train)

pca = PCA(n_components=2, random_state=67)
X_train_pca = pca.fit_transform(X_train_scaled)

x_min = X_train_pca[:, 0].min() - 0.8
x_max = X_train_pca[:, 0].max() + 0.8
y_min = X_train_pca[:, 1].min() - 0.8
y_max = X_train_pca[:, 1].max() + 0.8

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 220),
    np.linspace(y_min, y_max, 220)
)

grade_pca = np.c_[xx.ravel(), yy.ravel()]
grade_escalada = pca.inverse_transform(grade_pca)
grade_original = scaler_visual.inverse_transform(grade_escalada)
grade_original = pd.DataFrame(grade_original, columns=X.columns)

for nome, modelo in modelos:
    Z = modelo.predict(grade_original).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(
        xx, yy, Z,
        alpha=0.25,
        levels=[-0.5, 0.5, 1.5],
        cmap="coolwarm"
    )
    plt.scatter(
        X_train_pca[:, 0],
        X_train_pca[:, 1],
        c=y_train,
        cmap="coolwarm",
        edgecolors="k",
        s=28,
        alpha=0.8
    )
    plt.xlabel("Componente principal 1 (PCA)")
    plt.ylabel("Componente principal 2 (PCA)")
    plt.title(f"Fronteira de decisão - {nome}")
    plt.tight_layout()
    plt.show()
