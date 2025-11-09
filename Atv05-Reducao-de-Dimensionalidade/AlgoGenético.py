import pygad
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.metrics import accuracy_score

# Carregar o conjunto de dados
df_resnet101 = pd.read_csv("modificado_result_final_resnet101.csv")

# Separar características e rótulos
labels_resnet101 = df_resnet101["Class"].to_numpy()
df_resnet101.drop(columns=["Class", "Filename"], inplace=True)
df_resnet101_numpy = df_resnet101.to_numpy()

# Normalizar os dados
scaler = StandardScaler()
data_normalized = scaler.fit_transform(df_resnet101_numpy)
num_features = data_normalized.shape[1]

# Dividir dados em treino, validação e teste
X_train, X_temp, y_train, y_temp = train_test_split(
    data_normalized, labels_resnet101, test_size=0.4, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# --- Função de fitness ---
def fitness_func(ga_instance, solution, solution_idx):
    # Garante que solution é um array NumPy 1D
    solution = np.atleast_1d(solution)

    # Selecionar índices das features onde o gene = 1
    selected_features = np.where(solution == 1)[0]

    # Se nenhuma feature for selecionada, fitness = 0
    if len(selected_features) == 0:
        return 0

    # Treinar o classificador
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score

    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(X_train[:, selected_features], y_train)

    # Avaliar no conjunto de validação
    y_pred = classifier.predict(X_val[:, selected_features])
    acc = accuracy_score(y_val, y_pred)

    return acc


# --- Configurações do algoritmo genético ---
gene_space = [0, 1]

ga_instance = pygad.GA(
    num_generations=50,
    num_parents_mating=10,
    fitness_func=fitness_func,
    sol_per_pop=20,
    num_genes=num_features,
    gene_space=gene_space,
    parent_selection_type="rank",
    crossover_type="single_point",
    mutation_type="random",
    mutation_percent_genes=5
)

# Executar o algoritmo genético
ga_instance.run()

# Melhor solução encontrada
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(f"\nMelhor fitness: {solution_fitness}")
print(f"Número de features selecionadas: {np.sum(solution)}")

# Exibir quais colunas foram escolhidas
selected_features = np.where(solution == 1)[0]
print("Features selecionadas:", selected_features)
