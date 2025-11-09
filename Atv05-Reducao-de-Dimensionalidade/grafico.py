import matplotlib.pyplot as plt
import numpy as np

# Dados
t_512_seg = {"Knn": 23, "RF": 38, "SVM": 14, "MLP": 134}
a_512 = {"Knn": 0.72, "RF": 0.74, "SVM": 0.83, "MLP": 0.8375}

t_256_seg = {"Knn": 26, "MLP": 96, "SVM": 8.7, "RF": 30.9}
a_256 = {"Knn": 0.775, "MLP": 0.83, "SVM": 0.83, "RF": 0.77}

t_128_seg = {"Knn": 36, "MLP": 95, "SVM": 7, "RF": 31}
a_128 = {"Knn": 0.79, "MLP": 0.79, "SVM": 0.83, "RF": 0.79}

t_64_seg = {"Knn": 18, "MLP": 83, "SVM": 4.7, "RF": 22}
a_64 = {"Knn": 0.79, "MLP": 0.8, "RF": 0.79, "SVM": 0.83}

t_32_seg = {"Knn": 21, "MLP": 94, "SVM": 6, "RF": 15}
a_32 = {"Knn": 0.77, "MLP": 0.79, "RF": 0.78, "SVM": 0.79}

t_16_seg = {"Knn": 23.6, "MLP": 120, "RF": 10.9, "SVM": 5}
a_16 = {"Knn": 0.73, "MLP": 0.77, "RF": 0.76, "SVM": 0.78}

# Dimensões do PCA
pca_dims = {
    256: 256,
    128: 128,
    64: 64,
    32: 32,
    16: 16
}

# Configuração de segmentações
segments = [512, 256, 128, 64, 32, 16]
segments_with_pca = [256, 128, 64, 32, 16]
models = ["Knn", "RF", "SVM", "MLP"]
colors = {'Knn': '#2E86AB', 'RF': '#A23B72', 'SVM': '#F18F01', 'MLP': '#C73E1D'}

# Dados organizados por modelo
accuracy_data = {
    "Knn": [a_512["Knn"], a_256["Knn"], a_128["Knn"], a_64["Knn"], a_32["Knn"], a_16["Knn"]],
    "RF": [a_512["RF"], a_256["RF"], a_128["RF"], a_64["RF"], a_32["RF"], a_16["RF"]],
    "SVM": [a_512["SVM"], a_256["SVM"], a_128["SVM"], a_64["SVM"], a_32["SVM"], a_16["SVM"]],
    "MLP": [a_512["MLP"], a_256["MLP"], a_128["MLP"], a_64["MLP"], a_32["MLP"], a_16["MLP"]]
}

time_data = {
    "Knn": [t_512_seg["Knn"], t_256_seg["Knn"], t_128_seg["Knn"], t_64_seg["Knn"], t_32_seg["Knn"], t_16_seg["Knn"]],
    "RF": [t_512_seg["RF"], t_256_seg["RF"], t_128_seg["RF"], t_64_seg["RF"], t_32_seg["RF"], t_16_seg["RF"]],
    "SVM": [t_512_seg["SVM"], t_256_seg["SVM"], t_128_seg["SVM"], t_64_seg["SVM"], t_32_seg["SVM"], t_16_seg["SVM"]],
    "MLP": [t_512_seg["MLP"], t_256_seg["MLP"], t_128_seg["MLP"], t_64_seg["MLP"], t_32_seg["MLP"], t_16_seg["MLP"]]
}

# Criar figura com subplots
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

fig.suptitle('Comparação de Modelos: Acurácia, Tempo de Treinamento e Dimensões PCA', 
             fontsize=16, fontweight='bold')

# 1. Acurácia por Segmentação
ax1 = fig.add_subplot(gs[0, 0])
for model in models:
    ax1.plot(segments, accuracy_data[model], marker='o', linewidth=2, 
             label=model, color=colors[model], markersize=8)
ax1.set_xlabel('Número de Segmentos', fontsize=11)
ax1.set_ylabel('Acurácia', fontsize=11)
ax1.set_title('Acurácia vs Número de Segmentos', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xscale('log', base=2)
ax1.invert_xaxis()

# 2. Tempo de Treinamento por Segmentação
ax2 = fig.add_subplot(gs[0, 1])
for model in models:
    ax2.plot(segments, time_data[model], marker='s', linewidth=2, 
             label=model, color=colors[model], markersize=8)
ax2.set_xlabel('Número de Segmentos', fontsize=11)
ax2.set_ylabel('Tempo de Treinamento (s)', fontsize=11)
ax2.set_title('Tempo de Treinamento vs Número de Segmentos', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xscale('log', base=2)
ax2.invert_xaxis()

# 3. Dimensões PCA vs Acurácia Média
ax3 = fig.add_subplot(gs[1, 0])
pca_segments = [256, 128, 64, 32, 16]
avg_accuracy_pca = []
for seg in pca_segments:
    seg_idx = segments.index(seg)
    avg_acc = np.mean([accuracy_data[model][seg_idx] for model in models])
    avg_accuracy_pca.append(avg_acc)

ax3.plot(pca_segments, avg_accuracy_pca, marker='D', linewidth=3, 
         markersize=10, color='#2C7A4F', label='Acurácia Média')
ax3.set_xlabel('Dimensões PCA (= Número de Segmentos)', fontsize=11)
ax3.set_ylabel('Acurácia Média', fontsize=11)
ax3.set_title('Impacto das Dimensões PCA na Acurácia Média', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.set_xscale('log', base=2)
ax3.invert_xaxis()
ax3.legend()

# Adicionar valores nos pontos
for i, (seg, acc) in enumerate(zip(pca_segments, avg_accuracy_pca)):
    ax3.annotate(f'{acc:.3f}', (seg, acc), textcoords="offset points", 
                xytext=(0,10), ha='center', fontsize=9, fontweight='bold')

# 4. Acurácia vs Tempo (scatter plot para cada segmentação)
ax4 = fig.add_subplot(gs[1, 1])
for i, seg in enumerate(segments):
    size = 300 - i * 40  # Tamanho decrescente para visualizar segmentações
    for model in models:
        acc = accuracy_data[model][i]
        time = time_data[model][i]
        ax4.scatter(time, acc, s=size, alpha=0.6, color=colors[model], 
                   label=model if i == 0 else "", edgecolors='black', linewidth=0.5)

ax4.set_xlabel('Tempo de Treinamento (s)', fontsize=11)
ax4.set_ylabel('Acurácia', fontsize=11)
ax4.set_title('Acurácia vs Tempo de Treinamento\n(Tamanho indica segmentação: maior = mais segmentos)', 
              fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Heatmap de Acurácia por Modelo e Dimensão PCA
ax5 = fig.add_subplot(gs[2, 0])
accuracy_matrix = []
for seg in pca_segments:
    seg_idx = segments.index(seg)
    row = [accuracy_data[model][seg_idx] for model in models]
    accuracy_matrix.append(row)

im = ax5.imshow(accuracy_matrix, cmap='RdYlGn', aspect='auto', vmin=0.7, vmax=0.85)
ax5.set_xticks(np.arange(len(models)))
ax5.set_yticks(np.arange(len(pca_segments)))
ax5.set_xticklabels(models)
ax5.set_yticklabels(pca_segments)
ax5.set_xlabel('Modelo', fontsize=11)
ax5.set_ylabel('Dimensões PCA', fontsize=11)
ax5.set_title('Heatmap: Acurácia por Modelo e Dimensões PCA', fontsize=12, fontweight='bold')

# Adicionar valores no heatmap
for i in range(len(pca_segments)):
    for j in range(len(models)):
        text = ax5.text(j, i, f'{accuracy_matrix[i][j]:.3f}',
                       ha="center", va="center", color="black", fontsize=9, fontweight='bold')

cbar = plt.colorbar(im, ax=ax5)
cbar.set_label('Acurácia', rotation=270, labelpad=15)

# 6. Comparação de barras para diferentes dimensões PCA (SVM)
ax6 = fig.add_subplot(gs[2, 1])
svm_accuracies = [accuracy_data["SVM"][segments.index(seg)] for seg in pca_segments]
svm_times = [time_data["SVM"][segments.index(seg)] for seg in pca_segments]

x_pos = np.arange(len(pca_segments))
width = 0.35

bars1 = ax6.bar(x_pos - width/2, svm_accuracies, width, 
                label='Acurácia', color='#F18F01', alpha=0.7)
ax6_twin = ax6.twinx()
bars2 = ax6_twin.bar(x_pos + width/2, svm_times, width, 
                     label='Tempo (s)', color='#F18F01', 
                     alpha=0.4, edgecolor='black', linewidth=1.5)

ax6.set_xlabel('Dimensões PCA', fontsize=11)
ax6.set_ylabel('Acurácia (SVM)', fontsize=11, color='#F18F01')
ax6_twin.set_ylabel('Tempo de Treinamento (s)', fontsize=11, color='#F18F01')
ax6.set_title('SVM: Desempenho vs Dimensões PCA', fontsize=12, fontweight='bold')
ax6.set_xticks(x_pos)
ax6.set_xticklabels(pca_segments)
ax6.legend(loc='upper left')
ax6_twin.legend(loc='upper right')
ax6.grid(True, alpha=0.3, axis='y')
ax6.tick_params(axis='y', labelcolor='#F18F01')
ax6_twin.tick_params(axis='y', labelcolor='#F18F01')

plt.tight_layout()
plt.show()

# Análise resumida
print("="*70)
print("ANÁLISE RESUMIDA - DIMENSÕES PCA E DESEMPENHO DOS MODELOS")
print("="*70)

print("\n1. IMPACTO DAS DIMENSÕES PCA NA ACURÁCIA MÉDIA:")
for seg in [256, 128, 64, 32, 16]:
    seg_idx = segments.index(seg)
    avg_acc = np.mean([accuracy_data[model][seg_idx] for model in models])
    print(f"   {seg:3d} dimensões: {avg_acc:.4f}")

print("\n2. MELHOR ACURÁCIA POR DIMENSÃO PCA:")
for seg in [256, 128, 64, 32, 16]:
    if seg == 256:
        best_model = max(a_256, key=a_256.get)
        print(f"   {seg} dimensões: {best_model} ({a_256[best_model]:.4f})")
    elif seg == 128:
        best_model = max(a_128, key=a_128.get)
        print(f"   {seg} dimensões: {best_model} ({a_128[best_model]:.4f})")
    elif seg == 64:
        best_model = max(a_64, key=a_64.get)
        print(f"   {seg} dimensões: {best_model} ({a_64[best_model]:.4f})")
    elif seg == 32:
        best_model = max(a_32, key=a_32.get)
        print(f"   {seg} dimensões: {best_model} ({a_32[best_model]:.4f})")
    else:
        best_model = max(a_16, key=a_16.get)
        print(f"   {seg} dimensões: {best_model} ({a_16[best_model]:.4f})")

print("\n3. DESEMPENHO DO SVM (MELHOR CUSTO-BENEFÍCIO) POR DIMENSÃO PCA:")
for seg in [256, 128, 64, 32, 16]:
    seg_idx = segments.index(seg)
    svm_acc = accuracy_data["SVM"][seg_idx]
    svm_time = time_data["SVM"][seg_idx]
    print(f"   {seg:3d} dimensões: Acurácia = {svm_acc:.4f}, Tempo = {svm_time:5.1f}s")

print("\n4. OBSERVAÇÕES IMPORTANTES:")
print("   • SVM mantém acurácia consistente (0.78-0.83) em todas as dimensões")
print("   • Tempo de treinamento do SVM diminui com menos dimensões PCA")
print("   • 64-128 dimensões parecem ser o ponto ideal para maioria dos modelos")
print("   • Reduzir de 256 para 64 dimensões economiza ~50% do tempo de treinamento")
print("   • MLP é sempre o mais lento, independente das dimensões PCA")
print("="*70)