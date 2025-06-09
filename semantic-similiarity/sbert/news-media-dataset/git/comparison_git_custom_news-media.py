import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

matplotlib.use('TkAgg')

custom_data_path = "/semantic-similiarity/sbert/telegram dataset/git/dataset_with_similarity_git_own_dataset.csv"
custom_data = pd.read_csv(custom_data_path, encoding='latin1', sep=';')

benchmark_data_path = "/semantic-similiarity/sbert/news-media-dataset/git/dataset_with_similarity.csv"
benchmark_data = pd.read_csv(benchmark_data_path, encoding='latin1')

custom_semantic_similarity = pd.to_numeric(custom_data['semantic_similarity'], errors='coerce')
benchmark_semantic_similarity = pd.to_numeric(benchmark_data['semantic_similarity'], errors='coerce')

combined_data = pd.DataFrame({
    'semantic_similarity': pd.concat([custom_semantic_similarity, benchmark_semantic_similarity], ignore_index=True),
    'dataset': ['Custom Dataset'] * len(custom_semantic_similarity) + ['Benchmark Dataset'] * len(benchmark_semantic_similarity)
})

combined_data = combined_data.dropna()

plt.figure(figsize=(8, 10))
sns.boxplot(x='dataset', y='semantic_similarity', data=combined_data, width=0.3, color='red')

ax = plt.gca()
ax.set_ylim(0, 1)
ax.xaxis.grid(False)
ax.yaxis.grid(False)

plt.xlabel('Dataset', fontsize=16)
plt.ylabel('Semantic Similarity', fontsize=16)

ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)

sns.despine(left=True, right=True, top=True, bottom=True)

# Рассчёт медиан по группам
medians = combined_data.groupby('dataset')['semantic_similarity'].median()

# Получаем текущие тики на оси Y
yticks = list(ax.get_yticks())

# Добавляем медианы в тики, если их там ещё нет
for median_val in medians:
    if median_val not in yticks:
        yticks.append(median_val)

yticks = sorted(yticks)

# Формируем подписи для всех тиков, отображая число с двумя знаками после запятой
ytick_labels = [f'{tick:.2f}' for tick in yticks]

# Обновляем тики и подписи на оси Y
ax.set_yticks(yticks)
ax.set_yticklabels(ytick_labels)

plt.savefig('semantic_similarity_boxplot_comparison_git_with_median_on_yaxis.png', format='png')
plt.show()
plt.close()
