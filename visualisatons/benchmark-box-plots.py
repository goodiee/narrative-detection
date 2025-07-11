import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from matplotlib.ticker import MultipleLocator

matplotlib.use('TkAgg')

custom_data = pd.read_csv("path/to/custom_dataset.csv", encoding='latin1', sep=';')
benchmark_data = pd.read_csv("path/to/benchmark_dataset.csv")

custom_semantic_similarity = pd.to_numeric(custom_data['Column3'], errors='coerce')
benchmark_semantic_similarity = pd.to_numeric(benchmark_data['semantic_similarity'], errors='coerce')

combined_data = pd.DataFrame({
    'semantic_similarity': pd.concat([custom_semantic_similarity, benchmark_semantic_similarity], ignore_index=True),
    'dataset': ['Custom Dataset'] * len(custom_semantic_similarity) + ['Benchmark Dataset'] * len(benchmark_semantic_similarity)
})

combined_data = combined_data.dropna()

plt.figure(figsize=(8, 10))
sns.boxplot(x='dataset', y='semantic_similarity', data=combined_data, width=0.3, color='red')

ax = plt.gca()
ax.xaxis.grid(False)
ax.yaxis.grid(False)

plt.xlabel('Dataset', fontsize=16)
plt.ylabel('Semantic Similarity', fontsize=16)

ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)

sns.despine(left=True, right=True, top=True, bottom=True)
ax.yaxis.set_major_locator(MultipleLocator(0.05))

medians = combined_data.groupby('dataset')['semantic_similarity'].median()
yticks = list(ax.get_yticks())

for median_val in medians:
    if median_val not in yticks:
        yticks.append(median_val)

yticks = sorted(yticks)
ytick_labels = [f'{tick:.2f}' for tick in yticks]
ax.set_yticks(yticks)
ax.set_yticklabels(ytick_labels)

for i, dataset in enumerate(medians.index):
    median_val = medians[dataset]
    ax.text(i, median_val + 0.02, f'{median_val:.2f}',
            horizontalalignment='center', color='black', weight='semibold', fontsize=14)

plt.savefig('results/semantic_similarity_boxplot_with_median_on_yaxis.png', format='png')
plt.show()
plt.close()
