import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib as mpl

sns.set(font_scale=1.2)
mpl.rcParams['figure.dpi'] = 300

enemies = [[1, 2, 3], [4, 5, 6]]

df = pd.read_csv('neat_generalist.csv')
df2 = pd.read_csv('neuro-generalist-results.csv')

for e, enemy_group in enumerate(enemies):
    e_df = pd.concat([df.loc[df['enemy'] == e], df2.loc[df['enemy'] == e]], ignore_index=True)

    ax = sns.lineplot(data=e_df, x='gen', y='fitness', hue='algo', style='type')
    ax.set(xlabel='generation', ylabel='fitness', title=f'enemies {enemy_group}')

    plt.tight_layout()
    plt.savefig(f'duo-lineplot-group-{e}.png')
    plt.clf()
    plt.cla()

