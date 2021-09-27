import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import seaborn as sns


# sns.set(font_scale=1.2)
# mpl.rcParams['figure.dpi'] = 300

enemies = [1, 2, 3]

df = pd.read_csv('neat-results.csv')

for e in enemies:
    e_df = df.loc[df['enemy'] == e]

    ax = sns.lineplot(data=e_df, x='gen', y='value', hue='EA', style='metric')
    ax.set(xlabel='generation', ylabel='fitness', title=f'enemy {e}')

    plt.tight_layout()
    plt.savefig(f'results-enemy-group-{e}.png')
