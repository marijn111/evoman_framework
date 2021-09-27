import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# sns.set(font_scale=1.2)
# mpl.rcParams['figure.dpi'] = 300

enemies = [1, 2, 3]

df = pd.read_csv('neat.csv')

for e in enemies:
    e_df = df.loc[df['enemy'] == e]

    ax = sns.lineplot(data=e_df, x='gen', y='fitness', hue='algo', style='type')
    ax.set(xlabel='generation', ylabel='fitness', title=f'enemy {e}')

    plt.tight_layout()
    plt.savefig(f'ea1-lineplot-enemy-{e}.png')
    plt.clf()
    plt.cla()

