import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib as mpl

sns.set(font_scale=1.2)
mpl.rcParams['figure.dpi'] = 300

enemies = [1, 2, 3]

df = pd.read_csv('neat.csv')
df2 = pd.read_csv('custom_neat.csv')

for e in enemies:
    # e_df = df.loc[df['enemy'] == e]
    e_df = pd.concat([df.loc[df['enemy'] == e], df2.loc[df['enemy'] == e]], ignore_index=True)
    # print(e_df)
    # exit()

    ax = sns.lineplot(data=e_df, x='gen', y='fitness', hue='algo', style='type')
    ax.set(xlabel='generation', ylabel='fitness', title=f'enemy {e}')

    plt.tight_layout()
    # plt.savefig(f'ea1-lineplot-enemy-{e}.png')
    plt.savefig(f'duo-lineplot-enemy-{e}.png')
    plt.clf()
    plt.cla()

