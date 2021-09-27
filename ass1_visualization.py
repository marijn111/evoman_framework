import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import seaborn as sns


sns.set(font_scale=1.2)
mpl.rcParams['figure.dpi'] = 300

df = pd.read_csv('neat-results.csv')
e1 = df.loc[df['enemy'] == 1]

ax = sns.lineplot(data=e1, x='gen', y='value', hue='EA', style='metric')
ax.set(xlabel='generation', ylabel='fitness', title='enemy 1')

plt.tight_layout()
plt.savefig('results-enemy-group-0.png')
