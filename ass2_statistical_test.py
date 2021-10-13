import pandas as pd
import scipy.stats as sc

neuro_df = pd.read_csv('ass2_neuro_gain.csv')
neat_df = pd.read_csv('ass2_neat_gain.csv')

df = pd.concat(neuro_df, neat_df)

df_neat = df[df.algo == "neat"]
df_neuro = df[df.algo == "neuro"]
df1_neat = df[(df.group == 0) & (df.algo == "neat")]
df1_neuro = df[(df.group == 0) & (df.algo == "neuro")]
df2_neat = df[(df.group == 1) & (df.algo == "neat")]
df2_neuro = df[(df.group == 1) & (df.algo == "neuro")]


# Comparing results of both algorithms for both groups
value, pvalue = sc.mannwhitneyu(df_neat['gain'].values, df_neuro['gain'].values)
print(value, pvalue)

# Comparing results of both algorithms for first enemy group (1, 2, 3)
value1, pvalue1 = sc.mannwhitneyu(df1_neat['gain'].values, df1_neuro['gain'].values)
print(value1, pvalue1)

# Comparing results of both algorithms for second enemy group (4, 5, 6)
value2, pvalue2 = sc.mannwhitneyu(df2_neat['gain'].values, df2_neuro['gain'].values)
print(value2, pvalue2)
