import pandas as pd
import scipy

df = pd.read_csv('results-winner-gains.csv')

df_NEAT = df[df.EA == "neat"]
df_Neuro = df[df.EA == "neuro"]
df1_NEAT = df[(df.enemy == 1) & (df.EA == "neat")]
df1_Neuro = df[(df.enemy == 1) & (df.EA == "neuro")]
df2_NEAT = df[(df.enemy == 2) & (df.EA == "neat")]
df2_Neuro = df[(df.enemy == 2) & (df.EA == "neuro")]
df3_NEAT = df[(df.enemy == 3) & (df.EA == "neat")]
df3_Neuro = df[(df.enemy == 3) & (df.EA == "neuro")]


# Comparing results of both algorithms for both groups of enemies
value, pvalue = scipy.stats.mannwhitneyu(df_NEAT['gain'].values, df_Neuro['gain'].values)
print(value, pvalue)

# Comparing results of both algorithms for first enemy
value1, pvalue1 = scipy.stats.mannwhitneyu(df1_NEAT['gain'].values, df1_Neuro['gain'].values)
print(value1, pvalue1)

# Comparing results of both algorithms for second enemy
value2, pvalue2 = scipy.stats.mannwhitneyu(df2_NEAT['gain'].values, df2_Neuro['gain'].values)
print(value2, pvalue2)

#Comparing results of both algorithms for third enemy
value2, pvalue2 = scipy.stats.mannwhitneyu(df3_NEAT['gain'].values, df3_Neuro['gain'].values)
print(value2, pvalue2)
