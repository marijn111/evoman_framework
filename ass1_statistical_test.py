import pandas as pd
import scipy.stats as sc

df = pd.read_csv('gains_winners_duo.csv')

df_neat = df[df.algo == "neat"]
df_custom_neat = df[df.algo == "custom_neat"]
df1_neat = df[(df.enemy_group == 1) & (df.algo == "neat")]
df1_custom_neat = df[(df.enemy_group == 1) & (df.algo == "custom_neat")]
df2_neat = df[(df.enemy_group == 2) & (df.algo == "neat")]
df2_custom_neat = df[(df.enemy_group == 2) & (df.algo == "custom_neat")]
df3_neat = df[(df.enemy_group == 3) & (df.algo == "neat")]
df3_custom_neat = df[(df.enemy_group == 3) & (df.algo == "custom_neat")]


# Comparing results of both algorithms for all enemies
value, pvalue = sc.mannwhitneyu(df_neat['gain'].values, df_custom_neat['gain'].values)
print(value, pvalue)

# Comparing results of both algorithms for first enemy
value1, pvalue1 = sc.mannwhitneyu(df1_neat['gain'].values, df1_custom_neat['gain'].values)
print(value1, pvalue1)

# Comparing results of both algorithms for second enemy
value2, pvalue2 = sc.mannwhitneyu(df2_neat['gain'].values, df2_custom_neat['gain'].values)
print(value2, pvalue2)

#Comparing results of both algorithms for third enemy
value2, pvalue2 = sc.mannwhitneyu(df3_neat['gain'].values, df3_custom_neat['gain'].values)
print(value2, pvalue2)
