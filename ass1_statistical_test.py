import pandas as pd
import scipy

df = pd.read_csv('gains_winners_duo.csv')

df_neat = df[df.EA == "neat"]
df_custom_neat = df[df.EA == "custom_neat"]
df1_neat = df[(df.enemy == 1) & (df.EA == "neat")]
df1_custom_neat = df[(df.enemy == 1) & (df.EA == "custom_neat")]
df2_neat = df[(df.enemy == 2) & (df.EA == "neat")]
df2_custom_neat = df[(df.enemy == 2) & (df.EA == "custom_neat")]
df3_neat = df[(df.enemy == 3) & (df.EA == "neat")]
df3_custom_neat = df[(df.enemy == 3) & (df.EA == "custom_neat")]


# Comparing results of both algorithms for all enemies
value, pvalue = scipy.stats.mannwhitneyu(df_neat['gain'].values, df_custom_neat['gain'].values)
print(value, pvalue)

# Comparing results of both algorithms for first enemy
value1, pvalue1 = scipy.stats.mannwhitneyu(df1_neat['gain'].values, df1_custom_neat['gain'].values)
print(value1, pvalue1)

# Comparing results of both algorithms for second enemy
value2, pvalue2 = scipy.stats.mannwhitneyu(df2_neat['gain'].values, df2_custom_neat['gain'].values)
print(value2, pvalue2)

#Comparing results of both algorithms for third enemy
value2, pvalue2 = scipy.stats.mannwhitneyu(df3_neat['gain'].values, df3_custom_neat['gain'].values)
print(value2, pvalue2)
