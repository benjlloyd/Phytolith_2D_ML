import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

import common

df = pd.read_csv('features_compare.csv', comment='#', skipinitialspace=True)

names = list(df)

M = len(df)
N = len(df.columns)

ind = np.arange(M)  # the x locations for the groups
width = 0.1       # the width of the bars

fig, ax = plt.subplots()

ax.yaxis.grid(True)
ax.xaxis.grid(False)

ax.set_ylim([50, 100])

for i in xrange(N):
    ax.bar(ind + width * i, df.iloc[:, i], width, label=names[i])

ax.set_ylabel('Accuracy')
ax.legend(fontsize=12)
plt.xticks(ind + width * i / 2, ('Tribe', 'Genus'))

fig.savefig('features_cmp.eps')

plt.show()
