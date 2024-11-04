import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from npbench.infrastructure import utilities as util

# create a database connection
database = r"npbench.db"
conn = util.create_connection(database)
data = pd.read_sql_query("SELECT * FROM lcounts", conn)

# get rid of kind and dwarf, we don't use them
data = data.drop(['timestamp', 'kind', 'dwarf', 'version'],
                 axis=1).reset_index(drop=True)

# Remove everything that does not have a domain
data = data[data["domain"] != ""]

# for each framework and benchmark, choose only the best details,mode (based on min npdiff), then get rid of those
aggdata = data.groupby(
    ["benchmark", "domain", "framework", "mode", "details", "count"],
    dropna=False).agg({
        "npdiff": np.min
    }).reset_index()
best = aggdata.sort_values("npdiff").groupby(
    ["benchmark", "domain", "framework", "mode"],
    dropna=False).first().reset_index()
best = best.drop(['domain', 'mode', 'details'], axis=1).reset_index(drop=True)

frmwrks = list(best['framework'].unique())
print(frmwrks)
assert ('numpy' in frmwrks)
frmwrks.remove('numpy')
frmwrks.append('numpy')
percs = ["{}_perc".format(f) for f in frmwrks]

# get improvement over numpy (keep times in best_wide_time for numpy column), reorder columns
data = best.pivot_table(index=["benchmark"],
                        columns="framework",
                        values="count").reset_index()  # pivot to wide form
data = data[['benchmark'] + frmwrks].reset_index(drop=True)
# get improvement over numpy (keep times in best_wide_time for numpy column), reorder columns
diffs = best.pivot_table(index=["benchmark"],
                         columns="framework",
                         values="npdiff").reset_index()  # pivot to wide form
diffs = diffs[frmwrks].reset_index(drop=True)

for f in frmwrks:
    data["{}_perc".format(f)] = (diffs[f] / data['numpy']) * 100

# color of the heatmap is percentage changed
colors = data[percs]
# rename the columns, drop the " Perc" for labelling
colors = colors.rename(columns={a: b for a, b in zip(percs, frmwrks)})

# number in the heatmap is change to NumPy (except for NumPy, where it is the total)
numbers = data[frmwrks]
for f in frmwrks:
    if f == 'numpy':
        continue
    numbers[f] = numbers[f] - numbers['numpy']

plt.style.use('classic')
figsz = (len(frmwrks) + 1, 12)
fig, ax0 = plt.subplots(nrows=1, ncols=1, figsize=figsz)

# plot benchmark heatmap
im = ax0.imshow(colors.to_numpy(),
                cmap='RdYlGn_r',
                interpolation='nearest',
                aspect="auto",
                vmin=0,
                vmax=100)

for i in range(len(data['benchmark'])):
    for j in range(len(colors.columns)):
        l = numbers.to_numpy()[i, j]
        lo = l
        p = colors.to_numpy()[i, j]
        if not math.isnan(p):
            p = str(int(p))
        if j < len(colors.columns) - 1:
            if math.isnan(l):
                text = ax0.text(j,
                                i,
                                "missing",
                                ha="center",
                                va="center",
                                color="red",
                                fontsize=7)
            elif l >= 0:
                l = "+" + str(int(l))
            else:
                l = str(int(l))  #+ ", " + str(p) + "%"
            if not math.isnan(lo):
                text = ax0.text(j,
                                i,
                                l,
                                ha="center",
                                va="center",
                                color="white",
                                fontsize=10)
        else:
            if not math.isnan(lo):
                text = ax0.text(j,
                                i,
                                int(l),
                                ha="center",
                                va="center",
                                color="white",
                                fontweight='bold',
                                fontsize=10)

# We want to show all ticks...
ticks = ax0.set_xticks(np.arange(len(colors.columns)))
ticks = ax0.set_yticks(np.arange(len(data['benchmark'])))
# ... and label them with the respective list entries
ticks = ax0.set_xticklabels(colors.columns)
ticks = ax0.set_yticklabels(data['benchmark'])

# Rotate the tick labels and set their alignment.
plt.setp(ax0.get_xticklabels(),
         rotation=45,
         ha="right",
         rotation_mode="anchor")

plt.tight_layout()
plt.savefig("plot2.pdf", dpi=600)
plt.show()
