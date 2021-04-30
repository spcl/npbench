import math
import pandas as pd
import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt

from scipy.stats.mstats import gmean
from npbench.infrastructure import utilities as util


# geomean which ignores NA values
def my_geomean(x):
    x = x.dropna()
    res = gmean(x)
    return res


# make nice/short numbers with up/down indicator
def my_speedup_abbr(x):
    prefix = ""
    label = ""
    if math.isnan(x):
        return ""
    if x < 1:
        prefix = u"\u2191"
        x = 1 / x
    elif x > 1:
        prefix = u"\u2193"
    if x > 100:
        x = int(x)
    if x > 1000:
        label = prefix + str(round(x / 1000, 1)) + "k"
    else:
        label = prefix + str(round(x, 1))
    return str(label)


# make nice/short runtime numbers with seconds / milliseconds
def my_runtime_abbr(x):
    suffix = " s"
    if math.isnan(x):
        return ""
    if x < 0.1:
        x = x * 1000
        suffix = " ms"
    return str(round(x, 2)) + suffix


def bootstrap_ci(data, statfunction=np.median, alpha=0.05, n_samples=300):
    """inspired by https://github.com/cgevans/scikits-bootstrap"""
    import warnings

    def bootstrap_ids(data, n_samples=100):
        for _ in range(n_samples):
            yield np.random.randint(data.shape[0], size=(data.shape[0], ))

    alphas = np.array([alpha / 2, 1 - alpha / 2])
    nvals = np.round((n_samples - 1) * alphas).astype(int)
    if np.any(nvals < 10) or np.any(nvals >= n_samples - 10):
        warnings.warn(
            "Some values used extremal samples; results are probably unstable. "
            "Try to increase n_samples")

    data = np.array(data)
    if np.prod(data.shape) != max(data.shape):
        raise ValueError("Data must be 1D")
    data = data.ravel()

    boot_indexes = bootstrap_ids(data, n_samples)
    stat = np.asarray([statfunction(data[_ids]) for _ids in boot_indexes])
    stat.sort(axis=0)

    return stat[nvals][1] - stat[nvals][0]


# create a database connection
database = r"npbench.db"
conn = util.create_connection(database)
data = pd.read_sql_query("SELECT * FROM results", conn)

# get rid of kind and dwarf, we don't use them
data = data.drop(['timestamp', 'kind', 'dwarf', 'version'],
                 axis=1).reset_index(drop=True)

# Remove everything that does not have a domain
data = data[data["domain"] != ""]

# remove everything that does not validate, then get rid of validated column
data = data[data['validated'] == True]
data = data.drop(['validated'], axis=1).reset_index(drop=True)

# for each framework and benchmark, choose only the best details,mode (based on median runtime), then get rid of those
aggdata = data.groupby(["benchmark", "domain", "framework", "mode", "details"],
                       dropna=False).agg({
                           "time": np.median
                       }).reset_index()
best = aggdata.sort_values("time").groupby(
    ["benchmark", "domain", "framework", "mode"],
    dropna=False).first().reset_index()
bestgroup = best.drop(
    ["time"],
    axis=1)  # remove time, we don't need it and it is actually a median
data = pd.merge(left=bestgroup,
                right=data,
                on=["benchmark", "domain", "framework", "mode", "details"],
                how="inner")  # do a join on data and best
data = data.drop(['mode', 'details'], axis=1).reset_index(drop=True)

frmwrks = list(data['framework'].unique())
print(frmwrks)
assert ('numpy' in frmwrks)
frmwrks.remove('numpy')
frmwrks.append('numpy')
lfilter = ['benchmark', 'domain'] + frmwrks

# get improvement over numpy (keep times in best_wide_time for numpy column), reorder columns
best_wide = best.pivot_table(index=["benchmark", "domain"],
                             columns="framework",
                             values="time").reset_index()  # pivot to wide form
best_wide = best_wide[lfilter].reset_index(drop=True)
best_wide_time = best_wide.copy(deep=True)
for f in frmwrks:
    best_wide[f] = best_wide[f] / best_wide_time['numpy']

# compute ci-size for each
cidata = data.groupby(["benchmark", "domain", "framework"], dropna=False).agg({
    "time": [bootstrap_ci, np.median]
}).reset_index()
cidata.columns = ['_'.join(col).strip() for col in cidata.columns.values]
cidata['perc'] = (cidata['time_bootstrap_ci'] / cidata['time_median']) * 100

overall = best_wide.drop(['domain'], axis=1)
overall = pd.melt(overall, [
    'benchmark',
])
overall = overall.groupby(['framework']).value.apply(my_geomean).reset_index(
)  #this throws warnings if NA is found, which is ok
overall_wide = overall.pivot_table(columns="framework",
                                   values="value",
                                   dropna=False).reset_index(drop=True)
overall_wide = overall_wide[frmwrks]

overall_time = best_wide_time.drop(['domain'], axis=1)
overall_time = pd.melt(overall_time, ['benchmark'])
overall_time = overall_time.groupby(
    ['framework']).value.apply(my_geomean).reset_index(
    )  #this throws warnings if NA is found, which is ok
overall_time_wide = overall_time.pivot_table(
    columns="framework", values="value", dropna=False).reset_index(drop=True)
overall_time_wide = overall_wide[frmwrks]

plt.style.use('classic')
figsz = (len(frmwrks) + 1, 12)
fig, (ax2, ax1) = plt.subplots(2,
                               1,
                               figsize=figsz,
                               sharex=True,
                               gridspec_kw={'height_ratios': [0.1, 5.7]})

hm_data_all = overall_wide
im0 = ax2.imshow(hm_data_all.to_numpy(),
                 cmap='RdYlGn_r',
                 interpolation='nearest',
                 vmin=0,
                 vmax=2,
                 aspect="auto")
ax2.set_yticks(np.arange(1))
ax2.set_yticklabels(["Total"])
for j in range(len(overall_wide.columns)):
    if j < len(overall_wide.columns) - 1:
        label = hm_data_all.to_numpy()[0, j]
        t = label
        if t < 1:
            t = 1 / t
        if t < 1.3:
            text = ax2.text(j,
                            0,
                            my_speedup_abbr(label),
                            ha="center",
                            va="center",
                            color="grey",
                            fontsize=8)
        else:
            text = ax2.text(j,
                            0,
                            my_speedup_abbr(label),
                            ha="center",
                            va="center",
                            color="white",
                            fontsize=8)
    else:
        label = overall_time_wide['numpy'].to_numpy()[0]

# plot benchmark heatmap
hm_data = best_wide.drop(['benchmark', 'domain'], axis=1)
im = ax1.imshow(hm_data.to_numpy(),
                cmap='RdYlGn_r',
                interpolation='nearest',
                vmin=0,
                vmax=2,
                aspect="auto")

# We want to show all ticks...
ticks = ax1.set_xticks(np.arange(len(hm_data.columns)))
ticks = ax1.set_yticks(np.arange(len(best_wide['benchmark'])))
# ... and label them with the respective list entries
ticks = ax1.set_xticklabels(hm_data.columns)
ticks = ax1.set_yticklabels(best_wide['benchmark'])

# Rotate the tick labels and set their alignment.
plt.setp(ax1.get_xticklabels(),
         rotation=90,
         ha="right",
         rotation_mode="anchor")

for i in range(len(best_wide['benchmark'])):
    # annotate with improvement over numpy
    for j in range(len(hm_data.columns)):
        b = best_wide['benchmark'][i]
        f = hm_data.columns[j]
        if j < len(hm_data.columns) - 1:
            label = hm_data.to_numpy()[i, j]
            if math.isnan(label):
                r = ""
                if len(r) > 0:
                    text = ax1.text(j,
                                    i,
                                    str(r.to_numpy()[0]),
                                    ha="center",
                                    va="center",
                                    color="red",
                                    fontsize=7)
            else:
                p = cidata[(cidata['framework_'] == f)
                           & (cidata['benchmark_'] == b)]['perc']
                ci = int(p.to_numpy()[0])
                if ci > 0:
                    ci = "$^{(" + str(ci) + ")}$"
                else:
                    ci = ""
                t = label
                if t < 1:
                    t = 1 / t
                if t < 1.3:
                    text = ax1.text(j,
                                    i,
                                    my_speedup_abbr(label) + ci,
                                    ha="center",
                                    va="center",
                                    color="grey",
                                    fontsize=8)
                else:
                    text = ax1.text(j,
                                    i,
                                    my_speedup_abbr(label) + ci,
                                    ha="center",
                                    va="center",
                                    color="white",
                                    fontsize=8)
        else:
            label = best_wide_time['numpy'].to_numpy()[i]
            p = cidata[(cidata['framework_'] == f)
                       & (cidata['benchmark_'] == b)]['perc']
            try:
                ci = int(p.to_numpy()[0])
                if ci > 0:
                    ci = "$^{(" + str(ci) + ")}$"
                else:
                    ci = ""
            except:
                pass
            finally:
                ci = ""
            text = ax1.text(j,
                            i,
                            my_runtime_abbr(label) + ci,
                            ha="center",
                            va="center",
                            color="black",
                            fontsize=8)

ax1.set_ylabel("Benchmarks", labelpad=0)

plt.tight_layout()
plt.savefig("heatmap.pdf", dpi=600)
# plt.show()
