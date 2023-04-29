from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .data import Data

def performance_bar_panel(log, ax):
    Hit = ((log['S(left)']==1)&(log['R(left)']==1)).mean()
    Correct_Reject = ((log['S(left)']==0)&(log['R(left)']==0)).mean()
    false = ((log['S(left)']==0)&(log['R(left)']==1)).mean()
    miss = ((log['S(left)']==1)&(log['R(left)']==0)).mean()

    labels = ['Hit', 'Correct Reject', 'Miss', 'False']
    colors = ['b', 'r', 'cyan', 'coral']
    nums = [Hit, Correct_Reject, miss, false]
    ax.set_yticks([])
    ax.set_xlabel('% of trials')
    ax.set_title('Performance of the mouse')

    offsets = 0
    for i in range(len(nums)):
        # 棒グラフを描画する。
        bar = ax.barh(0, nums[i], left=offsets, color=colors[i], height=0.01)
        offsets += nums[i]
    ax.legend(labels)
    
    
def lick_history_panel(log, ax):
    ax.bar(log.Trial, ((log.type=='go')&log.lick)*2 - 1, width=0.5)
    ax.bar(log.Trial, ((log.type=='nogo')&log.lick)*2 - 1, width=0.5)
    ax.legend(['go', 'no go'])
    ax.set_yticks([1,-1],['lick', 'not lick'])
    ax.set_xlabel('trial')
    
    
def lick_rasterplot_panel(log, ax, xlabel='Time(ms)', ylabel='# of Trial', legend=True):
    positions = []
    for i,row in enumerate(log["Response_Time_Points(left)"]):
        if type(row) is str:
            if ',' in row:
                events = np.array(row.split(','), dtype=int)
            else:
                events = [np.array(row, dtype=int)]
            for event in events:
                positions.append([event, i+1])
    positions = np.array(positions)
    if len(positions)>0:
        ax.scatter(*zip(positions.T), s=5, c='k', marker='.')
    ax.set_ylabel(ylabel, fontsize=20)
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_xlim((-0.5,16999+0.5))
    ax.set_ylim((0.5,len(log)+0.5))
    ax.fill_betweenx([0, len(log)], 4000, 7000, alpha=0.1)
    ax.fill_betweenx([0, len(log)], 7000, 9000, alpha=0.1)
    if legend:
        ax.legend(['Lick', 'Odor Emi', 'Responce'], bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=10)
    ax.invert_yaxis()

    
def footprint_panel(cnm, n, ax):
    A = np.array(cnm.estimates.A.todense()).reshape(cnm.dims + (-1,), order='F').transpose((2,0,1))[cnm.estimates.idx_components]
    ax.set_title('Foot print')
    ax.imshow(A[n])
    
def footprint_with_all_panel(data, n, ax):
    ax.imshow(data.A.max(axis=0))
    ax.contour(data.A[n], levels=[0.1,1],colors='w')
    
def responce_panel(df, n, title, ax, fig, method='z-score', vmin=None, vmax=None):
    im0 = ax.imshow(df.pivot_table(values=n, index='trial', columns='t'), aspect='auto', vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_ylabel('Trial')
    ax.set_xlabel('Frame')
    ax.set_yticks([0,len(df.trial.unique())-1])
    ax.set_yticklabels([1, len(df.trial.unique())])
    ax.set_xticks([60, 105, 135])
    ax.set_xticklabels(['odor ->', 'res ->', 'ITI ->'], rotation=0)
    cb0 = fig.colorbar(im0, ax=ax, shrink=0.3, anchor=(0.0, 0.0))
    cb0.ax.set_xlabel(method)
    
    
def mean_responce_panel(mean, std, n, title, ax, method='z-score'):
    ax.plot(mean[n])
    ax.set_title(title)
    ax.set_ylabel(f'Mean {method}')
    ax.set_xlabel('Frame')
    ax.set_xticks([60, 105, 135])
    ax.set_xticklabels(['odor ->', 'res ->', 'ITI ->'], rotation=0)
    ax.fill_between(range(mean.shape[0]), mean[n]-std[n], mean[n]+std[n], alpha=0.5)
    ax.legend(['Mean', 'SD'])
    

def performance_panel(log, ax, points_per_day=4, ylabel="Performance Index"):
    xs = range(points_per_day)
    ys = [performance_panel_performance_index(l) for l in performance_panel_split_log(log, points_per_day)]
    
    ax.plot(xs, ys, marker='*', c='k')
    ax.set_ylim(-1,1.02)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.set_xticks([])
    
def performance_panel_performance_index(log):
    n_go = len(log[log.type=='go'])
    n_nogo = len(log[log.type=='nogo'])    
    n_golick = len(log[(log.type=='go')&(log.lick==1)])    
    n_nogonolick = len(log[(log.type=='nogo')&(log.lick==1)])
    return n_golick/n_go - n_nogonolick/n_nogo

def performance_panel_split_log(log, points_per_day):
    l0 = len(log)
    n = points_per_day
    ls = [(l0+i)//n for i in range(n)]
    logs = []
    for l in ls:
        logs.append(log.iloc[:l])
        log = log.iloc[l:]
    return logs

def performance_across_day_panel(logs, ax, points_per_day, c, title, x_start=0):
    xs = []
    ys = []
    for log in logs:
        xs.append(range(x_start, x_start+points_per_day))
        ys.append([performance_index(l) for l in split_log(log, points_per_day)])
        x_start += points_per_day
    for x, y in zip(xs, ys):
        ax.plot(x, y, marker='*', c='k')
    gaps = [i*points_per_day-0.5 for i in range(len(ys) +1)]
    # for d in gaps[1:]:
    #     plt.axvline(d,alpha=0.5,c='k')
    for i in range(len(gaps)-1):
        ax.fill_betweenx(np.linspace(-1.5,1.5), gaps[i], gaps[i+1], color=c[i], alpha=0.3)
    ax.set_ylim(-1,1.02)
    ax.set_ylabel("Performance Index", fontsize=20)
    ax.set_xticks([])
    ax.set_title(title)
    return ys
    
def performance_index(log):
    n_go = len(log[log.type=='go'])
    n_nogo = len(log[log.type=='nogo'])    
    n_golick = len(log[(log.type=='go')&(log.lick==1)])    
    n_nogonolick = len(log[(log.type=='nogo')&(log.lick==1)])
    return n_golick/n_go - n_nogonolick/n_nogo

def split_log(log, points_per_day):
    l0 = len(log)
    n = points_per_day
    ls = [(l0+i)//n for i in range(n)]
    logs = []
    for l in ls:
        logs.append(log.iloc[:l])
        log = log.iloc[l:]
    return logs