import warnings
warnings.simplefilter('ignore')
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from caiman.source_extraction.cnmf.cnmf import load_CNMF
from .utils import getFootprints
import tifffile as tf
import h5py

class Data():
    def __init__(self, path=None, cellregdir=False):
        # load Suit2p data & behavior Log
        print(path)
        if not cellregdir: # if using cnm data from 1 session.
            cnm = load_CNMF(f"{path}\cnm.hdf5")
            F = cnm.estimates.C[cnm.estimates.idx_components]
            A = getFootprints(cnm)
            maxprojection = tf.imread(f"{path}\mv_MC_MaxProjectionImage.tif")
            log = pd.read_csv(fr'{path}\log.csv')
            log["type"] = np.where(log["S(left)"], "go", "nogo")
            log["lick"] = log["R(left)"]
            log["day"] = 0
            logful = pd.read_csv(fr'{path}\log_full.csv')
            logful["type"] = np.where(logful["S(left)"], "go", "nogo")
            logful["lick"] = logful["R(left)"]
            logful["day"] = 0
            
            #make dataframe
            trials = [s for s in log.Trial for _ in range(198)]
            types = [s for s in log.type for _ in range(198)]
            licks = [s for s in log.lick for _ in range(198)]
            ts = [i for _ in range(log.shape[0]) for i in range(198)]
            df = pd.DataFrame(F.T)
            ncell = F.shape[1]
            df.insert(0, 'lick', licks)
            df.insert(0, 'phase', 0)
            df.insert(0, 'type', types)
            df.insert(0, 'trial', trials)
            df.insert(0, 't', ts)
            df.loc[df.t < 60, 'phase'] = 0
            df.loc[(60 <= df.t) & (df.t < 105), 'phase'] = 1
            df.loc[(105 <= df.t) & (df.t < 135), 'phase'] = 2
            df.loc[135 <= df.t, 'phase'] = 3

            # calculate z-score
            F = (F - np.mean(F, axis=1)[:, np.newaxis]) / (np.std(F, axis=1)[:, np.newaxis] + 1e-10)
            F = F.T
            
        else: # if using cnm data from more than 1 sessions.
            matchpath = list(Path(cellregdir).glob("cellRegistered*.mat"))[0]
            with h5py.File(matchpath, 'r') as f:
                matches = np.array(f['cell_registered_struct']['cell_to_index_map'])
            both_cells = (matches[:, (matches>0).all(axis=0)] - 1).astype(int)

            cnms, Fs, As, maxprojections, logs, dfs, logfuls = [], [], [], [], [], [], []
            trial_icr = 0
            for i, (p, both_cell) in enumerate(zip(path, both_cells)):
                cnm = load_CNMF(f"{p}\cnm.hdf5")
                F = cnm.estimates.C[cnm.estimates.idx_components]
                A = getFootprints(cnm)
                maxprojection = tf.imread(f"{p}\mvMC_mxprj.tif")
                log = pd.read_csv(fr'{p}\log.csv')
                log["type"] = np.where(log["S(left)"], "go", "nogo")
                log["lick"] = log["R(left)"]
                log["day"] = i
                log["trial"] += trial_itr
                logful = pd.read_csv(fr'{path}\log_full.csv')
                logful["type"] = np.where(logful["S(left)"], "go", "nogo")
                logful["lick"] = logful["R(left)"]
                logful["day"] = i
                logful["trial"] += trial_itr
                
                trials = [s for s in log.Trial for _ in range(198)]
                types = [s for s in log.type for _ in range(198)]
                licks = [s for s in log.lick for _ in range(198)]
                ts = [i for _ in range(log.shape[0]) for i in range(198)]
                df = pd.DataFrame(F.T)
                ncell = F.shape[1]
                df.insert(0, 'day', i)
                df.insert(0, 'lick', licks)
                df.insert(0, 'phase', 0)
                df.insert(0, 'type', types)
                df.insert(0, 'trial', trials)
                df.insert(0, 't', ts)
                df.loc[df.t < 60, 'phase'] = 0
                df.loc[(60 <= df.t) & (df.t < 105), 'phase'] = 1
                df.loc[(105 <= df.t) & (df.t < 135), 'phase'] = 2
                df.loc[135 <= df.t, 'phase'] = 3
                
                # calculate z-score
                F = (F - np.mean(F, axis=1)[:, np.newaxis]) / (np.std(F, axis=1)[:, np.newaxis] + 1e-10)
                F = F[both_cell].T 

                cnms.append(cnm)
                Fs.append(F)
                As.append(A)
                maxprojections.append(maxprojection)
                logs.append(log)
                dfs.append(df)
                
                trial_itr += log.trial.iloc[-1] + 2
                
            F = np.concatenate(Fs, axis=0)
            log = pd.concat(logs, axis=0).reset_index()
            cnm = cnms
            A = As
            maxprojection = maxprojections
            df = pd.condat(dfs, axis=0)
            logful = pd.concat(logfuls, axis=0).reset_index()
        
        # make dataframe
        ncell = F.shape[1]
        
        self.cnm = cnm
        self.A = A
        self.maxprojection = maxprojection
        self.F = F
        self.log = log
        self.df = df
        self.logful = logful
        self.ncell = ncell
        self.method = "z-score"
        self.go_mean = df[df.type=='go'].groupby('t').mean().loc[:, 0:]
        self.nogo_mean = df[df.type=='nogo'].groupby('t').mean().loc[:, 0:]
        self.go_std = df[df.type=='go'].groupby('t').std().loc[:, 0:]
        self.nogo_std = df[df.type=='nogo'].groupby('t').std().loc[:, 0:]
        
        # day result
        self.ID = Path(path).parent.parent.name
        self.date = [Path(path).parent.name, Path(path).name]
        self.hit_rate = np.sum((log.type=='go')&(log.lick==1))/np.sum(log.type=='go')
        self.false_rate = np.sum((log.type=='nogo')&(log.lick==0))/np.sum(log.type=='nogo')
        self.hit_index = self.hit_rate - self.false_rate
        

class Data_suite2p():
    def __init__(self, path=None):
        # load Suit2p data & behavior Log
        print(path)
        Fpath = f"{path}\plane0\F.npy"
        iscellpath = f"{path}\plane0\iscell.npy"
        F = np.load(Fpath)
        iscell = np.load(iscellpath)
        F = F[iscell.T[0].astype('bool')]
        log = pd.read_csv(fr'{path}\log.csv')
        log["type"] = np.where(log["S(left)"], "go", "nogo")
        log["lick"] = log["R(left)"]
        log["day"] = 0
        logful = pd.read_csv(fr'{path}\log_full.csv')
        logful["type"] = np.where(logful["S(left)"], "go", "nogo")
        logful["lick"] = logful["R(left)"]
        logful["day"] = 0
        
        #make dataframe
        trials = [s for s in log.Trial for _ in range(198)]
        types = [s for s in log.type for _ in range(198)]
        licks = [s for s in log.lick for _ in range(198)]
        ts = [i for _ in range(log.shape[0]) for i in range(198)]
        df = pd.DataFrame(F.T)
        ncell = F.shape[1]
        df.insert(0, 'lick', licks)
        df.insert(0, 'phase', 0)
        df.insert(0, 'type', types)
        df.insert(0, 'trial', trials)
        df.insert(0, 't', ts)
        df.loc[df.t < 60, 'phase'] = 0
        df.loc[(60 <= df.t) & (df.t < 105), 'phase'] = 1
        df.loc[(105 <= df.t) & (df.t < 135), 'phase'] = 2
        df.loc[135 <= df.t, 'phase'] = 3
            
        # make dataframe
        ncell = F.shape[1]
        
        self.F = F
        self.log = log
        self.df = df
        self.logful = logful
        self.ncell = ncell
        self.method = "z-score"
        self.go_mean = df[df.type=='go'].groupby('t').mean().loc[:, 0:]
        self.nogo_mean = df[df.type=='nogo'].groupby('t').mean().loc[:, 0:]
        self.go_std = df[df.type=='go'].groupby('t').std().loc[:, 0:]
        self.nogo_std = df[df.type=='nogo'].groupby('t').std().loc[:, 0:]
        
        # day result
        self.ID = Path(path).parent.parent.name
        self.date = [Path(path).parent.name, Path(path).name]
        self.hit_rate = np.sum((log.type=='go')&(log.lick==1))/np.sum(log.type=='go')
        self.false_rate = np.sum((log.type=='nogo')&(log.lick==0))/np.sum(log.type=='nogo')
        self.hit_index = self.hit_rate - self.false_rate