import h5py
from pathlib import Path
import numpy as np
import pandas as pd
    
def load_cellreg_match(cellreg_path):
    matchpath = list(Path(cellreg_path).glob("cellRegistered*.mat"))[0]
    with h5py.File(matchpath, 'r') as f:
        matches = np.array(f['cell_registered_struct']['cell_to_index_map']) - 1
    return matches

def pick_correct_trials(df):
    df = df.query("(lick==1&type=='go')or(lick==0&type=='nogo')")
    return df

def load_log_from_raw_file(behavior_path, double=False):
    #load final log in the bpath
    logpaths = list(Path(behavior_path).glob("Log*"))
    if len(logpaths) > 1:
        log = pd.read_csv(logpaths[-2], delimiter='\t').iloc[1:].reset_index(drop=True)
    else:
        log = pd.read_csv(logpaths[0], delimiter='\t').iloc[1:].reset_index(drop=True)
    if double:
        log.iloc[101:]["Trial"] += 102
        
    log["type"] = np.where(log["S(left)"], "go", "nogo")
    log["lick"] = np.where(log["R(left)"], True, False)
    log["correct"] = np.where(((log["type"]=='go')&log["lick"]) | ((log["type"]=='nogo')&(~log["lick"])), True, False)
    return log

def getFootprints(cnm):
    return np.array(cnm.estimates.A.todense()).reshape(cnm.dims + (-1,), order='F').transpose((2,0,1))[cnm.estimates.idx_components]

def renameSnapToMv(path):
    files = list(Path(path).glob('*'))
    for file in files[1:]:
        file.rename(Path(str(file).replace('snap', 'mv')))