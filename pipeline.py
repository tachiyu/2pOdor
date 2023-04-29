#!/usr/bin/env python
# coding: utf-8

# In[24]:
import warnings
warnings.simplefilter('ignore')
import time
from .data import Data, Data_suite2p
from pathlib import Path
import numpy as np
import caiman as cm
import pandas as pd
from .plot import *
import imagej
import matplotlib.pyplot as plt
from caiman.base.movies import movie as cm_movie
from caiman.source_extraction import cnmf
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import params as params
from scipy.io import savemat

#Main function
def pipeline(imaging_path, behavior_path, cor_thr=0.2e6, manual_drop_movie_num=None, desired_movie_len=102):
    #%% start a cluster for parallel processing (if a cluster already exists it will be closed and a new session will be opened)
    if 'dview' in locals():
        cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=10, single_thread=False)

    try:
        start_time = time.time()
        ds_rate = 2 # a factor of spatial down sampling of the movie
        result_path = behavior_path.replace('behavior', 'results')
        Path(result_path).mkdir(exist_ok=True, parents=True)

        movies = load_movies(imaging_path, desired_movie_len)
        
        calc_dropped_frame(movies, result_path, cor_thr, manual_drop_movie_num)
        
        make_logfile(behavior_path, result_path, double=True)
        
        mv_name = movie_concatenated(movies, result_path, ds_rate)
        
        mv_mmap_name = motion_correction(mv_name, result_path, dview)
        
        cnmfe(mv_mmap_name, result_path, n_processes, dview)
        
        make_data_object(result_path)
        
        plot(result_path)
        
        #remove unnecessary files
        print("Clearing unnecessary files")
        remove_files_in_directory(result_path, f"mv.avi")
        remove_files_in_directory(result_path, f"*.mmap")

        print("Done")
        print(f"Wall Time: {(time.time() - start_time) // 60} min, {(time.time() - start_time) % 60} sec")
        
    except Exception as e:
        dview.terminate()
        raise e
    dview.terminate()




# ### Related functions

def load_movies(imaging_path, desired_movie_len=102):
    # initialize imagej 
    ij = imagej.init('sc.fiji:fiji')

    movie_paths = list(Path(imaging_path).glob('mv*.oir'))

    print("Loading movies")
    movies = []
    # load raw avi movies and concatenate the movies into one movie.
    for path in movie_paths:
        oir = ij.io().open(str(path))
        narr = np.squeeze(ij.py.from_java(oir).values).astype('float32')
        movies.append(cm_movie(narr, fr=15))
    # Error if the number of movies is different.
    if len(movies) != desired_movie_len:
        raise ValueError(f"Error:movies len is {len(movies)}, not {desired_movie_len}")
    movies = movies[:-1]
    return movies


def calc_dropped_frame(movies, result_path, cor_thr=0.2e6, manual_drop_movie_num=None):
    
    """
    Argument
    
    movies: list of movies (eg. [mv, mv001, ..., mv101] (mv: movie array in each trial))
    result_path: path to save txt file in which the number of the trial where any dropped frame exists and the number of dropped frames in the trial will be written
    (ex. 3 198
         6 120
         ...
         101 198).
    cor_thr: 
    manual_drop_movie_num: if you want to also specify the number of trial where no dropped frame exists but you want to remove, set this.
    
    Return
    
    None ('dropped.txt' and 'dropped.png' will be saved in 'result_path')
    """
    def cor(mv):
        cors = []
        for frame in mv:
            mean = np.mean(frame)
            cor = np.mean((frame[:-1,:-1]- mean)*(frame[1:,1:] - mean)*(frame[1:,1:] - mean)*(frame[1:,1:] - mean))
            cors.append(cor)
        return np.array(cors)

    print("Detecting dropped frames")
    start_time = time.time()
        
    # the dropped.png, indicating a histogram of auto correlations in all frames
    al = []
    for i, mv in enumerate(movies):
        cors = cor(mv)
        al.append(cors)
    al = np.concatenate(al)
    plt.hist(al[al < 1000000],bins=500)
    plt.savefig(f"{result_path}/dropped.png")
    plt.show()

    # if the histgram above has values around 0 (meaning there are dropped frame), the dropped.txt will be written
    dropped_text = []
    if (al < cor_thr).any():
        for i, mv in enumerate(movies):
            flag = []
            cors = cor(mv)
            for j, c in enumerate(cors):
                if c < cor_thr:
                    flag.append(j)
            if flag:
                print(i, len(flag))
                dropped_text.append([i,len(flag)])
    if manual_drop_movie_num:
        print("There are dropped frames")
        if type(manual_drop_movie_num) == int:
            manual_drop_movie_num = [manual_drop_movie_num]
        for n in manual_drop_movie_num:
            dropped_text.append([n, -1])
            print(n, -1)
    else:
        print("No dropped frame")

    dropped_text = np.array(dropped_text)
    np.savetxt(f"{result_path}/dropped.txt", dropped_text, fmt='%.i')
    print(f"Output: dropped.png, dropped.txt")
    print(f"{(time.time() - start_time) // 60} min, {(time.time() - start_time) % 60} sec")
    print("")
    

def make_logfile(behavior_path, result_path, double=False):
    print("Making a log csv")
    #load the last log file in the bpath
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
                
    log.to_csv(f"{result_path}/log_full.csv", index=False)
    dropped_text = np.loadtxt(f"{result_path}/dropped.txt").astype(int)
    if len(dropped_text) > 0:
        if dropped_text.ndim == 1:
            dropped_text = dropped_text[0]    
        else:
            dropped_text = dropped_text[:,0]
        dropped_idx = np.ones(len(log)).astype(bool)
        dropped_idx[dropped_text] = 0
        log = log.iloc[dropped_idx]
    log.to_csv(f"{result_path}/log.csv", index=False)
    print("Output: log.csv, log_full.csv")
    print("")
    

def movie_concatenated(movies, result_path, ds_rate):
    print("Concatenating Movies")
    dropped_text = np.loadtxt(f"{result_path}/dropped.txt").astype(int)
    # if there are dropped trials, exclude those from movies.
    if len(dropped_text) > 0:
        if dropped_text.ndim == 1:
            movies = [mv for i, mv in enumerate(movies) if i != dropped_text[0]]
        else:
            movies = [mv for i, mv in enumerate(movies) if i not in dropped_text[:,0]]
    movie = cm.concatenate(movies)
    # do spatial downsampling and save movie for visualization
    movie2 = movie.resize(fx=1/ds_rate, fy=1/ds_rate)
    mv_name = fr"{result_path}\mv.avi"
    movie2.save(mv_name, q_max=100, q_min=0)
    print("Output: mv.avi")
    print("")
    
    return mv_name


def motion_correction(mv_name, result_path, dview):
    start_time = time.time()
    print("Performing Motion Correction")
    # set params
    fnames = [mv_name]
    # dataset dependent parameters
    frate = 15
    decay_time = 0.4                 # length of a typical transient in seconds
    # motion correction parameters
    pw_rigid = True         # flag for performing piecewise-rigid motion correction (otherwise just rigid)
    max_shifts = (6, 6)      # maximum allowed rigid shift
    strides = (48, 48)       # start a new patch for pw-rigid motion correction every x pixels
    overlaps = (24, 24)      # overlap between pathes (size of patch strides+overlaps)
    max_deviation_rigid = 3  # maximum deviation allowed for patch with respect to rigid shifts
    border_nan = 'copy'      # replicate values along the boundaries

    mc_dict = {
        'fr': frate,
        'decay_time': decay_time,
        'pw_rigid': pw_rigid,
        'max_shifts': max_shifts,
        'strides': strides,
        'overlaps': overlaps,
        'max_deviation_rigid': max_deviation_rigid,
        'border_nan': border_nan
    }
    bord_px = 0
    opts = params.CNMFParams(params_dict=mc_dict)

    # do motion correction rigid
    mc = MotionCorrect(fnames, dview=dview, **opts.get_group('motion'))
    # estimate motion correction
    mc.motion_correct()
    # make the motion corrected movie
    mv_mc = mc.apply_shifts_movie(fnames[0])
    # save the motion corrected movie and the max projection image for visualization
    mv_mc.save(fr"{result_path}\mv_MotionCorrected.avi", q_max=100, q_min=0)
    mv_mc.max(axis=0).save(fr"{result_path}\mv_MC_MaxProjectionImage.tif")
    # save the motion corrected movie as the memmap file for further processing
    mv_mmap_name = mv_mc.save(fr"{result_path}\mvMMAP.mmap", order='C')
    print("Output: mv_MotionCorrected.avi, mv_MC_MaxProjectionImage.jpg, mvMMAP.mmap")
    print(f"{(time.time() - start_time) // 60} min, {(time.time() - start_time) % 60} sec")
    print("")
    
    return mv_mmap_name


def cnmfe(mv_mmap_name, result_path, n_processes, dview):
    print("Performing CNMFe")
    start_time = time.time()
        
    # dataset dependent parameters
    frate = 15
    decay_time = 0.4                 # length of a typical transient in seconds
    # parameters for source extraction and deconvolution
    p = 1               # order of AR model
    K = None            # upper bound on number of components per patch, in general None
    gSig = (3, 3)       # gaussian width of a 2D gaussian kernel, which approximates a neuron
    gSiz = (13, 13)     # average diameter of a neuron, in general 4*gSig+1
    Ain = None          # possibility to seed with predetermined binary masks
    merge_thr = .7      # merging threshold, max correlation allowed
    rf = 40             # half-size of the patches in pixels. e.g., if rf=40, patches are 80x80. default=49
    stride_cnmf = 20    # amount of overlap between the patches in pixels. default=20
    #                     (keep it at least large as gSiz, i.e 4 times the neuron size gSig)
    tsub = 2            # downsampling factor in time for initialization,
    #                     increase if you have memory problems
    ssub = 1            # downsampling factor in space for initialization,
    #                     increase if you have memory problems
    #                     you can pass them here as boolean vectors
    low_rank_background = None  # None leaves background of each patch intact,
    #                     True performs global low-rank approximation if gnb>0
    gnb = 0             # number of background components (rank) if positive,
    #                     else exact ring model with following settings
    #                         gnb= 0: Return background as b and W
    #                         gnb=-1: Return full rank background B
    #                         gnb<-1: Don't return background
    nb_patch = 0        # number of background components (rank) per patch if gnb>0,
    #                     else it is set automatically
    min_corr = .8       # min peak value from correlation image
    min_pnr = 10        # min peak to noise ration from PNR image
    ssub_B = 2          # additional downsampling factor in space for background
    ring_size_factor = 1.4  # radius of ring is gSiz*ring_size_factor

    opts = params.CNMFParams(params_dict={'fr': frate,
                                    'decay_time': decay_time,
                                    'method_init': 'corr_pnr',  # use this for 1 photon
                                    'K': K,
                                    'gSig': gSig,
                                    'gSiz': gSiz,
                                    'merge_thr': merge_thr,
                                    'p': p,
                                    'tsub': tsub,
                                    'ssub': ssub,
                                    'rf': rf,
                                    'stride': stride_cnmf,
                                    'only_init': True,    # set it to True to run CNMF-E
                                    'nb': gnb,
                                    'nb_patch': nb_patch,
                                    'method_deconvolution': 'oasis',       # could use 'cvxpy' alternatively
                                    'low_rank_background': low_rank_background,
                                    'update_background_components': True,  # sometimes setting to False improve the results
                                    'min_corr': min_corr,
                                    'min_pnr': min_pnr,
                                    'normalize_init': False,               # just leave as is
                                    'center_psf': True,                    # leave as is for 1 photon
                                    'ssub_B': ssub_B,
                                    'ring_size_factor': ring_size_factor,
                                    'del_duplicates': True,                # whether to remove duplicates from initialization
                                    'border_pix': 0})                # number of pixels to not consider in the borders)

    # load memory mappable file
    fname_new = mv_mmap_name

    Yr, dims, T = cm.load_memmap(fname_new)
    images = Yr.T.reshape((T,) + dims, order='F')

    # do cnmf-e
    opts.change_params(params_dict={'dims': dims})
    cnm = cnmf.CNMF(n_processes=n_processes, dview=dview, Ain=Ain, params=opts)
    cnm.fit(images)

    #%% COMPONENT EVALUATION
    # the components are evaluated in three ways:
    #   a) the shape of each component must be correlated with the data
    #   b) a minimum peak SNR is required over the length of a transient
    #   c) each shape passes a CNN based classifier

    min_SNR = 3            # adaptive way to set threshold on the transient size
    r_values_min = 0.85    # threshold on space consistency (if you lower more components
    #                        will be accepted, potentially with worst quality)
    cnm.params.set('quality', {'min_SNR': min_SNR,
                               'rval_thr': r_values_min,
                               'use_cnn': False})
    cnm.estimates.evaluate_components(images, cnm.params, dview=dview)

    print(' ***** ')
    print('Number of total components: ', len(cnm.estimates.C))
    print('Number of accepted components: ', len(cnm.estimates.idx_components))

    # save a cnmf-e result
    cnm.save(fr"{result_path}/cnm.hdf5")

    # save a contours plot of components
    cm.utils.visualization.plot_contours(cnm.estimates.A, images.max(axis=0))
    plt.savefig(fr"{result_path}/cell_contours.png")

    # save spatial footprints for cellreg
    A = np.array(cnm.estimates.A.todense()).reshape(cnm.dims + (-1,), order='F').transpose((2,0,1))[cnm.estimates.idx_components]
    savemat(fr"{result_path}/footprints.mat", {'footprints':A})

    print("Output: cnm.hdf5, cell_contours.png, footprints.mat")
    print(f"{(time.time() - start_time) // 60} min, {(time.time() - start_time) % 60} sec")
    print("")

    
def make_data_object(result_path):
    print("Making data object")
    start_time = time.time()
    # making data.pickle
    data = Data(path=result_path, cellregdir=False)
    pd.to_pickle(data, f"{result_path}/data.pkl")
    print("Output: data.pkl")
    print(f"{(time.time() - start_time) // 60} min, {(time.time() - start_time) % 60} sec")
    print("")

def make_data_object_suite2p(result_path):
    print("Making data object")
    start_time = time.time()
    # making data.pickle
    data = Data_suite2p(path=result_path)
    pd.to_pickle(data, f"{result_path}/data.pkl")
    print("Output: data.pkl")
    print(f"{(time.time() - start_time) // 60} min, {(time.time() - start_time) % 60} sec")
    print("")
    

def plot(result_path):
    print("Outputting result plots")
    start_time = time.time()
        
    figpath = f"{result_path}/Figs"
    Path(figpath).mkdir(parents=True, exist_ok=True)
    data = pd.read_pickle(f"{result_path}/data.pkl")
    logful = data.logful
    log = data.log
    
    fig, ax = plt.subplots(1,1)
    performance_bar_panel(logful, ax)
    fig.tight_layout()
    fig.savefig(f"{figpath}/performance.png")
    plt.close()
    
    fig, ax = plt.subplots(1,1)
    performance_bar_panel(log, ax)
    fig.tight_layout()
    fig.savefig(f"{figpath}/performance_only_valid_trials.png")
    plt.close()
    
    fig, ax = plt.subplots(1,1)
    lick_history_panel(logful, ax)
    fig.tight_layout()
    fig.savefig(f"{figpath}/lick_history.png")
    plt.close()
    
    fig, ax = plt.subplots(1,1)
    lick_history_panel(log, ax)
    fig.tight_layout()
    fig.savefig(f"{figpath}/lick_history_only_valid_trials.png")
    plt.close()
    
    fig, ax = plt.subplots(1,2, figsize=(10,6))
    lick_rasterplot_panel(logful[logful.type=='go'], ax[0])
    ax[0].set_title('Go')
    if len(logful[logful.type=='nogo']) > 0:
        lick_rasterplot_panel(logful[logful.type=='nogo'], ax[1])
        ax[1].set_title('NoGo')
    fig.tight_layout()
    fig.savefig(f"{figpath}/lick_raster.png")
    plt.close()    
    
    fig, ax = plt.subplots(1,2, figsize=(10,6))
    lick_rasterplot_panel(log[log.type=='go'], ax[0])
    ax[0].set_title('Go')
    if len(logful[logful.type=='nogo']) > 0:
        lick_rasterplot_panel(log[log.type=='nogo'], ax[1])
        ax[1].set_title('NoGo')
    fig.tight_layout()
    fig.savefig(f"{figpath}/lick_raster_only_valid_trials.png")
    plt.close()    
    
    df = data.df
    go = df[df.type=='go']
    nogo = df[df.type=='nogo']
    cellfigpath = f"{figpath}/Cells"
    Path(cellfigpath).mkdir(parents=True, exist_ok=True)
    for n in range(data.ncell):
        fig, ax = plt.subplots(1,5, figsize=(20,6))
        footprint_panel(data.cnm, n, ax[0])
        responce_panel(go, n, 'Go', ax[1], fig)
        if len(nogo) > 0:
            responce_panel(nogo, n, 'NoGo', ax[2], fig)
        mean_responce_panel(data.go_mean, data.go_std, n, 'Go (mean)', ax[3], method='z-score')
        if len(nogo) > 0:
            mean_responce_panel(data.nogo_mean, data.nogo_std, n, 'NoGo (mean)', ax[4], method='z-score')
        fig.tight_layout()
        fig.savefig(f"{cellfigpath}/cell{n:04}.png")
        plt.close()
    
    print(f"Output: Figs")
    print(f"{(time.time() - start_time) // 60} min, {(time.time() - start_time) % 60} sec")
    print("")
    
def plot_suite2p(result_path):
    print("Outputting result plots")
    start_time = time.time()
        
    figpath = f"{result_path}/Figs"
    Path(figpath).mkdir(parents=True, exist_ok=True)
    data = pd.read_pickle(f"{result_path}/data.pkl")
    logful = data.logful
    log = data.log
    
    fig, ax = plt.subplots(1,1)
    performance_bar_panel(logful, ax)
    fig.tight_layout()
    fig.savefig(f"{figpath}/performance.png")
    plt.close()
    
    fig, ax = plt.subplots(1,1)
    performance_bar_panel(log, ax)
    fig.tight_layout()
    fig.savefig(f"{figpath}/performance_only_valid_trials.png")
    plt.close()
    
    fig, ax = plt.subplots(1,1)
    lick_history_panel(logful, ax)
    fig.tight_layout()
    fig.savefig(f"{figpath}/lick_history.png")
    plt.close()
    
    fig, ax = plt.subplots(1,1)
    lick_history_panel(log, ax)
    fig.tight_layout()
    fig.savefig(f"{figpath}/lick_history_only_valid_trials.png")
    plt.close()
    
    fig, ax = plt.subplots(1,2, figsize=(10,6))
    lick_rasterplot_panel(logful[logful.type=='go'], ax[0])
    ax[0].set_title('Go')
    if len(logful[logful.type=='nogo']) > 0:
        lick_rasterplot_panel(logful[logful.type=='nogo'], ax[1])
        ax[1].set_title('NoGo')
    fig.tight_layout()
    fig.savefig(f"{figpath}/lick_raster.png")
    plt.close()    
    
    fig, ax = plt.subplots(1,2, figsize=(10,6))
    lick_rasterplot_panel(log[log.type=='go'], ax[0])
    ax[0].set_title('Go')
    if len(logful[logful.type=='nogo']) > 0:
        lick_rasterplot_panel(log[log.type=='nogo'], ax[1])
        ax[1].set_title('NoGo')
    fig.tight_layout()
    fig.savefig(f"{figpath}/lick_raster_only_valid_trials.png")
    plt.close()    
    
    df = data.df
    go = df[df.type=='go']
    nogo = df[df.type=='nogo']
    cellfigpath = f"{figpath}/Cells"
    Path(cellfigpath).mkdir(parents=True, exist_ok=True)
    for n in range(data.ncell):
        fig, ax = plt.subplots(1,4, figsize=(20,6))
        responce_panel(go, n, 'Go', ax[0], fig)
        if len(nogo) > 0:
            responce_panel(nogo, n, 'NoGo', ax[1], fig)
        mean_responce_panel(data.go_mean, data.go_std, n, 'Go (mean)', ax[2], method='z-score')
        if len(nogo) > 0:
            mean_responce_panel(data.nogo_mean, data.nogo_std, n, 'NoGo (mean)', ax[3], method='z-score')
        fig.tight_layout()
        fig.savefig(f"{cellfigpath}/cell{n:04}.png")
        plt.close()
    
    print(f"Output: Figs")
    print(f"{(time.time() - start_time) // 60} min, {(time.time() - start_time) % 60} sec")
    print("")
    
def remove_files_in_directory(dirpath, match):
    filepaths = list(Path(dirpath).glob(match))
    for path in filepaths:
        path.unlink()