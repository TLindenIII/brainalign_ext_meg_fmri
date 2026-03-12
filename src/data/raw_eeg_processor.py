import os
from pathlib import Path
import numpy as np
import mne
from sklearn.utils import shuffle
from sklearn.discriminant_analysis import _cov
import scipy.linalg
from tqdm import tqdm

def preprocess_raw_eeg(sub, project_dir, sfreq=100, seed=20200220):
    """
    Epochs the raw THINGSEEG2 63-channel data downsampled to 100Hz without
    discarding spatial channels. Apples MVNN covariance scaling.
    """
    print(f"Starting raw preprocessing for Subject {sub:02d}...")
    project_dir = Path(project_dir)
    epoched_test, epoched_train = [], []
    img_conditions_train = []
    
    ch_names_final = None
    times_final = None
    
    # 1. EPOCHING
    for s in range(4):
        print(f"  Processing Session {s+1}...")
        for data_part in ['test', 'training']:
            raw_file = project_dir / 'data' / 'things-eeg2' / 'raw_data' / f'sub-{sub:02d}' / f'ses-{s+1:02d}' / f'raw_eeg_{data_part}.npy'
            
            if not raw_file.exists():
                print(f"    File not found: {raw_file}")
                continue
                
            eeg_data = np.load(raw_file, allow_pickle=True).item()
            ch_names = eeg_data['ch_names']
            ch_types = eeg_data['ch_types']
            raw_eeg_data = eeg_data['raw_eeg_data']
            
            info = mne.create_info(ch_names, eeg_data['sfreq'], ch_types)
            raw = mne.io.RawArray(raw_eeg_data, info, verbose=False)
            
            events = mne.find_events(raw, stim_channel='stim', verbose=False)
            
            # Reject 99999 (target catch trials)
            events = events[events[:, 2] != 99999]
            
            # Epoch the data 
            # BrainAlign baseline uses (-0.2, 0.8) which naturally yields 100 features at 100 Hz
            epochs = mne.Epochs(raw, events, tmin=-0.2, tmax=0.8, baseline=(None, 0), preload=True, verbose=False)
            
            if sfreq < 1000:
                epochs.resample(sfreq, verbose=False)
                
            data = epochs.get_data(copy=False)
            event_ids = epochs.events[:, 2]
            img_cond = np.unique(event_ids)
            
            max_rep = 20 if data_part == 'test' else 2
            
            # Shape: Image conditions × EEG repetitions × 63 Channels × 100 Time points
            sorted_data = np.zeros((len(img_cond), max_rep, data.shape[1], data.shape[2]))
            
            for i in range(len(img_cond)):
                idx = np.where(event_ids == img_cond[i])[0]
                idx = shuffle(idx, random_state=seed, n_samples=min(len(idx), max_rep))
                sorted_data[i, :len(idx)] = data[idx]
                
            if data_part == 'test':
                epoched_test.append(sorted_data)
                ch_names_final = epochs.info['ch_names']
                times_final = epochs.times
            else:
                epoched_train.append(sorted_data)
                img_conditions_train.append(img_cond)

    if not epoched_test or not epoched_train:
        print("  Missing data. Aborting.")
        return

    # 2. MVNN
    print("  Applying Multivariate Noise Normalization (MVNN)...")
    whitened_test = []
    whitened_train = []
    for s in range(4):
        session_data = [epoched_test[s], epoched_train[s]]
        
        # Data partitions covariance matrix of shape:
        # Data partitions × EEG channels × EEG channels
        sigma_part = np.empty((len(session_data), session_data[0].shape[2], session_data[0].shape[2]))
        
        for p in range(sigma_part.shape[0]):
            sigma_cond = np.empty((session_data[p].shape[0], session_data[0].shape[2], session_data[0].shape[2]))
            for i in tqdm(range(session_data[p].shape[0]), desc=f"Session {s+1} Partition {p}"):
                cond_data = session_data[p][i]
                sigma_cond[i] = np.mean([_cov(cond_data[:,:,t], shrinkage='auto') for t in range(cond_data.shape[2])], axis=0)
            
            sigma_part[p] = sigma_cond.mean(axis=0)
            
        sigma_tot = sigma_part.mean(axis=0)
        
        # Tikhonov regularization: add small diagonal term to prevent singular matrix
        # A near-singular covariance matrix (e.g., sub-05) causes fractional_matrix_power
        # to return NaN. This is the standard fix used in robust MVNN implementations.
        reg = 1e-6 * np.eye(sigma_tot.shape[0]) 
        sigma_tot_reg = sigma_tot + reg
        sigma_inv = scipy.linalg.fractional_matrix_power(sigma_tot_reg, -0.5)
        
        # Safety guard: if inversion still produced NaN/Inf, fall back to identity (no whitening)
        if not np.isfinite(sigma_inv).all():
            print(f"  WARNING: MVNN inversion failed for Session {s+1} even with regularization. Using identity (no whitening).")
            sigma_inv = np.eye(sigma_tot.shape[0])
        
        whitened_test.append(np.reshape((np.reshape(session_data[0], (-1, session_data[0].shape[2], session_data[0].shape[3])).swapaxes(1, 2) @ sigma_inv).swapaxes(1, 2), session_data[0].shape))
        whitened_train.append(np.reshape((np.reshape(session_data[1], (-1, session_data[1].shape[2], session_data[1].shape[3])).swapaxes(1, 2) @ sigma_inv).swapaxes(1, 2), session_data[1].shape))

    # 3. MERGING AND SAVING
    print("  Merging sessions and saving preprocessed arrays...")
    
    # Merge Test
    merged_test = whitened_test[0]
    for s in range(1, 4):
        merged_test = np.append(merged_test, whitened_test[s], 1)
        
    idx = shuffle(np.arange(0, merged_test.shape[1]), random_state=seed)
    merged_test = merged_test[:, idx]
    
    test_dict = {
        'preprocessed_eeg_data': merged_test,
        'ch_names': ch_names_final,
        'times': times_final
    }
    
    save_dir = project_dir / 'data' / 'things-eeg2' / 'preprocessed' / f'sub-{sub:02d}'
    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_dir / 'preprocessed_eeg_test.npy', test_dict)
    
    # Merge Train
    white_data = whitened_train[0]
    img_cond = img_conditions_train[0]
    for s in range(1, 4):
        white_data = np.append(white_data, whitened_train[s], 0)
        img_cond = np.append(img_cond, img_conditions_train[s], 0)
        
    merged_train = np.zeros((len(np.unique(img_cond)), white_data.shape[1] * 2, white_data.shape[2], white_data.shape[3]))
    unique_conds = np.unique(img_cond)
    
    for i in range(len(unique_conds)):
        idx_train = np.where(img_cond == unique_conds[i])[0]
        for r in range(len(idx_train)):
            if r == 0:
                ordered_data = white_data[idx_train[r]]
            else:
                ordered_data = np.append(ordered_data, white_data[idx_train[r]], 0)
        merged_train[i] = ordered_data
        
    idx_train_shuf = shuffle(np.arange(0, merged_train.shape[1]), random_state=seed)
    merged_train = merged_train[:, idx_train_shuf]
    
    train_dict = {
        'preprocessed_eeg_data': merged_train,
        'ch_names': ch_names_final,
        'times': times_final
    }
    
    np.save(save_dir / 'preprocessed_eeg_training.npy', train_dict)
    print(f"  Successfully saved 63-channel representations for Subject {sub:02d} to {save_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--project_dir", type=str, default=".")
    args = parser.parse_args()
    
    preprocess_raw_eeg(args.subject, args.project_dir)
