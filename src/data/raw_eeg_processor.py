from pathlib import Path
import numpy as np
import mne
from sklearn.utils import shuffle
from sklearn.discriminant_analysis import _cov
from tqdm import tqdm


def _build_whitener_from_train(train_packed, peak_threshold_v=500e-6):
    # train_packed: [n_cond, n_rep, n_ch, n_time]
    n_cond, n_rep, n_ch, n_time = train_packed.shape
    trials = train_packed.reshape(-1, n_ch, n_time)

    finite_mask = np.isfinite(trials).all(axis=(1, 2))
    peak_mask = np.abs(trials).max(axis=(1, 2)) < peak_threshold_v
    good_mask = finite_mask & peak_mask

    n_total = len(trials)
    n_good = int(good_mask.sum())

    if n_good >= max(50, n_ch):
        cov_trials = trials[good_mask]
        print(f"    MVNN: using {n_good}/{n_total} training trials after peak filter (< {peak_threshold_v*1e6:.0f} µV)")
    else:
        cov_trials = trials[finite_mask]
        print(f"    MVNN: peak filter left too few trials ({n_good}/{n_total}); using finite-only trials {cov_trials.shape[0]}/{n_total}")

    if cov_trials.shape[0] < 2:
        print("    WARNING: too few trials for MVNN; using identity.")
        return np.eye(n_ch, dtype=np.float64)

    sigma_t = np.empty((n_time, n_ch, n_ch), dtype=np.float64)
    for t in range(n_time):
        sigma_t[t] = _cov(cov_trials[:, :, t], shrinkage="auto")

    sigma_tot = sigma_t.mean(axis=0)
    sigma_tot = 0.5 * (sigma_tot + sigma_tot.T)

    evals, evecs = np.linalg.eigh(sigma_tot)

    if (not np.isfinite(evals).all()) or (evals.max() <= 0):
        print("    WARNING: eigendecomposition unstable; using diagonal whitening.")
        channel_var = np.var(cov_trials, axis=(0, 2), ddof=1)
        channel_var = np.maximum(channel_var, 1e-12)
        return np.diag(1.0 / np.sqrt(channel_var))

    lam_floor = max(float(evals.max()) * 1e-6, 1e-12)
    evals_clip = np.clip(evals, lam_floor, None)
    cond_num = float(evals_clip.max() / evals_clip.min())

    if cond_num > 1e8:
        print(f"    WARNING: covariance ill-conditioned (cond={cond_num:.2e}); using diagonal whitening.")
        channel_var = np.var(cov_trials, axis=(0, 2), ddof=1)
        channel_var = np.maximum(channel_var, 1e-12)
        sigma_inv = np.diag(1.0 / np.sqrt(channel_var))
    else:
        sigma_inv = (evecs @ np.diag(evals_clip ** -0.5) @ evecs.T).astype(np.float64)
        max_abs = float(np.abs(sigma_inv).max())
        print(f"    MVNN: eig floor={lam_floor:.3e}, cond={cond_num:.2e}, |W|_max={max_abs:.3e}")

    if not np.isfinite(sigma_inv).all():
        print("    WARNING: whitening matrix contains NaN/Inf; using diagonal whitening.")
        channel_var = np.var(cov_trials, axis=(0, 2), ddof=1)
        channel_var = np.maximum(channel_var, 1e-12)
        sigma_inv = np.diag(1.0 / np.sqrt(channel_var))

    return sigma_inv


def _apply_whitener(packed, sigma_inv):
    # packed: [n_cond, n_rep, n_ch, n_time]
    flat = packed.reshape(-1, packed.shape[2], packed.shape[3])          # [N, C, T]
    flat_w = (flat.swapaxes(1, 2) @ sigma_inv).swapaxes(1, 2)            # [N, C, T]
    return flat_w.reshape(packed.shape)


def preprocess_raw_eeg(sub, project_dir, sfreq=100, seed=20200220):
    print(f"Starting raw preprocessing for Subject {sub:02d}...")
    project_dir = Path(project_dir)

    epoched_test, epoched_train = [], []
    img_conditions_train = []

    ch_names_final = None
    times_final = None

    for s in range(4):
        print(f"  Processing Session {s+1}...")
        for data_part in ["test", "training"]:
            raw_file = (
                project_dir
                / "data"
                / "things-eeg2"
                / "raw_data"
                / f"sub-{sub:02d}"
                / f"ses-{s+1:02d}"
                / f"raw_eeg_{data_part}.npy"
            )

            if not raw_file.exists():
                print(f"    File not found: {raw_file}")
                continue

            eeg_data = np.load(raw_file, allow_pickle=True).item()
            ch_names = eeg_data["ch_names"]
            ch_types = eeg_data["ch_types"]
            raw_eeg_data = eeg_data["raw_eeg_data"]

            info = mne.create_info(ch_names, eeg_data["sfreq"], ch_types)
            raw = mne.io.RawArray(raw_eeg_data, info, verbose=False)

            events = mne.find_events(raw, stim_channel="stim", verbose=False)
            events = events[events[:, 2] != 99999]

            epochs = mne.Epochs(
                raw,
                events,
                tmin=-0.2,
                tmax=0.8,
                baseline=(None, 0),
                preload=True,
                verbose=False,
            )

            if sfreq < 1000:
                epochs.resample(sfreq, verbose=False)

            eeg_ch_idx = mne.pick_types(epochs.info, eeg=True, stim=False)
            data = epochs.get_data(copy=False)[:, eeg_ch_idx, :]
            event_ids = epochs.events[:, 2]
            img_cond = np.unique(event_ids)

            max_rep = 20 if data_part == "test" else 2
            sorted_data = np.zeros((len(img_cond), max_rep, data.shape[1], data.shape[2]), dtype=np.float32)

            for i in range(len(img_cond)):
                idx = np.where(event_ids == img_cond[i])[0]
                idx = shuffle(idx, random_state=seed, n_samples=min(len(idx), max_rep))
                sorted_data[i, :len(idx)] = data[idx]

            if data_part == "test":
                epoched_test.append(sorted_data)
                ch_names_final = [epochs.info["ch_names"][i] for i in eeg_ch_idx]
                times_final = epochs.times
            else:
                epoched_train.append(sorted_data)
                img_conditions_train.append(img_cond)

    if not epoched_test or not epoched_train:
        print("  Missing data. Aborting.")
        return

    print("  Applying Multivariate Noise Normalization (MVNN)...")
    whitened_test = []
    whitened_train = []

    for s in range(4):
        print(f"  Session {s+1} MVNN...")
        sigma_inv = _build_whitener_from_train(epoched_train[s], peak_threshold_v=500e-6)
        whitened_train.append(_apply_whitener(epoched_train[s], sigma_inv))
        whitened_test.append(_apply_whitener(epoched_test[s], sigma_inv))

        train_std_before = float(epoched_train[s].std())
        train_std_after = float(whitened_train[s].std())
        test_std_before = float(epoched_test[s].std())
        test_std_after = float(whitened_test[s].std())

        print(
            f"    train std: {train_std_before:.4f} -> {train_std_after:.4f} | "
            f"test std: {test_std_before:.4f} -> {test_std_after:.4f}"
        )

        if not np.isfinite(whitened_train[s]).all() or not np.isfinite(whitened_test[s]).all():
            raise ValueError(f"NaN/Inf detected after whitening in session {s+1}")

    print("  Merging sessions and saving preprocessed arrays...")

    merged_test = whitened_test[0]
    for s in range(1, 4):
        merged_test = np.append(merged_test, whitened_test[s], 1)

    idx = shuffle(np.arange(0, merged_test.shape[1]), random_state=seed)
    merged_test = merged_test[:, idx]

    test_dict = {
        "preprocessed_eeg_data": merged_test,
        "ch_names": ch_names_final,
        "times": times_final,
    }

    save_dir = project_dir / "data" / "things-eeg2" / "preprocessed" / f"sub-{sub:02d}"
    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_dir / "preprocessed_eeg_test.npy", test_dict)

    white_data = whitened_train[0]
    img_cond = img_conditions_train[0]
    for s in range(1, 4):
        white_data = np.append(white_data, whitened_train[s], 0)
        img_cond = np.append(img_cond, img_conditions_train[s], 0)

    merged_train = np.zeros(
        (len(np.unique(img_cond)), white_data.shape[1] * 2, white_data.shape[2], white_data.shape[3]),
        dtype=np.float32,
    )
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

    if not np.isfinite(merged_train).all():
        raise ValueError("NaN/Inf detected in merged_train")
    if not np.isfinite(merged_test).all():
        raise ValueError("NaN/Inf detected in merged_test")

    train_dict = {
        "preprocessed_eeg_data": merged_train,
        "ch_names": ch_names_final,
        "times": times_final,
    }

    np.save(save_dir / "preprocessed_eeg_training.npy", train_dict)
    print(f"  Successfully saved 63-channel representations for Subject {sub:02d} to {save_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--project_dir", type=str, default=".")
    args = parser.parse_args()

    preprocess_raw_eeg(args.subject, args.project_dir)