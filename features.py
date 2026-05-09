import numpy as np
import librosa

TARGET_SR = 16000
N_MFCC    = 40


def extract_mfcc(source, max_len: int, sr: int = TARGET_SR, include_delta: bool = False) -> np.ndarray | None:
    """
    Extract MFCC features from a file path or a raw numpy audio array.

    Args:
        source        : str (file path) or np.ndarray (raw audio signal)
        max_len       : fixed length along the time axis (number of frames)
        sr            : sample rate — resamples if it differs from TARGET_SR
        include_delta : if True, appends delta + delta2 -> (max_len, N_MFCC*3)

    Returns:
        np.ndarray of shape (max_len, N_MFCC) or (max_len, N_MFCC*3), or None on error
    """
    try:
        if isinstance(source, str):
            audio, file_sr = librosa.load(source, sr=None, mono=True)
            if file_sr != TARGET_SR:
                audio = librosa.resample(audio, orig_sr=file_sr, target_sr=TARGET_SR)
        else:
            audio = source.astype(np.float32)
            if sr != TARGET_SR:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)

        mfcc = librosa.feature.mfcc(y=audio, sr=TARGET_SR, n_mfcc=N_MFCC)

        if include_delta:
            delta  = librosa.feature.delta(mfcc)
            delta2 = librosa.feature.delta(mfcc, order=2)
            features = np.vstack([mfcc, delta, delta2])  # (N_MFCC*3, T)
        else:
            features = mfcc                              # (N_MFCC, T)

        if features.shape[1] < max_len:
            features = np.pad(features, ((0, 0), (0, max_len - features.shape[1])), mode="constant")
        else:
            features = features[:, :max_len]

        return features.T.astype(np.float32)   # (max_len, N_MFCC) or (max_len, N_MFCC*3)

    except Exception as e:
        print(f"[features] error: {e}")
        return None
