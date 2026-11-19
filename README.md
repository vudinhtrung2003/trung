# trung
# Mandarin Tone F0 &amp; DTW Prototype  Prototype module for extracting pitch contours ($F_0$ via `pYIN`), normalizing semitone/z-score contours, and scoring Mandarin tones with Dynamic Time Warping (DTW) for `taikhoanpremium.io.vn`.
# Mandarin Tone F0 & DTW Prototype

Prototype module for extracting pitch contours ($F_0$ via `pYIN`), normalizing semitone/z-score contours, and scoring Mandarin tones with Dynamic Time Warping (DTW) for `taikhoanpremium.io.vn`.
import numpy as np
import librosa
from typing import Dict, Tuple

# Mẫu đường cong cao độ chuẩn hóa (5-level Chao tone scale -> Z-score)
TONE_TEMPLATES: Dict[int, np.ndarray] = {
    1: np.array([1.2, 1.2, 1.2, 1.2, 1.2]),     # Thanh 1 (55): ngang cao
    2: np.array([-0.4, 0.0, 0.4, 0.8, 1.2]),    # Thanh 2 (35): đi lên
    3: np.array([-0.5, -1.2, -1.5, -0.8, 0.2]), # Thanh 3 (214): giáng rồi thăng
    4: np.array([1.4, 0.6, -0.2, -0.9, -1.5]),  # Thanh 4 (51): giáng mạnh
}

def extract_normalized_f0(audio_path: str, sr: int = 16000) -> np.ndarray:
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    y_trimmed, _ = librosa.effects.trim(y, top_db=25)

    f0, voiced_flag, _ = librosa.pyin(
        y_trimmed, fmin=65.0, fmax=500.0, sr=sr
    )
    voiced_f0 = f0[voiced_flag & ~np.isnan(f0)]
    if len(voiced_f0) < 5:
        raise ValueError("Không đủ dữ liệu hữu thanh (voiced frames) để phân tích thanh điệu")

    semitones = 12.0 * np.log2(voiced_f0 / 100.0)
    z_contour = (semitones - np.mean(semitones)) / (np.std(semitones) + 1e-6)
    return z_contour

def score_mandarin_tone(user_contour: np.ndarray, target_tone: int) -> Tuple[float, str]:
    template = TONE_TEMPLATES[target_tone]
    D, _ = librosa.sequence.dtw(
        user_contour.reshape(1, -1),
        template.reshape(1, -1),
        metric="euclidean"
    )
    normalized_cost = D[-1, -1] / max(len(user_contour), len(template))
    score = float(np.clip(100.0 * np.exp(-0.65 * normalized_cost), 0.0, 100.0))

    slope = float(user_contour[-1] - user_contour[0])
    feedback = "Đạt chuẩn" if score >= 75 else f"Lệch hình dạng thanh {target_tone} (độ dốc đo được: {slope:.2f})"
    return round(score, 1), feedback

if __name__ == "__main__":
    simulated_tone4 = np.array([1.3, 0.7, 0.1, -0.7, -1.3, -1.4])
    score, msg = score_mandarin_tone(simulated_tone4, target_tone=4)
    print({"target_tone": 4, "score": score, "feedback": msg})
