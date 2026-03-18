import cv2
import numpy as np
from dataclasses import dataclass, asdict

@dataclass
class SafetyProxyResult:
    score: float
    verdict: str
    visibility_score: float
    hallucination_score: float
    photometric_score: float
    haze_index: float
    contrast: float
    edge_density: float
    laplacian_var: float
    clipping_rate: float
    exposure_mean: float
    color_cast: float
    # Pair-mode extras
    haze_reduction_ratio: float | None = None
    edge_inflation_ratio: float | None = None
    hf_ratio: float | None = None
    clipping_growth_ratio: float | None = None


class AVSafetyProxyEvaluator:
    def __init__(
        self,
        dark_channel_kernel: int = 15,
        edge_low: int = 80,
        edge_high: int = 160,
        gamma_H: float = 0.85,   # acceptable haze ratio threshold
        gamma_E: float = 1.50,   # acceptable edge inflation threshold
        gamma_F: float = 2.00,   # acceptable HF inflation threshold
        gamma_C: float = 1.50,   # acceptable clipping growth threshold
    ):
        self.dark_channel_kernel = dark_channel_kernel
        self.edge_low = edge_low
        self.edge_high = edge_high
        self.gamma_H = gamma_H
        self.gamma_E = gamma_E
        self.gamma_F = gamma_F
        self.gamma_C = gamma_C

    # -------------------------
    # Public API
    # -------------------------
    def evaluate(self, image, hazy_reference=None) -> SafetyProxyResult:
        img = self._load_image(image)
        metrics = self._compute_single_metrics(img)

        # Absolute, single-image score
        visibility_score = self._single_visibility_score(metrics)
        hallucination_score = self._single_hallucination_score(metrics)
        photometric_score = self._single_photometric_score(metrics)

        haze_reduction_ratio = None
        edge_inflation_ratio = None
        hf_ratio = None
        clipping_growth_ratio = None

        # If hazy reference exists, compute pairwise SPI-like metrics
        if hazy_reference is not None:
            hazy = self._load_image(hazy_reference)
            ref = self._compute_single_metrics(hazy)

            pair_scores = self._pair_scores(metrics, ref)

            visibility_score = 0.6 * visibility_score + 0.4 * pair_scores["visibility_score"]
            hallucination_score = 0.5 * hallucination_score + 0.5 * pair_scores["hallucination_score"]
            photometric_score = 0.5 * photometric_score + 0.5 * pair_scores["photometric_score"]

            haze_reduction_ratio = pair_scores["haze_reduction_ratio"]
            edge_inflation_ratio = pair_scores["edge_inflation_ratio"]
            hf_ratio = pair_scores["hf_ratio"]
            clipping_growth_ratio = pair_scores["clipping_growth_ratio"]

        # Final weighted score
        score = 100.0 * (
            0.45 * visibility_score +
            0.30 * hallucination_score +
            0.25 * photometric_score
        )
        score = float(np.clip(score, 0.0, 100.0))

        if score >= 80:
            verdict = "good"
        elif score >= 65:
            verdict = "acceptable"
        elif score >= 50:
            verdict = "borderline"
        else:
            verdict = "high-risk"

        return SafetyProxyResult(
            score=score,
            verdict=verdict,
            visibility_score=float(visibility_score),
            hallucination_score=float(hallucination_score),
            photometric_score=float(photometric_score),
            haze_index=float(metrics["haze_index"]),
            contrast=float(metrics["contrast"]),
            edge_density=float(metrics["edge_density"]),
            laplacian_var=float(metrics["laplacian_var"]),
            clipping_rate=float(metrics["clipping_rate"]),
            exposure_mean=float(metrics["exposure_mean"]),
            color_cast=float(metrics["color_cast"]),
            haze_reduction_ratio=None if haze_reduction_ratio is None else float(haze_reduction_ratio),
            edge_inflation_ratio=None if edge_inflation_ratio is None else float(edge_inflation_ratio),
            hf_ratio=None if hf_ratio is None else float(hf_ratio),
            clipping_growth_ratio=None if clipping_growth_ratio is None else float(clipping_growth_ratio),
        )

    def _compute_single_metrics(self, img_bgr: np.ndarray) -> dict:
        img = img_bgr.astype(np.float32) / 255.0
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        # Dark channel haze proxy: hazier images tend to have brighter dark channel
        min_rgb = np.min(img, axis=2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.dark_channel_kernel, self.dark_channel_kernel))
        dark_channel = cv2.erode(min_rgb, kernel)
        haze_index = float(np.mean(dark_channel))  # lower is better

        # Contrast
        contrast = float(np.std(gray))

        # Edge density
        edges = cv2.Canny((gray * 255).astype(np.uint8), self.edge_low, self.edge_high)
        edge_density = float(np.mean(edges > 0))

        # High-frequency energy proxy
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        laplacian_var = float(np.var(lap))

        # Clipping
        clipping_rate = float(np.mean((img <= 0.01) | (img >= 0.99)))

        # Exposure
        exposure_mean = float(np.mean(gray))

        # Color cast: channel mean deviation from global mean
        ch_mean = np.mean(img, axis=(0, 1))
        color_cast = float(np.mean(np.abs(ch_mean - np.mean(ch_mean))))

        return {
            "haze_index": haze_index,
            "contrast": contrast,
            "edge_density": edge_density,
            "laplacian_var": laplacian_var,
            "clipping_rate": clipping_rate,
            "exposure_mean": exposure_mean,
            "color_cast": color_cast,
        }

    def _single_visibility_score(self, m: dict) -> float:
        # Lower haze index is better
        haze_term = self._clamp01(1.0 - (m["haze_index"] - 0.05) / 0.35)

        # Moderate-to-high contrast is good
        contrast_term = self._clamp01((m["contrast"] - 0.08) / 0.18)

        # Edge density: too low means blurry; too high may mean over-sharpening/noise
        edge_term = self._bell_score(m["edge_density"], center=0.08, width=0.08)

        return 0.45 * haze_term + 0.35 * contrast_term + 0.20 * edge_term

    def _single_hallucination_score(self, m: dict) -> float:
        # Penalize excessive high-frequency energy and too many edges
        hf_term = self._bell_score(m["laplacian_var"], center=0.02, width=0.04)
        edge_term = self._bell_score(m["edge_density"], center=0.08, width=0.08)
        return 0.6 * hf_term + 0.4 * edge_term

    def _single_photometric_score(self, m: dict) -> float:
        # Exposure close to mid-range is preferred
        exposure_term = self._bell_score(m["exposure_mean"], center=0.50, width=0.25)

        # Lower clipping is better
        clip_term = self._clamp01(1.0 - m["clipping_rate"] / 0.08)

        # Lower color cast is better
        cast_term = self._clamp01(1.0 - m["color_cast"] / 0.10)

        return 0.45 * exposure_term + 0.35 * clip_term + 0.20 * cast_term

    def _pair_scores(self, out_m: dict, hazy_m: dict) -> dict:
        # SPI-1 visibility improvement: haze reduction ratio
        haze_ratio = out_m["haze_index"] / max(hazy_m["haze_index"], 1e-6)
        haze_reduction_ratio = self._clamp01((1.0 - haze_ratio) / max(1.0 - self.gamma_H, 1e-6))

        # SPI-2 hallucination proxies
        edge_inflation_ratio = out_m["edge_density"] / max(hazy_m["edge_density"], 1e-6)
        hf_ratio = out_m["laplacian_var"] / max(hazy_m["laplacian_var"], 1e-6)

        edge_ok = self._clamp01(1.0 - max(edge_inflation_ratio - self.gamma_E, 0.0) / self.gamma_E)
        hf_ok = self._clamp01(1.0 - max(hf_ratio - self.gamma_F, 0.0) / self.gamma_F) 
        hallucination_score = 0.5 * edge_ok + 0.5 * hf_ok

        # SPI-3 photometric stability
        clipping_growth_ratio = out_m["clipping_rate"] / max(hazy_m["clipping_rate"] + 1e-6, 1e-6)
        clip_ok = self._clamp01(1.0 - max(clipping_growth_ratio - self.gamma_C, 0.0) / self.gamma_C)

        mean_shift = abs(out_m["exposure_mean"] - hazy_m["exposure_mean"])
        exposure_ok = self._clamp01(1.0 - mean_shift / 0.25)

        photometric_score = 0.6 * clip_ok + 0.4 * exposure_ok

        return {
            "visibility_score": haze_reduction_ratio,
            "hallucination_score": hallucination_score,
            "photometric_score": photometric_score,
            "haze_reduction_ratio": haze_reduction_ratio,
            "edge_inflation_ratio": edge_inflation_ratio,
            "hf_ratio": hf_ratio,
            "clipping_growth_ratio": clipping_growth_ratio,
        }

    def _load_image(self, x) -> np.ndarray:
        if isinstance(x, str):
            img = cv2.imread(x, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Could not read image: {x}")
            return img
        elif isinstance(x, np.ndarray):
            img = x.copy()
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            return img
        else:
            raise TypeError("Input must be a file path or a numpy array.")

    @staticmethod
    def _clamp01(x):
        return float(np.clip(x, 0.0, 1.0))

    @staticmethod
    def _bell_score(x, center, width):
        return float(np.exp(-((x - center) ** 2) / (2.0 * width * width)))


if __name__ == "__main__":
    evaluator = AVSafetyProxyEvaluator()

    result_single = evaluator.evaluate("hazy.png")
    print("hazy")
    print(asdict(result_single))

    result_single = evaluator.evaluate("baseline-dehaze.png")
    print("baseline")
    print(asdict(result_single))

    result_single = evaluator.evaluate("baseline-20-dehaze.png")
    print("vanilla 20")
    print(asdict(result_single))

    result_single = evaluator.evaluate("our-20-dehaze.png")
    print("20")
    print(asdict(result_single))

    result_single = evaluator.evaluate("our-50-dehaze.png")
    print("50")
    print(asdict(result_single))
