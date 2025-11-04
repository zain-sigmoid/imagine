# core/post_processing.py (or inline in your app)
import cv2
import numpy as np
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)


class PostProcessing:
    @staticmethod
    def _rgb_to_lab(img_rgb):
        return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)

    @staticmethod
    def _lab_to_rgb(img_lab):
        return cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)

    @staticmethod
    def _vibrance_hsv(img_rgb, vib=0.18):
        """Vibrance: boost saturation more for low-sat pixels, protect high-sat."""
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        h, s, v = cv2.split(hsv)
        s_norm = s / 255.0
        boost = vib * (1.0 - s_norm)  # more boost where s is low
        s = np.clip((s_norm + boost) * 255.0, 0, 255)
        hsv = cv2.merge([h, s, v]).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    @staticmethod
    def _white_point(img_rgb, percentile=99.2, target=245):
        """Gentle per-channel white balance by stretching top percentile to target."""
        arr = img_rgb.astype(np.float32)
        for c in range(3):
            p = np.percentile(arr[..., c], percentile)
            if p < 1:  # avoid div by zero
                continue
            gain = target / p
            arr[..., c] = np.clip(arr[..., c] * gain, 0, 255)
        return arr.astype(np.uint8)

    @staticmethod
    def _unsharp(img_rgb, amount=0.35, sigma=1.0):
        blur = cv2.GaussianBlur(img_rgb, (0, 0), sigma)
        return cv2.addWeighted(img_rgb, 1 + amount, blur, -amount, 0)

    @staticmethod
    def _white_point_neutral(img_rgb, percentile=99.2, target=245):
        """Neutral-preserving white point: compute a single gain from luminance."""
        arr = img_rgb.astype(np.float32)
        # luminance approximation (keeps grays neutral if scaled equally)
        Y = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
        p = np.percentile(Y, percentile)
        gain = target / max(p, 1.0)
        arr = np.clip(arr * gain, 0, 255)
        return arr.astype(np.uint8)

    @staticmethod
    @staticmethod
    def _vibrance_hsv_bg_protected(
        img_rgb, vib=0.18, bg_s_thresh=0.12, bg_v_thresh=0.90
    ):
        # to HSV as float for math
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        h, s, v = cv2.split(hsv)
        s_n = s / 255.0
        v_n = v / 255.0

        # near-white background mask
        bg = (s_n < bg_s_thresh) & (v_n > bg_v_thresh)

        # vibrance curve
        s_new = np.clip((s_n + vib * (1.0 - s_n)) * 255.0, 0, 255)

        # protect background
        s_new[bg] = s[bg]

        # *** ensure SAME TYPE for merge ***
        h_u8 = h.astype(np.uint8)
        s_u8 = s_new.astype(np.uint8)
        v_u8 = v.astype(np.uint8)

        hsv_u8 = cv2.merge([h_u8, s_u8, v_u8])  # all uint8, same size
        return cv2.cvtColor(hsv_u8, cv2.COLOR_HSV2RGB)

    @staticmethod
    def enhance_image(pil_img: Image.Image, strength: str = "low") -> Image.Image:
        """
        Safer enhancer for watercolor/foil napkins.
        strength: 'low' | 'medium' | 'high' (medium is recommended)
        """
        logger.info(f"PostProcessing: enhancing with strength={strength}")
        # Params tuned for watercolor; adjust if needed
        cfg = {
            "low": dict(
                clahe_clip=1.4, vib=0.12, unsharp_amt=0.25, sigma=0.9, wp_pct=99.0
            ),
            "medium": dict(
                clahe_clip=1.6, vib=0.18, unsharp_amt=0.35, sigma=1.0, wp_pct=99.2
            ),
            "high": dict(
                clahe_clip=1.9, vib=0.25, unsharp_amt=0.45, sigma=1.1, wp_pct=99.4
            ),
        }[strength]

        img = pil_img.convert("RGB")
        rgb = np.array(img)

        # 1) White-point gently (protect texture by not aiming full 255)
        # rgb = PostProcessing._white_point(rgb, percentile=cfg["wp_pct"], target=245)
        rgb = PostProcessing._white_point_neutral(
            rgb, percentile=cfg["wp_pct"], target=255
        )

        # 2) Local contrast on L channel (CLAHE) with highlight protection
        lab = PostProcessing._rgb_to_lab(rgb)
        L, A, B = cv2.split(lab)
        # protect highlight areas (avoid blowing paper emboss)
        highlight_mask = (L > 235).astype(np.uint8) * 255
        clahe = cv2.createCLAHE(clipLimit=cfg["clahe_clip"], tileGridSize=(8, 8))
        L_enh = clahe.apply(L)
        # blend back highlights
        L = np.where(highlight_mask == 255, L, L_enh)
        lab = cv2.merge([L, A, B])
        rgb = PostProcessing._lab_to_rgb(lab)

        # 3) Vibrance (not plain saturation) to keep pastels clean
        # rgb = PostProcessing._vibrance_hsv(rgb, vib=cfg["vib"])
        rgb = PostProcessing._vibrance_hsv_bg_protected(rgb, vib=cfg["vib"])

        # 4) Very light unsharp mask
        rgb = PostProcessing._unsharp(
            rgb, amount=cfg["unsharp_amt"], sigma=cfg["sigma"]
        )

        # 5) Final clamp and return
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        return Image.fromarray(rgb)

    @staticmethod
    def apply_post_processing(img: Image.Image) -> Image.Image:
        """
        Apply post-processing based on the selected option.
        'Low' | 'Medium' | 'High'
        """
        enhanced_img_l = PostProcessing.enhance_image(img, strength="low")
        enhanced_img_m = PostProcessing.enhance_image(img, strength="medium")
        enhanced_img_h = PostProcessing.enhance_image(img, strength="high")
        return enhanced_img_l, enhanced_img_m, enhanced_img_h
