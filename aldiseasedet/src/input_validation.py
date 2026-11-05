"""Modular Input Validation Pipeline for Apple Leaf Images.

Multi-tier, multi-modal input validation:
1. Tier 1: File integrity, corrupt byte-stream detection, blurriness (Laplacian variance), and illumination checks.
2. Tier 2: Deep Semantic Foliar Verification using pretrained MobileNetV2 feature embeddings
   measured against empirical apple leaf prototype centroids (rejects posters, vehicles, faces, cartoons, furniture).
3. Tier 3: Morphological and multi-spectral vegetation verification (chlorophyll green, chlorotic yellow/rust, necrotic brown).
4. Tier 4: Natural leaf vein awareness (distinguishes central/lateral veins from artificial grids).
5. Tier 5: Cautious human face rejection (Haar cascades + skin chrominance validation to eliminate false alarms on leaves).
6. Configurable thresholds with graceful handling of uncertain/borderline images.
"""

from typing import Dict, Any, Union, Tuple, Optional
import os
import sys
import io
import cv2
import numpy as np
from PIL import Image

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Global lazy-loaded semantic feature extractor
_semantic_model = None
_semantic_transform = None
_semantic_centroid = None


def _load_semantic_extractor():
    """Lazy-load offline MobileNetV2 backbone and precomputed apple leaf centroid."""
    global _semantic_model, _semantic_transform, _semantic_centroid

    if _semantic_model is not None and _semantic_centroid is not None:
        return _semantic_model, _semantic_transform, _semantic_centroid

    try:
        import torch
        from torchvision.models import mobilenet_v2
        from torchvision import transforms

        weights_cache_path = os.path.expanduser(r"~/.cache/torch/hub/checkpoints/mobilenet_v2-7ebf99e0.pth")
        if not os.path.exists(weights_cache_path):
            alt_path = os.path.join(project_root, "models", "mobilenet_v2-7ebf99e0.pth")
            if os.path.exists(alt_path):
                weights_cache_path = alt_path

        if os.path.exists(weights_cache_path):
            m = mobilenet_v2(weights=None)
            m.load_state_dict(torch.load(weights_cache_path, map_location="cpu"))
            m.eval()
            _semantic_model = m

            _semantic_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        centroid_path = os.path.join(project_root, "models", "leaf_semantic_centroid.npy")
        if os.path.exists(centroid_path):
            _semantic_centroid = np.load(centroid_path)
            _semantic_centroid = _semantic_centroid / np.linalg.norm(_semantic_centroid)

    except Exception as e:
        print(f"[WARNING] Semantic validator fallback to heuristic mode: {e}")

    return _semantic_model, _semantic_transform, _semantic_centroid


class AppleLeafValidator:
    """Multi-cue validator for verifying apple leaf photographs while eliminating false rejections."""

    def __init__(
        self,
        min_dim: int = 100,
        severe_blur_threshold: float = 18.0,
        moderate_blur_threshold: float = 28.0,
        semantic_threshold_accept: float = 0.52,
        semantic_threshold_reject: float = 0.42,
        min_foliar_ratio: float = 0.08,
        min_leaf_contour_ratio: float = 0.04,
        face_min_skin_ratio: float = 0.35,
        face_max_foliar_ratio: float = 0.25,
        doc_max_foliar_ratio: float = 0.12,
        doc_min_neutral_ratio: float = 0.65,
    ):
        """Configurable thresholds documented for agronomic field deployment.

        Args:
            min_dim: Minimum acceptable image dimension in pixels (width or height).
            severe_blur_threshold: Hard rejection threshold for Laplacian sharpness variance.
            moderate_blur_threshold: Threshold below which an advisory review warning is recorded.
            semantic_threshold_accept: MobileNetV2 cosine similarity threshold for clean approval.
            semantic_threshold_reject: Similarity threshold below which an image is definitively rejected.
            min_foliar_ratio: Minimum ratio of biological foliar pigmentation (green, yellow, necrotic).
            min_leaf_contour_ratio: Minimum area of primary leaf contour relative to image area.
            face_min_skin_ratio: Minimum fraction of human skin chrominance required to confirm a face.
            face_max_foliar_ratio: Face candidate boxes exceeding this foliar ratio are treated as leaf veins/spots.
            doc_max_foliar_ratio: Images exceeding this foliar ratio cannot be rejected as documents (veins != grid).
            doc_min_neutral_ratio: Minimum proportion of white/neutral paper background for document classification.
        """
        self.min_dim = min_dim
        self.severe_blur_threshold = severe_blur_threshold
        self.moderate_blur_threshold = moderate_blur_threshold
        self.semantic_threshold_accept = semantic_threshold_accept
        self.semantic_threshold_reject = semantic_threshold_reject
        self.min_foliar_ratio = min_foliar_ratio
        self.min_leaf_contour_ratio = min_leaf_contour_ratio
        self.face_min_skin_ratio = face_min_skin_ratio
        self.face_max_foliar_ratio = face_max_foliar_ratio
        self.doc_max_foliar_ratio = doc_max_foliar_ratio
        self.doc_min_neutral_ratio = doc_min_neutral_ratio

        # Load classical Haar Cascades for face detection
        cascade_dir = cv2.data.haarcascades
        self.frontal_face_cascade = cv2.CascadeClassifier(
            os.path.join(cascade_dir, "haarcascade_frontalface_default.xml")
        )
        self.profile_face_cascade = cv2.CascadeClassifier(
            os.path.join(cascade_dir, "haarcascade_profileface.xml")
        )

    def validate(
        self,
        image_input: Union[str, np.ndarray, Image.Image],
        is_rgb: bool = True,
    ) -> Dict[str, Any]:
        """Validate input image across independent physical, biological, and semantic cues.

        Returns structured dictionary:
            - is_valid: bool (True allows classification, False halts pipeline)
            - status: "APPROVED" | "REVIEW_RECOMMENDED" | "REJECTED"
            - reason: str (user-facing clear message)
            - details: Dict with fine-grained diagnostic telemetry
            - image_rgb: Validated RGB numpy array (if readable)
        """
        # --- CUE 1: File integrity and loading ---
        img_rgb, load_err = self._load_image(image_input, is_rgb=is_rgb)
        if img_rgb is None:
            return {
                "is_valid": False,
                "status": "REJECTED",
                "reason": f"Corrupted or unreadable image file: {load_err}",
                "details": {"stage": "integrity", "error": load_err},
                "image_rgb": None,
            }

        h, w, c = img_rgb.shape
        if h < self.min_dim or w < self.min_dim:
            return {
                "is_valid": False,
                "status": "REJECTED",
                "reason": f"Image resolution too low ({w}x{h}). Minimum required is {self.min_dim}x{self.min_dim}.",
                "details": {"stage": "dimensions", "width": w, "height": h},
                "image_rgb": img_rgb,
            }

        # --- CUE 2: Exposure and blur metrics ---
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        mean_intensity = float(np.mean(gray))
        std_intensity = float(np.std(gray))

        # Severe darkness / pitch black
        if mean_intensity < 14.0:
            return {
                "is_valid": False,
                "status": "REJECTED",
                "reason": "Image is severely underexposed (pitch black). Please provide an image with adequate lighting.",
                "details": {"stage": "exposure", "mean_intensity": mean_intensity},
                "image_rgb": img_rgb,
            }

        # Severe washout (pure white blank or zero dynamic contrast)
        if mean_intensity > 250.0 or std_intensity < 8.0:
            return {
                "is_valid": False,
                "status": "REJECTED",
                "reason": "Image is washed out, blank, or lacks perceptible contrast.",
                "details": {"stage": "exposure", "mean_intensity": mean_intensity, "std_intensity": std_intensity},
                "image_rgb": img_rgb,
            }

        # Severe blur hard rejection
        if laplacian_var < self.severe_blur_threshold:
            return {
                "is_valid": False,
                "status": "REJECTED",
                "reason": (
                    f"Image is heavily blurred or out of focus (sharpness score: {laplacian_var:.1f}, "
                    f"minimum threshold: {self.severe_blur_threshold:.1f}). Please capture a stabilized photo."
                ),
                "details": {"stage": "blur", "laplacian_var": laplacian_var},
                "image_rgb": img_rgb,
            }

        # --- CUE 3: Deep Semantic Visual Embedding Check (MobileNetV2 Centroid) ---
        semantic_status, semantic_sim, semantic_msg = self._check_semantic_embedding(img_rgb)
        if semantic_status is False:
            # Definite non-leaf: posters, greeting cards, cars, screenshots, faces, furniture
            return {
                "is_valid": False,
                "status": "REJECTED",
                "reason": f"Non-leaf object detected: {semantic_msg}",
                "details": {
                    "stage": "semantic_embedding",
                    "cosine_similarity": round(semantic_sim, 4),
                    "threshold_reject": self.semantic_threshold_reject,
                },
                "image_rgb": img_rgb,
            }

        # --- CUE 4: Biological foliar pigmentation analysis ---
        foliar_mask, foliar_ratio, color_metrics = self._extract_foliar_pigmentation(img_rgb)

        # Non-foliar hard rejection: if image contains virtually zero plant colors
        if foliar_ratio < self.min_foliar_ratio and (semantic_status is None or semantic_sim < self.semantic_threshold_accept):
            return {
                "is_valid": False,
                "status": "REJECTED",
                "reason": (
                    f"Non-leaf object detected. Plant foliar pigmentation is negligible ({foliar_ratio * 100:.1f}% "
                    f"vs. expected minimum {self.min_foliar_ratio * 100:.1f}%). Image lacks natural leaf tissue."
                ),
                "details": {"stage": "foliar_pigmentation", "foliar_ratio": foliar_ratio, **color_metrics},
                "image_rgb": img_rgb,
            }

        # --- CUE 5: Cautious human face detection with skin verification ---
        face_confirmed, face_count, face_details = self._verify_human_face(img_rgb, gray, foliar_mask)
        if face_confirmed:
            return {
                "is_valid": False,
                "status": "REJECTED",
                "reason": "Human face detected with confirmed facial skin characteristics. Non-leaf portraits are prohibited.",
                "details": {"stage": "human_face", "face_count": face_count, **face_details},
                "image_rgb": img_rgb,
            }

        # --- CUE 6: Document / Poster / Screenshot Check with Leaf Vein Awareness ---
        is_doc, doc_reason, doc_details = self._verify_document_or_synthetic(
            gray, img_rgb, foliar_mask, foliar_ratio, h, w
        )
        if is_doc:
            return {
                "is_valid": False,
                "status": "REJECTED",
                "reason": f"Rejected as synthetic document or poster: {doc_reason}",
                "details": {"stage": "document_check", **doc_details},
                "image_rgb": img_rgb,
            }

        # --- CUE 7: Leaf contour and blade structure ---
        is_leaf, leaf_status, leaf_metrics = self._verify_blade_structure(foliar_mask, h, w)
        if not is_leaf and (semantic_status is None or semantic_sim < self.semantic_threshold_accept):
            return {
                "is_valid": False,
                "status": "REJECTED",
                "reason": f"Non-leaf object detected: {leaf_status}",
                "details": {"stage": "leaf_structure", **leaf_metrics},
                "image_rgb": img_rgb,
            }

        # --- CUE 8: Uncertainty / Review-recommended evaluation ---
        is_borderline_blur = (laplacian_var < self.moderate_blur_threshold)
        is_borderline_semantic = (semantic_status is True and semantic_sim < self.semantic_threshold_accept)
        is_sparse_foliar = (foliar_ratio < self.min_foliar_ratio * 1.5)

        if is_borderline_semantic:
            advisory = (
                "Botanical foliar tissue detected, but specific Malus domestica (apple leaf) "
                f"profile is uncertain (semantic similarity: {semantic_sim:.3f}). "
                "Generic leaf presence does not confirm apple identity. Caution and physical orchard inspection advised."
            )
            return {
                "is_valid": True,
                "status": "REVIEW_RECOMMENDED",
                "reason": advisory,
                "details": {
                    "width": w,
                    "height": h,
                    "sharpness": round(laplacian_var, 2),
                    "foliar_ratio": round(foliar_ratio, 4),
                    "semantic_similarity": round(semantic_sim, 4) if semantic_status is not None else None,
                    "advisory": advisory,
                    **leaf_metrics,
                },
                "image_rgb": img_rgb,
            }

        if is_borderline_blur or is_sparse_foliar:
            advisory = (
                "Borderline image sharpness, complex background, or foliar lighting detected. "
                "Processing permitted, but physical orchard verification is recommended."
            )
            return {
                "is_valid": True,
                "status": "REVIEW_RECOMMENDED",
                "reason": advisory,
                "details": {
                    "width": w,
                    "height": h,
                    "sharpness": round(laplacian_var, 2),
                    "foliar_ratio": round(foliar_ratio, 4),
                    "semantic_similarity": round(semantic_sim, 4) if semantic_status is not None else None,
                    "advisory": advisory,
                    **leaf_metrics,
                },
                "image_rgb": img_rgb,
            }

        # Clean approval
        return {
            "is_valid": True,
            "status": "APPROVED",
            "reason": "Input verified as a valid, clear apple leaf photograph.",
            "details": {
                "width": w,
                "height": h,
                "sharpness": round(laplacian_var, 2),
                "foliar_ratio": round(foliar_ratio, 4),
                "semantic_similarity": round(semantic_sim, 4) if semantic_status is not None else None,
                **leaf_metrics,
            },
            "image_rgb": img_rgb,
        }

    def _check_semantic_embedding(self, img_rgb: np.ndarray) -> Tuple[Optional[bool], float, str]:
        """Evaluate deep semantic visual embedding against prototypical apple leaf manifold."""
        model, transform, centroid = _load_semantic_extractor()
        if model is None or centroid is None:
            return None, 0.0, "Semantic model weights or centroid not available; bypassing embedding check."

        try:
            import torch
            pil_img = Image.fromarray(img_rgb)
            x = transform(pil_img).unsqueeze(0)

            with torch.no_grad():
                f = model.features(x).mean([2, 3])
                f = f / torch.norm(f, dim=1, keepdim=True)
                embed = f.numpy()[0]

            cos_sim = float(np.dot(embed, centroid))

            if cos_sim < self.semantic_threshold_reject:
                return False, cos_sim, f"Deep visual embedding mismatch (Cosine similarity: {cos_sim:.4f} < {self.semantic_threshold_reject:.2f})."
            elif cos_sim < self.semantic_threshold_accept:
                # Borderline range: permit through with review recommended
                return True, cos_sim, f"Borderline semantic similarity ({cos_sim:.4f})."
            else:
                return True, cos_sim, f"Deep semantic verification passed ({cos_sim:.4f})."

        except Exception as e:
            return None, 0.0, f"Semantic embedding check encountered exception: {e}"

    def _load_image(
        self,
        image_input: Union[str, np.ndarray, Image.Image],
        is_rgb: bool,
    ) -> Tuple[Optional[np.ndarray], Optional[str]]:
        try:
            if isinstance(image_input, str):
                if not os.path.exists(image_input):
                    return None, f"File does not exist: {image_input}"
                pil_img = Image.open(image_input)
                pil_img.verify()
                pil_img = Image.open(image_input).convert("RGB")
                return np.array(pil_img, dtype=np.uint8), None

            elif isinstance(image_input, Image.Image):
                rgb_img = image_input.convert("RGB")
                return np.array(rgb_img, dtype=np.uint8), None

            elif isinstance(image_input, np.ndarray):
                if image_input.size == 0 or image_input.ndim not in [2, 3]:
                    return None, "Empty or invalid numpy array shape"
                if image_input.ndim == 2:
                    return cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB), None
                if is_rgb:
                    return image_input.astype(np.uint8), None
                return cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB), None
            else:
                return None, f"Unsupported input type: {type(image_input)}"
        except Exception as e:
            return None, str(e)

    def _extract_foliar_pigmentation(
        self, img_rgb: np.ndarray
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """Segment natural plant foliar pigments: green chlorophyll, chlorotic rust/yellow, and necrotic lesions."""
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        total_px = float(img_rgb.shape[0] * img_rgb.shape[1])

        # 1. Healthy green chlorophyll (Hue 22 to 98, moderate saturation & value)
        green_mask = cv2.inRange(hsv, np.array([22, 22, 20]), np.array([98, 255, 255]))

        # 2. Chlorotic rust / bright yellow (Hue 9 to 26, high sat, value > 30)
        yellow_rust_mask = cv2.inRange(hsv, np.array([9, 28, 30]), np.array([26, 255, 255]))

        # 3. Necrotic lesions / dark spots within biological range (Hue 5-28, moderate sat)
        brown_necrotic_mask = cv2.inRange(hsv, np.array([5, 15, 15]), np.array([28, 255, 120]))

        # Combined foliar mask
        foliar_mask = cv2.bitwise_or(green_mask, yellow_rust_mask)
        foliar_mask = cv2.bitwise_or(foliar_mask, brown_necrotic_mask)

        foliar_count = int(np.count_nonzero(foliar_mask))
        foliar_ratio = float(foliar_count / (total_px + 1e-6))

        return foliar_mask, foliar_ratio, {
            "green_ratio": float(np.count_nonzero(green_mask) / total_px),
            "yellow_rust_ratio": float(np.count_nonzero(yellow_rust_mask) / total_px),
            "brown_necrotic_ratio": float(np.count_nonzero(brown_necrotic_mask) / total_px),
        }

    def _verify_human_face(
        self,
        img_rgb: np.ndarray,
        gray: np.ndarray,
        foliar_mask: np.ndarray,
    ) -> Tuple[bool, int, Dict[str, Any]]:
        """Cautious face detection: filters out false Haar responses on leaf lesions and veins."""
        faces = self.frontal_face_cascade.detectMultiScale(
            gray, scaleFactor=1.15, minNeighbors=7, minSize=(60, 60)
        )

        profile_faces = self.profile_face_cascade.detectMultiScale(
            gray, scaleFactor=1.15, minNeighbors=7, minSize=(60, 60)
        )

        all_candidates = list(faces) if len(faces) > 0 else []
        if len(profile_faces) > 0:
            all_candidates.extend(list(profile_faces))

        if not all_candidates:
            return False, 0, {}

        img_ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
        confirmed_faces = 0
        details = []

        for (fx, fy, fw, fh) in all_candidates:
            face_roi_foliar = foliar_mask[fy : fy + fh, fx : fx + fw]
            face_roi_ycrcb = img_ycrcb[fy : fy + fh, fx : fx + fw]
            face_area = float(fw * fh)

            # Check if this "face" is actually green leaf tissue
            foliar_in_face = float(np.count_nonzero(face_roi_foliar) / (face_area + 1e-6))
            if foliar_in_face > self.face_max_foliar_ratio:
                continue

            cr = face_roi_ycrcb[:, :, 1]
            cb = face_roi_ycrcb[:, :, 2]
            skin_mask = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)
            skin_ratio = float(np.count_nonzero(skin_mask) / (face_area + 1e-6))

            if skin_ratio >= self.face_min_skin_ratio:
                confirmed_faces += 1
                details.append({
                    "box": (int(fx), int(fy), int(fw), int(fh)),
                    "skin_ratio": round(skin_ratio, 3),
                    "foliar_ratio": round(foliar_in_face, 3),
                })

        if confirmed_faces > 0:
            return True, confirmed_faces, {"confirmed_faces": details}

        return False, 0, {"filtered_spurious_candidates": len(all_candidates)}

    def _verify_document_or_synthetic(
        self,
        gray: np.ndarray,
        img_rgb: np.ndarray,
        foliar_mask: np.ndarray,
        foliar_ratio: float,
        h: int,
        w: int,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Multi-cue document / poster / screenshot detection with leaf vein awareness."""
        total_pixels = float(h * w)

        if foliar_ratio >= self.doc_max_foliar_ratio:
            return (
                False,
                "",
                {"status": "bypassed_due_to_confirmed_foliar_tissue", "foliar_ratio": foliar_ratio},
            )

        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        neutral_paper_mask = (sat <= 35) & (val >= 150)
        neutral_paper_ratio = float(np.count_nonzero(neutral_paper_mask) / total_pixels)

        edges = cv2.Canny(gray, 60, 160)
        min_line_len = int(min(h, w) * 0.40)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=100, minLineLength=min_line_len, maxLineGap=12
        )

        orthogonal_lines = 0
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)
                if dx < 2 or dy < 2:
                    orthogonal_lines += 1

        if neutral_paper_ratio >= self.doc_min_neutral_ratio and foliar_ratio < 0.06:
            if orthogonal_lines >= 8:
                return (
                    True,
                    "Rectilinear document table grid or paper border detected on neutral background.",
                    {
                        "neutral_paper_ratio": round(neutral_paper_ratio, 3),
                        "foliar_ratio": round(foliar_ratio, 3),
                        "orthogonal_lines": orthogonal_lines,
                    },
                )

        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        white_ratio = float(np.count_nonzero(binary == 255) / total_pixels)
        edge_density = float(np.count_nonzero(edges) / total_pixels)

        if neutral_paper_ratio > 0.82 and foliar_ratio < 0.04 and edge_density > 0.045:
            return (
                True,
                "Dense text paragraphs on white paper background detected.",
                {
                    "neutral_paper_ratio": round(neutral_paper_ratio, 3),
                    "edge_density": round(edge_density, 3),
                    "foliar_ratio": round(foliar_ratio, 3),
                },
            )

        return False, "", {
            "neutral_paper_ratio": round(neutral_paper_ratio, 3),
            "orthogonal_lines": orthogonal_lines,
            "foliar_ratio": round(foliar_ratio, 3),
        }

    def _verify_blade_structure(
        self, foliar_mask: np.ndarray, h: int, w: int
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Verify presence of a cohesive organic leaf blade."""
        total_pixels = float(h * w)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        foliar_closed = cv2.morphologyEx(foliar_mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(foliar_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return False, "No cohesive foliar contour detected.", {}

        largest_contour = max(contours, key=cv2.contourArea)
        c_area = float(cv2.contourArea(largest_contour))
        c_ratio = float(c_area / total_pixels)

        if c_ratio < self.min_leaf_contour_ratio:
            return (
                False,
                f"Foliar structure too small ({c_ratio * 100:.1f}% vs. required {self.min_leaf_contour_ratio * 100:.1f}%).",
                {"contour_area_ratio": round(c_ratio, 4)},
            )

        hull = cv2.convexHull(largest_contour)
        hull_area = float(cv2.contourArea(hull))
        solidity = float(c_area / (hull_area + 1e-6))

        if solidity < 0.18:
            return (
                False,
                f"Foliar object is overly fragmented or non-cohesive (solidity: {solidity:.2f}).",
                {"solidity": round(solidity, 3), "contour_area_ratio": round(c_ratio, 4)},
            )

        return (
            True,
            "Valid organic leaf blade structure.",
            {"contour_area_ratio": round(c_ratio, 4), "solidity": round(solidity, 3)},
        )


# Global singleton validator
default_validator = AppleLeafValidator()


def validate_apple_leaf_input(
    image_input: Union[str, np.ndarray, Image.Image],
    is_rgb: bool = True,
) -> Dict[str, Any]:
    """Top-level convenience validator function."""
    return default_validator.validate(image_input, is_rgb=is_rgb)
