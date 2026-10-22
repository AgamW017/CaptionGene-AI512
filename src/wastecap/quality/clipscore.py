"""
CLIPScore computation for WasteCap candidates.

CLIPScore measures the cosine similarity between a CLIP image embedding
and a CLIP text embedding. Higher = more aligned caption. It is used as
a filter signal and quality indicator — not as a ground-truth metric.

Reference: Hessel et al. (EMNLP 2021), https://aclanthology.org/2021.emnlp-main.595

Implementation notes
--------------------
- Lazy loading: CLIP model is NOT imported at module level. It is loaded
  on first call and cached via @lru_cache (one instance per process).
- Graceful degradation: if open_clip / torch are not installed (local dev
  without ML extras), returns None for every score. Never raises.
- The canonical model is ViT-B/32 from openai via open_clip. This is the
  same model used in the original CLIPScore paper.
- Scores are multiplied by 2.5 to match the w·max(cos,0) scaling in the
  paper (w=2.5). Values typically fall in [0, 0.5]; treat ≥0.28 as good.

DB write-back
-------------
score_candidates_for_record(rec, db_path) computes scores for all
candidates that have clip_score=None and writes them back to the DB
via store.update_record. Call this from scripts/compute_clipscore.py or
lazily from the app.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

logger = logging.getLogger(__name__)

# CLIPScore weight from the paper (w · max(cos_sim, 0))
_W = 2.5
_MODEL_NAME = "ViT-B-32"
_PRETRAINED = "openai"


# ---------------------------------------------------------------------------
# Lazy model loading
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _load_clip() -> tuple | None:
    """Load and cache the CLIP model + preprocess transform.

    Returns (model, preprocess, tokenizer, device) or None if open_clip unavailable.
    """
    try:
        import open_clip

        device = "cuda" if _cuda_available() else "cpu"
        model, _, preprocess = open_clip.create_model_and_transforms(
            _MODEL_NAME, pretrained=_PRETRAINED
        )
        model = model.to(device).eval()
        tokenizer = open_clip.get_tokenizer(_MODEL_NAME)
        logger.info("CLIP (%s/%s) loaded on %s", _MODEL_NAME, _PRETRAINED, device)
        return model, preprocess, tokenizer, device
    except ImportError:
        logger.debug(
            "open_clip not installed — CLIPScore will return None. "
            "Install with: pip install -e '.[ml]'"
        )
        return None
    except Exception as exc:
        logger.warning("Failed to load CLIP model: %s — CLIPScore disabled.", exc)
        return None


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def reload_clip() -> None:
    """Clear the CLIP cache (useful in tests or after switching devices)."""
    _load_clip.cache_clear()


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def compute_clipscore(image: PILImage, text: str) -> float | None:
    """Compute CLIPScore for a single (image, text) pair.

    Returns:
        Float in roughly [0, 0.5] (higher = better aligned), or None if
        CLIP is not available.
    """
    clip = _load_clip()
    if clip is None:
        return None

    import torch

    model, preprocess, tokenizer, device = clip

    try:
        img_tensor = preprocess(image).unsqueeze(0).to(device)
        txt_tokens = tokenizer([text]).to(device)

        with torch.inference_mode():
            img_feat = model.encode_image(img_tensor)
            txt_feat = model.encode_text(txt_tokens)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
            cos_sim = (img_feat * txt_feat).sum(dim=-1).item()

        return float(_W * max(cos_sim, 0.0))
    except Exception as exc:
        logger.warning("CLIPScore computation failed: %s", exc)
        return None


def batch_clipscore(
    image_paths: list[Path],
    texts: list[str],
    batch_size: int = 32,
) -> list[float | None]:
    """Compute CLIPScore for a batch of (image_path, text) pairs.

    Returns a list of scores in the same order as the inputs.
    Entries where the image cannot be loaded return None.

    Args:
        image_paths: List of paths to image files.
        texts:       Corresponding caption strings (same length).
        batch_size:  Number of items per CLIP forward pass.
    """
    if len(image_paths) != len(texts):
        raise ValueError("image_paths and texts must have the same length")

    clip = _load_clip()
    if clip is None:
        return [None] * len(image_paths)

    import torch
    from PIL import Image

    model, preprocess, tokenizer, device = clip
    results: list[float | None] = []

    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        batch_texts = texts[start : start + batch_size]

        imgs: list = []
        valid_idx: list[int] = []
        for i, p in enumerate(batch_paths):
            try:
                img = Image.open(p).convert("RGB")
                imgs.append(preprocess(img))
                valid_idx.append(i)
            except Exception as exc:
                logger.warning("Cannot load image %s: %s", p, exc)

        batch_results: list[float | None] = [None] * len(batch_paths)

        if not imgs:
            results.extend(batch_results)
            continue

        try:
            img_tensor = torch.stack(imgs).to(device)
            txt_tokens = tokenizer([batch_texts[i] for i in valid_idx]).to(device)

            with torch.inference_mode():
                img_feats = model.encode_image(img_tensor)
                txt_feats = model.encode_text(txt_tokens)
                img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
                txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)
                cos_sims = (img_feats * txt_feats).sum(dim=-1).tolist()

            for j, orig_idx in enumerate(valid_idx):
                batch_results[orig_idx] = float(_W * max(cos_sims[j], 0.0))
        except Exception as exc:
            logger.warning("Batch CLIPScore failed: %s", exc)

        results.extend(batch_results)

    return results


# ---------------------------------------------------------------------------
# DB write-back
# ---------------------------------------------------------------------------


def score_candidates_for_record(
    rec: object,
    db_path: Path | None = None,
) -> int:
    """Compute CLIPScore for all candidates with clip_score=None in *rec*.

    Writes the updated record back to the DB. Returns the number of scores
    computed (0 if CLIP unavailable or all already scored).

    This is the function called by scripts/compute_clipscore.py and by the
    lazy-load path in the Compare and Review pages.
    """
    from pathlib import Path as _Path

    from PIL import Image

    from wastecap.store import update_record

    img_path = _Path(rec.file)
    if not img_path.exists():
        logger.debug("Image not found, cannot score: %s", img_path)
        return 0

    unscored = [(i, c) for i, c in enumerate(rec.candidates) if c.clip_score is None]
    if not unscored:
        return 0

    clip = _load_clip()
    if clip is None:
        return 0

    try:
        image = Image.open(img_path).convert("RGB")
    except Exception as exc:
        logger.warning("Cannot open %s for CLIPScore: %s", img_path, exc)
        return 0

    computed = 0
    for i, cand in unscored:
        # Use the best caption text available: raw_parsed caption, else full text
        caption = _extract_caption_text(cand)
        score = compute_clipscore(image, caption)
        if score is not None:
            rec.candidates[i] = cand.model_copy(update={"clip_score": score})
            computed += 1

    if computed > 0:
        update_record(rec, db_path)

    return computed


def _extract_caption_text(cand: object) -> str:
    """Extract the best caption string from a Candidate for CLIPScore.

    Prefers the 'caption' field from raw_parsed objects list, then falls
    back to the full raw text.
    """
    if cand.raw_parsed:
        # Scene-level: objects[0].caption
        objects = cand.raw_parsed.get("objects") or []
        if objects and isinstance(objects, list) and objects[0].get("caption"):
            return str(objects[0]["caption"])
        # Object-level flat dict
        if cand.raw_parsed.get("caption"):
            return str(cand.raw_parsed["caption"])
    return cand.text[:500]  # truncate very long raw text
