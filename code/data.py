"""Dataset loading: claims, user history, evidence requirements, and images.

Resolves paths relative to the repo root so the code runs from anywhere. Images are
read once and base64-encoded for the OpenAI-style ``image_url`` content blocks.
"""

from __future__ import annotations

import base64
import csv
import os
from dataclasses import dataclass, field
from functools import lru_cache
from io import BytesIO

# Repo root = parent of the code/ directory that holds this file.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(REPO_ROOT, "dataset")

_MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}

# Image preprocessing (normalization): convert every image to true RGB JPEG and bound the
# long edge. Fixes mislabeled/RGBA/grayscale files (correct MIME) and caps token cost/latency
# on oversized photos. Toggle/tune via env vars; on by default.
IMG_PREPROCESS = os.environ.get("IMG_PREPROCESS", "1") != "0"
IMG_MAX_EDGE = int(os.environ.get("IMG_MAX_EDGE", "1536"))
IMG_JPEG_QUALITY = int(os.environ.get("IMG_JPEG_QUALITY", "85"))


def _normalize_image_bytes(abs_path: str) -> tuple[bytes, str]:
    """Return (jpeg_bytes, mime). Falls back to raw bytes if Pillow is unavailable/fails."""
    try:
        from PIL import Image, ImageOps
        with Image.open(abs_path) as im:
            im = ImageOps.exif_transpose(im)          # honor EXIF orientation (safe even if none)
            if im.mode != "RGB":
                im = im.convert("RGB")                 # drop alpha / promote grayscale -> real RGB
            w, h = im.size
            if max(w, h) > IMG_MAX_EDGE:
                scale = IMG_MAX_EDGE / max(w, h)
                im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
            buf = BytesIO()
            im.save(buf, format="JPEG", quality=IMG_JPEG_QUALITY)
            return buf.getvalue(), "image/jpeg"
    except Exception:  # noqa: BLE001 - any decode/IO error -> use raw bytes
        with open(abs_path, "rb") as fh:
            ext = os.path.splitext(abs_path)[1].lower()
            return fh.read(), _MIME.get(ext, "image/jpeg")


@dataclass
class Image:
    image_id: str          # filename without extension, e.g. "img_1"
    rel_path: str          # path as written in the CSV, e.g. "images/test/case_001/img_1.jpg"
    abs_path: str
    exists: bool
    data_uri: str = ""     # "data:image/jpeg;base64,...." (empty if missing)


@dataclass
class Claim:
    user_id: str
    image_paths: str       # raw CSV value (semicolon separated), preserved verbatim for output
    user_claim: str
    claim_object: str
    images: list[Image] = field(default_factory=list)
    # Gold fields (only present for sample_claims.csv); kept as a dict for evaluation.
    gold: dict = field(default_factory=dict)


def _abs(rel_path: str) -> str:
    # CSV paths look like "images/test/...". They live under dataset/.
    return os.path.join(DATASET_DIR, rel_path)


def load_image(rel_path: str) -> Image:
    rel_path = rel_path.strip()
    image_id = os.path.splitext(os.path.basename(rel_path))[0]
    abs_path = _abs(rel_path)
    exists = os.path.isfile(abs_path)
    data_uri = ""
    if exists:
        if IMG_PREPROCESS:
            raw, mime = _normalize_image_bytes(abs_path)
        else:
            ext = os.path.splitext(abs_path)[1].lower()
            mime = _MIME.get(ext, "image/jpeg")
            with open(abs_path, "rb") as fh:
                raw = fh.read()
        b64 = base64.b64encode(raw).decode("ascii")
        data_uri = f"data:{mime};base64,{b64}"
    return Image(image_id=image_id, rel_path=rel_path, abs_path=abs_path, exists=exists, data_uri=data_uri)


def load_claims(csv_path: str, with_images: bool = True) -> list[Claim]:
    """Load claims from a CSV. Works for both claims.csv (inputs only) and
    sample_claims.csv (inputs + gold outputs)."""
    claims: list[Claim] = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        gold_cols = [c for c in (reader.fieldnames or []) if c not in
                     ("user_id", "image_paths", "user_claim", "claim_object")]
        for row in reader:
            claim = Claim(
                user_id=row["user_id"],
                image_paths=row["image_paths"],
                user_claim=row["user_claim"],
                claim_object=row["claim_object"],
                gold={c: row[c] for c in gold_cols} if gold_cols else {},
            )
            if with_images:
                claim.images = [load_image(p) for p in row["image_paths"].split(";") if p.strip()]
            claims.append(claim)
    return claims


@lru_cache(maxsize=1)
def load_user_history() -> dict[str, dict]:
    path = os.path.join(DATASET_DIR, "user_history.csv")
    out: dict[str, dict] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[row["user_id"]] = row
    return out


@lru_cache(maxsize=1)
def load_evidence_requirements() -> list[dict]:
    path = os.path.join(DATASET_DIR, "evidence_requirements.csv")
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def requirements_for(claim_object: str) -> list[dict]:
    """Evidence requirements applicable to a claim object: object-specific + 'all'."""
    reqs = load_evidence_requirements()
    return [r for r in reqs if r["claim_object"] in (claim_object, "all")]
