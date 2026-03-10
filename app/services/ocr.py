from __future__ import annotations
import os
import re
from typing import Dict, Optional, Tuple, List
import cv2
import numpy as np
from functools import lru_cache

# --- Prefer pytesseract, fallback to EasyOCR ---
try:
    import pytesseract
except Exception:
    pytesseract = None

# Where to store OCR models/caches on HF Spaces (writable)
EASYOCR_DIR = os.getenv("EASYOCR_DIR", "/tmp/.easyocr")
os.environ.setdefault("TORCH_HOME", os.getenv("TORCH_HOME", "/tmp/torch"))


def _ensure_dir(p: str):
    try:
        os.makedirs(p, exist_ok=True)
    except Exception:
        pass


@lru_cache(maxsize=1)
def _easyocr():
    _ensure_dir(EASYOCR_DIR)
    _ensure_dir(os.path.join(EASYOCR_DIR, "user_network"))

    import easyocr
    reader = easyocr.Reader(
        lang_list=["en"],
        gpu=False,
        download_enabled=True,
        model_storage_directory=EASYOCR_DIR,
        user_network_directory=os.path.join(EASYOCR_DIR, "user_network"),
    )
    return reader


# ---------- OCR ----------
def ocr_text(bgr: np.ndarray) -> str:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if pytesseract is not None:
        try:
            return pytesseract.image_to_string(rgb) or ""
        except Exception:
            pass

    reader = _easyocr()
    lines = reader.readtext(rgb, detail=0) or []
    return "\n".join(lines)


# ---------- MAC / RSN helpers ----------
MAC_RE = re.compile(r"\b([0-9A-Fa-f]{2}[:\-\.]){5}[0-9A-Fa-f]{2}\b")
MAC_LINE_HINT = re.compile(r"\bMAC(?:\s*ID)?\b", re.IGNORECASE)

OUI_PREFER = {"CC:54:FE"}

RSN_LABELED_RE = re.compile(
    r"\b(?:RSN|S\/N|SERIAL)\s*[:#\-]?\s*([A-Z0-9\-]{6,24})\b",
    re.IGNORECASE,
)
RSN_TOKEN_RE = re.compile(r"[A-Z0-9\-]{6,24}")
RSN_STOPWORDS = {
    "INDIA", "MODEL", "WARRANTY", "PRODUCT", "POWER", "VOLT", "EAN"
}


def _normalize_mac(raw: str) -> Optional[str]:
    if not raw:
        return None

    t = raw.upper().translate(str.maketrans({
        "O": "0", "Q": "0", "I": "1", "L": "1",
        "S": "5", "B": "8", "Z": "2"
    }))

    pairs = re.findall(r"[0-9A-F]{2}", t)
    if len(pairs) < 6:
        return None

    best = None
    best_score = -1

    for i in range(len(pairs) - 5):
        mac = ":".join(pairs[i:i+6])
        score = 0
        if mac[:8] in OUI_PREFER:
            score += 2
        if ":" in raw or "-" in raw:
            score += 1
        if score > best_score:
            best_score = score
            best = mac

    return best


def _extract_mac_from_lines(lines: List[str]) -> Optional[str]:
    candidates: List[str] = []

    # A) MAC keyword lines
    for ln in lines:
        if not MAC_LINE_HINT.search(ln):
            continue

        norm = _normalize_mac(ln)
        if norm:
            candidates.append(norm)

    # B) Strict MAC regex anywhere
    if not candidates:
        for ln in lines:
            m = MAC_RE.search(ln)
            if m:
                norm = _normalize_mac(m.group(0))
                if norm:
                    candidates.append(norm)

    # C) FINAL fallback: spaced hex anywhere
    if not candidates:
        all_pairs = re.findall(r"[0-9A-F]{2}", " ".join(lines).upper())
        if len(all_pairs) >= 6:
            return ":".join(all_pairs[:6])

    if not candidates:
        return None

    candidates.sort(
        key=lambda m: (m[:8] in OUI_PREFER, ":" in m),
        reverse=True
    )
    return candidates[0]


def extract_mac(text: str) -> Optional[str]:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    return _extract_mac_from_lines(lines)


def _is_probable_rsn(token: str) -> bool:
    s = token.upper()
    if len(s) < 8 or len(s) > 24:
        return False
    if s in RSN_STOPWORDS:
        return False
    return sum(ch.isdigit() for ch in s) >= 3


def extract_rsn(text: str, lines: Optional[List[str]] = None) -> Optional[str]:
    m = RSN_LABELED_RE.search(text)
    if m and _is_probable_rsn(m.group(1)):
        return m.group(1).upper()

    tokens = RSN_TOKEN_RE.findall(text.upper())
    candidates = [t for t in tokens if _is_probable_rsn(t)]

    if not candidates:
        return None

    candidates.sort(key=lambda t: (sum(c.isdigit() for c in t), len(t)), reverse=True)
    return candidates[0]


# ---------- Public API ----------
def ocr_text_block(img: np.ndarray) -> str:
    reader = _easyocr()
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    lines = reader.readtext(rgb, detail=0) or []
    return "\n".join(lines)

def ocr_single_line(img: np.ndarray) -> str:
    """
    Returns the single longest OCR line.
    Used by validate.py as a fallback hint.
    """
    reader = _easyocr()
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    lines = reader.readtext(rgb, detail=0) or []
    if not lines:
        return ""
    return max(lines, key=len)

def extract_label_fields(text: str) -> Dict[str, Optional[str]]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return {
        "macId": _extract_mac_from_lines(lines),
        "rsn": extract_rsn(text, lines),
    }


# ---------- Azimuth ----------
ANGLE_RE = re.compile(
    r"\b(?P<deg>\d{1,3})\s*°?\s*(?P<dir>N|NE|E|SE|S|SW|W|NW)?\b",
    re.IGNORECASE,
)

def extract_angle(text: str) -> Tuple[Optional[int], Optional[str]]:
    for m in ANGLE_RE.finditer(text or ""):
        deg = int(m.group("deg"))
        if 0 <= deg <= 360:
            return deg, (m.group("dir") or "").upper() or None
    return None, None

def extract_azimuth(text: str) -> Dict[str, Optional[object]]:
    """
    Wrapper expected by validate.py.
    Extracts azimuth degree and direction from OCR text.
    """
    deg, ddir = extract_angle(text or "")
    return {
        "azimuthDeg": deg,
        "azimuthDir": ddir
    }

