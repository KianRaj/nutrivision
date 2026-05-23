"""NutriVision — AWS Lambda Container (Flask under apig-wsgi).

Same Flask routes as the HF-Space variant, except the checkpoint is read
from the baked-in `/var/task/best_model.pt` instead of downloaded from HF.
Avoids the 829 MB download on every Lambda cold start.
"""
from __future__ import annotations
import io
import os
import sys
import time
import traceback
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
from models.om_igsm_net import OmIGSMNet  # noqa: E402

DOCS_DIR      = os.path.join(THIS_DIR, "docs")
LOCAL_CKPT    = os.path.join(THIS_DIR, "best_model.pt")
DEVICE        = torch.device("cpu")
EXPECTED_PMAE = 13.48

IMG_NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def preprocess_rgb(pil_img: Image.Image, img_size: int = 224) -> torch.Tensor:
    pil = pil_img.convert("RGB").resize((img_size, img_size), Image.BILINEAR)
    arr = np.array(pil).astype(np.float32) / 255.0
    t   = torch.from_numpy(arr).permute(2, 0, 1)
    return IMG_NORM(t).unsqueeze(0)


def preprocess_depth_v1(depth_hw: np.ndarray, img_size: int = 224) -> torch.Tensor:
    d = depth_hw.astype(np.float32)
    lo, hi = d.min(), d.max()
    d = (d - lo) / (hi - lo) if hi > lo else np.zeros_like(d)
    d_pil = Image.fromarray((d * 255).astype(np.uint8), mode="L")
    d_pil = d_pil.resize((img_size, img_size), Image.BILINEAR)
    d_np  = np.array(d_pil).astype(np.float32) / 255.0
    return torch.from_numpy(d_np).unsqueeze(0).unsqueeze(0)


# ─── lazy singletons ───────────────────────────────────────────────────────
_model = None
_depth = None
_clip  = None
_ingr_suggester = None


def get_model() -> nn.Module:
    global _model
    if _model is not None:
        return _model
    print(f"[load] checkpoint from {LOCAL_CKPT} ({os.path.getsize(LOCAL_CKPT)/1e6:.0f} MB)")
    print("[load] building OmIGSMNet…")
    m = OmIGSMNet(
        backbone_name="convnext_base.fb_in22k_ft_in1k",
        out_c=256, cls_dim=512, unfreeze_stages=2,
        ism_blocks=2, ism_heads=4, ism_window=7,
        clip_dim=512, tau_freq=0.20, fafm_heads=4,
        mph_hidden=512, num_tasks=5, drop_rate=0.1,
    ).to(DEVICE).eval()
    ckpt = torch.load(LOCAL_CKPT, map_location=DEVICE, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    missing, _ = m.load_state_dict(sd, strict=False)
    if missing:
        print(f"[load] {len(missing)} missing keys (first 3): {missing[:3]}")
    _model = m
    print("[load] model ready.")
    return _model


def get_depth_pipe():
    global _depth
    if _depth is not None:
        return _depth
    from transformers import pipeline
    print("[load] Depth-Anything V2-Small…")
    _depth = pipeline(
        "depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
        device=-1,
    )
    return _depth


def get_clip():
    global _clip
    if _clip is not None:
        return _clip
    import clip
    print("[load] CLIP ViT-B/32…")
    m, _ = clip.load("ViT-B/32", device=str(DEVICE),
                     download_root=os.environ.get("HF_HOME", "/tmp/clip"))
    m.eval()
    _clip = m
    return _clip


def get_ingr_suggester():
    global _ingr_suggester
    if _ingr_suggester is not None:
        return _ingr_suggester
    from transformers import BlipForConditionalGeneration, BlipProcessor
    import clip as _clip
    print("[load] BLIP captioner…")
    blip_proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(DEVICE).eval()
    vocab = [
        "chicken", "beef", "pork", "salmon", "tuna", "shrimp", "egg", "tofu",
        "rice", "brown rice", "pasta", "noodles", "bread", "tortilla",
        "potato", "fries", "mashed potato", "quinoa", "oats",
        "broccoli", "lettuce", "spinach", "kale", "tomato", "cucumber",
        "carrot", "bell pepper", "onion", "garlic", "mushroom", "corn",
        "green beans", "peas", "avocado", "cheese", "butter", "yogurt",
        "beans", "chickpea", "olive oil", "salad", "soup",
        "sandwich", "burger", "pizza", "salsa", "soy sauce", "hummus",
    ]
    clip_m = get_clip()
    with torch.no_grad():
        tok = _clip.tokenize([f"a photo of {t}" for t in vocab]).to(DEVICE)
        txt = clip_m.encode_text(tok).float()
        txt = txt / txt.norm(dim=-1, keepdim=True)
    _ingr_suggester = (blip, blip_proc, vocab, txt)
    return _ingr_suggester


@torch.no_grad()
def encode_ingredients(text: str) -> torch.Tensor:
    import clip
    if not text or not text.strip():
        return torch.zeros(1, 512, device=DEVICE)
    tok = clip.tokenize([text]).to(DEVICE)
    return get_clip().encode_text(tok).float()


@torch.no_grad()
def run_predict(image: Image.Image, ingredients: str) -> dict:
    image = image.convert("RGB")
    rgb_t = preprocess_rgb(image).to(DEVICE)
    depth_out = get_depth_pipe()(image)
    depth_np = np.array(depth_out["predicted_depth"]).astype(np.float32)
    if depth_np.ndim == 3:
        depth_np = depth_np[0]
    depth_t = preprocess_depth_v1(depth_np).to(DEVICE)
    ingr_t = encode_ingredients(ingredients or "")
    pred, _ = get_model()(rgb_t, depth_t, ingr_t)
    cal, mass, fat, carb, prot = pred.squeeze(0).cpu().numpy().tolist()
    return {
        "predictions": {
            "calories_kcal": float(cal),
            "mass_g":        float(mass),
            "fat_g":         float(fat),
            "carbs_g":       float(carb),
            "protein_g":     float(prot),
        },
        "depth_min": float(depth_np.min()),
        "depth_max": float(depth_np.max()),
        "checkpoint": "best_model.pt",
        "expected_pmae": EXPECTED_PMAE,
    }


@torch.no_grad()
def run_suggest(image: Image.Image, k: int = 6) -> dict:
    image = image.convert("RGB")
    import clip as _clip
    blip, blip_proc, vocab, vocab_emb = get_ingr_suggester()
    inputs = blip_proc(image, "a photo of", return_tensors="pt").to(DEVICE)
    cap_ids = blip.generate(**inputs, max_new_tokens=30)
    caption = blip_proc.decode(cap_ids[0], skip_special_tokens=True)
    clip_m = get_clip()
    preproc = T.Compose([
        T.Resize(224, interpolation=Image.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711)),
    ])
    img_t = preproc(image).unsqueeze(0).to(DEVICE)
    img_emb = clip_m.encode_image(img_t).float()
    img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
    sims = (img_emb @ vocab_emb.T).squeeze(0).cpu().numpy()
    idx = np.argsort(sims)[::-1][:k]
    top = [vocab[i] for i in idx if sims[i] > 0.2][:k]
    return {"ingredients": ", ".join(top), "caption": caption}


# ─── Flask app ─────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=DOCS_DIR, static_url_path="")
CORS(app, resources={r"/api/*": {"origins": "*"}})


@app.get("/")
def root():
    return send_from_directory(DOCS_DIR, "index.html")


@app.get("/healthz")
def healthz():
    return jsonify({
        "ok": True,
        "model_loaded": _model is not None,
        "expected_pmae": EXPECTED_PMAE,
        "checkpoint": "best_model.pt",
        "runtime": "aws-lambda-container",
    })


def _open_uploaded(field: str = "image") -> Image.Image:
    f = request.files.get(field)
    if not f:
        raise ValueError("missing 'image' upload")
    return Image.open(io.BytesIO(f.read()))


@app.post("/api/predict")
def api_predict():
    try:
        image = _open_uploaded("image")
        ingredients = (request.form.get("ingredients") or "").strip()
        t0 = time.time()
        out = run_predict(image, ingredients)
        out["latency_ms"] = int((time.time() - t0) * 1000)
        return jsonify({"ok": True, "ingredients": ingredients, **out})
    except Exception:
        return jsonify({"ok": False, "error": traceback.format_exc()}), 500


@app.post("/api/suggest")
def api_suggest():
    try:
        image = _open_uploaded("image")
        out = run_suggest(image, k=6)
        return jsonify({"ok": True, **out})
    except Exception:
        return jsonify({"ok": False, "error": traceback.format_exc()}), 500
