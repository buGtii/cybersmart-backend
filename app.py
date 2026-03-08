"""
CyberSmart - Flask API v2
Production-ready phishing detection REST API
"""

import re
import time
import warnings
import logging
import urllib.parse

warnings.filterwarnings("ignore")

from flask import Flask, request, jsonify
from flask_cors import CORS
from feature_extractor import FeatureExtractor
from model import ModelManager

# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CyberSmart")

# ─────────────────────────────────────────────
#  TRUSTED DOMAIN WHITELIST
#  These are verified legitimate domains. Subdomains are also trusted.
#  e.g. accounts.google.com, login.microsoftonline.com, store.steampowered.com
# ─────────────────────────────────────────────
TRUSTED_DOMAINS = {
    # Google
    "google.com", "googleapis.com", "googleusercontent.com",
    "youtube.com", "gmail.com", "google.co.uk", "google.com.au",
    # Microsoft
    "microsoft.com", "microsoftonline.com", "live.com",
    "outlook.com", "office.com", "azure.com", "office365.com",
    # Apple
    "apple.com", "icloud.com",
    # Amazon / AWS
    "amazon.com", "aws.amazon.com", "aws.com", "amazonaws.com",
    # Social
    "facebook.com", "instagram.com", "twitter.com",
    "linkedin.com", "reddit.com", "tiktok.com",
    # Tech
    "github.com", "stackoverflow.com", "wikipedia.org",
    "mozilla.org", "w3.org", "cloudflare.com",
    # Streaming / software
    "netflix.com", "spotify.com", "steampowered.com",
    "twitch.tv", "discord.com", "slack.com", "zoom.us",
    # Storage
    "dropbox.com", "drive.google.com", "onedrive.live.com",
    # Finance
    "paypal.com", "stripe.com", "chase.com", "wellsfargo.com",
    "bankofamerica.com", "citibank.com",
    # Dev / docs
    "python.org", "docs.python.org", "developer.apple.com",
    "docs.microsoft.com", "developer.mozilla.org",
}

def _get_root_domain(url: str) -> str:
    """Extract root domain (e.g. accounts.google.com → google.com)"""
    try:
        if not url.startswith(("http://", "https://")):
            url = "http://" + url
        host = urllib.parse.urlparse(url).hostname or ""
        host = host.lower()
        parts = host.split(".")
        if len(parts) >= 2:
            return ".".join(parts[-2:])
        return host
    except Exception:
        return ""

def _is_trusted(url: str) -> bool:
    """Return True if the URL's root domain is in the trusted whitelist."""
    return _get_root_domain(url) in TRUSTED_DOMAINS

# ─────────────────────────────────────────────
#  APP INIT
# ─────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

logger.info("Initializing CyberSmart API...")
extractor     = FeatureExtractor(timeout=5)
model_manager = ModelManager()
logger.info("CyberSmart API ready!")


# ─────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "name":    "CyberSmart API",
        "version": "2.0.0",
        "status":  "running",
        "endpoints": {
            "POST /predict":          "Analyze a URL for phishing",
            "POST /predict/batch":    "Analyze multiple URLs (max 50)",
            "POST /features/extract": "Get raw features for a URL",
            "GET  /features":         "List all feature names",
            "GET  /health":           "Health check"
        }
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "timestamp": time.time()})


@app.route("/features", methods=["GET"])
def features():
    return jsonify({
        "feature_count": len(FeatureExtractor.FEATURE_NAMES),
        "features":      FeatureExtractor.FEATURE_NAMES
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "url" not in data:
        return jsonify({"error": "Missing 'url' field in request body"}), 400

    url = data["url"].strip()
    if not url:
        return jsonify({"error": "URL cannot be empty"}), 400

    try:
        start = time.time()

        # ── Whitelist check (trusted domains bypass ML model) ────────────────
        if _is_trusted(url):
            elapsed_ms = round((time.time() - start) * 1000, 2)
            logger.info(f"[WHITELIST] {url[:50]} → safe (trusted domain)")
            return jsonify({
                "url":                url,
                "label":              "safe",
                "probability":        0.01,
                "confidence":         99.0,
                "risk_level":         "LOW",
                "method":             "whitelist",
                "features":           {},
                "processing_time_ms": elapsed_ms
            })

        # ── ML prediction ────────────────────────────────────────────────────
        features_list = extractor.extract(url)
        features_dict = dict(zip(FeatureExtractor.FEATURE_NAMES, features_list))
        prediction    = model_manager.predict(features_list)
        elapsed_ms    = round((time.time() - start) * 1000, 2)

        logger.info(f"[PREDICT] {url[:50]} → {prediction['label']} "
                    f"({prediction['probability']:.3f}) [{elapsed_ms}ms]")

        return jsonify({
            "url":                url,
            "label":              prediction["label"],
            "probability":        prediction["probability"],
            "confidence":         prediction["confidence"],
            "risk_level":         prediction["risk_level"],
            "method":             "ml_model",
            "features":           features_dict,
            "processing_time_ms": elapsed_ms
        })

    except Exception as e:
        logger.error(f"[ERROR] {url} — {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    data = request.get_json()
    if not data or "urls" not in data:
        return jsonify({"error": "Missing 'urls' field"}), 400

    urls = data["urls"]
    if not isinstance(urls, list) or len(urls) == 0:
        return jsonify({"error": "'urls' must be a non-empty list"}), 400
    if len(urls) > 50:
        return jsonify({"error": "Maximum 50 URLs per batch request"}), 400

    results = []
    for url in urls:
        try:
            if _is_trusted(url):
                results.append({
                    "url":        url,
                    "label":      "safe",
                    "probability": 0.01,
                    "risk_level": "LOW",
                    "method":     "whitelist"
                })
            else:
                features_list = extractor.extract(url)
                pred = model_manager.predict(features_list)
                results.append({
                    "url":         url,
                    "label":       pred["label"],
                    "probability": pred["probability"],
                    "risk_level":  pred["risk_level"],
                    "method":      "ml_model"
                })
        except Exception as e:
            results.append({"url": url, "error": str(e)})

    return jsonify({"total": len(urls), "results": results})


@app.route("/features/extract", methods=["POST"])
def extract_features():
    data = request.get_json()
    if not data or "url" not in data:
        return jsonify({"error": "Missing 'url' field"}), 400
    url = data["url"].strip()
    try:
        features_dict = extractor.extract_with_names(url)
        return jsonify({
            "url":           url,
            "feature_count": len(features_dict),
            "trusted":       _is_trusted(url),
            "features":      features_dict
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
#  ERROR HANDLERS
# ─────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed"}), 405

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════╗
║        CyberSmart API v2.0.0             ║
║   AI-Powered Phishing Detection System   ║
╚══════════════════════════════════════════╝

  Local:   http://localhost:5000
  Network: http://0.0.0.0:5000

  Use ngrok for public access:
    ngrok http 5000

  Detection layers:
    1. Trusted domain whitelist (instant)
    2. ML model — 56 URL features (91% accuracy)
""")
    app.run(host="0.0.0.0", port=5000, debug=False)
