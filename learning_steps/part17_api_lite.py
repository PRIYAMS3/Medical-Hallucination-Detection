from __future__ import annotations

from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import argparse
import json
import logging
from uuid import uuid4

import joblib
import numpy as np


ARTIFACT_DIR = Path("learning_steps") / "outputs" / "part12_artifacts"
LOG_PATH = Path("learning_steps") / "outputs" / "part20_api.log"
MAX_BODY_BYTES = 1_000_000

logger = logging.getLogger("phishing_api")


class PhishingEngine:
    def __init__(self, artifact_dir: Path) -> None:
        cfg = json.loads((artifact_dir / "inference_config.json").read_text(encoding="utf-8"))
        self.feature_order: list[str] = cfg["feature_order"]
        self.model_names: list[str] = cfg["member_models"]
        self.hybrid_threshold: float = float(cfg.get("hybrid_threshold", 0.8))

        self.models = {
            name: joblib.load(artifact_dir / f"{name}.joblib")
            for name in self.model_names
        }
        self.feature_index = {name: idx for idx, name in enumerate(self.feature_order)}

    def _rule_engine(self, x: np.ndarray) -> list[str]:
        rules = []
        if x[self.feature_index["having_IP_Address"]] == 1:
            rules.append("IP address used")
        if x[self.feature_index["SSLfinal_State"]] == -1:
            rules.append("Invalid SSL")
        if x[self.feature_index["URL_of_Anchor"]] == 1:
            rules.append("Suspicious anchor URLs")
        if x[self.feature_index["Prefix_Suffix"]] == 1:
            rules.append("Hyphen in domain")
        return rules

    def predict_one(self, features: dict[str, float], mode: str = "ensemble") -> dict[str, object]:
        if mode not in {"ensemble", "hybrid"}:
            raise ValueError("mode must be 'ensemble' or 'hybrid'")
        if not isinstance(features, dict):
            raise ValueError("features must be a JSON object")

        missing = [f for f in self.feature_order if f not in features]
        if missing:
            raise ValueError(f"Missing features: {missing}")

        values = []
        for f in self.feature_order:
            try:
                v = float(features[f])
            except Exception as exc:
                raise ValueError(f"Feature '{f}' must be numeric") from exc
            if not np.isfinite(v):
                raise ValueError(f"Feature '{f}' must be a finite number")
            values.append(v)

        x = np.array(values, dtype=float).reshape(1, -1)
        probs = [model.predict_proba(x)[:, 1][0] for model in self.models.values()]
        ensemble_prob = float(np.mean(probs))
        pred = int(ensemble_prob >= 0.5)
        confidence = float(max(ensemble_prob, 1.0 - ensemble_prob))
        rules = self._rule_engine(x[0])
        decision_type = "model_only"

        if mode == "hybrid" and confidence < self.hybrid_threshold and len(rules) > 0 and pred != 1:
            pred = 1
            decision_type = "rule_override"

        return {
            "mode": mode,
            "prediction_label": int(pred),
            "prediction_text": "Phishing" if pred == 1 else "Legitimate",
            "prob_positive_class": round(ensemble_prob, 6),
            "confidence": round(confidence, 6),
            "decision_type": decision_type,
            "rules_triggered": rules,
        }


def make_handler(engine: PhishingEngine):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, payload: dict[str, object]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            if self.path == "/health":
                self._send(200, {"status": "ok"})
                return
            self._send(404, {"error": "Not found"})

        def do_POST(self):  # noqa: N802
            req_id = str(uuid4())[:8]
            if self.path != "/predict":
                self._send(404, {"error": "Not found"})
                logger.warning("req=%s path=%s status=404", req_id, self.path)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0:
                    raise ValueError("Empty request body")
                if length > MAX_BODY_BYTES:
                    self._send(413, {"error": "Payload too large", "request_id": req_id})
                    logger.warning("req=%s status=413 bytes=%s", req_id, length)
                    return
                raw = self.rfile.read(length).decode("utf-8")
                payload = json.loads(raw)
                features = payload.get("features", {})
                mode = payload.get("mode", "ensemble")
                result = engine.predict_one(features=features, mode=mode)
                result["request_id"] = req_id
                self._send(200, result)
                logger.info(
                    "req=%s status=200 mode=%s pred=%s conf=%s",
                    req_id,
                    result.get("mode"),
                    result.get("prediction_text"),
                    result.get("confidence"),
                )
            except Exception as exc:  # keep server robust for local use
                self._send(400, {"error": str(exc), "request_id": req_id})
                logger.warning("req=%s status=400 error=%s", req_id, str(exc))

        def log_message(self, format: str, *args):  # noqa: A003
            return

    return Handler


def configure_logging() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return
    handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)


def run_server(host: str, port: int) -> None:
    configure_logging()
    engine = PhishingEngine(ARTIFACT_DIR)
    server = HTTPServer((host, port), make_handler(engine))
    print(f"Server running on http://{host}:{port}")
    print("Endpoints: GET /health, POST /predict")
    print(f"Logs: {LOG_PATH}")
    server.serve_forever()


def run_self_test() -> None:
    import pandas as pd

    engine = PhishingEngine(ARTIFACT_DIR)
    df = pd.read_csv(r"C:\Users\PRIYAMVADA NAMBIAR\Downloads\output.csv")
    sample = df.drop(columns=["Result"]).iloc[0].to_dict()
    out_ens = engine.predict_one(sample, mode="ensemble")
    out_hyb = engine.predict_one(sample, mode="hybrid")
    print(json.dumps({"ensemble": out_ens, "hybrid": out_hyb}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Lightweight phishing API server (no extra dependencies).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8008)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return
    run_server(args.host, args.port)


if __name__ == "__main__":
    main()
