"""Compare released-model outputs across a package rename on real smoke inputs."""

import argparse
import gc
import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", default="rnavolve")
    parser.add_argument("--class-prefix", default="RNAvolve")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    torch.manual_seed(2222)
    torch.set_num_threads(4)
    api = importlib.import_module(f"{args.package}.evaluation.inference_api")
    predictor_class = getattr(api, args.class_prefix + "Predictor")
    frame = pd.read_parquet(ROOT / "assets/smoke/student/human_test.parquet")
    result = {"transcript_ids": frame.transcript_id.tolist(), "predictions": {}}
    for name in ("w0", "w1", "w2", "final_tdc", "hl_new"):
        predictor = predictor_class(name, device=args.device)
        values = predictor.predict(frame.mrna.tolist(), frame.utr5_size.tolist(), batch_size=1)
        assert np.isfinite(values).all(), name
        result["predictions"][name] = values
        del predictor
        gc.collect()
        torch.cuda.empty_cache()
    embedding = importlib.import_module(f"{args.package}.evaluation.embedding")
    tokenizer_module = importlib.import_module(f"{args.package}.models.tokenization")
    tokenizer = getattr(tokenizer_module, args.class_prefix + "Tokenizer")(model_max_length=10000)
    dataset = embedding.AlignmentDataset(frame.iloc[:1], tokenizer)
    ids = dataset[0]["input_ids"].unsqueeze(0).to(args.device)
    encoder = embedding.build_backbone(embedding.DEFAULT_CONFIG, embedding.DEFAULT_WEIGHTS, args.device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        output = encoder(ids)
    hidden = output.last_hidden_state if hasattr(output, "last_hidden_state") else output[0]
    assert torch.isfinite(hidden).all()
    result["embedding_shape"] = list(hidden.shape)
    result["embedding_sha256"] = hashlib.sha256(hidden.float().cpu().numpy().tobytes()).hexdigest()
    teacher_module = importlib.import_module(f"{args.package}.models.teacher")
    teacher = teacher_module.load_teacher(ROOT / "assets/checkpoints/teacher/privileged_teacher.pkl")
    names = json.loads((ROOT / "assets/checkpoints/teacher/metadata/feature_names.json").read_text())
    features = pd.read_csv(ROOT / "assets/smoke/teacher/validation.csv.gz")
    result["teacher_predictions"] = teacher.predict(features[names]).tolist()
    assert np.isfinite(result["teacher_predictions"]).all()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.reference:
        reference = json.loads(args.reference.read_text())
        assert result == reference, "Renaming changed inference, embedding, or teacher outputs"
        print("PASS exact rename equivalence: five predictors, backbone embeddings, teacher scoring")
    else:
        print(f"Saved baseline outputs to {args.output}")


if __name__ == "__main__":
    main()
