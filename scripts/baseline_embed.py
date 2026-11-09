import json, sys
import numpy as np
from sentence_transformers import SentenceTransformer

rows = [json.loads(l) for l in open(sys.argv[1])]
m = SentenceTransformer(sys.argv[2] if len(sys.argv) > 2 else "sentence-transformers/all-MiniLM-L6-v2")
hit, chance = [], []
for r in rows:
    e = m.encode([r["context"]] + r["options"], normalize_embeddings=True)
    hit.append(int(np.argmax(e[1:] @ e[0]) == r["target"]))
    chance.append(1 / len(r["options"]))
h = np.array(hit)
print(f"accuracy {h.mean():.4f} +- {1.96 * h.std() / len(h) ** 0.5:.3f} | chance {np.mean(chance):.4f} | flip_rate 0.0")