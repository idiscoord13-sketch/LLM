"""Build the production scanner from whatever artefacts exist locally.

    from myllm.aruz.engine import build_scanner
    scanner = build_scanner()          # lexicon + metre prior + neural classifier, if present

Artefacts (all derived from Ganjoor, all git-ignored, regenerated with the commands in
docs/product/aruz.md):
  data/processed/aruz_lexicon.json   word-weight lexicon        (myllm.aruz.lexicon learn)
  data/processed/aruz_ensemble.json  α, β and the metre prior   (scripts/tune_aruz.py)
  out/aruz/classifier.pt             neural metre classifier    (myllm.aruz.classifier train)
Missing files simply switch that component off, so the bot always starts.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

from myllm.aruz.lexicon import load_lexicon
from myllm.aruz.scan import Scanner

DEFAULT_LEXICON = "data/processed/aruz_lexicon.json"
DEFAULT_ENSEMBLE = "data/processed/aruz_ensemble.json"
DEFAULT_CLASSIFIER = "out/aruz/classifier.pt"


def metre_prior(poem_keys: list[str], smoothing: float = 1.0) -> dict[str, float]:
    """log P(metre) from one vote per *poem* (so the 60k-line Shahnameh counts once per poem, not per line)."""
    counts = Counter(poem_keys)
    total = sum(counts.values()) + smoothing * len(counts)
    return {k: math.log((n + smoothing) / total) for k, n in counts.items()}


def load_ensemble(path: str | Path | None) -> dict:
    if not path or not Path(path).exists():
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_scanner(lexicon: str | None = DEFAULT_LEXICON, ensemble: str | None = DEFAULT_ENSEMBLE,
                  classifier: str | None = DEFAULT_CLASSIFIER, threads: int | None = None) -> Scanner:
    ens = load_ensemble(ensemble)
    clf = None
    if classifier and Path(classifier).exists() and ens.get("classifier_weight", 0) > 0:
        import torch

        from myllm.aruz.classifier import MeterClassifier

        if threads:
            torch.set_num_threads(threads)
        clf = MeterClassifier.load(classifier)
    return Scanner(load_lexicon(lexicon), prior=ens.get("prior"), prior_weight=ens.get("prior_weight", 0.0),
                   classifier=clf, classifier_weight=ens.get("classifier_weight", 0.0))
