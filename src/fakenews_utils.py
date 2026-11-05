"""Shared helpers for PCS26/84 - Deep Learning-Based Fake News Detection.

Import-safe module (no notebook state). Used by notebooks 01-03 and by any
demo/inference script. Everything here is deterministic and documented so that
the same preprocessing is applied at training time and at prediction time.

IMPORTANT LIMITATION: models built with this module predict the *dataset label*
(Fake=0 / Real=1). They do not verify factual truth.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import Pipeline

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
LABEL_MAP = {"Fake": 0, "Real": 1}          # project convention (fixed)
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
GROUP_SEPARATOR = " || "                    # same separator as Notebook 01
DISCLAIMER = (
    "This model predicts whether an article resembles the 'Fake' or 'Real' class "
    "of its training dataset. It does not verify factual claims."
)

# Same TF-IDF configuration as the original Notebook 02 baseline.
TFIDF_KWARGS = dict(
    lowercase=True,
    strip_accents="unicode",
    ngram_range=(1, 2),
    max_features=100_000,
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    dtype=np.float32,
)


# --------------------------------------------------------------------------
# Paths, hashing, protected writes
# --------------------------------------------------------------------------
def find_project_root(start=None, marker: str = "dataset") -> Path:
    """Walk upwards from `start` (default: cwd) until a folder containing
    `marker/` is found. Works from the project root or from notebooks/."""
    here = Path(start).resolve() if start else Path.cwd().resolve()
    for candidate in [here, *here.parents]:
        if (candidate / marker).is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not find a project folder containing '{marker}/' above {here}. "
        "Open the notebook from the project root or from the notebooks/ folder."
    )


def sha256_file(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


class HashRegistry:
    """Records SHA-256 digests of protected files (split manifest, clean CSV,
    model files) the first time they are seen and verifies them afterwards, so
    a silently changed manifest or dataset cannot go unnoticed."""

    def __init__(self, registry_path, project_root):
        self.path = Path(registry_path)
        self.root = Path(project_root).resolve()
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {}

    def check(self, file_path, strict: bool = True) -> str:
        file_path = Path(file_path).resolve()
        rel = file_path.relative_to(self.root).as_posix()
        digest = sha256_file(file_path)
        if rel not in self.data:
            self.data[rel] = {
                "sha256": digest,
                "recorded_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            }
            self.path.write_text(json.dumps(self.data, indent=2))
            return "recorded"
        if self.data[rel]["sha256"] == digest:
            return "verified"
        msg = f"HASH MISMATCH for {rel}: file changed since it was first recorded."
        if strict:
            raise RuntimeError(msg)
        return "MISMATCH"


def write_protected_csv(df: pd.DataFrame, path, index: bool = False) -> str:
    """Write `df` only if `path` does not exist. If it exists and the content is
    identical nothing is written; if it differs, the new content is saved next
    to it with a '.REBUILT' suffix and the original is left untouched."""
    path = Path(path)
    if not path.exists():
        df.to_csv(path, index=index)
        return "written"
    existing = pd.read_csv(path, dtype=str, keep_default_na=False)
    new = df.reset_index(drop=True).astype(str) if not index else df.astype(str)
    if existing.shape == new.shape and list(existing.columns) == list(new.columns) \
            and all(existing[c].equals(new[c].reset_index(drop=True)) for c in existing.columns):
        return "identical (not rewritten)"
    alt = path.with_name(path.stem + ".REBUILT" + path.suffix)
    df.to_csv(alt, index=index)
    return f"DIFFERS from existing file; new content saved to {alt.name}"


# --------------------------------------------------------------------------
# Article keys / groups
# --------------------------------------------------------------------------
def _norm(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.lower()


def article_key(title: pd.Series, text: pd.Series) -> pd.Series:
    """Normalised 'title || body' key - identical to the key used to build the
    group-aware split in Notebook 01."""
    return _norm(title) + GROUP_SEPARATOR + _norm(text)


def group_ids(keys: pd.Series) -> pd.Series:
    return keys.map(lambda k: hashlib.md5(k.encode("utf-8")).hexdigest()[:16])


# --------------------------------------------------------------------------
# Publisher / formatting marker rules
# --------------------------------------------------------------------------
# These are CANDIDATE rules. None is assumed to matter: ArticleTextBuilder keeps
# a rule only if the TRAINING partition shows it is strongly class-skewed
# (see compute_rule_report). Order matters: rules are applied in this order.
_RULE_DEFS = [
    ("dateline_reuters",
     r"^\s*(?:[A-Z][A-Za-z0-9.,'/&\- ]{0,60}?)?\(\s*Reuters\s*\)\s*[-\u2013\u2014:]*\s*", 0,
     "Leading '[CITY] (Reuters) -' dateline (seen in the sample rows of Notebook 01)"),
    ("reuters_token", r"\bReuters\b", 0, "Any remaining 'Reuters' token"),
    ("reporting_by_credit",
     r"\(\s*(?:Additional\s+)?(?:Reporting|Writing|Editing)\s+by[^)]{0,250}\)", re.I,
     "Wire-style '(Reporting by ...; Editing by ...)' credit"),
    ("century_wire", r"\b21st\s+Century\s+Wire\b(?:\s+says)?", re.I,
     "'21st Century Wire (says)' (seen in a Fake sample row of Notebook 01)"),
    ("featured_image", r"\bfeatured\s+image(?:\s+(?:via|credit|courtesy\s+of))?", re.I,
     "'Featured image (via ...)' caption phrase"),
    ("image_credit", r"\b(?:photo|image)\s+(?:via|credit|courtesy\s+of)\b", re.I,
     "'Photo/Image via|credit' caption phrase"),
    ("getty_images", r"\bgetty\s+images?\b", re.I, "'Getty Images' credit"),
    ("pic_twitter", r"\bpic\.twitter\.com/\S+", re.I, "Embedded tweet picture link"),
    ("url", r"https?://\S+|\bwww\.\S+", re.I, "URLs"),
    ("twitter_handle", r"(?<![\w.])@\w{1,15}\b", 0, "@handles"),
    ("watch_label", r"\bWATCH\b\s*[:\-\u2013\u2014]", 0, "'WATCH:' embed label"),
    ("read_more_label", r"\bREAD\s+MORE\b\s*[:\-\u2013\u2014]?", 0, "'READ MORE:' label"),
    ("generic_dateline", r"^\s*[A-Z][A-Z .,'/\-]{2,40}\s*[-\u2013\u2014]\s+", 0,
     "Generic 'CITY - ' dateline without a named agency"),
]
RULE_ORDER = [name for name, *_ in _RULE_DEFS]
RULES_BY_NAME = {
    name: {"regex": re.compile(pattern, flags), "pattern": pattern, "description": desc}
    for name, pattern, flags, desc in _RULE_DEFS
}


def compute_rule_report(titles, bodies, y, min_prevalence: float = 0.01,
                        min_ratio: float = 5.0) -> pd.DataFrame:
    """Prevalence of every candidate rule per class (share of documents in which
    the rule matches the title or the body) and whether it is selected.
    Call this on TRAINING data only."""
    titles = pd.Series(titles).fillna("").astype(str).tolist()
    bodies = pd.Series(bodies).fillna("").astype(str).tolist()
    y = np.asarray(y).astype(int)
    rows = []
    for name in RULE_ORDER:
        rx = RULES_BY_NAME[name]["regex"]
        hit = np.fromiter(
            (bool(rx.search(t)) or bool(rx.search(b)) for t, b in zip(titles, bodies)),
            dtype=bool, count=len(titles))
        p_fake = float(hit[y == 0].mean()) if (y == 0).any() else 0.0
        p_real = float(hit[y == 1].mean()) if (y == 1).any() else 0.0
        ratio = (max(p_fake, p_real) + 1e-3) / (min(p_fake, p_real) + 1e-3)
        rows.append({
            "rule": name,
            "description": RULES_BY_NAME[name]["description"],
            "prevalence_fake": p_fake,
            "prevalence_real": p_real,
            "skew_ratio": ratio,
            "selected": bool(max(p_fake, p_real) >= min_prevalence and ratio >= min_ratio),
        })
    return pd.DataFrame(rows)


class ArticleTextBuilder(BaseEstimator, TransformerMixin):
    """Turns a DataFrame with columns ['title', 'text'] into model input strings.

    use_title / use_body : which fields enter the model.
    strip_markers        : remove publisher/format markers (see _RULE_DEFS).
    rule_names           : explicit list of rules to apply (frozen); if None and
                           strip_markers=True, rules are selected in fit() from
                           the training data via compute_rule_report.
    """

    def __init__(self, use_title=True, use_body=True, strip_markers=False,
                 rule_names=None, min_prevalence=0.01, min_ratio=5.0):
        self.use_title = use_title
        self.use_body = use_body
        self.strip_markers = strip_markers
        self.rule_names = rule_names
        self.min_prevalence = min_prevalence
        self.min_ratio = min_ratio

    @staticmethod
    def _fields(X):
        if not isinstance(X, pd.DataFrame) or not {"title", "text"} <= set(X.columns):
            raise TypeError("ArticleTextBuilder expects a DataFrame with 'title' and 'text' columns.")
        return (X["title"].fillna("").astype(str).tolist(),
                X["text"].fillna("").astype(str).tolist())

    def fit(self, X, y=None):
        if not (self.use_title or self.use_body):
            raise ValueError("At least one of use_title/use_body must be True.")
        self.active_rules_ = []
        self.rule_report_ = None
        if self.strip_markers:
            if self.rule_names is not None:
                unknown = set(self.rule_names) - set(RULE_ORDER)
                if unknown:
                    raise ValueError(f"Unknown rule names: {sorted(unknown)}")
                self.active_rules_ = [r for r in RULE_ORDER if r in set(self.rule_names)]
            else:
                if y is None:
                    raise ValueError("y is required to select rules when rule_names is None.")
                titles, bodies = self._fields(X)
                self.rule_report_ = compute_rule_report(
                    titles, bodies, y, self.min_prevalence, self.min_ratio)
                self.active_rules_ = self.rule_report_.loc[self.rule_report_.selected, "rule"].tolist()
        return self

    def _clean(self, s: str) -> str:
        for name in self.active_rules_:
            s = RULES_BY_NAME[name]["regex"].sub(" ", s)
        return re.sub(r"\s+", " ", s).strip()

    def transform(self, X):
        if not hasattr(self, "active_rules_"):
            raise RuntimeError("ArticleTextBuilder is not fitted.")
        titles, bodies = self._fields(X)
        out = []
        for t, b in zip(titles, bodies):
            if self.strip_markers:
                t, b = self._clean(t), self._clean(b)
            else:
                t, b = t.strip(), b.strip()
            parts = []
            if self.use_title and t:
                parts.append(t)
            if self.use_body and b:
                parts.append(b)
            out.append(" ".join(parts))
        return out


# --------------------------------------------------------------------------
# Input variants used in the ablation study
# --------------------------------------------------------------------------
VARIANTS = {
    "V0": dict(use_title=True,  use_body=True,  strip_markers=False, description="Raw title + body"),
    "V1": dict(use_title=False, use_body=True,  strip_markers=False, description="Body only (raw)"),
    "V2": dict(use_title=True,  use_body=False, strip_markers=False, description="Title only (raw)"),
    "V3": dict(use_title=True,  use_body=True,  strip_markers=True,  description="Title + body, publisher markers removed"),
    "V4": dict(use_title=False, use_body=True,  strip_markers=True,  description="Body only, publisher markers removed"),
}


def make_builder(variant: str, rule_names=None) -> ArticleTextBuilder:
    spec = VARIANTS[variant]
    return ArticleTextBuilder(
        use_title=spec["use_title"], use_body=spec["use_body"],
        strip_markers=spec["strip_markers"],
        rule_names=rule_names if spec["strip_markers"] else None)


def make_tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(**TFIDF_KWARGS)


def assemble_pipeline(builder, tfidf, clf) -> Pipeline:
    """Combine already-fitted steps into one end-to-end Pipeline
    (DataFrame[title,text] -> prediction)."""
    return Pipeline([("text", builder), ("tfidf", tfidf), ("clf", clf)])


def positive_class_score(clf, X) -> np.ndarray:
    """Score for the positive class (Real=1): decision margin if available,
    otherwise predicted probability."""
    if hasattr(clf, "decision_function"):
        return np.asarray(clf.decision_function(X)).ravel()
    return np.asarray(clf.predict_proba(X))[:, 1]


# --------------------------------------------------------------------------
# Information-matched ML baseline (same head-truncation as the neural models)
# --------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"[a-z]+|[0-9]+|[^\w\s]")


class TokenTruncator(BaseEstimator, TransformerMixin):
    """Keeps only the first `max_tokens` tokens of each text, using the SAME
    tokenisation as the LSTM/BiLSTM (fakenews_dl.tokenize). Used to give the
    classical model exactly the information the neural models see."""

    def __init__(self, max_tokens=200):
        self.max_tokens = max_tokens

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [" ".join(_TOKEN_RE.findall(t.lower())[: self.max_tokens]) for t in X]


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def evaluate_binary(y_true, y_pred, y_score=None) -> dict:
    """Accuracy, per-class precision/recall/F1, macro-F1 and ROC-AUC.
    Real (1) is the positive class for ROC-AUC."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0)
    out = {
        "n": int(len(y_true)),
        "accuracy": float((y_true == y_pred).mean()),
        "precision_fake": float(p[0]), "recall_fake": float(r[0]), "f1_fake": float(f[0]),
        "precision_real": float(p[1]), "recall_real": float(r[1]), "f1_real": float(f[1]),
        "macro_f1": float(f.mean()),
        "roc_auc": float("nan"),
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
    return out


def bootstrap_ci(y_true, y_pred, metric, groups=None, n_boot=1000, seed=42, alpha=0.05):
    """Percentile bootstrap CI for metric(y_true, y_pred). If `groups` is given,
    whole duplicate-groups are resampled (avoids over-confident intervals when
    repeated articles exist)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if groups is None:
        idx_sets = [np.array([i]) for i in range(n)]
    else:
        codes, uniques = pd.factorize(pd.Series(groups))
        idx_sets = [np.flatnonzero(codes == g) for g in range(len(uniques))]
    G = len(idx_sets)
    stats = []
    for _ in range(n_boot):
        pick = rng.integers(0, G, G)
        idx = np.concatenate([idx_sets[g] for g in pick])
        stats.append(metric(y_true[idx], y_pred[idx]))
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def mcnemar_exact(correct_a, correct_b) -> dict:
    """Exact McNemar test on two paired boolean 'is correct' vectors."""
    from scipy.stats import binomtest
    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    only_a = int((a & ~b).sum())
    only_b = int((~a & b).sum())
    n = only_a + only_b
    p = 1.0 if n == 0 else float(binomtest(min(only_a, only_b), n, 0.5).pvalue)
    return {"a_right_b_wrong": only_a, "a_wrong_b_right": only_b, "p_value": p}


def top_linear_features(tfidf, clf, n: int = 30) -> pd.DataFrame:
    """Most influential n-grams of a linear model. Positive weight -> Real(1),
    negative weight -> Fake(0)."""
    names = np.asarray(tfidf.get_feature_names_out())
    coef = np.asarray(clf.coef_).ravel()
    order = np.argsort(coef)
    fake_idx, real_idx = order[:n], order[::-1][:n]
    return pd.DataFrame({
        "rank": np.arange(1, n + 1),
        "toward_Fake": names[fake_idx], "weight_Fake": coef[fake_idx],
        "toward_Real": names[real_idx], "weight_Real": coef[real_idx],
    })


# --------------------------------------------------------------------------
# Saving / loading / prediction
# --------------------------------------------------------------------------
def load_bundle(path):
    import joblib
    return joblib.load(path)


def predict_article(title: str, text: str, bundle: dict, min_words: int = 20) -> dict:
    """Predict the dataset label of ONE article with a saved bundle.
    Applies exactly the preprocessing used in training (the bundled pipeline)."""
    pipe = bundle["pipeline"]
    frame = pd.DataFrame({"title": [title or ""], "text": [text or ""]})
    model_input = pipe.named_steps["text"].transform(frame)[0]
    n_words = len(model_input.split())
    if n_words == 0:
        return {"label": None, "label_id": None, "note": "No usable text after preprocessing.",
                "disclaimer": DISCLAIMER}
    label_id = int(pipe.predict(frame)[0])
    result = {"label": ID_TO_LABEL[label_id], "label_id": label_id,
              "n_words_used": n_words, "disclaimer": DISCLAIMER}
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        result["probability_real"] = float(pipe.predict_proba(frame)[0, 1])
    elif hasattr(clf, "decision_function"):
        result["decision_margin"] = float(pipe.decision_function(frame)[0])
        result["margin_note"] = "Signed margin (positive -> Real). Not a calibrated probability."
    if n_words < min_words:
        result["warning"] = (f"Only {n_words} words after preprocessing; the model was trained on "
                             "full articles, so this prediction is unreliable.")
    return result
