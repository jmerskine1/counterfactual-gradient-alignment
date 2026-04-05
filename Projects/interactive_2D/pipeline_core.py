"""
Interactive 2D Pipeline – core logic.

Uses counterfactual_alignment library for models, loss functions, and training.
Data is generated synthetically; no pickled datasets required.
"""
from __future__ import annotations

import os
import sys
import time
import threading
import numpy as np
import jax
import jax.numpy as jnp
import optax
from functools import partial
import torch.utils.data as torch_data
from sklearn import datasets as sk_datasets

# ── locate package root (two levels up from this file) ───────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from flax import linen as nn
from counterfactual_alignment.custom_models import SimpleClassifier
from counterfactual_alignment.loss_functions import loss_functions
from counterfactual_alignment.utilities import create_train_state, train_step


class SmallClassifier(nn.Module):
    """SimpleClassifier minus the 50% dropout — much better for tiny datasets."""
    num_hidden  : int
    num_outputs : int

    def setup(self):
        self.linear1 = nn.Dense(features=self.num_hidden)
        self.linear2 = nn.Dense(features=self.num_outputs)

    def __call__(self, x, train=False):
        x = self.linear1(x)
        x = nn.tanh(x)
        x = self.linear2(x)
        return x, None


# ─── Synthetic 2-D datasets ───────────────────────────────────────────────────

def _gaussian(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = np.vstack([
        rng.multivariate_normal([1.5,  1.5], np.eye(2), n // 2),
        rng.multivariate_normal([-1.5, -1.5], np.eye(2), n // 2),
    ]).astype(np.float32)
    Y = np.array([0] * (n // 2) + [1] * (n // 2), dtype=np.int32)
    return X, Y


def _xor(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    per = n // 4
    corners = [(0.0, 0.0, 0), (0.0, 1.0, 1), (1.0, 0.0, 1), (1.0, 1.0, 0)]
    X = np.vstack([
        np.column_stack([cx * np.ones(per), cy * np.ones(per)])
        + 0.18 * rng.standard_normal((per, 2))
        for cx, cy, _ in corners
    ]).astype(np.float32)
    Y = np.array([lab for _, _, lab in corners for _ in range(per)], dtype=np.int32)
    return X, Y


def _two_moons(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    X, Y = sk_datasets.make_moons(n, noise=0.12, random_state=seed)
    return X.astype(np.float32), Y.astype(np.int32)


def _circles(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    X, Y = sk_datasets.make_circles(n, noise=0.08, factor=0.45, random_state=seed)
    return X.astype(np.float32), Y.astype(np.int32)


DATASET_NAMES: list[str] = ["Two Moons", "Gaussian", "XOR", "Circles"]

_DATASET_FNS: dict = {
    "Gaussian":  _gaussian,
    "XOR":       _xor,
    "Two Moons": _two_moons,
    "Circles":   _circles,
}

# Loss functions that work with K['vector'] shape (B, 2) and return a scalar
LOSS_FN_NAMES: list[str] = [
    "cross_entropy",
    "combined_loss_softplus",
    "combined_loss_relu",
    "combined_loss_sign",
]


# ─── 2-D Dataset ──────────────────────────────────────────────────────────────

class Dataset2D(torch_data.Dataset):
    """
    Wraps a synthetic 2-D point cloud for interactive annotation.

    Each training point can hold **multiple** direction-vector annotations.
    During training, ``__getitem__`` randomly samples one vector per point
    (or returns a zero vector if unannotated), keeping batch size constant
    and avoiding cross-entropy imbalance.
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray) -> None:
        self.X = np.asarray(X, dtype=np.float32)
        self.Y = np.asarray(Y, dtype=np.int32)
        n = len(Y)
        # Per-point list of unit direction vectors
        self.annotations: list[list[np.ndarray]] = [[] for _ in range(n)]

    # ── Annotation API ──────────────────────────────────────────────────────

    def annotate(self, idx: int, raw_vec) -> bool:
        """Append a new direction annotation to point *idx*."""
        raw_vec = np.asarray(raw_vec, dtype=np.float32)
        mag = float(np.linalg.norm(raw_vec))
        if mag < 1e-6:
            return False
        unit = (raw_vec / mag).astype(np.float32)
        self.annotations[idx].append(unit)
        return True

    def clear(self, idx: int) -> None:
        """Remove all annotations from point *idx*."""
        self.annotations[idx].clear()

    def clear_all(self) -> None:
        for lst in self.annotations:
            lst.clear()

    def annotated_mask(self) -> np.ndarray:
        """Boolean mask – True where at least one annotation exists."""
        return np.array([len(a) > 0 for a in self.annotations], dtype=bool)

    def annotation_count(self) -> int:
        """Total number of annotations across all points."""
        return sum(len(a) for a in self.annotations)

    def nearest(self, x: float, y: float) -> int:
        """Index of the training point closest to (x, y)."""
        d2 = (self.X[:, 0] - x) ** 2 + (self.X[:, 1] - y) ** 2
        return int(np.argmin(d2))

    # ── PyTorch Dataset interface ────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.Y)

    def __getitem__(self, idx: int) -> dict:
        vecs = self.annotations[idx]
        if vecs:
            vec_arr = np.stack(vecs)          # (n_ann, 2)
        else:
            vec_arr = np.zeros((1, 2), dtype=np.float32)  # single zero vec
        return {
            "X": self.X[idx],
            "Y": self.Y[idx],
            "K": {"vector": vec_arr},         # (n_ann, 2) – variable per point
        }


def _collate(batch: list[dict]) -> dict:
    """Stack 2-D tabular samples.  K['vector'] → (B, max_n, 2), zero-padded."""
    vecs = [b["K"]["vector"] for b in batch]           # list of (n_i, 2)
    max_n = max(v.shape[0] for v in vecs)
    padded = []
    for v in vecs:
        if v.shape[0] < max_n:
            pad = np.zeros((max_n - v.shape[0], 2), dtype=np.float32)
            v = np.concatenate([v, pad], axis=0)
        padded.append(v)
    return {
        "X": jnp.stack([b["X"] for b in batch]),
        "Y": jnp.stack([b["Y"] for b in batch]).astype(jnp.int32),
        "K": {"vector": jnp.stack(padded)},             # (B, max_n, 2)
    }


# ─── Pipeline ─────────────────────────────────────────────────────────────────

class Pipeline2D:
    """
    Ensemble of SimpleClassifiers trained on a synthetic 2-D problem.

    A small training set (n_train points) is used for gradient updates and
    interactive annotation.  A large validation set (n_val points) is drawn
    from the same distribution and used only for accuracy reporting.
    """

    def __init__(
        self,
        *,
        dataset_name : str   = "Two Moons",
        n_models     : int   = 3,
        n_hidden     : int   = 16,
        lr           : float = 0.01,
        loss_fn_name : str   = "combined_loss_softplus",
        alpha        : float = 0.5,
        batch_size   : int   = 8,
        seed         : int   = 42,
        n_train      : int   = 10,
        n_val        : int   = 500,
    ) -> None:
        self.dataset_name = dataset_name
        self.n_models     = n_models
        self.n_hidden     = n_hidden
        self.lr           = lr
        self.loss_fn_name = loss_fn_name
        self.alpha        = alpha
        self.batch_size   = batch_size
        self.seed         = seed
        self.n_train      = n_train
        self.n_val        = n_val

        self.epoch   = 0
        self.history : dict[str, list] = {
            "train_loss":     [],
            "train_acc":      [],
            "val_acc":        [],
            "loss_fn":        [],
            "epoch_time_s":   [],   # wall-clock seconds per epoch
            "cumulative_s":   [],   # cumulative training time
            "grad_steps":     [],   # gradient updates per epoch
        }
        self._cumulative_time = 0.0
        self._stop = threading.Event()

        self._build()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _build(self) -> None:
        X_all, Y_all = _DATASET_FNS[self.dataset_name](
            n=self.n_train + self.n_val, seed=self.seed
        )
        # Shuffle so the train/val split has both classes
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(len(Y_all))
        X_all, Y_all = X_all[perm], Y_all[perm]

        self.dataset = Dataset2D(X_all[:self.n_train], Y_all[:self.n_train])
        self.val_X   = X_all[self.n_train:].astype(np.float32)
        self.val_Y   = Y_all[self.n_train:].astype(np.int32)
        self._make_loader()

        master = jax.random.PRNGKey(self.seed)
        keys   = jax.random.split(master, self.n_models + 1)
        self._rng = keys[0]

        opt = optax.adam(self.lr)
        self.models = [
            SmallClassifier(num_hidden=self.n_hidden, num_outputs=1)
            for _ in range(self.n_models)
        ]
        self.states = [
            create_train_state(m, opt, vector_length=2, key=k)
            for m, k in zip(self.models, keys[1:])
        ]

    @property
    def n_params(self) -> int:
        """Total trainable parameters per model."""
        if not self.states:
            return 0
        return sum(
            p.size for p in jax.tree_util.tree_leaves(self.states[0].params)
        )

    def _make_loader(self) -> None:
        self.loader = torch_data.DataLoader(
            self.dataset,
            batch_size = min(self.batch_size, len(self.dataset)),
            shuffle    = True,
            collate_fn = _collate,
            drop_last  = False,
        )

    def _loss_fn(self):
        fn = loss_functions[self.loss_fn_name]
        if self.loss_fn_name == "cross_entropy":
            return fn
        return partial(fn, alpha=self.alpha)

    # ── Training ──────────────────────────────────────────────────────────────

    def train_epochs(
        self,
        n_epochs : int,
        callback = None,   # callable(ep, total, train_loss, train_acc, val_acc)
    ) -> tuple[float, float, float]:
        """
        Train all ensemble members for *n_epochs*.
        Returns (final_train_loss, final_train_acc, final_val_acc).
        """
        self._stop.clear()
        loss_fn = self._loss_fn()
        last_loss = last_train_acc = last_val_acc = 0.0

        for ep in range(n_epochs):
            if self._stop.is_set():
                break

            t0 = time.perf_counter()
            losses: list[float] = []
            accs:   list[float] = []
            steps   = 0

            for m_idx, (model, state) in enumerate(zip(self.models, self.states)):
                self._rng, step_rng = jax.random.split(self._rng)

                for batch in self.loader:
                    state, metrics = train_step(state, model, batch, loss_fn, step_rng)
                    steps += 1

                losses.append(float(metrics["loss"]))
                accs.append(float(metrics["accuracy"]))
                self.states[m_idx] = state

            epoch_time = time.perf_counter() - t0
            self._cumulative_time += epoch_time

            self.epoch       += 1
            last_loss         = float(np.mean(losses))
            last_train_acc    = float(np.mean(accs))

            # Validation accuracy on the held-out set
            val_mean, _  = self.ensemble_probs(self.val_X)
            val_preds    = (val_mean > 0.5).astype(np.int32)
            last_val_acc = float(np.mean(val_preds == self.val_Y))

            self.history["train_loss"].append(last_loss)
            self.history["train_acc"].append(last_train_acc)
            self.history["val_acc"].append(last_val_acc)
            self.history["loss_fn"].append(self.loss_fn_name)
            self.history["epoch_time_s"].append(round(epoch_time, 4))
            self.history["cumulative_s"].append(round(self._cumulative_time, 4))
            self.history["grad_steps"].append(steps)

            if callback:
                callback(ep + 1, n_epochs, last_loss, last_train_acc, last_val_acc)

        return last_loss, last_train_acc, last_val_acc

    # ── Inference ─────────────────────────────────────────────────────────────

    def ensemble_probs(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Ensemble mean and std of P(class = 1) evaluated at *points* (N, 2).
        Returns (mean, std), each of shape (N,).
        """
        pts     = np.asarray(points, dtype=np.float32)
        inf_rng = jax.random.PRNGKey(0)
        all_p: list[np.ndarray] = []

        for model, state in zip(self.models, self.states):
            logits, _ = model.apply(
                {"params": state.params}, pts,
                train=False, rngs={"dropout": inf_rng},
            )
            all_p.append(np.array(jax.nn.sigmoid(logits).squeeze(-1)))

        stack = np.stack(all_p)       # (n_models, N)
        return stack.mean(0), stack.std(0)

    # ── Knowledge API ─────────────────────────────────────────────────────────

    def annotate(self, idx: int, raw_vec: np.ndarray) -> bool:
        """Annotate point *idx* with a direction vector.  Returns True on success."""
        return self.dataset.annotate(idx, raw_vec)

    def clear_annotation(self, idx: int) -> None:
        self.dataset.clear(idx)

    def clear_all_annotations(self) -> None:
        self.dataset.clear_all()

    def nearest_point(self, x: float, y: float) -> int:
        return self.dataset.nearest(x, y)

    def annotation_count(self) -> int:
        return self.dataset.annotation_count()

    def stop_training(self) -> None:
        """Signal the training loop to exit after the current epoch."""
        self._stop.set()

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Re-generate data and reinitialise all models (keeps current settings)."""
        self.epoch   = 0
        self.history = {
            "train_loss": [], "train_acc": [], "val_acc": [], "loss_fn": [],
            "epoch_time_s": [], "cumulative_s": [], "grad_steps": [],
        }
        self._cumulative_time = 0.0
        self._build()
