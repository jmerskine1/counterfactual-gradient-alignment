"""
Interactive Gradient Alignment Explorer – 2D
PyQt6 GUI for interactive annotation and real-time training.

Usage:
    cd Projects/interactive_2D
    python gui.py
"""
from __future__ import annotations

import os
import sys
import json
import datetime
import numpy as np

# ── force Agg backend before any pyplot import ───────────────────────────────
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget,
    QHBoxLayout, QVBoxLayout, QGroupBox,
    QLabel, QComboBox, QSpinBox, QDoubleSpinBox,
    QPushButton, QSlider, QSizePolicy, QStatusBar,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QSplitter,
)
from PyQt6.QtCore import Qt, QThread, QObject, pyqtSignal
from PyQt6.QtGui import QPalette, QColor, QFont

# ── local pipeline ────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline_core import Pipeline2D, DATASET_NAMES, LOSS_FN_NAMES


# ─── Palette ──────────────────────────────────────────────────────────────────

_C0      = "#E05252"   # class 0 – warm red
_C1      = "#4A90D9"   # class 1 – sky blue
_C_ARROW = "#2ECC71"   # annotation arrows – emerald
_C_SEL   = "#F39C12"   # selected point ring – amber
_C_DRAG  = "#F7DC6F"   # live drag arrow – yellow
_BG_DARK = "#1A1A2E"   # figure background
_BG_MID  = "#16213E"   # panel / axes background
_BG_CTRL = "#0F3460"   # group-box tint
_ACC     = "#E94560"   # accent (pressed buttons, contour)


# ─── Loss-function colour map ────────────────────────────────────────────────

_LOSS_FN_COLORS: dict[str, str] = {
    "cross_entropy":           "#FF6B6B",
    "combined_loss_softplus":  "#4ECDC4",
    "combined_loss_relu":      "#FFE66D",
    "combined_loss_sign":      "#C38FFF",
}
_LOSS_FN_SHORT: dict[str, str] = {
    "cross_entropy":           "CE",
    "combined_loss_softplus":  "Softplus",
    "combined_loss_relu":      "ReLU",
    "combined_loss_sign":      "Sign",
}


def _loss_fn_segments(loss_fns: list[str]) -> list[tuple[int, int, str]]:
    """Return (start, end_exclusive, fn_name) for each consecutive run."""
    if not loss_fns:
        return []
    segs: list[tuple[int, int, str]] = []
    start = 0
    for i in range(1, len(loss_fns)):
        if loss_fns[i] != loss_fns[i - 1]:
            segs.append((start, i, loss_fns[start]))
            start = i
    segs.append((start, len(loss_fns), loss_fns[start]))
    return segs


# ─── Run-history colour palette ──────────────────────────────────────────────

_RUN_PALETTE = [
    "#E05252", "#4A90D9", "#2ECC71", "#F39C12",
    "#9B59B6", "#1ABC9C", "#E67E22", "#3498DB",
    "#E74C3C", "#27AE60", "#F1C40F", "#8E44AD",
]


def _run_color(idx: int) -> str:
    return _RUN_PALETTE[idx % len(_RUN_PALETTE)]


# ─── Training thread ─────────────────────────────────────────────────────────

class _TrainWorker(QObject):
    """Runs Pipeline2D.train_epochs on a background QThread."""
    progress = pyqtSignal(int, int, float, float, float)  # ep, total, train_loss, train_acc, val_acc
    finished = pyqtSignal()

    def __init__(self, pipeline: Pipeline2D, n_epochs: int) -> None:
        super().__init__()
        self._pipeline = pipeline
        self._n_epochs = n_epochs

    def run(self) -> None:
        def _cb(ep, total, train_loss, train_acc, val_acc):
            self.progress.emit(ep, total, train_loss, train_acc, val_acc)
        self._pipeline.train_epochs(self._n_epochs, callback=_cb)
        self.finished.emit()


# ─── Decision-boundary canvas ────────────────────────────────────────────────

class BoundaryCanvas(FigureCanvas):
    """
    Embedded matplotlib canvas showing the ensemble decision boundary.

    Interactions
    ------------
    Left-click + drag near a training point → annotate direction.
    Right-click near a training point → clear annotation.
    """

    SNAP_FRAC = 0.10

    def __init__(self, pipeline: Pipeline2D) -> None:
        self._fig = Figure(figsize=(6, 5.5), facecolor=_BG_DARK)
        super().__init__(self._fig)
        self._ax = self._fig.add_subplot(111)

        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self._fig.tight_layout(pad=0.6)

        self._pipeline    = pipeline
        self._sel_idx     : int | None               = None
        self._drag_xy     : tuple[float, float] | None = None

        self.mpl_connect("button_press_event",   self._on_press)
        self.mpl_connect("motion_notify_event",  self._on_motion)
        self.mpl_connect("button_release_event", self._on_release)

        self.redraw()

    # ── Public API ──────────────────────────────────────────────────────────

    def set_pipeline(self, pipeline: Pipeline2D) -> None:
        self._pipeline = pipeline
        self._sel_idx  = None
        self._drag_xy  = None
        self.redraw()

    def redraw(self) -> None:
        """Full repaint of decision surface, points and annotations."""
        ax = self._ax
        p  = self._pipeline
        ax.clear()
        ax.set_facecolor(_BG_MID)

        X = p.dataset.X
        Y = p.dataset.Y

        # ── Meshgrid (bounds from val set – covers the full distribution) ──
        bX  = p.val_X
        pad = 0.3
        x0, x1 = bX[:, 0].min() - pad, bX[:, 0].max() + pad
        y0, y1 = bX[:, 1].min() - pad, bX[:, 1].max() + pad
        step   = max((x1 - x0), (y1 - y0)) / 140.0
        xx, yy = np.meshgrid(
            np.arange(x0, x1, step),
            np.arange(y0, y1, step),
        )
        grid = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)

        mean_p, std_p = p.ensemble_probs(grid)
        Z    = mean_p.reshape(xx.shape)
        Zstd = std_p.reshape(xx.shape)

        # ── Decision surface ───────────────────────────────────────────────
        ax.contourf(xx, yy, Z, levels=60, cmap="RdBu_r",
                    alpha=0.85, vmin=0.0, vmax=1.0)
        ax.contourf(xx, yy, Zstd, levels=8, cmap="Greys",
                    alpha=0.22, vmin=0.0)
        try:
            cs = ax.contour(xx, yy, Z, levels=[0.5],
                            colors=[_ACC], linewidths=1.8, alpha=0.95)
            plt.setp(cs.collections, path_effects=[
                pe.withStroke(linewidth=3.5, foreground=_BG_DARK)])
        except Exception:
            pass

        # ── Validation scatter (small, semi-transparent) ───────────────────
        vX = p.val_X
        vY = p.val_Y
        cls_cfg = [(0, _C0, "o"), (1, _C1, "s")]
        for cls, col, marker in cls_cfg:
            mask = vY == cls
            ax.scatter(
                vX[mask, 0], vX[mask, 1],
                c=col, edgecolors="none",
                s=10, marker=marker, zorder=3, alpha=0.30,
            )

        # ── Training scatter (larger, opaque, labelled) ────────────────────
        for cls, col, marker in cls_cfg:
            mask = Y == cls
            lbl  = f"Class {cls} (train)"
            ax.scatter(
                X[mask, 0], X[mask, 1],
                c=col, edgecolors="#ffffff", linewidths=0.5,
                s=60, marker=marker, zorder=4, label=lbl, alpha=0.95,
            )

        # ── Annotation arrows (multiple per point) ──────────────────────────
        scale = max(float(np.ptp(bX[:, 0])), float(np.ptp(bX[:, 1]))) * 0.08
        ax_arr, ay_arr, dx_arr, dy_arr = [], [], [], []
        for i, vecs in enumerate(p.dataset.annotations):
            for vec in vecs:
                ax_arr.append(X[i, 0])
                ay_arr.append(X[i, 1])
                dx_arr.append(vec[0] * scale)
                dy_arr.append(vec[1] * scale)
        if ax_arr:
            ax.quiver(
                ax_arr, ay_arr, dx_arr, dy_arr,
                color=_C_ARROW, angles="xy", scale_units="xy", scale=1.0,
                width=0.007, headwidth=5, headlength=6,
                zorder=5, alpha=0.95,
            )

        # ── Selected-point highlight ───────────────────────────────────────
        if self._sel_idx is not None:
            xi, yi = X[self._sel_idx]
            ax.scatter([xi], [yi], s=280,
                       facecolors="none", edgecolors=_C_SEL,
                       linewidths=2.5, zorder=6)

        # ── Live drag arrow ────────────────────────────────────────────────
        if self._sel_idx is not None and self._drag_xy is not None:
            xi, yi = X[self._sel_idx]
            cx, cy = self._drag_xy
            if abs(cx - xi) + abs(cy - yi) > 0.02:
                ax.annotate(
                    "",
                    xy=(cx, cy), xytext=(xi, yi),
                    arrowprops=dict(
                        arrowstyle="->", color=_C_DRAG, lw=2.2,
                        mutation_scale=14,
                    ),
                    zorder=7,
                )

        # ── Axes styling ───────────────────────────────────────────────────
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        for spine in ax.spines.values():
            spine.set_edgecolor("#3a3a5c")
        ax.tick_params(colors="#aaa", labelsize=8)
        ax.set_xlabel("x₁", color="#bbb", fontsize=9)
        ax.set_ylabel("x₂", color="#bbb", fontsize=9)
        leg = ax.legend(
            loc="upper right", fontsize=8,
            facecolor=_BG_MID, edgecolor="#3a3a5c",
            labelcolor="white", framealpha=0.9,
        )

        self._fig.tight_layout(pad=0.6)
        self.draw()

    # ── Mouse events ────────────────────────────────────────────────────────

    def _axes_xy(self, event) -> tuple[float, float] | None:
        if event.inaxes is not self._ax or event.xdata is None:
            return None
        return float(event.xdata), float(event.ydata)

    def _snap(self, x: float, y: float) -> int | None:
        idx = self._pipeline.nearest_point(x, y)
        xi, yi = self._pipeline.dataset.X[idx]
        rx = float(np.ptp(self._pipeline.dataset.X[:, 0]))
        ry = float(np.ptp(self._pipeline.dataset.X[:, 1]))
        radius = self.SNAP_FRAC * max(rx, ry, 0.1)
        dist   = ((xi - x) ** 2 + (yi - y) ** 2) ** 0.5
        return idx if dist < radius else None

    def _on_press(self, event) -> None:
        xy = self._axes_xy(event)
        if xy is None:
            return
        idx = self._snap(*xy)
        if event.button == 1:
            if idx is not None:
                self._sel_idx = idx
                self._drag_xy = xy
                self.redraw()
        elif event.button == 3:
            if idx is not None:
                self._pipeline.clear_annotation(idx)
                if self._sel_idx == idx:
                    self._sel_idx = None
                self.redraw()

    def _on_motion(self, event) -> None:
        if self._sel_idx is None:
            return
        xy = self._axes_xy(event)
        if xy is None:
            return
        self._drag_xy = xy
        self.redraw()

    def _on_release(self, event) -> None:
        if self._sel_idx is None or event.button != 1:
            return
        xy = self._axes_xy(event)
        if xy is not None:
            xi, yi = self._pipeline.dataset.X[self._sel_idx]
            raw = np.array([xy[0] - xi, xy[1] - yi], dtype=np.float32)
            self._pipeline.annotate(self._sel_idx, raw)
        self._sel_idx = None
        self._drag_xy = None
        self.redraw()


# ─── Small metrics canvas (Explorer sidebar) ─────────────────────────────────

class MetricsCanvas(FigureCanvas):
    """Compact training/val accuracy + loss display for the Explorer tab."""

    def __init__(self) -> None:
        self._fig = Figure(figsize=(4, 2.4), facecolor=_BG_DARK)
        super().__init__(self._fig)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self._ax_acc  = self._fig.add_subplot(111)
        self._ax_loss = self._ax_acc.twinx()
        self._style_axes()

    def _style_axes(self) -> None:
        for ax in (self._ax_acc, self._ax_loss):
            ax.set_facecolor(_BG_MID)
            ax.tick_params(colors="#aaa", labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor("#3a3a5c")
        self._ax_acc.set_xlabel("Epoch",    color="#bbb", fontsize=8)
        self._ax_acc.set_ylabel("Accuracy", color=_C1,    fontsize=8)
        self._ax_acc.tick_params(axis="y", labelcolor=_C1)
        self._ax_loss.set_ylabel("Loss", color="#F0B429", fontsize=8)
        self._ax_loss.yaxis.set_label_position("right")
        self._ax_loss.tick_params(axis="y", labelcolor="#F0B429")

    def update_history(
        self,
        history  : dict[str, list],
        past_runs: list[dict] | None = None,
    ) -> None:
        acc      = history["train_acc"]
        loss     = history["train_loss"]
        loss_fns = history.get("loss_fn", [])
        n        = len(acc)
        epochs   = list(range(1, n + 1))

        for ax in (self._ax_acc, self._ax_loss):
            ax.cla()
        self._style_axes()

        # ── Past runs as faint ghost lines ────────────────────────────────
        if past_runs:
            for run in past_runs[-5:]:
                r_ep = list(range(1, len(run["train_acc"]) + 1))
                if run["train_acc"]:
                    self._ax_acc.plot(
                        r_ep, run["train_acc"],
                        color="#aaaaaa", lw=0.9, alpha=0.25, zorder=1,
                    )
                if run.get("val_acc"):
                    self._ax_acc.plot(
                        r_ep, run["val_acc"],
                        color="#dddddd", lw=0.9, alpha=0.25, linestyle=":", zorder=1,
                    )
                if run["train_loss"]:
                    self._ax_loss.plot(
                        r_ep, run["train_loss"],
                        color="#aaaaaa", lw=0.9, alpha=0.25,
                        linestyle="--", zorder=1,
                    )

        if not acc:
            self._fig.tight_layout(pad=0.5)
            self.draw()
            return

        val_acc = history.get("val_acc", [])

        # ── Current run – segmented by loss function ──────────────────────
        if loss_fns:
            segs = _loss_fn_segments(loss_fns)
            seen: set[str] = set()
            for start, end, fn_name in segs:
                color = _LOSS_FN_COLORS.get(fn_name, "#aaaaaa")
                short = _LOSS_FN_SHORT.get(fn_name, fn_name)
                hi    = min(end + 1, n)
                s_ep   = epochs[start:hi]
                s_acc  = acc[start:hi]
                s_loss = loss[start:hi]
                lbl = short if fn_name not in seen else None
                seen.add(fn_name)
                self._ax_acc.plot( s_ep, s_acc,  color=color, lw=2.0, zorder=3, label=lbl)
                self._ax_loss.plot(s_ep, s_loss, color=color, lw=1.5, linestyle="--", zorder=2)
            self._ax_acc.fill_between(epochs, acc, alpha=0.10, color=_C1, zorder=0)
        else:
            self._ax_acc.plot( epochs, acc,  color=_C1,       lw=2.0, zorder=3, label="Train acc")
            self._ax_loss.plot(epochs, loss, color="#F0B429",  lw=1.5, linestyle="--", zorder=2)
            self._ax_acc.fill_between(epochs, acc, alpha=0.18, color=_C1)

        if val_acc:
            self._ax_acc.plot(
                epochs, val_acc,
                color="#2ECC71", lw=2.2, linestyle="-", zorder=5,
                label="Val acc", alpha=0.9,
            )

        self._ax_acc.set_ylim(0.0, 1.05)
        self._ax_acc.legend(
            loc="lower right", fontsize=7,
            facecolor=_BG_MID, edgecolor="#3a3a5c",
            labelcolor="white", framealpha=0.85,
        )
        self._fig.tight_layout(pad=0.5)
        self.draw()


# ─── Run History panel ────────────────────────────────────────────────────────

_HISTORY_COLS = ["", "#", "Dataset", "Loss fn", "Train", "Epochs", "Val acc", "Train acc"]


class HistoryCanvas(FigureCanvas):
    """Large comparison plot for the Run History tab."""

    def __init__(self) -> None:
        self._fig = Figure(figsize=(9, 4), facecolor=_BG_DARK)
        super().__init__(self._fig)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self._ax_acc  = self._fig.add_subplot(111)
        self._ax_loss = self._ax_acc.twinx()
        self._style_axes()

    def _style_axes(self) -> None:
        for ax in (self._ax_acc, self._ax_loss):
            ax.set_facecolor(_BG_MID)
            ax.tick_params(colors="#aaa", labelsize=9)
            for spine in ax.spines.values():
                spine.set_edgecolor("#3a3a5c")
        self._ax_acc.set_xlabel("Epoch", color="#bbb", fontsize=10)
        self._ax_acc.set_ylabel("Accuracy", color="#bbb", fontsize=10)
        self._ax_acc.tick_params(axis="y", labelcolor="#bbb")
        self._ax_loss.set_ylabel("Loss", color="#F0B429", fontsize=10)
        self._ax_loss.yaxis.set_label_position("right")
        self._ax_loss.tick_params(axis="y", labelcolor="#F0B429")

    def plot_runs(self, runs: list[dict], selected: set[int]) -> None:
        """Redraw with only the selected runs visible."""
        for ax in (self._ax_acc, self._ax_loss):
            ax.cla()
        self._style_axes()

        if not selected:
            self._fig.tight_layout(pad=0.8)
            self.draw()
            return

        for idx in sorted(selected):
            if idx >= len(runs):
                continue
            run   = runs[idx]
            color = _run_color(idx)
            ep    = list(range(1, len(run["train_acc"]) + 1))
            label = f"Run {idx + 1}"

            # Val accuracy — solid, prominent
            if run.get("val_acc"):
                self._ax_acc.plot(
                    ep, run["val_acc"],
                    color=color, lw=2.2, zorder=3,
                    label=f"{label} val",
                )
            # Train accuracy — dashed, thinner
            if run["train_acc"]:
                self._ax_acc.plot(
                    ep, run["train_acc"],
                    color=color, lw=1.4, linestyle="--", alpha=0.7, zorder=2,
                    label=f"{label} train",
                )
            # Loss — dotted, on right axis
            if run["train_loss"]:
                self._ax_loss.plot(
                    ep, run["train_loss"],
                    color=color, lw=1.2, linestyle=":", alpha=0.6, zorder=2,
                )

        self._ax_acc.set_ylim(0.0, 1.05)
        self._ax_acc.legend(
            loc="lower right", fontsize=8, ncol=2,
            facecolor=_BG_MID, edgecolor="#3a3a5c",
            labelcolor="white", framealpha=0.9,
        )
        self._fig.tight_layout(pad=0.8)
        self.draw()


class HistoryPanel(QWidget):
    """Tab containing a run table and a large comparison canvas."""

    def __init__(self, past_runs: list[dict]) -> None:
        super().__init__()
        self._runs = past_runs     # shared reference with MainWindow

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        splitter = QSplitter(Qt.Orientation.Vertical)

        # ── Run table ──────────────────────────────────────────────────────
        self._table = QTableWidget(0, len(_HISTORY_COLS))
        self._table.setHorizontalHeaderLabels(_HISTORY_COLS)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setStyleSheet(
            "QTableWidget {"
            "  background: #16213E; color: #ddd; gridline-color: #3a3a5c;"
            "  font-size: 9pt;"
            "}"
            "QHeaderView::section {"
            "  background: #1e1e35; color: #ccc; border: 1px solid #3a3a5c;"
            "  padding: 4px; font-weight: bold; font-size: 9pt;"
            "}"
        )
        # Resize columns to fit
        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)          # checkbox
        self._table.setColumnWidth(0, 30)
        for c in range(1, len(_HISTORY_COLS)):
            hdr.setSectionResizeMode(c, QHeaderView.ResizeMode.Stretch)

        self._table.itemChanged.connect(self._on_check_changed)

        splitter.addWidget(self._table)

        # ── Comparison canvas ──────────────────────────────────────────────
        self._canvas = HistoryCanvas()
        splitter.addWidget(self._canvas)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 5)

        # ── Select-all / clear buttons ─────────────────────────────────────
        btn_row = QHBoxLayout()
        self._btn_all   = QPushButton("Select All")
        self._btn_none  = QPushButton("Clear Selection")
        self._btn_clear = QPushButton("Delete All Runs")
        for b in (self._btn_all, self._btn_none, self._btn_clear):
            _style_btn(b)
        btn_row.addWidget(self._btn_all)
        btn_row.addWidget(self._btn_none)
        btn_row.addStretch()
        btn_row.addWidget(self._btn_clear)
        self._btn_all.clicked.connect(self._select_all)
        self._btn_none.clicked.connect(self._select_none)
        self._btn_clear.clicked.connect(self._delete_all)

        layout.addLayout(btn_row)
        layout.addWidget(splitter)

    # ── Refresh table from runs list ───────────────────────────────────────

    def refresh(self) -> None:
        """Rebuild the table rows from self._runs."""
        self._table.blockSignals(True)
        self._table.setRowCount(len(self._runs))

        for i, run in enumerate(self._runs):
            color = _run_color(i)

            # Checkbox column
            chk = QTableWidgetItem()
            chk.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            chk.setCheckState(Qt.CheckState.Checked)
            self._table.setItem(i, 0, chk)

            # Color swatch + run number
            num_item = QTableWidgetItem(f"  {i + 1}")
            num_item.setForeground(QColor(color))
            fnt = num_item.font()
            fnt.setBold(True)
            num_item.setFont(fnt)
            self._table.setItem(i, 1, num_item)

            # Metadata columns
            loss_fns_used = sorted(set(run.get("loss_fn", [])))
            short_fns = ", ".join(_LOSS_FN_SHORT.get(f, f) for f in loss_fns_used)

            vals = [
                run.get("dataset", "?"),
                short_fns or "?",
                str(run.get("n_train", "?")),
                str(run.get("epochs", len(run["train_acc"]))),
                f"{run['val_acc'][-1] * 100:.1f}%" if run.get("val_acc") else "—",
                f"{run['train_acc'][-1] * 100:.1f}%" if run["train_acc"] else "—",
            ]
            for c, v in enumerate(vals, start=2):
                self._table.setItem(i, c, QTableWidgetItem(v))

        self._table.blockSignals(False)
        self._redraw()

    def _checked_indices(self) -> set[int]:
        out: set[int] = set()
        for i in range(self._table.rowCount()):
            item = self._table.item(i, 0)
            if item and item.checkState() == Qt.CheckState.Checked:
                out.add(i)
        return out

    def _redraw(self) -> None:
        self._canvas.plot_runs(self._runs, self._checked_indices())

    def _on_check_changed(self, item: QTableWidgetItem) -> None:
        if item.column() == 0:
            self._redraw()

    def _select_all(self) -> None:
        self._table.blockSignals(True)
        for i in range(self._table.rowCount()):
            item = self._table.item(i, 0)
            if item:
                item.setCheckState(Qt.CheckState.Checked)
        self._table.blockSignals(False)
        self._redraw()

    def _select_none(self) -> None:
        self._table.blockSignals(True)
        for i in range(self._table.rowCount()):
            item = self._table.item(i, 0)
            if item:
                item.setCheckState(Qt.CheckState.Unchecked)
        self._table.blockSignals(False)
        self._redraw()

    def _delete_all(self) -> None:
        self._runs.clear()
        self.refresh()


# ─── Main window ──────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Interactive Gradient Alignment — 2D Explorer")
        self.setMinimumSize(1150, 720)

        self._pipeline      : Pipeline2D | None = None
        self._worker        : _TrainWorker | None = None
        self._thread        : QThread | None = None
        self._past_runs     : list[dict] = []
        self._pending_reset : bool = False
        self._run_counter   : int  = 0

        self._pipeline = self._make_pipeline_defaults()
        self._build_ui()
        self._update_status()

    # ── Pipeline factory ────────────────────────────────────────────────────

    @staticmethod
    def _make_pipeline_defaults() -> Pipeline2D:
        return Pipeline2D(
            dataset_name = "Two Moons",
            n_models     = 3,
            n_hidden     = 16,
            lr           = 0.01,
            loss_fn_name = "combined_loss_softplus",
            alpha        = 0.50,
            batch_size   = 8,
            seed         = 42,
            n_train      = 10,
            n_val        = 500,
        )

    def _make_pipeline_from_widgets(self) -> Pipeline2D:
        return Pipeline2D(
            dataset_name = self._w_dataset.currentText(),
            n_models     = self._w_models.value(),
            n_hidden     = self._w_hidden.value(),
            lr           = self._w_lr.value(),
            loss_fn_name = self._w_loss.currentText(),
            alpha        = self._w_alpha.value() / 100.0,
            batch_size   = self._w_batch.value(),
            seed         = self._w_seed.value(),
            n_train      = self._w_ntrain.value(),
            n_val        = self._w_nval.value(),
        )

    # ── UI construction ─────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self._tabs = QTabWidget()
        self._tabs.setStyleSheet(
            "QTabWidget::pane { border: none; }"
            "QTabBar::tab {"
            "  background: #1e1e35; color: #aaa; padding: 6px 18px;"
            "  border: 1px solid #3a3a5c; border-bottom: none;"
            "  border-top-left-radius: 6px; border-top-right-radius: 6px;"
            "  font-size: 10pt;"
            "}"
            "QTabBar::tab:selected {"
            "  background: #16213E; color: #fff; font-weight: bold;"
            "}"
        )
        self.setCentralWidget(self._tabs)

        # ── Tab 1: Explorer ────────────────────────────────────────────────
        explorer = QWidget()
        root = QHBoxLayout(explorer)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(10)

        self._boundary = BoundaryCanvas(self._pipeline)
        root.addWidget(self._boundary, stretch=7)

        right = QWidget()
        right.setFixedWidth(310)
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(8)
        rl.addWidget(self._build_config_group())
        rl.addWidget(self._build_train_group())
        rl.addWidget(self._build_help_label())

        self._metrics = MetricsCanvas()
        rl.addWidget(self._metrics, stretch=1)
        root.addWidget(right)

        self._tabs.addTab(explorer, "Explorer")

        # ── Tab 2: Run History ─────────────────────────────────────────────
        self._history_panel = HistoryPanel(self._past_runs)
        self._tabs.addTab(self._history_panel, "Run History")

        # ── Status bar ─────────────────────────────────────────────────────
        self._status_bar = QStatusBar()
        self._status_bar.setStyleSheet("color: #ccc; font-size: 8pt;")
        self.setStatusBar(self._status_bar)

    def _build_config_group(self) -> QGroupBox:
        gb = QGroupBox("Configuration")
        _style_group(gb)
        vl = QVBoxLayout(gb)
        vl.setSpacing(5)

        vl.addWidget(_lbl("Dataset"))
        self._w_dataset = QComboBox()
        self._w_dataset.addItems(DATASET_NAMES)
        self._w_dataset.setCurrentText("Two Moons")
        _style_combo(self._w_dataset)
        vl.addWidget(self._w_dataset)

        vl.addWidget(_lbl("Loss function"))
        self._w_loss = QComboBox()
        self._w_loss.addItems(LOSS_FN_NAMES)
        self._w_loss.setCurrentText("combined_loss_softplus")
        _style_combo(self._w_loss)
        vl.addWidget(self._w_loss)

        self._w_alpha_lbl = _lbl("α (direction mix): 0.50")
        vl.addWidget(self._w_alpha_lbl)
        self._w_alpha = QSlider(Qt.Orientation.Horizontal)
        self._w_alpha.setRange(0, 100)
        self._w_alpha.setValue(50)
        self._w_alpha.setTickInterval(25)
        self._w_alpha.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._w_alpha.setStyleSheet(
            "QSlider::groove:horizontal { height:4px; background:#3a3a5c; }"
            "QSlider::handle:horizontal { width:12px; height:12px;"
            "  background:#7C3AED; border-radius:6px; margin:-4px 0; }"
        )
        vl.addWidget(self._w_alpha)
        self._w_alpha.valueChanged.connect(self._on_alpha_changed)

        row = QHBoxLayout()
        row.setSpacing(6)
        row.addWidget(_lbl("Models"))
        self._w_models = QSpinBox()
        self._w_models.setRange(1, 10)
        self._w_models.setValue(3)
        _style_spin(self._w_models)
        row.addWidget(self._w_models)
        row.addWidget(_lbl("Hidden"))
        self._w_hidden = QSpinBox()
        self._w_hidden.setRange(4, 256)
        self._w_hidden.setValue(16)
        self._w_hidden.setSingleStep(4)
        _style_spin(self._w_hidden)
        row.addWidget(self._w_hidden)
        vl.addLayout(row)

        row2 = QHBoxLayout()
        row2.setSpacing(6)
        row2.addWidget(_lbl("LR"))
        self._w_lr = QDoubleSpinBox()
        self._w_lr.setRange(1e-5, 1.0)
        self._w_lr.setValue(0.01)
        self._w_lr.setSingleStep(0.001)
        self._w_lr.setDecimals(4)
        _style_spin(self._w_lr)
        row2.addWidget(self._w_lr)
        row2.addWidget(_lbl("Batch"))
        self._w_batch = QSpinBox()
        self._w_batch.setRange(4, 512)
        self._w_batch.setValue(32)
        _style_spin(self._w_batch)
        row2.addWidget(self._w_batch)
        vl.addLayout(row2)

        row3 = QHBoxLayout()
        row3.setSpacing(6)
        row3.addWidget(_lbl("Train pts"))
        self._w_ntrain = QSpinBox()
        self._w_ntrain.setRange(2, 10000)
        self._w_ntrain.setValue(10)
        _style_spin(self._w_ntrain)
        row3.addWidget(self._w_ntrain)
        row3.addWidget(_lbl("Val pts"))
        self._w_nval = QSpinBox()
        self._w_nval.setRange(50, 5000)
        self._w_nval.setValue(500)
        self._w_nval.setSingleStep(50)
        _style_spin(self._w_nval)
        row3.addWidget(self._w_nval)
        vl.addLayout(row3)

        row4 = QHBoxLayout()
        row4.setSpacing(6)
        row4.addWidget(_lbl("Seed"))
        self._w_seed = QSpinBox()
        self._w_seed.setRange(0, 9999)
        self._w_seed.setValue(42)
        _style_spin(self._w_seed)
        row4.addWidget(self._w_seed)
        row4.addStretch()
        vl.addLayout(row4)

        # Connect after all widgets initialised
        self._w_dataset.currentIndexChanged.connect(self._on_dataset_changed)
        self._w_loss.currentIndexChanged.connect(self._on_loss_fn_changed)

        return gb

    def _build_train_group(self) -> QGroupBox:
        gb = QGroupBox("Training")
        _style_group(gb)
        vl = QVBoxLayout(gb)
        vl.setSpacing(6)

        row = QHBoxLayout()
        row.addWidget(_lbl("Epochs per run"))
        self._w_epochs = QSpinBox()
        self._w_epochs.setRange(1, 500)
        self._w_epochs.setValue(10)
        _style_spin(self._w_epochs)
        row.addWidget(self._w_epochs)
        vl.addLayout(row)

        btn_row = QHBoxLayout()
        self._btn_train = QPushButton("▶  Train")
        self._btn_reset = QPushButton("↺  Reset")
        for b in (self._btn_train, self._btn_reset):
            _style_btn(b)
        btn_row.addWidget(self._btn_train)
        btn_row.addWidget(self._btn_reset)
        vl.addLayout(btn_row)

        self._btn_clear = QPushButton("✕  Clear All Annotations")
        _style_btn(self._btn_clear)
        vl.addWidget(self._btn_clear)

        self._btn_train.clicked.connect(self._on_train)
        self._btn_reset.clicked.connect(self._on_reset)
        self._btn_clear.clicked.connect(self._on_clear)

        return gb

    @staticmethod
    def _build_help_label() -> QLabel:
        lbl = QLabel(
            "<span style='color:#888; font-size:8pt;'>"
            "<b>🖱 Left-click + drag</b> near a point → annotate direction<br>"
            "<b>🖱 Right-click</b> near a point → remove annotation"
            "</span>"
        )
        lbl.setWordWrap(True)
        lbl.setContentsMargins(4, 2, 4, 4)
        return lbl

    # ── Button handlers ─────────────────────────────────────────────────────

    def _on_train(self) -> None:
        if self._thread and self._thread.isRunning():
            self._pipeline.stop_training()
            self._btn_train.setEnabled(False)
            self._btn_train.setText("Stopping…")
            return

        self._btn_train.setText("■  Stop")

        n_ep = self._w_epochs.value()
        self._worker = _TrainWorker(self._pipeline, n_ep)
        self._thread = QThread()
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_epoch_done)
        self._worker.finished.connect(self._on_train_finished)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._clear_thread)

        self._thread.start()

    def _clear_thread(self) -> None:
        self._thread = None

    def _on_epoch_done(self, ep: int, total: int, loss: float, train_acc: float, val_acc: float) -> None:
        self._status_bar.showMessage(
            f"Training… epoch {self._pipeline.epoch}  |  "
            f"loss: {loss:.4f}  |  "
            f"train acc: {train_acc * 100:.1f}%  |  "
            f"val acc: {val_acc * 100:.1f}%"
        )

    def _on_train_finished(self) -> None:
        self._btn_train.setEnabled(True)
        self._btn_train.setText("▶  Train")
        if self._pending_reset:
            self._pending_reset = False
            self._execute_reset()
            return
        self._boundary.redraw()
        self._metrics.update_history(self._pipeline.history, self._past_runs)
        self._update_status()

    def _save_run(self) -> None:
        """Snapshot the current run into past_runs and persist to disk."""
        p = self._pipeline
        if not p or not p.history["train_acc"]:
            return

        self._run_counter += 1
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # ── Build the run record ───────────────────────────────────────────
        record: dict = {
            "run_id":      self._run_counter,
            "timestamp":   ts,
            # ── Experiment metadata ────────────────────────────────────────
            "dataset":     p.dataset_name,
            "n_train":     p.n_train,
            "n_val":       p.n_val,
            "n_hidden":    p.n_hidden,
            "n_models":    p.n_models,
            "alpha":       p.alpha,
            "lr":          p.lr,
            "batch_size":  p.batch_size,
            "seed":        p.seed,
            "epochs":      p.epoch,
            # ── Training curves ────────────────────────────────────────────
            "train_acc":    list(p.history["train_acc"]),
            "train_loss":   list(p.history["train_loss"]),
            "val_acc":      list(p.history.get("val_acc", [])),
            "loss_fn":      list(p.history.get("loss_fn", [])),
            # ── Compute ───────────────────────────────────────────────────
            "epoch_time_s":    list(p.history.get("epoch_time_s", [])),
            "cumulative_s":    list(p.history.get("cumulative_s", [])),
            "grad_steps":      list(p.history.get("grad_steps", [])),
            "total_grad_steps": sum(p.history.get("grad_steps", [])),
            "total_time_s":    round(p._cumulative_time, 4),
            "n_params":        p.n_params,
            # ── Dataset (for reproducing plots) ────────────────────────────
            "train_X":     p.dataset.X.tolist(),
            "train_Y":     p.dataset.Y.tolist(),
            "val_X":       p.val_X.tolist(),
            "val_Y":       p.val_Y.tolist(),
            # ── Annotations (per-point list of unit direction vectors) ─────
            "annotations": [
                [v.tolist() for v in vecs]
                for vecs in p.dataset.annotations
            ],
        }

        self._past_runs.append(record)
        self._history_panel.refresh()

        # ── Write to disk ──────────────────────────────────────────────────
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
        os.makedirs(out_dir, exist_ok=True)
        fname = f"run_{self._run_counter:03d}_{ts}.json"
        fpath = os.path.join(out_dir, fname)
        try:
            with open(fpath, "w") as f:
                json.dump(record, f, indent=2)
            self._status_bar.showMessage(f"Run {self._run_counter} saved → {fname}", 4000)
        except Exception as e:
            self._status_bar.showMessage(f"Save failed: {e}", 4000)

    def _execute_reset(self) -> None:
        self._btn_reset.setEnabled(True)
        self._save_run()
        self._pipeline = self._make_pipeline_from_widgets()
        self._boundary.set_pipeline(self._pipeline)
        self._metrics.update_history(self._pipeline.history, self._past_runs)
        self._update_status()

    def _on_reset(self) -> None:
        if self._thread and self._thread.isRunning():
            self._pending_reset = True
            self._pipeline.stop_training()
            self._btn_reset.setEnabled(False)
            return
        self._execute_reset()

    def _on_dataset_changed(self) -> None:
        if self._thread and self._thread.isRunning():
            self._pending_reset = True
            self._pipeline.stop_training()
            return
        self._execute_reset()

    def _on_loss_fn_changed(self) -> None:
        if self._pipeline:
            self._pipeline.loss_fn_name = self._w_loss.currentText()
        self._update_status()

    def _on_clear(self) -> None:
        if self._pipeline:
            self._pipeline.clear_all_annotations()
        self._boundary.redraw()
        self._update_status()

    def _on_alpha_changed(self, val: int) -> None:
        a = val / 100.0
        self._w_alpha_lbl.setText(f"α (direction mix): {a:.2f}")
        if self._pipeline:
            self._pipeline.alpha = a

    # ── Status bar ──────────────────────────────────────────────────────────

    def _update_status(self) -> None:
        if self._pipeline is None:
            return
        h        = self._pipeline.history
        tr_acc   = f"{h['train_acc'][-1] * 100:.1f}%" if h["train_acc"]  else "—"
        val_acc  = f"{h['val_acc'][-1]   * 100:.1f}%" if h.get("val_acc") else "—"
        loss     = f"{h['train_loss'][-1]:.4f}"        if h["train_loss"] else "—"
        self._status_bar.showMessage(
            f"Epoch: {self._pipeline.epoch}  │  "
            f"Annotations: {self._pipeline.annotation_count()}  │  "
            f"Train acc: {tr_acc}  │  Val acc: {val_acc}  │  Loss: {loss}  │  "
            f"Dataset: {self._pipeline.dataset_name}  │  "
            f"Loss fn: {self._pipeline.loss_fn_name}"
        )


# ─── Qt style helpers ─────────────────────────────────────────────────────────

def _lbl(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet("color: #ccc; font-size: 9pt;")
    return lbl


def _style_group(gb: QGroupBox) -> None:
    gb.setStyleSheet(
        "QGroupBox {"
        "  color: #ddd; font-weight: bold; font-size: 9pt;"
        "  border: 1px solid #3a3a5c; border-radius: 7px;"
        "  margin-top: 10px; padding: 8px 6px 6px 6px;"
        "  background: #1e1e35;"
        "}"
        "QGroupBox::title {"
        "  subcontrol-origin: margin; padding: 0 6px;"
        "}"
    )


def _style_combo(w: QComboBox) -> None:
    w.setStyleSheet(
        "QComboBox {"
        "  background: #16213E; color: #eee; border: 1px solid #3a3a5c;"
        "  border-radius: 5px; padding: 3px 8px; font-size: 9pt;"
        "}"
        "QComboBox::drop-down { border: none; }"
        "QComboBox QAbstractItemView {"
        "  background: #16213E; color: #eee; selection-background-color: #7C3AED;"
        "}"
    )


def _style_spin(w) -> None:
    w.setStyleSheet(
        "QSpinBox, QDoubleSpinBox {"
        "  background: #16213E; color: #eee; border: 1px solid #3a3a5c;"
        "  border-radius: 5px; padding: 2px 4px; font-size: 9pt;"
        "}"
    )


def _style_btn(btn: QPushButton) -> None:
    btn.setStyleSheet(
        "QPushButton {"
        "  background: #1e1e45; color: #ddd; border: 1px solid #3a3a6c;"
        "  border-radius: 6px; padding: 5px 10px; font-size: 9pt;"
        "}"
        "QPushButton:hover   { background: #2e2e65; }"
        "QPushButton:pressed { background: #E94560; color: white; }"
        "QPushButton:disabled { background: #1a1a2e; color: #555; }"
    )


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    pal = QPalette()
    dark = QColor(26, 26, 46)
    mid  = QColor(22, 33, 62)
    text = QColor(220, 220, 220)
    pal.setColor(QPalette.ColorRole.Window,          dark)
    pal.setColor(QPalette.ColorRole.WindowText,      text)
    pal.setColor(QPalette.ColorRole.Base,            mid)
    pal.setColor(QPalette.ColorRole.AlternateBase,   dark)
    pal.setColor(QPalette.ColorRole.Text,            text)
    pal.setColor(QPalette.ColorRole.Button,          QColor(30, 30, 69))
    pal.setColor(QPalette.ColorRole.ButtonText,      text)
    pal.setColor(QPalette.ColorRole.Highlight,       QColor(124, 58, 237))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    app.setPalette(pal)

    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
