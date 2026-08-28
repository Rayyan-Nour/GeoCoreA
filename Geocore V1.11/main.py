"""
GeoCore Analytics Studio — desktop application.

The Studio shell (Project Explorer / Dataset Config / Command Log docks,
menu bar, toolbar) around the tested geocore engine, with the embedded
WebGL 3-D terrain map.

Dataset slots accept GeoTIFF (.tif), shapefile (.shp), and CSV point data.

Launch with run_geocore.bat (Windows) or:  python -m app.main
"""
from __future__ import annotations

import math
import sys
import traceback
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QUrl, QDir  # noqa: E402
from PyQt6.QtGui import (QPixmap, QPainter, QColor, QFont,            # noqa: E402
                         QLinearGradient, QAction, QPen, QFileSystemModel)
from PyQt6.QtWidgets import (                                         # noqa: E402
    QApplication, QMainWindow, QSplashScreen, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QPushButton, QLineEdit, QComboBox, QFileDialog,
    QProgressBar, QPlainTextEdit, QTabWidget, QTextBrowser, QTableWidget,
    QTableWidgetItem, QMessageBox, QFormLayout, QToolButton, QDockWidget,
    QTreeView, QScrollArea, QToolBar, QStyle,
)

try:
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    HAS_WEBENGINE = True
except ImportError:
    HAS_WEBENGINE = False

from geocore.config import PipelineConfig, COMMODITIES               # noqa: E402
from geocore.pipeline import run_pipeline                            # noqa: E402
from geocore.projectdb import ProjectDB                              # noqa: E402

APP_NAME = "GeoCore Analytics Studio"
APP_VERSION = "5.4.0"
COPPER = "#d98e4a"
COPPER_HI = "#f3b87c"
BASALT = "#16130f"
PANEL = "#1d1915"
LINE = "#3a3128"
BONE = "#e9e2d6"
DIM = "#9a8f7e"

STUDIO_QSS = f"""
QMainWindow, QWidget {{ background:{BASALT}; color:{BONE};
    font-family:'Space Grotesk','Segoe UI'; font-size:13px; }}
QMenuBar {{ background:#121009; border-bottom:1px solid {LINE}; }}
QMenuBar::item {{ padding:6px 12px; }}
QMenuBar::item:selected {{ background:{PANEL}; color:{COPPER_HI}; }}
QMenu {{ background:{PANEL}; border:1px solid {LINE}; }}
QMenu::item {{ padding:6px 22px; }}
QMenu::item:selected {{ background:#2a241d; color:{COPPER_HI}; }}
QToolBar {{ background:#121009; border-bottom:2px solid {COPPER};
    spacing:8px; padding:6px 8px; }}
QToolBar QToolButton {{ background:{PANEL}; border:1px solid {LINE};
    padding:7px 14px; color:{BONE}; letter-spacing:0.4px; }}
QToolBar QToolButton:hover {{ border-color:{COPPER}; color:{COPPER_HI}; }}
QDockWidget {{ font-family:'Space Grotesk'; }}
QDockWidget::title {{ background:#121009; padding:7px 10px;
    border-bottom:1px solid {LINE}; font-weight:600;
    letter-spacing:1.5px; text-transform:uppercase; }}
QLineEdit {{ background:#14110d; border:1px solid {LINE}; padding:6px;
    color:{BONE}; selection-background-color:{COPPER}; }}
QLineEdit:read-only {{ color:#c9c0b2; }}
QLineEdit:focus {{ border-color:{COPPER}; }}
QComboBox {{ background:#14110d; border:1px solid {LINE}; padding:6px; }}
QComboBox:hover {{ border-color:{COPPER}; }}
QComboBox QAbstractItemView {{ background:{PANEL};
    selection-background-color:#2a241d; selection-color:{COPPER_HI};
    border:1px solid {LINE}; }}
QToolButton#browse {{ background:{COPPER}; border:none; color:#1a120a;
    padding:6px; min-width:28px; font-weight:700; }}
QToolButton#browse:hover {{ background:{COPPER_HI}; }}
QToolButton#clear {{ background:transparent; border:1px solid {LINE};
    color:{DIM}; padding:6px; min-width:24px; }}
QToolButton#clear:hover {{ border-color:#a85a3a; color:#e0826a; }}
QPushButton {{ background:{PANEL}; border:1px solid {LINE};
    padding:7px 14px; }}
QPushButton:hover {{ border-color:{COPPER}; color:{COPPER_HI}; }}
QPushButton#run {{ background:{COPPER}; border:none; color:#1a120a;
    font-weight:700; font-size:14px; padding:12px;
    letter-spacing:1.2px; }}
QPushButton#run:hover {{ background:{COPPER_HI}; }}
QPushButton#run:disabled {{ background:#4a4036; color:#7a6f60; }}
QProgressBar {{ border:1px solid {LINE}; background:#14110d;
    text-align:center; height:16px; color:{BONE};
    font-family:'JetBrains Mono'; font-size:11px; }}
QProgressBar::chunk {{ background:{COPPER}; }}
QTabWidget::pane {{ border:1px solid {LINE}; }}
QTabBar::tab {{ background:#121009; padding:9px 22px;
    border:1px solid {LINE}; border-bottom:none;
    letter-spacing:0.8px; }}
QTabBar::tab:selected {{ background:{BASALT}; color:{COPPER_HI};
    border-top:2px solid {COPPER}; }}
QTreeView, QTableWidget, QTextBrowser {{ background:#14110d;
    border:1px solid {LINE}; alternate-background-color:#191510; }}
QTreeView::item:selected, QTableWidget::item:selected {{
    background:#2a241d; color:{COPPER_HI}; }}
QHeaderView::section {{ background:#121009; border:none;
    border-bottom:1px solid {LINE}; padding:7px;
    letter-spacing:1px; text-transform:uppercase; font-size:10px;
    color:{DIM}; }}
QPlainTextEdit#cmdlog {{ background:#0d0b08; color:{COPPER_HI};
    font-family:'JetBrains Mono',Consolas,monospace; font-size:12px;
    border:none; border-top:2px solid {COPPER}; padding:6px; }}
QScrollArea {{ border:none; }}
QScrollBar:vertical {{ background:{BASALT}; width:10px; }}
QScrollBar::handle:vertical {{ background:{LINE}; min-height:24px; }}
QScrollBar::handle:vertical:hover {{ background:{COPPER}; }}
QStatusBar {{ background:#121009; border-top:1px solid {LINE};
    color:{DIM}; font-family:'JetBrains Mono'; font-size:11px; }}
QLabel#commod {{ color:{COPPER_HI}; font-weight:600; font-size:11pt;
    letter-spacing:0.5px; }}
"""

# Dataset Config slots: (key, label, file filter, kind)
#   kind: 'dem' | 'deposits' | 'raster' (tif->feature_rasters) |
#         'any' (tif->rasters, shp/csv->vectors)
DATASET_SLOTS = [
    ("dem",        "DEM (required)",        "GeoTIFF (*.tif *.tiff)", "dem"),
    ("deposits",   "Deposits (required)",   "Deposits (*.csv *.shp)", "deposits"),
    ("magnetics",  "Magnetics",             "GeoTIFF (*.tif *.tiff)", "raster"),
    ("gravity",    "Gravity",               "GeoTIFF (*.tif *.tiff)", "raster"),
    ("geochem_cu", "Geochemistry Cu",       "Data (*.tif *.tiff *.shp *.csv)", "any"),
    ("geochem_au", "Geochemistry Au",       "Data (*.tif *.tiff *.shp *.csv)", "any"),
    ("radiom_k",   "Radiometric K",         "Data (*.tif *.tiff *.shp *.csv)", "any"),
    ("radiom_th",  "Radiometric Th",        "Data (*.tif *.tiff *.shp *.csv)", "any"),
    ("radiom_u",   "Radiometric U",         "Data (*.tif *.tiff *.shp *.csv)", "any"),
    ("faults",     "Faults / structure",    "Shapefile (*.shp)", "any"),
    ("alteration", "Alteration footprint",  "Shapefile (*.shp)", "any"),
    ("custom1",    "Custom layer 1",        "Data (*.tif *.tiff *.shp *.csv)", "any"),
    ("custom2",    "Custom layer 2",        "Data (*.tif *.tiff *.shp *.csv)", "any"),
]


def load_fonts():
    """Register bundled typefaces (Space Grotesk / JetBrains Mono)."""
    from PyQt6.QtGui import QFontDatabase
    fdir = Path(__file__).parent / "fonts"
    for f in fdir.glob("*.ttf"):
        QFontDatabase.addApplicationFont(str(f))


def make_splash_pixmap() -> QPixmap:
    pm = QPixmap(600, 360)
    p = QPainter(pm)
    g = QLinearGradient(0, 0, 600, 360)
    g.setColorAt(0.0, QColor("#1d1915"))
    g.setColorAt(1.0, QColor("#16130f"))
    p.fillRect(pm.rect(), g)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)

    # Strata: layered sediment lines with one copper seam
    for i in range(7):
        y = 210 + i * 18
        c = QColor("#d98e4a") if i == 2 else QColor(58, 49, 40)
        p.setPen(QPen(c, 2 if i == 2 else 1))
        last = None
        for x in range(0, 601, 10):
            yy = y - 10 * math.sin(x / 90 + i * 1.7) \
                   - 5 * math.sin(x / 31 + i)
            if last:
                p.drawLine(last[0], int(last[1]), x, int(yy))
            last = (x, yy)

    p.setPen(QColor("#f3b87c"))
    p.setFont(QFont("Space Grotesk", 30, QFont.Weight.Bold))
    p.drawText(44, 110, "GeoCore")
    p.setPen(QColor("#e9e2d6"))
    p.setFont(QFont("Space Grotesk", 30, QFont.Weight.Light))
    p.drawText(208, 110, "Analytics")
    p.setPen(QColor("#9a8f7e"))
    p.setFont(QFont("Space Grotesk", 11))
    p.drawText(46, 142, "S T U D I O")
    p.setFont(QFont("JetBrains Mono", 9))
    p.setPen(QColor("#6f6555"))
    p.drawText(46, 176, "prospectivity / depth-to-source / "
                        "spatial validation")
    p.drawText(46, 338, f"v{APP_VERSION}")
    p.end()
    return pm


class PipelineWorker(QThread):
    progress = pyqtSignal(float, str)
    log = pyqtSignal(str)
    finished_ok = pyqtSignal(object, str)
    failed = pyqtSignal(str)

    def __init__(self, cfg: PipelineConfig, db_path: str, project: str):
        super().__init__()
        self.cfg, self.db_path, self.project = cfg, db_path, project

    def run(self):
        try:
            self.log.emit(f"[CONFIG] Commodity: {self.cfg.commodity.upper()}")
            for n, p in self.cfg.feature_rasters.items():
                self.log.emit(f"[FEATURE] {n}: {p}")
            for n, p in self.cfg.feature_vectors.items():
                self.log.emit(f"[VECTOR] {n}: {p}")

            db = ProjectDB(self.db_path)
            result = run_pipeline(
                self.cfg, progress=lambda p, m: self.progress.emit(p, m),
                db=db, project_name=self.project)

            self.progress.emit(96, "Exporting 3D map")
            from viewer.export_3d import export_viewer_3d
            from viewer.export_web import export_viewer
            out = Path(self.cfg.results_dir)
            export_viewer_3d(out, out / "geocore_viewer_3d.html",
                             deposits_csv=self.cfg.deposit_csv,
                             commodity=self.cfg.commodity)
            export_viewer(out, out / "geocore_viewer_2d.html",
                          deposits_csv=self.cfg.deposit_csv)
            for w in result.warnings:
                self.log.emit(f"[WARN] {w}")
            self.progress.emit(100, "Complete")
            self.finished_ok.emit(result, str(out))
        except Exception:
            self.failed.emit(traceback.format_exc())


class DemoWorker(QThread):
    progress = pyqtSignal(float, str)
    log = pyqtSignal(str)
    finished_ok = pyqtSignal(object, str)
    failed = pyqtSignal(str)

    def run(self):
        try:
            self.progress.emit(2, "Building synthetic world")
            from demo.run_demo import build_world
            out_root = ROOT / "demo" / "out"
            out_root.mkdir(parents=True, exist_ok=True)
            cfg = build_world(out_root)
            w = PipelineWorker(cfg, str(out_root / "geocore_projects.db"),
                               "Synthetic demo")
            w.progress.connect(self.progress.emit)
            w.log.connect(self.log.emit)
            w.finished_ok.connect(self.finished_ok.emit)
            w.failed.connect(self.failed.emit)
            w.run()
        except Exception:
            self.failed.emit(traceback.format_exc())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION} "
                            f"\u2014 3D Terrain Visualization")
        self.resize(1500, 920)
        self.worker = None
        self.slot_edits: dict[str, QLineEdit] = {}
        self._last_results: Path | None = None
        self._build_central()
        self._build_docks()
        self._build_menus_toolbar()
        self.statusBar().showMessage("Ready")
        self.cmd("[3D] Viewer ready - run an analysis or the synthetic demo")

    # --- central tabs -----------------------------------------------------
    def _build_central(self):
        self.tabs = QTabWidget()
        if HAS_WEBENGINE:
            self.map3d = QWebEngineView()
            # Dark page background prevents the white flash while the
            # viewer HTML is loading (the source of the reported flicker)
            self.map3d.page().setBackgroundColor(QColor("#15100b"))
            self.map3d.setHtml(
                "<body style='background:#1b1b1b;color:#7da79c;"
                "font-family:sans-serif;display:grid;place-items:center;"
                "height:100vh'><div style='text-align:center'>"
                "<h2 style='color:#1de9b6'>GeoCore 3D Terrain</h2>"
                "<p>Run an analysis (or the synthetic demo from the "
                "Analysis menu) to load the map.</p></div>")
            self.tabs.addTab(self.map3d, "3D Map")
        else:
            holder = QWidget(); hl = QVBoxLayout(holder)
            lab = QLabel("PyQt6-WebEngine is not installed; the 3D map opens "
                         "in your browser instead.\n"
                         "To embed it here:  pip install PyQt6-WebEngine")
            lab.setAlignment(Qt.AlignmentFlag.AlignCenter)
            hl.addWidget(lab)
            self.btn_browser = QPushButton("Open 3D map in browser")
            self.btn_browser.setEnabled(False)
            self.btn_browser.clicked.connect(self._open_in_browser)
            hl.addWidget(self.btn_browser,
                         alignment=Qt.AlignmentFlag.AlignCenter)
            self.tabs.addTab(holder, "3D Map")
            self.map3d = None

        self.report = QTextBrowser()
        self.tabs.addTab(self.report, "Validation report")
        self.model_tab = QTableWidget()
        self.model_tab.setAlternatingRowColors(True)
        self.tabs.addTab(self.model_tab, "Model")

        self.targets_tab = QTableWidget()
        self.targets_tab.setAlternatingRowColors(True)
        self.tabs.addTab(self.targets_tab, "Targets")

        self.runs = QTableWidget()
        self.runs.setAlternatingRowColors(True)
        self.tabs.addTab(self.runs, "Run history")
        self.tabs.currentChanged.connect(
            lambda i: self._load_runs()
            if self.tabs.tabText(i) == "Run history" else None)
        self.setCentralWidget(self.tabs)

    # --- docks ------------------------------------------------------------
    def _build_docks(self):
        # Project Explorer (left)
        dock = QDockWidget("Project Explorer", self)
        self.fs_model = QFileSystemModel()
        self.fs_model.setRootPath(str(ROOT))
        tree = QTreeView()
        tree.setModel(self.fs_model)
        tree.setRootIndex(self.fs_model.index(str(ROOT)))
        for col in (1, 2, 3):
            tree.hideColumn(col)
        tree.doubleClicked.connect(self._tree_open)
        dock.setWidget(tree)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
        self.dock_explorer = dock

        # Dataset Config (right)
        dock = QDockWidget("Dataset Config", self)
        panel = QWidget(); form = QFormLayout(panel)
        form.setHorizontalSpacing(8); form.setVerticalSpacing(8)

        lab = QLabel("Commodity"); lab.setObjectName("commod")
        self.cmb_comm = QComboBox()
        self.cmb_comm.addItems(sorted(COMMODITIES))
        form.addRow(lab, self.cmb_comm)

        for key, label, filt, kind in DATASET_SLOTS:
            self.slot_edits[key] = self._slot_row(form, label, filt)

        self.cmb_res = QComboBox()
        self.cmb_res.addItems(["Fast (800 px)", "Balanced (1200 px)",
                               "Detailed (2000 px)", "Native (slow)"])
        self.cmb_res.setCurrentIndex(1)
        form.addRow("Analysis resolution", self.cmb_res)

        self.ed_out = QLineEdit(str(ROOT / "results" / "run1"))
        form.addRow("Results folder", self.ed_out)

        self.btn_run = QPushButton("Run Analysis")
        self.btn_run.setObjectName("run")
        self.btn_run.clicked.connect(self._run)
        form.addRow(self.btn_run)
        self.pb = QProgressBar()
        form.addRow(self.pb)
        self.eta = QLabel("")
        self.eta.setStyleSheet("color:#9a8f7e; font-family:'JetBrains Mono';"
                               "font-size:11px;")
        form.addRow(self.eta)
        hint = QLabel("Slots accept GeoTIFF, shapefile (.shp + sidecars), "
                      "or CSV points. Anything left empty is simply skipped.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#999; font-size:10px;")
        form.addRow(hint)

        scroll = QScrollArea(); scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        dock.setWidget(scroll)
        dock.setMinimumWidth(360)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self.dock_dataset = dock

        # Command Log (bottom)
        dock = QDockWidget("Command Log", self)
        self.cmdlog = QPlainTextEdit(); self.cmdlog.setReadOnly(True)
        self.cmdlog.setObjectName("cmdlog")
        self.cmdlog.setMaximumBlockCount(2000)
        dock.setWidget(self.cmdlog)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)
        self.dock_log = dock

    def _slot_row(self, form: QFormLayout, label: str, filt: str) -> QLineEdit:
        ed = QLineEdit(); ed.setReadOnly(True)
        b = QToolButton(); b.setText("\u2026"); b.setObjectName("browse")
        x = QToolButton(); x.setText("\u2715"); x.setObjectName("clear")

        def pick():
            f, _ = QFileDialog.getOpenFileName(self, label, "", filt)
            if f:
                ed.setText(f); ed.setToolTip(f)
        b.clicked.connect(pick)
        x.clicked.connect(lambda: (ed.clear(), ed.setToolTip("")))

        w = QWidget(); h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0); h.setSpacing(4)
        h.addWidget(ed); h.addWidget(b); h.addWidget(x)
        form.addRow(label, w)
        return ed

    # --- menus & toolbar ---------------------------------------------------
    def _build_menus_toolbar(self):
        mb = self.menuBar()
        pm = mb.addMenu("&Project")
        pm.addAction(QAction("Open results folder\u2026", self,
                             triggered=self._open_results))
        pm.addAction(QAction("Exit", self, triggered=self.close))

        am = mb.addMenu("&Analysis")
        am.addAction(QAction("Run analysis", self, triggered=self._run))
        am.addAction(QAction("Run synthetic demo (no data needed)", self,
                             triggered=self._run_demo))

        rm = mb.addMenu("&Reports")
        rm.addAction(QAction("View validation report", self,
                             triggered=lambda: self.tabs.setCurrentWidget(
                                 self.report)))
        rm.addAction(QAction("Open 3D map in browser", self,
                             triggered=self._open_in_browser))
        rm.addAction(QAction("View executive summary (plain language)",
                             self, triggered=self._show_exec_summary))

        vm = mb.addMenu("&View")
        for d, name in [(self.dock_explorer, "Project Explorer"),
                        (self.dock_dataset, "Dataset Config"),
                        (self.dock_log, "Command Log")]:
            vm.addAction(d.toggleViewAction())

        hm = mb.addMenu("&Help")
        hm.addAction(QAction("About", self, triggered=self._about))

        tb = QToolBar("Main"); tb.setMovable(False)
        self.addToolBar(tb)
        tb.addAction(QAction("Run Analysis", self, triggered=self._run))
        tb.addAction(QAction("Run Demo", self, triggered=self._run_demo))
        tb.addSeparator()
        tb.addAction(QAction("Open Results Folder", self,
                             triggered=self._open_results_in_os))
        tb.addAction(QAction("Reload Map", self, triggered=self._reload_map))
        tb.addAction(QAction("Reset Camera", self, triggered=self._reload_map))

    # --- command log --------------------------------------------------------
    def cmd(self, msg: str):
        self.cmdlog.appendPlainText(msg)

    # --- running -------------------------------------------------------------
    def _busy(self, busy: bool):
        self.btn_run.setEnabled(not busy)
        self.btn_run.setText("Running\u2026" if busy else "Run Analysis")

    def _collect_cfg(self) -> PipelineConfig:
        rasters, vectors = {}, {}
        for key, label, filt, kind in DATASET_SLOTS:
            path = self.slot_edits[key].text().strip()
            if not path or kind in ("dem", "deposits"):
                continue
            if path.lower().endswith((".tif", ".tiff")):
                rasters[key] = path
            else:
                vectors[key] = path
        return PipelineConfig(
            dem_path=self.slot_edits["dem"].text().strip(),
            deposit_csv=self.slot_edits["deposits"].text().strip(),
            feature_rasters=rasters,
            feature_vectors=vectors,
            commodity=self.cmb_comm.currentText(),
            results_dir=self.ed_out.text().strip()
                        or str(ROOT / "results" / "run1"),
            analysis_max_px={0: 800, 1: 1200, 2: 2000, 3: 0}[
                self.cmb_res.currentIndex()],
        )

    def _run(self):
        cfg = self._collect_cfg()
        problems = cfg.validate()
        if problems:
            QMessageBox.warning(self, "Cannot run",
                                "Fix these first:\n\n- "
                                + "\n- ".join(problems))
            return
        self._start(PipelineWorker(cfg, str(ROOT / "geocore_projects.db"),
                                   Path(cfg.results_dir).name))

    def _run_demo(self):
        self._start(DemoWorker())

    def _start(self, worker):
        self.worker = worker
        self._busy(True)
        self.pb.setValue(0)
        worker.progress.connect(self._on_progress)
        worker.log.connect(self.cmd)
        worker.finished_ok.connect(self._on_done)
        worker.failed.connect(self._on_fail)
        worker.start()

    def _on_progress(self, p: float, msg: str):
        self.pb.setValue(int(p))
        if not msg.endswith("%)"):          # don't spam chunk updates
            self.cmd(f"[{p:5.1f}%] {msg}")
        import time as _t
        if not hasattr(self, "_t0") or p <= 2:
            self._t0 = _t.time()
        elif p > 5:
            el = _t.time() - self._t0
            rem = el / max(p, 1) * (100 - p)
            self.eta.setText(f"elapsed {el:5.0f}s   /   est. remaining "
                             f"{rem:4.0f}s   /   {msg}")
        if p >= 100:
            self.eta.setText("")
        self.statusBar().showMessage(msg)

    def _on_fail(self, tb: str):
        self._busy(False)
        self.cmd("[ERROR] " + tb.strip().splitlines()[-1])
        self.cmd(tb)
        QMessageBox.critical(self, "Pipeline failed",
                             tb.strip().splitlines()[-1])

    def _on_done(self, result, results_dir: str):
        self._busy(False)
        self._last_results = Path(results_dir)
        cv = result.cv_report
        self.cmd(f"[RESULT] Spatial CV AUC {cv.auc_mean:.3f} "
                 f"\u00b1 {cv.auc_std:.3f}")
        ho = result.holdout_metrics.get("contrast_auc")
        if ho:
            self.cmd(f"[RESULT] Holdout contrast AUC {ho:.3f}")
        self.cmd(f"[RESULT] Artifacts: {results_dir}")
        self._fill_model_tab(result)
        self._load_results(self._last_results)

    def _fill_model_tab(self, result):
        imps = list(result.importances)
        rows = [("Spatial CV AUC",
                 f"{result.cv_report.auc_mean:.3f} "
                 f"\u00b1 {result.cv_report.auc_std:.3f}"),
                ("Training deposits", str(result.n_deposits_train)),
                ("Holdout deposits", str(result.n_deposits_holdout))]
        ho = result.holdout_metrics.get("contrast_auc")
        if ho:
            rows.append(("Holdout contrast AUC", f"{ho:.3f}"))
        rows.append(("", ""))
        rows += [(f"Feature: {n}", f"{v:.3f}") for n, v in imps]
        self.model_tab.setColumnCount(2)
        self.model_tab.setHorizontalHeaderLabels(["Metric / feature",
                                                  "Value / importance"])
        self.model_tab.setRowCount(len(rows))
        for r, (k, v) in enumerate(rows):
            self.model_tab.setItem(r, 0, QTableWidgetItem(k))
            self.model_tab.setItem(r, 1, QTableWidgetItem(v))
        self.model_tab.resizeColumnsToContents()

    # --- results -------------------------------------------------------------
    def _load_results(self, folder: Path, force: bool = False):
        html3d = folder / "geocore_viewer_3d.html"
        if html3d.exists():
            if self.map3d is not None:
                url = QUrl.fromLocalFile(str(html3d.resolve()))
                # Only (re)load when the target actually changed - repeated
                # loads of the same file cause a visible white flash.
                if force or url != getattr(self, "_loaded_url", None):
                    self.map3d.load(url)
                    self._loaded_url = url
                    self.cmd(f"[3D] Loaded terrain with overlays: "
                             f"{html3d.name}")
            else:
                self.btn_browser.setEnabled(True)
        rep = folder / "geocore_validation_report.md"
        if rep.exists():
            self.report.setMarkdown(rep.read_text(encoding="utf-8"))
        self._load_targets(folder)
        self._load_runs()
        self.tabs.setCurrentIndex(0)

    def _reload_map(self):
        if self._last_results:
            self._load_results(self._last_results, force=True)

    def _show_exec_summary(self):
        if not self._last_results:
            QMessageBox.information(self, "No results",
                                    "Run an analysis first.")
            return
        p = self._last_results / "geocore_executive_summary.md"
        if p.exists():
            self.report.setMarkdown(p.read_text(encoding="utf-8"))
            self.tabs.setCurrentWidget(self.report)
        else:
            QMessageBox.information(self, "Not found",
                                    "This results folder predates v5.4 - "
                                    "rerun to generate the summary.")

    def _open_in_browser(self):
        if self._last_results:
            f = self._last_results / "geocore_viewer_3d.html"
            if f.exists():
                webbrowser.open(f.resolve().as_uri())

    def _open_results(self):
        d = QFileDialog.getExistingDirectory(self, "Results folder")
        if d:
            self._last_results = Path(d)
            self._load_results(Path(d))

    def _open_results_in_os(self):
        target = self._last_results or ROOT
        import subprocess
        if sys.platform == "win32":
            subprocess.Popen(["explorer", str(target)])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(target)])
        else:
            subprocess.Popen(["xdg-open", str(target)])

    def _tree_open(self, index):
        path = Path(self.fs_model.filePath(index))
        if path.is_dir() and (path / "geocore_viewer_3d.html").exists():
            self._last_results = path
            self._load_results(path)
        elif path.suffix == ".html":
            webbrowser.open(path.resolve().as_uri())
        elif path.suffix == ".md":
            self.report.setMarkdown(path.read_text(encoding="utf-8"))
            self.tabs.setCurrentWidget(self.report)

    def _load_targets(self, folder: Path):
        """Fill the Targets tab from geocore_targets.csv."""
        import csv as _csv
        tcsv = folder / "geocore_targets.csv"
        self.targets_tab.setRowCount(0)
        if not tcsv.exists():
            return
        with open(tcsv, newline="", encoding="utf-8") as fh:
            rows = [r for r in _csv.reader(fh) if r and r[0] != ""]
        if len(rows) < 2 or rows[0][0] != "rank":
            return
        header, data = rows[0], [r for r in rows[1:]
                                 if r[0].strip().isdigit()]
        pretty = {"rank": "#", "latitude": "Latitude",
                  "longitude": "Longitude", "probability": "Probability",
                  "uncertainty": "Uncertainty",
                  "target_class": "Class",
                  "depth_spectral_m": "Depth spectral (m)",
                  "depth_euler_m": "Depth Euler (m)"}
        cols = [pretty.get(h, h.replace("_", " ")) for h in header]
        self.targets_tab.setColumnCount(len(cols))
        self.targets_tab.setHorizontalHeaderLabels(cols)
        self.targets_tab.setRowCount(len(data))
        for r, row in enumerate(data):
            for c, val in enumerate(row):
                self.targets_tab.setItem(r, c, QTableWidgetItem(val))
        self.targets_tab.resizeColumnsToContents()
        self.cmd(f"[TARGETS] {len(data)} ranked targets loaded "
                 f"(full file: {tcsv.name})")

    def _load_runs(self):
        rows = []
        for dbp in [ROOT / "geocore_projects.db",
                    ROOT / "demo" / "out" / "geocore_projects.db"]:
            if dbp.exists():
                try:
                    rows += ProjectDB(str(dbp)).list_runs()
                except Exception:
                    pass
        cols = ["run", "project", "started", "commodity", "cv_auc", "status"]
        self.runs.setColumnCount(len(cols))
        self.runs.setHorizontalHeaderLabels(cols)
        self.runs.setRowCount(len(rows))
        for r, run in enumerate(rows):
            for c, key in enumerate(cols):
                self.runs.setItem(r, c,
                                  QTableWidgetItem(str(run.get(key, ""))))
        self.runs.resizeColumnsToContents()

    def _about(self):
        QMessageBox.about(
            self, "About",
            f"<b>{APP_NAME} v{APP_VERSION}</b><br>"
            "Prospectivity ranking with spatial cross-validation and "
            "physics-based depth-to-source estimation.<br><br>"
            "Methods: Carranza &amp; Laborte 2015; Valavi et al. 2019; "
            "Spector &amp; Grant 1970; Reid et al. 1990.<br>"
            "See docs/HONEST_CLAIMS.md before quoting numbers.")


def main() -> int:
    import os
    # Windows QtWebEngine flicker fix: Chromium's native window occlusion
    # tracking blanks/flickers embedded views; GL context sharing is
    # required for stable WebEngine compositing.
    flags = os.environ.get("QTWEBENGINE_CHROMIUM_FLAGS", "")
    if "CalculateNativeWinOcclusion" not in flags:
        os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = (
            flags + " --disable-features=CalculateNativeWinOcclusion"
                    " --ignore-gpu-blocklist").strip()
    QApplication.setAttribute(
        Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
    app = QApplication(sys.argv)
    load_fonts()
    app.setFont(QFont("Space Grotesk", 10))
    app.setStyleSheet(STUDIO_QSS)
    app.setApplicationName(APP_NAME)

    splash = QSplashScreen(make_splash_pixmap())
    splash.show()
    for i, m in enumerate(["Loading geocore engine\u2026",
                           "Preparing validation suite\u2026",
                           "Starting Studio workspace\u2026"]):
        QTimer.singleShot(350 * i, lambda m=m: splash.showMessage(
            "   " + m, Qt.AlignmentFlag.AlignBottom, QColor("#9a8f7e")))
    app.processEvents()

    win = MainWindow()
    QTimer.singleShot(1200, lambda: (win.showMaximized(), splash.finish(win)))
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
