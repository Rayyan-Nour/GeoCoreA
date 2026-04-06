"""
GeoCore Analytics Studio - Professional Mineral Exploration Platform
Multi-project SaaS for geological analysis and 3D terrain visualization.
"""

import sys
import os
import json
import subprocess
from pathlib import Path

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QTimer, pyqtSignal as Signal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QProgressBar, QMessageBox

# Import project management
from project_manager import ProjectManagerDialog, ProjectDatabase, Project
from data_import_hub import DataImportHub

# Import SaaS features
from user_auth import UserManager, User
from cloud_storage import CloudStorageManager
from report_generator import ReportGenerator, ReportDialog
from collaboration import CollaborationPanel
from subscription_manager import SubscriptionManager, SubscriptionDialog

# Import enterprise features
from ai_model_trainer import ModelTrainingDialog, ModelTrainer
from mobile_integration import FieldDataWidget, MobileAPIServer
from web_dashboard import WebDashboardWidget, APITestWidget
from enterprise_security import SecurityManager, SecuritySettingsDialog, ComplianceWidget

# =====================================================
# PATHS / CONSTANTS
# =====================================================

APP_DIR = Path(__file__).resolve().parent

DEFAULT_PYTHON = r"C:\Program Files\QGIS 3.40.12\apps\Python312\python.exe"
DEFAULT_BASE = r"c:\Users\rayya\OneDrive\Desktop\projects\first"
DEFAULT_CORE = str(Path(__file__).resolve().parent / "oresinsight_v4_ml.py")  # v4 ML with auto-training

DEFAULT_STACK = str(Path(DEFAULT_BASE) / "AZ_multi_layer_maggrav.npy")
DEFAULT_REF = str(Path(DEFAULT_BASE).parent / "AZ_DEM_10m" / "USGS_13_n34w111_20240402.tif")  # 10m resolution with REAL deposits!
DEFAULT_MRDS = str(Path(DEFAULT_BASE) / "mrds-csv" / "mrds.csv")
DEFAULT_GEO = str(Path(DEFAULT_BASE) / "MAPS" / "500geo_utm" / "utm500geo.shp")
DEFAULT_RESULTS = str(Path(DEFAULT_BASE) / "results_v3")  # v3 results
DEFAULT_OVERLAY = "oreinsight_v3_probability.tif"  # v3 output

QGIS_EXE = r"C:\Program Files\QGIS 3.40.12\bin\qgis-ltr-bin.exe"

COPPER_ACCENT = "#B87333"
REE_ACCENT = "#00C9A7"


# =====================================================================
# Startup splash
# =====================================================================

class StartupSplash(QWidget):
    finished = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowFlag(Qt.FramelessWindowHint)
        self.setWindowFlag(Qt.WindowStaysOnTopHint)
        self.setFixedSize(900, 540)

        root = QVBoxLayout(self)
        root.setContentsMargins(60, 40, 60, 40)
        root.setSpacing(20)

        self.setStyleSheet("""
        QWidget {
            background: qlineargradient(
                x1:0, y1:0, x2:1, y2:1,
                stop:0   #1e3c72,
                stop:0.5 #2a5298,
                stop:1   #667eea
            );
            color: white;
        }
        """)

        title_block = QVBoxLayout()
        title_block.setSpacing(4)
        title_block.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        title = QLabel("GeoCore Analytics")
        title.setFont(QFont("Segoe UI", 42, QFont.Bold))
        title.setAlignment(Qt.AlignLeft)
        title.setStyleSheet("background: transparent; color: white;")

        product = QLabel("Mineral Explorer™")
        product.setFont(QFont("Segoe UI", 24, QFont.Bold))
        product.setAlignment(Qt.AlignLeft)
        product.setStyleSheet("background: transparent; color: white;")

        tagline = QLabel("Version 2.0 – 3D Terrain Visualization")
        tagline.setFont(QFont("Segoe UI", 14))
        tagline.setAlignment(Qt.AlignLeft)
        tagline.setStyleSheet("background: transparent; color: rgba(255,255,255,220);")

        title_block.addWidget(title)
        title_block.addWidget(product)
        title_block.addWidget(tagline)

        root.addStretch(3)
        root.addLayout(title_block)
        root.addStretch(4)

        bottom = QVBoxLayout()
        bottom.setSpacing(6)

        self.status_label = QLabel("Initializing QGIS…")
        self.status_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.status_label.setFont(QFont("Segoe UI", 10))
        self.status_label.setStyleSheet("background: transparent; color: rgba(255,255,255,210);")

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(12)
        self.progress.setStyleSheet("""
            QProgressBar {
                background-color: rgba(0, 0, 0, 70);
                border-radius: 6px;
            }
            QProgressBar::chunk {
                border-radius: 6px;
                background: rgba(255, 255, 255, 230);
            }
        """)

        footer_row = QVBoxLayout()
        copyright_label = QLabel("Copyright © 2025 Rayyan Nour")
        copyright_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        copyright_label.setFont(QFont("Segoe UI", 9))
        copyright_label.setStyleSheet("background: transparent; color: rgba(255,255,255,190);")

        footer_row.addWidget(copyright_label)
        footer_row.addStretch(1)

        bottom.addWidget(self.status_label)
        bottom.addWidget(self.progress)
        bottom.addLayout(footer_row)

        root.addLayout(bottom)

        self._value = 0
        self._current_status = ""
        # Don't auto-increment - we'll manually control progress
        # self._timer = QTimer(self)
        # self._timer.timeout.connect(self._tick)
        # self._timer.start(250)
    
    def update_progress(self, value, message=""):
        """Update progress bar and status message."""
        self._value = value
        self.progress.setValue(value)
        if message:
            self.status_label.setText(message)
            self._current_status = message
        QApplication.processEvents()  # Force UI update

    def _tick(self):
        if self._value < 100:
            self._value += 2
            self.progress.setValue(self._value)

            v = self._value
            if v < 20:
                msg = "Initializing 3D engine…"
            elif v < 45:
                msg = "Loading visualization modules…"
            elif v < 70:
                msg = "Starting Mineral Explorer engine…"
            elif v < 90:
                msg = "Preparing 3D terrain viewer…"
            else:
                msg = "Finalizing startup…"

            if msg != self._current_status:
                self._current_status = msg
                self.status_label.setText(msg)
        else:
            # Stop timer and emit finished ONLY ONCE
            if self._timer.isActive():
                self._timer.stop()
                self.finished.emit()



# =====================================================================
# Worker thread
# =====================================================================

class PipelineWorker(QtCore.QThread):
    log_line = Signal(str)
    finished = Signal(int)
    progress = Signal(int, str)

    def __init__(self, python_exe: str, core_script: str, env: dict, parent=None):
        super().__init__(parent)
        self.python_exe = python_exe
        self.core_script = core_script
        self.env = env
        self.current_progress = 0

    def _stage_progress(self, line_lower: str):
        if "loading 10m" in line_lower or "loading dem" in line_lower:
            return 10, "Loading 10m DEM…"
        if "loading deposits" in line_lower or "loading copper" in line_lower:
            return 25, "Loading copper deposits…"
        if "loading geochem" in line_lower:
            return 35, "Loading geochemistry data…"
        if "creating probability" in line_lower or "computing distance" in line_lower:
            return 50, "Creating probability map…"
        if "creating grade" in line_lower:
            return 70, "Creating grade map…"
        if "smoothing" in line_lower:
            return 85, "Smoothing maps…"
        if "saving" in line_lower or "save" in line_lower:
            return 95, "Saving outputs…"
        if "complete" in line_lower or "success" in line_lower:
            return 100, "Analysis complete!"
        return None

    def _bump_progress(self, target: int, label: str):
        if target <= self.current_progress:
            return
        self.current_progress = min(target, 95)
        self.progress.emit(self.current_progress, label)

    def run(self):
        cmd = [self.python_exe, self.core_script]
        try:
            self.current_progress = 3
            self.progress.emit(self.current_progress, "Starting ML analysis engine…")

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=self.env,
            )

            for raw_line in proc.stdout:
                line = raw_line.rstrip("\n")
                self.log_line.emit(line)

                # Check for [PROGRESS:X:Message] markers
                if "[PROGRESS:" in line:
                    try:
                        # Extract progress value and message
                        # Format: [PROGRESS:50:Training model...]
                        parts = line.split("[PROGRESS:")[1].split("]")[0]
                        value_str, message = parts.split(":", 1)
                        value = int(value_str)
                        self._bump_progress(value, message)
                        continue
                    except:
                        pass  # Fall through to old parsing

                # Fallback to old stage-based parsing
                l = line.lower()
                mapped = self._stage_progress(l)
                if mapped is not None:
                    pct, label = mapped
                    self._bump_progress(pct, label)
                else:
                    if self.current_progress < 90:
                        self.current_progress += 1
                        self.progress.emit(self.current_progress, "Running analysis…")

            proc.wait()
            self.finished.emit(proc.returncode)
        except Exception as e:
            self.log_line.emit(f"[ERROR] Failed to start pipeline: {e}")
            self.finished.emit(-1)


# =====================================================================
# Pipeline progress dialog
# =====================================================================

class PipelineProgressDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Running OreInsight v4 ML Analysis…")
        self.setModal(True)
        self.setWindowFlags(Qt.Dialog | Qt.WindowTitleHint | Qt.CustomizeWindowHint)

        self.label = QtWidgets.QLabel("Starting ML analysis engine…")
        self.label.setStyleSheet("color:#dddddd; font-size: 11pt;")
        self.bar = QtWidgets.QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #555;
                border-radius: 5px;
                text-align: center;
                background-color: #2a2a2a;
                height: 20px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #0d7377, stop:1 #14a085);
                border-radius: 4px;
            }
        """)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(20, 15, 20, 15)
        layout.addWidget(self.label)
        layout.addWidget(self.bar)

        self.resize(480, 120)
        
        # Smooth animation timer — fills gaps between progress updates
        self._target = 0
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(100)  # 100ms tick
        self._timer.timeout.connect(self._animate)
        self._timer.start()

    def _animate(self):
        """Smoothly animate progress bar toward target value."""
        current = self.bar.value()
        if current < self._target:
            # Move 1-2% per tick toward target
            step = max(1, (self._target - current) // 5)
            self.bar.setValue(min(current + step, self._target))

    def update_progress(self, value: int, stage: str):
        if value < self._target and value != 100:
            return
        self._target = value
        if value == 100:
            self.bar.setValue(100)
        self.label.setText(stage)


# =====================================================================
# Dataset config dock widget
# =====================================================================

class DatasetConfigWidget(QtWidgets.QWidget):
    config_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        # FORCE these values - don't load from settings
        self.stack_path = DEFAULT_STACK
        self.ref_raster = DEFAULT_REF  # This will now use the NEW DEM file
        self.mrds_csv = DEFAULT_MRDS
        self.geology_shp = DEFAULT_GEO
        self.results_dir = DEFAULT_RESULTS
        self.overlay_name = DEFAULT_OVERLAY

        # Apply dark theme styling to ensure visibility
        self.setStyleSheet("""
            DatasetConfigWidget {
                background-color: #2b2b2b;
            }
            DatasetConfigWidget QLineEdit {
                background-color: #3c3c3c;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 6px;
                color: #ffffff;
                min-height: 20px;
            }
            DatasetConfigWidget QLineEdit:read-only {
                background-color: #333333;
                color: #cccccc;
            }
            DatasetConfigWidget QToolButton {
                background-color: #0078d4;
                border: 1px solid #0078d4;
                border-radius: 3px;
                color: #ffffff;
                padding: 6px;
                min-width: 30px;
                min-height: 20px;
                font-weight: bold;
            }
            DatasetConfigWidget QToolButton:hover {
                background-color: #106ebe;
            }
            DatasetConfigWidget QToolButton:pressed {
                background-color: #005a9e;
            }
            DatasetConfigWidget QLabel {
                color: #ffffff;
                background-color: transparent;
                padding: 2px;
            }
        """)

        form = QtWidgets.QFormLayout(self)
        form.setLabelAlignment(Qt.AlignLeft)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(8)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.ExpandingFieldsGrow)

        def add_path_row(label, initial, attr_name, file_mode=True):
            line = QtWidgets.QLineEdit(initial)
            line.setReadOnly(True)  # Make read-only to prevent editing
            line.setToolTip(initial)  # Show full path on hover
            
            btn = QtWidgets.QToolButton()
            btn.setText("…")
            btn.setFixedWidth(30)
            btn.setToolTip("Browse for file")

            def browse():
                if file_mode:
                    p, _ = QtWidgets.QFileDialog.getOpenFileName(self, f"Select {label}")
                else:
                    p = QtWidgets.QFileDialog.getExistingDirectory(self, f"Select {label}")
                if p:
                    line.setText(p)
                    line.setToolTip(p)  # Update tooltip with new path
                    setattr(self, attr_name, p)
                    self.config_changed.emit()

            btn.clicked.connect(browse)

            h = QtWidgets.QHBoxLayout()
            h.setSpacing(4)
            h.addWidget(line, 1)  # Give line edit stretch factor
            h.addWidget(btn, 0)   # Button doesn't stretch

            container = QtWidgets.QWidget()
            container.setLayout(h)
            form.addRow(label, container)
            return line

        self.stack_edit = add_path_row("Stack .npy:", self.stack_path, "stack_path", True)
        self.ref_edit = add_path_row("Reference raster:", self.ref_raster, "ref_raster", True)
        self.mrds_edit = add_path_row("MRDS CSV:", self.mrds_csv, "mrds_csv", True)
        self.geo_edit = add_path_row("Geology SHP:", self.geology_shp, "geology_shp", True)
        self.results_edit = add_path_row("Results folder:", self.results_dir, "results_dir", False)

        self.overlay_edit = QtWidgets.QLineEdit(self.overlay_name)
        self.overlay_edit.textChanged.connect(self._overlay_changed)
        form.addRow("Overlay file name:", self.overlay_edit)

        # Feature Layers Section
        separator = QtWidgets.QLabel("─" * 40)
        separator.setStyleSheet("color: #555; background-color: transparent;")
        form.addRow(separator)
        
        # Commodity selector
        commodity_label = QtWidgets.QLabel("Target Commodity:")
        commodity_label.setStyleSheet("color: #e6a817; font-weight: bold; font-size: 11pt; background-color: transparent;")
        form.addRow(commodity_label)
        
        self.commodity_combo = QtWidgets.QComboBox()
        self.commodity_combo.addItems(["Copper (Porphyry Cu)", "Rare Earth Elements (REE)"])
        self.commodity_combo.setStyleSheet("""
            QComboBox {
                background-color: #3c3c3c;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 6px;
                color: #ffffff;
                min-height: 20px;
                font-size: 11pt;
            }
            QComboBox:hover { border: 1px solid #14a085; }
            QComboBox QAbstractItemView {
                background-color: #2a2a2a;
                color: #ffffff;
                selection-background-color: #14a085;
            }
        """)
        self.commodity_combo.currentIndexChanged.connect(self.config_changed.emit)
        form.addRow(self.commodity_combo)
        
        feature_label = QtWidgets.QLabel("Optional Feature Layers:")
        feature_label.setStyleSheet("color: #14a085; font-weight: bold; font-size: 11pt; background-color: transparent;")
        form.addRow(feature_label)
        
        # Initialize feature layers dictionary
        self.feature_layers = {}
        self.feature_edits = {}
        
        # Add feature layer rows — includes both Cu and REE features
        feature_types = [
            # Shared features
            ("Faults:", "faults"),
            ("Geology:", "geology"),
            ("Rivers:", "rivers"),
            ("Streams:", "streams"),
            ("Magnetics:", "magnetics"),
            ("Gravity:", "gravity"),
            ("Landsat/Remote Sensing:", "landsat"),
            # Copper-specific
            ("Geochemistry Cu:", "geochem_cu"),
            ("Geochemistry Au:", "geochem_au"),
            ("Geochemistry Ag:", "geochem_ag"),
            ("ASTER Argillic:", "alteration_argillic"),
            ("ASTER Phyllic:", "alteration_phyllic"),
            ("ASTER Propylitic:", "alteration_propylitic"),
            ("ASTER Silica:", "alteration_silica"),
            ("NURE Cu Sediment:", "nure_cu"),
            # REE-specific (Lawley et al., 2024)
            ("Radiometric Th:", "radiometric_th"),
            ("Radiometric K:", "radiometric_k"),
            ("Radiometric U:", "radiometric_u"),
            ("NURE Phosphorus:", "nure_p"),
            ("NURE Niobium:", "nure_nb"),
            ("NURE Thorium:", "nure_th"),
            ("Dist. to Alkaline:", "dist_alkaline"),
        ]
        
        for label, key in feature_types:
            edit = self._add_feature_row(label, key, form)
            self.feature_edits[key] = edit

        hint = QtWidgets.QLabel(
            "Hint: OreInsight v6 reads these paths from\n"
            "environment variables (ORE_*). You can point\n"
            "this at any project folder / dataset."
        )
        hint.setStyleSheet("color:#bbbbbb; font-size:10px; background-color: transparent;")
        hint.setWordWrap(True)
        form.addRow(hint)
        
        # Force update all fields to show current values
        self._refresh_fields()
    
    def _add_feature_row(self, label, key, form):
        """Add a feature layer selection row."""
        line = QtWidgets.QLineEdit("")
        line.setPlaceholderText("Optional - click ... to select")
        line.setReadOnly(True)
        line.setStyleSheet("""
            QLineEdit {
                background-color: #2a2a2a;
                border: 1px solid #444;
                color: #aaa;
            }
        """)
        
        btn = QtWidgets.QToolButton()
        btn.setText("…")
        btn.setFixedWidth(30)
        btn.setToolTip(f"Browse for {label}")
        
        clear_btn = QtWidgets.QToolButton()
        clear_btn.setText("✕")
        clear_btn.setFixedWidth(30)
        clear_btn.setToolTip("Clear")
        clear_btn.setStyleSheet("""
            QToolButton {
                background-color: #5a2a2a;
                border: 1px solid #8a4a4a;
            }
            QToolButton:hover {
                background-color: #7a3a3a;
            }
        """)
        
        def browse():
            p, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, 
                f"Select {label}", 
                "", 
                "GeoTIFF (*.tif *.tiff);;Shapefiles (*.shp);;CSV (*.csv);;All Files (*.*)"
            )
            if p:
                line.setText(p)
                line.setToolTip(p)
                self.feature_layers[key] = p
                self.config_changed.emit()
        
        def clear():
            line.setText("")
            line.setToolTip("")
            if key in self.feature_layers:
                del self.feature_layers[key]
            self.config_changed.emit()
        
        btn.clicked.connect(browse)
        clear_btn.clicked.connect(clear)
        
        h = QtWidgets.QHBoxLayout()
        h.setSpacing(4)
        h.addWidget(line, 1)
        h.addWidget(btn, 0)
        h.addWidget(clear_btn, 0)
        
        container = QtWidgets.QWidget()
        container.setLayout(h)
        form.addRow(label, container)
        
        return line
    
    def _refresh_fields(self):
        """Force refresh all fields with current values"""
        self.stack_edit.setText(self.stack_path)
        self.ref_edit.setText(self.ref_raster)
        self.mrds_edit.setText(self.mrds_csv)
        self.geo_edit.setText(self.geology_shp)
        self.results_edit.setText(self.results_dir)
        self.overlay_edit.setText(self.overlay_name)

    def _overlay_changed(self, text):
        self.overlay_name = text.strip()
        self.config_changed.emit()

    def get_config(self):
        commodity_map = {0: "copper", 1: "ree"}
        return {
            "commodity": commodity_map.get(self.commodity_combo.currentIndex(), "copper"),
            "stack": self.stack_path,
            "ref": self.ref_raster,
            "mrds": self.mrds_csv,
            "geo": self.geology_shp,
            "results": self.results_dir,
            "overlay": self.overlay_name,
            "feature_layers": self.feature_layers,  # Include feature layers
        }

    def apply_config(self, cfg: dict):
        self.stack_path = cfg.get("stack", self.stack_path)
        self.ref_raster = cfg.get("ref", self.ref_raster)
        self.mrds_csv = cfg.get("mrds", self.mrds_csv)
        self.geology_shp = cfg.get("geo", self.geology_shp)
        self.results_dir = cfg.get("results", self.results_dir)
        self.overlay_name = cfg.get("overlay", self.overlay_name)

        self.stack_edit.setText(self.stack_path)
        self.ref_edit.setText(self.ref_raster)
        self.mrds_edit.setText(self.mrds_csv)
        self.geo_edit.setText(self.geology_shp)
        self.results_edit.setText(self.results_dir)
        self.overlay_edit.setText(self.overlay_name)

        self.config_changed.emit()



# =====================================================================
# Main window with QGIS canvas
# =====================================================================

class GeoCoreAnalyticsMainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.settings = QtCore.QSettings("GeoCore", "AnalyticsStudio")
        self.python_exe = self.settings.value("python_exe", DEFAULT_PYTHON)
        
        # FORCE update to v4 ML script
        self.core_script = DEFAULT_CORE
        self.settings.setValue("core_script", DEFAULT_CORE)
        print(f"[INIT] Using analysis script: {self.core_script}")

        # SaaS features
        self.user_manager = UserManager()
        self.cloud_storage = CloudStorageManager()
        self.report_generator = ReportGenerator()
        self.subscription_manager = SubscriptionManager()
        
        # Enterprise features
        self.security_manager = SecurityManager()
        self.model_trainer = ModelTrainer()
        self.mobile_api = MobileAPIServer()
        
        # Project management
        self.current_project = None
        try:
            self.project_db = ProjectDatabase()
        except Exception as e:
            print(f"[ERROR] Failed to initialize project database: {e}")
            self.project_db = None

        # Set basic window properties first
        self.setWindowTitle("GeoCore Analytics Studio – Professional Mineral Exploration")
        self.resize(1500, 850)
        self._init_palette()

        # Central widget with simple placeholder first
        central = QtWidgets.QWidget()
        cl = QtWidgets.QVBoxLayout(central)
        cl.setContentsMargins(4, 4, 4, 4)
        cl.setSpacing(2)

        # Create a simple placeholder initially
        self.canvas = QtWidgets.QLabel("Initializing 3D viewer...")
        self.canvas.setAlignment(QtCore.Qt.AlignCenter)
        self.canvas.setStyleSheet("background-color: #2b2b2b; color: white; font-size: 14px;")
        self.viewer_type = "Initializing"
        
        cl.addWidget(self.canvas)

        # Inline progress bar (under map, replaces popup dialog)
        self.inline_progress_container = QtWidgets.QWidget()
        self.inline_progress_container.setFixedHeight(40)
        self.inline_progress_container.setStyleSheet("background-color: #1e1e1e; border-top: 1px solid #333;")
        prog_layout = QtWidgets.QHBoxLayout(self.inline_progress_container)
        prog_layout.setContentsMargins(10, 4, 10, 4)
        prog_layout.setSpacing(10)
        
        self.inline_progress_label = QtWidgets.QLabel("Ready")
        self.inline_progress_label.setStyleSheet("color: #aaa; font-size: 10pt; background: transparent;")
        self.inline_progress_label.setMinimumWidth(200)
        prog_layout.addWidget(self.inline_progress_label)
        
        self.inline_progress_bar = QtWidgets.QProgressBar()
        self.inline_progress_bar.setRange(0, 100)
        self.inline_progress_bar.setValue(0)
        self.inline_progress_bar.setFixedHeight(16)
        self.inline_progress_bar.setTextVisible(False)
        self.inline_progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #444;
                border-radius: 8px;
                background-color: #2a2a2a;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #0d7377, stop:1 #14a085);
                border-radius: 7px;
            }
        """)
        prog_layout.addWidget(self.inline_progress_bar, 1)
        
        self.inline_eta_label = QtWidgets.QLabel("")
        self.inline_eta_label.setStyleSheet("color: #888; font-size: 9pt; background: transparent;")
        self.inline_eta_label.setMinimumWidth(80)
        prog_layout.addWidget(self.inline_eta_label)
        
        self.inline_progress_container.hide()
        cl.addWidget(self.inline_progress_container)
        
        # Smooth animation timer for inline progress
        self._inline_target = 0
        self._inline_timer = QtCore.QTimer(self)
        self._inline_timer.setInterval(80)
        self._inline_timer.timeout.connect(self._animate_inline_progress)
        
        # ETA tracking
        self._analysis_start_time = None

        self.map_path_label = QtWidgets.QLabel("No map loaded. Run pipeline to generate 3D terrain.")
        self.map_path_label.setStyleSheet("color:#bbbbbb; font-size:10px;")
        cl.addWidget(self.map_path_label)

        self.setCentralWidget(central)

        # Docks
        self._create_log_dock()
        self._create_project_dock()
        self._create_dataset_dock()
        self._create_model_metrics_dock()
        self._create_collaboration_dock()
        # self._create_field_data_dock()  # TODO: Implement
        self._create_web_dashboard_dock()  # Web Dashboard with API access
        # self._create_security_dock()  # TODO: Implement

        # Status bar
        self.status = self.statusBar()
        self.status.showMessage("Idle – QGIS Ready")

        # Menus + toolbars
        self._create_actions()
        self._create_menubar()
        self._create_toolbar()
        self._apply_commodity_theme(self.dataset_widget.get_config().get("commodity", "copper"))

        self.worker = None
        self.progress_dialog = None

        # Current data
        self.prob_layer = None
        self.geo_layer = None
        self.ref_layer = None

        print("[MAIN] Main window initialized with placeholder")

        # Initialize 3D viewer after window is shown
        QtCore.QTimer.singleShot(100, self._init_3d_viewer)
    
    def _animate_inline_progress(self):
        """Smoothly animate inline progress bar."""
        current = self.inline_progress_bar.value()
        if current < self._inline_target:
            step = max(1, (self._inline_target - current) // 4)
            self.inline_progress_bar.setValue(min(current + step, self._inline_target))
    

    def _update_inline_progress(self, value, message):
        """Update the inline progress bar and ETA."""
        import time
        self._inline_target = value
        self.inline_progress_label.setText(message)

        if value == 100:
            self.inline_progress_bar.setValue(100)
            self.inline_eta_label.setText("")
            self.inline_progress_label.setText("Analysis complete")
        elif value > 0 and self._analysis_start_time:
            elapsed = time.time() - self._analysis_start_time
            total_est = elapsed / max(value / 100.0, 0.01)
            remaining = max(0, total_est - elapsed)
            if remaining > 60:
                self.inline_eta_label.setText(f"~{remaining/60:.0f}m left")
            elif remaining > 5:
                self.inline_eta_label.setText(f"~{remaining:.0f}s left")
            else:
                self.inline_eta_label.setText("Almost done...")
        else:
            self.inline_eta_label.setText("")

    def _init_3d_viewer(self):
        """Initialize the 3D viewer after the main window is shown."""
        print("[MAIN] Initializing 3D viewer...")
        
        # Get the central widget layout
        central = self.centralWidget()
        layout = central.layout()
        
        # Remove the placeholder
        old_canvas = self.canvas
        layout.removeWidget(old_canvas)
        old_canvas.deleteLater()
        
        # Try VTK first (most reliable), then PyVista, then matplotlib
        try:
            from viewer_3d_vtk import Terrain3DViewer
            self.canvas = Terrain3DViewer()
            print("[MAIN] Using VTK 3D viewer (high performance)")
            self.viewer_type = "VTK"
        except Exception as e:
            print(f"[MAIN] VTK failed ({e}), trying PyVista")
            try:
                from viewer_3d import Terrain3DViewer
                self.canvas = Terrain3DViewer()
                print("[MAIN] Using PyVista 3D viewer")
                self.viewer_type = "PyVista"
            except Exception as e2:
                print(f"[MAIN] PyVista failed ({e2}), falling back to matplotlib")
                try:
                    from viewer_3d_matplotlib import Terrain3DViewer
                    self.canvas = Terrain3DViewer()
                    print("[MAIN] Using Matplotlib 3D viewer (compatibility mode)")
                    self.viewer_type = "Matplotlib"
                except Exception as e3:
                    print(f"[MAIN] All viewers failed! VTK: {e}, PyVista: {e2}, Matplotlib: {e3}")
                    # Create a simple label as last resort
                    self.canvas = QtWidgets.QLabel("3D Viewer failed to initialize")
                    self.canvas.setAlignment(QtCore.Qt.AlignCenter)
                    self.viewer_type = "Failed"
        
        # Add the new canvas to the layout
        layout.insertWidget(0, self.canvas)
        
        # Update window title
        self.setWindowTitle(f"GeoCore Analytics Studio – 3D Terrain Visualization ({self.viewer_type})")
        
        # Update status
        self.status.showMessage(f"3D Viewer ready ({self.viewer_type})", 3000)
        
        print(f"[MAIN] 3D viewer initialized: {self.viewer_type}")
        
        # Automatically load the last map if available
        if self.viewer_type != "Failed":
            print("[MAIN] Auto-loading terrain data...")
            try:
                self.reload_map()
            except Exception as e:
                print(f"[MAIN] Auto-load failed: {e}")
                self.append_log("[INFO] No terrain data found - use 'Run Analysis' to generate")
    
    def new_project(self):
        """Create a new project."""
        if not self.project_db:
            QtWidgets.QMessageBox.warning(
                self, "Database Error", 
                "Project database is not available. Cannot create projects."
            )
            return
            
        try:
            from project_manager import NewProjectDialog
            
            dialog = NewProjectDialog(self)
            if dialog.exec_() == QtWidgets.QDialog.Accepted:
                project = dialog.get_project()
                self.current_project = self.project_db.save_project(project)
                self.update_window_title()
                self.append_log(f"[PROJECT] Created new project: {project.name}")
                
                # Open data import hub for new project
                self.open_data_import()
        except Exception as e:
            self.append_log(f"[ERROR] Failed to create project: {e}")
            QtWidgets.QMessageBox.warning(
                self, "Project Error", 
                f"Failed to create project: {e}"
            )
    
    def open_project_manager(self):
        """Open the project manager."""
        if not self.project_db:
            QtWidgets.QMessageBox.warning(
                self, "Database Error", 
                "Project database is not available. Cannot manage projects."
            )
            return
            
        try:
            dialog = ProjectManagerDialog(self)
            dialog.project_selected.connect(self.load_project)
            dialog.exec_()
        except Exception as e:
            self.append_log(f"[ERROR] Failed to open project manager: {e}")
            QtWidgets.QMessageBox.warning(
                self, "Project Manager Error", 
                f"Failed to open project manager: {e}"
            )
    
    def load_project(self, project):
        """Load a project."""
        self.current_project = project
        self.update_window_title()
        self.append_log(f"[PROJECT] Loaded project: {project.name}")
        
        # Update dataset widget with project settings
        if hasattr(self, 'dataset_widget'):
            self.dataset_widget.apply_config(project.settings)
        
        # Reload map if data is available
        if project.settings.get("reference_raster"):
            self.reload_map()
    
    def save_current_project(self):
        """Save the current project."""
        if self.current_project:
            try:
                # Update project settings from current state
                if hasattr(self, 'dataset_widget'):
                    self.current_project.settings.update(self.dataset_widget.get_config())
                
                self.project_db.save_project(self.current_project)
                self.append_log(f"[PROJECT] Saved project: {self.current_project.name}")
            except Exception as e:
                self.append_log(f"[ERROR] Failed to save project: {e}")
                QtWidgets.QMessageBox.warning(
                    self, "Save Error", 
                    f"Failed to save project: {e}"
                )
        else:
            QtWidgets.QMessageBox.information(
                self, "No Project", 
                "No project is currently open. Create a new project first."
            )
    
    def open_data_import(self):
        """Open the data import hub."""
        if not self.current_project:
            reply = QtWidgets.QMessageBox.question(
                self, "No Project Open",
                "You need to open a project first. Would you like to create a new project?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
            )
            if reply == QtWidgets.QMessageBox.Yes:
                self.new_project()
            return
        
        dialog = DataImportHub(self.current_project, self)
        dialog.data_imported.connect(self.on_data_imported)
        dialog.exec_()
    
    def on_data_imported(self, file_type, file_path):
        """Handle data import."""
        if self.current_project:
            # Save project with new data
            self.project_db.save_project(self.current_project)
            
            # Update dataset widget if it's a key file
            if hasattr(self, 'dataset_widget'):
                if file_type == "reference":
                    # Update reference raster in dataset widget
                    pass  # Will implement dataset widget updates
                elif file_type == "dem":
                    # Update DEM path
                    pass
            
            self.append_log(f"[DATA] Imported {file_type}: {Path(file_path).name}")
    
    def update_window_title(self):
        """Update window title with current project."""
        base_title = f"GeoCore Analytics Studio – 3D Terrain Visualization ({self.viewer_type})"
        if self.current_project:
            self.setWindowTitle(f"{base_title} - {self.current_project.name}")
        else:
            self.setWindowTitle(base_title)
    
    def login_user(self):
        """Handle user login."""
        if self.user_manager.login(self):
            user = self.user_manager.get_current_user()
            self.append_log(f"[AUTH] Logged in as {user.username} ({user.role})")
            self.update_window_title()
            
            # Refresh collaboration panel
            self._refresh_collaboration_panel()
            
            # Update menu
            self._update_user_menu()
    
    def logout_user(self):
        """Handle user logout."""
        if self.user_manager.is_authenticated():
            username = self.user_manager.get_current_user().username
            self.user_manager.logout()
            self.append_log(f"[AUTH] Logged out {username}")
            
            # Refresh collaboration panel
            self._refresh_collaboration_panel()
            
            # Update menu
            self._update_user_menu()
    
    def _refresh_collaboration_panel(self):
        """Refresh collaboration panel based on auth status."""
        if hasattr(self, 'collaboration_panel'):
            # Remove old panel
            old_panel = self.collaboration_panel
            
            # Create new panel
            if self.user_manager.is_authenticated() and self.current_project:
                self.collaboration_panel = CollaborationPanel(
                    str(self.current_project.id), 
                    self.user_manager.get_current_user()
                )
            else:
                self.collaboration_panel = QLabel("Login and open a project to access collaboration features")
                self.collaboration_panel.setAlignment(Qt.AlignCenter)
                self.collaboration_panel.setStyleSheet("color: #666; font-style: italic;")
            
            # Update dock widget
            for dock in self.findChildren(QtWidgets.QDockWidget):
                if dock.windowTitle() == "Team Collaboration":
                    dock.setWidget(self.collaboration_panel)
                    break
            
            # Clean up old panel
            if old_panel:
                old_panel.deleteLater()
    
    def _update_user_menu(self):
        """Update user menu based on authentication status."""
        # This would normally refresh the menu, but for simplicity we'll just log
        if self.user_manager.is_authenticated():
            user = self.user_manager.get_current_user()
            self.status.showMessage(f"Logged in as {user.username}", 3000)
        else:
            self.status.showMessage("Not logged in", 3000)
    
    def _build_dynamic_report(self, validation_text, features_list):
        """Build a report that dynamically interprets the actual results."""
        import re
        from datetime import datetime

        # Parse values
        m = re.search(r'Real Deposits Used: (\d+)', validation_text)
        n_deposits = int(m.group(1)) if m else 0
        m = re.search(r'Test AUC: ([\d.]+)', validation_text)
        auc_val = float(m.group(1)) if m else 0
        m = re.search(r'Training AUC: ([\d.]+)', validation_text)
        train_auc_val = float(m.group(1)) if m else 0
        m = re.search(r'Training Samples: (\d+)', validation_text)
        n_train = int(m.group(1)) if m else 0
        m = re.search(r'Test Samples: (\d+)', validation_text)
        n_test = int(m.group(1)) if m else 0
        m = re.search(r'Features: (\d+)', validation_text)
        n_feat = int(m.group(1)) if m else 0

        is_transfer = "Transfer Prediction" in validation_text or n_deposits == 0

        # Parse features
        top_features = []
        for line in features_list[:10]:
            if ':' in line:
                parts = line.split(':')
                try:
                    top_features.append((parts[0].strip(), float(parts[1].strip())))
                except (ValueError, IndexError):
                    pass

        # Classify
        terrain_n = ['slope', 'aspect', 'curvature', 'tpi_relative_elevation']
        alter_n = ['alteration_silica', 'alteration_argillic', 'alteration_phyllic', 'alteration_propylitic', 'alteration_combined']
        geophys_n = ['magnetics', 'gravity']

        terrain_pct = sum(v for n, v in top_features if n in terrain_n) * 100
        alter_pct = sum(v for n, v in top_features if n in alter_n) * 100
        geophys_pct = sum(v for n, v in top_features if n in geophys_n) * 100
        other_pct = max(0, 100 - terrain_pct - alter_pct - geophys_pct)

        # AUC word
        if auc_val >= 0.90:
            auc_word = "excellent"
        elif auc_val >= 0.80:
            auc_word = "good"
        elif auc_val >= 0.70:
            auc_word = "moderate"
        else:
            auc_word = "limited"

        top_name = top_features[0][0] if top_features else "N/A"
        top_pct = top_features[0][1] * 100 if top_features else 0

        # Dynamic interpretation
        if alter_pct > 30:
            interp = (
                "Hydrothermal alteration features dominate ({:.0f}%), indicating the model "
                "detects real mineralisation signatures from ASTER satellite data. This aligns "
                "with the USGS methodology (Mars, 2019) for Arizona porphyry Cu prospectivity."
            ).format(alter_pct)
        elif geophys_pct > 30:
            interp = (
                "Geophysical features drive {:.0f}% of predictions, detecting subsurface "
                "intrusions and density contrasts. Adding ASTER alteration data would likely "
                "improve results further."
            ).format(geophys_pct)
        elif terrain_pct > 70:
            interp = (
                "Terrain features dominate at {:.0f}%. While valid for Arizona porphyry "
                "systems, adding geophysical and alteration layers would substantially "
                "improve predictions."
            ).format(terrain_pct)
        else:
            interp = (
                "The model uses a balanced multi-evidence approach: terrain ({:.0f}%), "
                "alteration ({:.0f}%), geophysics ({:.0f}%). This provides robust predictions."
            ).format(terrain_pct, alter_pct, geophys_pct)

        # Feature importance lines
        fi_lines = []
        for fname, fval in top_features:
            if fval > 0.001:
                fi_lines.append("   {}: {:.4f} ({:.1f}%)".format(fname, fval, fval * 100))
        fi_text = "\n".join(fi_lines) if fi_lines else "   No data available."

        # Build sections based on mode
        now = datetime.now()
        lines = []
        lines.append("=" * 70)
        lines.append("         GEOCORE ANALYTICS - TECHNICAL REPORT")
        lines.append("         Machine Learning Mineral Prospectivity")
        lines.append("=" * 70)
        lines.append("")
        lines.append("PROJECT INFORMATION")
        lines.append("-" * 70)
        lines.append("Analysis Date:    " + now.strftime('%B %d, %Y'))
        if is_transfer:
            lines.append("Mode:             TRANSFER PREDICTION (unexplored area)")
        else:
            lines.append("Mode:             TRAINED MODEL ({} deposits, {} features)".format(n_deposits, n_feat))
        lines.append("Mineral Target:   Copper (Cu) - Porphyry Systems")
        lines.append("Method:           Random Forest Classifier (200 trees)")
        lines.append("")

        lines.append("EXECUTIVE SUMMARY")
        lines.append("-" * 70)
        if is_transfer:
            lines.append("This analysis applied a previously trained model to a new area with")
            lines.append("no known deposits. The model identifies terrain with similar signatures")
            lines.append("to deposits in the training region. Results are EXPLORATORY.")
        else:
            lines.append("Trained on {} known Cu deposits with {} feature layers.".format(n_deposits, n_feat))
            lines.append("Test AUC of {:.4f} ({}) indicates {} predictive ability.".format(auc_val, auc_word, auc_word))
            lines.append(interp)
        lines.append("")

        lines.append("MODEL PERFORMANCE")
        lines.append("-" * 70)
        if is_transfer:
            lines.append("   Mode: Transfer Prediction (no local AUC)")
            lines.append("   Feature importance shown is from the training area.")
        else:
            lines.append("   Test AUC:         {:.4f} ({})".format(auc_val, auc_word))
            lines.append("   Training AUC:     {:.4f}".format(train_auc_val))
            lines.append("   Training Samples: {}".format(n_train))
            lines.append("   Test Samples:     {}".format(n_test))
            lines.append("   Deposits Used:    {}".format(n_deposits))
            lines.append("   Features:         {}".format(n_feat))
        lines.append("")

        lines.append("FEATURE IMPORTANCE")
        lines.append("-" * 70)
        lines.append(fi_text)
        lines.append("")
        lines.append("   Breakdown:")
        lines.append("   Terrain:     {:.1f}%".format(terrain_pct))
        lines.append("   Alteration:  {:.1f}%".format(alter_pct))
        lines.append("   Geophysics:  {:.1f}%".format(geophys_pct))
        lines.append("   Other:       {:.1f}%".format(other_pct))
        lines.append("")

        lines.append("INTERPRETATION")
        lines.append("-" * 70)
        lines.append("   " + interp)
        lines.append("   Top predictor: {} ({:.1f}% importance)".format(top_name, top_pct))
        lines.append("")

        lines.append("METHODOLOGY")
        lines.append("-" * 70)
        lines.append("   Algorithm:    Random Forest (scikit-learn), 200 trees, depth 15")
        lines.append("   Calibration:  Sigmoid probability calibration (Platt scaling)")
        lines.append("   Validation:   Stratified 5-fold cross-validation")
        lines.append("   Grade Model:  USGS Lognormal (Singer et al., 2008)")
        lines.append("                 Median: 0.44% Cu | P10: 0.25% | P90: 0.80%")
        lines.append("")
        lines.append("   References:")
        lines.append("   - Mars et al. (2019) Economic Geology - ASTER Cu prospectivity")
        lines.append("   - Singer et al. (2008) USGS OFR 2008-1155 - Grade-tonnage models")
        lines.append("   - Carranza & Laborte (2015) Ore Geology Reviews - RF for MPM")
        lines.append("   - Dong et al. (2024) JGR - Deep Forest for porphyry Cu MPM")
        lines.append("")

        lines.append("RECOMMENDATIONS")
        lines.append("-" * 70)
        if is_transfer:
            lines.append("   TRANSFER PREDICTION - EXPLORATORY TARGETS:")
            lines.append("   1. Identify areas showing elevated probability (>50%)")
            lines.append("   2. Cross-reference with published geological maps")
            lines.append("   3. Field reconnaissance in highest-probability zones")
            lines.append("   4. Collect rock chip and soil samples for confirmation")
            lines.append("   5. If confirmed, plan detailed geophysical surveys")
        else:
            lines.append("   DRILLING TARGETS:")
            lines.append("   1. Focus on RED/ORANGE zones (>70% probability)")
            lines.append("   2. Cross-reference high-prob zones with alteration data")
            lines.append("   3. Reconnaissance sampling in top zones")
            lines.append("   4. Initial drill program: 5-10 holes in highest zones")
            lines.append("   5. Target depth: 200-500m for porphyry copper")
            lines.append("   6. Retrain model with drilling results")
        lines.append("")

        lines.append("LIMITATIONS")
        lines.append("-" * 70)
        lines.append("   - Predictions are probabilistic, not deterministic")
        lines.append("   - Ground truthing required before investment")
        lines.append("   - Grade estimates are USGS reference ranges, NOT assays")
        if is_transfer:
            lines.append("   - TRANSFER MODE: Lower confidence than in-district predictions")
        if terrain_pct > 70:
            lines.append("   - Terrain-dominated model: add alteration/geophysics for improvement")
        lines.append("")

        lines.append("=" * 70)
        lines.append("Generated by: GeoCore Analytics v4.0")
        if is_transfer:
            lines.append("Mode: Transfer Prediction | Features: {}".format(n_feat))
        else:
            lines.append("AUC: {:.4f} | Deposits: {} | Features: {}".format(auc_val, n_deposits, n_feat))
        lines.append("Report Date: " + now.strftime('%Y-%m-%d %H:%M:%S'))
        lines.append("=" * 70)

        return "\n".join(lines)


    def generate_report(self):
        """Generate professional report with REAL analysis data."""
        try:
            cfg = self.dataset_widget.get_config()
            
            # Check if validation report exists
            validation_path = os.path.join(cfg['results'], 'oreinsight_v4_validation.txt')
            if not os.path.exists(validation_path):
                QtWidgets.QMessageBox.warning(
                    self,
                    "No Analysis Results",
                    "Please run the analysis first to generate a report.\n\n"
                    "Click 'Run Analysis' to create results."
                )
                return
            
            # Show info about what the report contains
            info_msg = QtWidgets.QMessageBox(self)
            info_msg.setIcon(QtWidgets.QMessageBox.Information)
            info_msg.setWindowTitle("Generating Comprehensive Report")
            info_msg.setText("Creating detailed technical report with:")
            info_msg.setInformativeText(
                "• Executive summary with key findings\n"
                "• Complete methodology explanation\n"
                "• Model performance metrics and validation\n"
                "• Feature importance analysis\n"
                "• Probability map interpretation\n"
                "• Drilling recommendations\n"
                "• Explanation of colored terrain display\n\n"
                "This is a FULL technical report, not just the validation text."
            )
            info_msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
            info_msg.setStyleSheet("""
                QMessageBox {
                    background-color: #2d2d2d;
                }
                QMessageBox QLabel {
                    color: #d4d4d4;
                    font-size: 11pt;
                }
                QPushButton {
                    background-color: #0d7377;
                    color: white;
                    border: none;
                    padding: 6px 20px;
                    border-radius: 4px;
                    min-width: 80px;
                }
                QPushButton:hover {
                    background-color: #14a085;
                }
            """)
            info_msg.exec_()
            
            # Read REAL validation data
            with open(validation_path, 'r') as f:
                validation_text = f.read()
            
            # Parse the validation data
            import re
            training_samples = re.search(r'Training Samples: (\d+)', validation_text)
            test_samples = re.search(r'Test Samples: (\d+)', validation_text)
            real_deposits = re.search(r'Real Deposits Used: (\d+)', validation_text)
            features = re.search(r'Features: (\d+)', validation_text)
            train_auc = re.search(r'Training AUC: ([\d.]+)', validation_text)
            test_auc = re.search(r'Test AUC: ([\d.]+)', validation_text)
            cv_auc = re.search(r'Cross-validation AUC: ([\d.]+) ± ([\d.]+)', validation_text)
            
            # Extract feature importance
            feature_section = re.search(r'Feature Importance:\n(.*?)\n\nTest Set', validation_text, re.DOTALL)
            features_list = []
            if feature_section:
                for line in feature_section.group(1).split('\n'):
                    if ':' in line:
                        features_list.append(line.strip())
            
            # Build dynamic report from actual results
            report = self._build_dynamic_report(validation_text, features_list)
            
            # Show in dialog
            dialog = QtWidgets.QDialog(self)
            dialog.setWindowTitle("Technical Report - GeoCore Analysis")
            dialog.resize(1000, 800)
            dialog.setStyleSheet("background-color: #2d2d2d;")
            
            layout = QtWidgets.QVBoxLayout(dialog)
            layout.setContentsMargins(10, 10, 10, 10)
            
            # Report content
            text_edit = QtWidgets.QTextEdit()
            text_edit.setReadOnly(True)
            text_edit.setPlainText(report)
            text_edit.setStyleSheet("""
                QTextEdit {
                    background-color: #1e1e1e;
                    color: #d4d4d4;
                    font-family: 'Consolas', 'Courier New', monospace;
                    font-size: 10pt;
                    border: 1px solid #555;
                    padding: 15px;
                    line-height: 1.4;
                }
            """)
            layout.addWidget(text_edit)
            
            # Buttons
            button_layout = QtWidgets.QHBoxLayout()
            
            save_btn = QtWidgets.QPushButton("Save as TXT")
            save_btn.clicked.connect(lambda: self._save_report_txt(report))
            save_btn.setStyleSheet("""
                QPushButton {
                    background-color: #0d7377;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #14a085;
                }
            """)
            button_layout.addWidget(save_btn)
            
            button_layout.addStretch()
            
            close_btn = QtWidgets.QPushButton("Close")
            close_btn.clicked.connect(dialog.close)
            close_btn.setStyleSheet("""
                QPushButton {
                    background-color: #404040;
                    color: white;
                    border: 1px solid #555;
                    padding: 8px 16px;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #4a4a4a;
                }
            """)
            button_layout.addWidget(close_btn)
            
            layout.addLayout(button_layout)
            
            dialog.exec_()
            
        except Exception as e:
            print(f"[ERROR] Failed to generate report: {e}")
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.warning(
                self,
                "Report Error",
                f"Failed to generate report:\n\n{e}"
            )
    
    def _save_report_txt(self, report_text):
        """Save report as text file."""
        try:
            cfg = self.dataset_widget.get_config()
            default_path = os.path.join(cfg['results'], 'geocore_technical_report.txt')
            
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Save Technical Report",
                default_path,
                "Text Files (*.txt);;All Files (*.*)"
            )
            
            if path:
                with open(path, 'w') as f:
                    f.write(report_text)
                QtWidgets.QMessageBox.information(
                    self,
                    "Report Saved",
                    f"Technical report saved to:\n\n{path}"
                )
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self,
                "Save Error",
                f"Failed to save report:\n\n{e}"
            )
    
    def _show_3d_help_dialog(self):
        """Show helpful explanation about the 3D viewer display."""
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Understanding Your 3D Results")
        dialog.resize(700, 500)
        dialog.setStyleSheet("background-color: #2d2d2d;")
        
        layout = QtWidgets.QVBoxLayout(dialog)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Title
        title = QtWidgets.QLabel("🎯 Understanding the Colored Terrain")
        title.setStyleSheet("""
            font-size: 18pt;
            font-weight: bold;
            color: #14a085;
            background-color: transparent;
        """)
        layout.addWidget(title)
        
        # Explanation text
        explanation = QtWidgets.QLabel(
            "<b>Why is the entire area colored?</b><br><br>"
            "This is <b>CORRECT behavior</b> - not a bug! The machine learning model "
            "predicts copper probability for <b>EVERY pixel</b> in the 54km × 54km "
            "analysis area (29.2 million predictions).<br><br>"
            
            "<b>What do the colors mean?</b><br>"
            "• <span style='color: #4444ff;'><b>BLUE/GREEN (11-40%)</b></span>: "
            "Lower probability background terrain<br>"
            "• <span style='color: #ffaa00;'><b>YELLOW/ORANGE (40-70%)</b></span>: "
            "Moderate probability zones<br>"
            "• <span style='color: #ff4444;'><b>RED (70-86%)</b></span>: "
            "<b>HIGH PRIORITY drilling targets</b><br><br>"
            
            "<b>What should you do?</b><br>"
            "1. Focus on <b>RED zones</b> - these match known copper deposit signatures<br>"
            "2. <b>Click on terrain</b> to get exact coordinates and probability<br>"
            "3. Use the color legend (right side) to interpret probability levels<br>"
            "4. Generate the Technical Report for detailed recommendations<br><br>"
            
            "<b>Is this scientifically valid?</b><br>"
            "Yes! The model uses terrain features "
            "(elevation, slope, aspect, curvature) which are strongly associated with "
            "Arizona porphyry copper deposits."
        )
        explanation.setWordWrap(True)
        explanation.setStyleSheet("""
            color: #d4d4d4;
            font-size: 11pt;
            background-color: transparent;
            line-height: 1.5;
        """)
        layout.addWidget(explanation)
        
        # Checkbox to not show again
        checkbox = QtWidgets.QCheckBox("Don't show this again")
        checkbox.setStyleSheet("""
            QCheckBox {
                color: #bbbbbb;
                font-size: 10pt;
                background-color: transparent;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
        """)
        layout.addWidget(checkbox)
        
        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        
        read_more_btn = QtWidgets.QPushButton("Read Full Guide")
        read_more_btn.clicked.connect(lambda: self._open_understanding_guide())
        read_more_btn.setStyleSheet("""
            QPushButton {
                background-color: #404040;
                color: white;
                border: 1px solid #555;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
            }
        """)
        button_layout.addWidget(read_more_btn)
        
        button_layout.addStretch()
        
        got_it_btn = QtWidgets.QPushButton("Got It!")
        got_it_btn.clicked.connect(dialog.accept)
        got_it_btn.setStyleSheet("""
            QPushButton {
                background-color: #0d7377;
                color: white;
                border: none;
                padding: 8px 24px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #14a085;
            }
        """)
        button_layout.addWidget(got_it_btn)
        
        layout.addLayout(button_layout)
        
        # Save preference if checkbox is checked
        def on_close():
            if checkbox.isChecked():
                self.settings.setValue("hide_3d_help", True)
            dialog.accept()
        
        got_it_btn.clicked.disconnect()
        got_it_btn.clicked.connect(on_close)
        
        dialog.exec_()
    
    def _open_understanding_guide(self):
        """Open the UNDERSTANDING_THE_ANALYSIS.md file."""
        guide_path = "UNDERSTANDING_THE_ANALYSIS.md"
        if os.path.exists(guide_path):
            import subprocess
            try:
                if sys.platform == 'win32':
                    os.startfile(guide_path)
                elif sys.platform == 'darwin':
                    subprocess.call(['open', guide_path])
                else:
                    subprocess.call(['xdg-open', guide_path])
            except Exception as e:
                QtWidgets.QMessageBox.information(
                    self,
                    "Guide Location",
                    f"Please open this file to read more:\n\n{os.path.abspath(guide_path)}"
                )
        else:
            QtWidgets.QMessageBox.warning(
                self,
                "Guide Not Found",
                "The UNDERSTANDING_THE_ANALYSIS.md guide file was not found."
            )

    def _save_report_txt(self, report_text):
        """Save report as text file."""
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Report",
            "GeoCore_Technical_Report.txt",
            "Text Files (*.txt);;All Files (*)"
        )
        if filename:
            try:
                with open(filename, 'w') as f:
                    f.write(report_text)
                QtWidgets.QMessageBox.information(
                    self,
                    "Report Saved",
                    f"Report saved successfully to:\n{filename}"
                )
            except Exception as e:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Save Error",
                    f"Failed to save report:\n{e}"
                )
            project_data = {
                'name': self.current_project.name,
                'location': self.current_project.location,
                'project_type': self.current_project.project_type,
                'description': self.current_project.description,
                'settings': self.current_project.settings
            }
        else:
            # Use default data if no project
            cfg = self.dataset_widget.get_config()
            project_data = {
                'name': 'GeoCore Analysis',
                'location': 'Arizona, USA',
                'project_type': 'Copper Exploration',
                'description': 'AI-powered mineral exploration analysis',
                'settings': cfg,
                'results_dir': cfg['results'],
                'probability_map': os.path.join(cfg['results'], 'oreinsight_v4_probability.tif'),
                'grade_map': os.path.join(cfg['results'], 'oreinsight_v4_grade.tif'),
                'uncertainty_map': os.path.join(cfg['results'], 'oreinsight_v4_uncertainty.tif'),
                'validation_report': os.path.join(cfg['results'], 'oreinsight_v4_validation.txt'),
                'deposit_type_map': os.path.join(cfg['results'], 'oreinsight_v3_deposit_type.tif')  # Legacy
            }
        
        # Show report dialog
        dialog = ReportDialog(project_data, self)
        dialog.exec_()
    
    def cloud_sync(self):
        """Sync project to cloud storage."""
        if not self.current_project:
            QtWidgets.QMessageBox.information(
                self, "No Project", 
                "Please open a project first to sync to cloud."
            )
            return
        
        if not self.cloud_storage.is_connected():
            QtWidgets.QMessageBox.warning(
                self, "Cloud Not Connected", 
                "Cloud storage is not configured. Please check cloud settings."
            )
            return
        
        # For now, just show a message (would implement actual sync)
        reply = QtWidgets.QMessageBox.question(
            self, "Cloud Sync",
            f"Sync project '{self.current_project.name}' to cloud storage?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
        
        if reply == QtWidgets.QMessageBox.Yes:
            # Would implement actual cloud sync here
            self.append_log(f"[CLOUD] Syncing project {self.current_project.name}...")
            QtWidgets.QMessageBox.information(
                self, "Cloud Sync", 
                "Project sync initiated. Check activity log for progress."
            )
    
    def cloud_settings(self):
        """Open cloud storage settings."""
        storage_info = self.cloud_storage.get_storage_info()
        
        info_text = f"""
Cloud Storage Status:
Connected: {'Yes' if storage_info['connected'] else 'No'}
Provider: {storage_info.get('provider', 'None')}
Auto Sync: {'Enabled' if storage_info.get('auto_sync') else 'Disabled'}

Configure cloud storage in the settings to enable:
• Automatic project backups
• Team collaboration sync
• Cross-device access
• Version history
        """
        
        QtWidgets.QMessageBox.information(
            self, "Cloud Storage Settings", info_text
        )
    
    def show_subscription(self):
        """Show subscription management dialog."""
        if not self.user_manager.is_authenticated():
            QtWidgets.QMessageBox.information(
                self, "Login Required", 
                "Please login to manage your subscription."
            )
            return
        
        dialog = SubscriptionDialog(self.user_manager.get_current_user(), self)
        dialog.exec_()
    
    def train_custom_model(self):
        """Open custom AI model training dialog."""
        if not self.user_manager.is_authenticated():
            QtWidgets.QMessageBox.information(
                self, "Login Required", 
                "Please login to access AI model training."
            )
            return
        
        user = self.user_manager.get_current_user()
        if not self.subscription_manager.check_feature_access(user.subscription_tier, "custom_models"):
            QtWidgets.QMessageBox.warning(
                self, "Feature Not Available", 
                "Custom AI model training is available in Enterprise tier.\n\n"
                "Upgrade your subscription to access this feature."
            )
            return
        
        dialog = ModelTrainingDialog(self)
        dialog.exec_()
    
    def show_security_settings(self):
        """Show security settings dialog."""
        if not self.user_manager.is_authenticated():
            QtWidgets.QMessageBox.information(
                self, "Login Required", 
                "Please login to access security settings."
            )
            return
        
        dialog = SecuritySettingsDialog(self.user_manager.get_current_user(), self)
        dialog.exec_()

    def _theme_for_commodity(self, commodity):
        commodity = (commodity or "copper").lower().strip()
        accent = REE_ACCENT if commodity == "ree" else COPPER_ACCENT
        accent_hover = "#1AD7B5" if commodity == "ree" else "#C9864A"
        accent_pressed = "#009F86" if commodity == "ree" else "#8E5A2B"
        return {
            "commodity": commodity,
            "accent": accent,
            "accent_hover": accent_hover,
            "accent_pressed": accent_pressed,
            "name": "REE" if commodity == "ree" else "COPPER",
        }

    def _apply_commodity_theme(self, commodity=None):
        if commodity is None:
            if hasattr(self, "dataset_widget"):
                commodity = self.dataset_widget.get_config().get("commodity", "copper")
            else:
                commodity = "copper"
        theme = self._theme_for_commodity(commodity)
        self._active_theme = theme
        accent = theme["accent"]
        accent_hover = theme["accent_hover"]
        accent_pressed = theme["accent_pressed"]

        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: #181818;
                color: #ffffff;
            }}
            QDockWidget {{
                background-color: #202020;
                color: #ffffff;
            }}
            QDockWidget::title {{
                background-color: #262626;
                color: #ffffff;
                padding: 5px 8px;
                border: 1px solid #3a3a3a;
                border-left: 4px solid {accent};
            }}
            QWidget {{
                background-color: #202020;
                color: #ffffff;
            }}
            QMenuBar {{
                background-color: #1d1d1d;
                color: #ffffff;
                border-bottom: 1px solid #303030;
            }}
            QMenuBar::item:selected {{
                background-color: {accent};
                border-radius: 4px;
            }}
            QMenu {{
                background-color: #252525;
                color: #ffffff;
                border: 1px solid #3a3a3a;
            }}
            QMenu::item:selected {{
                background-color: {accent};
            }}
            QStatusBar {{
                background-color: #1b1b1b;
                color: #d8d8d8;
                border-top: 1px solid #303030;
            }}
            QToolBar {{
                background-color: #1d1d1d;
                border: none;
                spacing: 4px;
            }}
            QToolButton, QPushButton {{
                background-color: #313131;
                border: 1px solid #4a4a4a;
                border-radius: 4px;
                color: #ffffff;
                padding: 4px 8px;
            }}
            QToolButton:hover, QPushButton:hover {{
                background-color: {accent_hover};
                border: 1px solid {accent_hover};
            }}
            QToolButton:pressed, QPushButton:pressed {{
                background-color: {accent_pressed};
                border: 1px solid {accent_pressed};
            }}
            QLineEdit, QPlainTextEdit, QTextEdit, QComboBox {{
                background-color: #2b2b2b;
                border: 1px solid #4d4d4d;
                border-radius: 4px;
                color: #ffffff;
                selection-background-color: {accent};
            }}
            QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus, QComboBox:focus {{
                border: 1px solid {accent};
            }}
            QComboBox QAbstractItemView {{
                background-color: #252525;
                color: #ffffff;
                selection-background-color: {accent};
            }}
            QProgressBar {{
                border: 1px solid #3e3e3e;
                border-radius: 8px;
                background-color: #2a2a2a;
            }}
            QProgressBar::chunk {{
                background: {accent};
                border-radius: 7px;
            }}
            QScrollBar:vertical, QScrollBar:horizontal {{
                background: #202020;
                border: none;
                margin: 0px;
            }}
            QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
                background: {accent};
                border-radius: 5px;
                min-height: 24px;
                min-width: 24px;
            }}
        """)

        if hasattr(self, "inline_progress_container"):
            self.inline_progress_container.setStyleSheet(
                f"background-color: #1a1a1a; border-top: 2px solid {accent};"
            )
        if hasattr(self, "inline_progress_label"):
            self.inline_progress_label.setStyleSheet(
                f"color: {accent}; font-size: 10pt; font-weight: 600; background: transparent;"
            )
        if hasattr(self, "inline_eta_label"):
            self.inline_eta_label.setStyleSheet(
                "color: #d9d9d9; font-size: 9pt; background: transparent;"
            )
        if hasattr(self, "inline_progress_bar"):
            self.inline_progress_bar.setStyleSheet(f"""
                QProgressBar {{
                    border: 1px solid #444;
                    border-radius: 8px;
                    background-color: #2a2a2a;
                }}
                QProgressBar::chunk {{
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 {accent_pressed}, stop:1 {accent});
                    border-radius: 7px;
                }}
            """)
        if hasattr(self, "map_path_label"):
            self.map_path_label.setStyleSheet(
                f"color:{accent}; font-size:10px; font-weight:600;"
            )
        if hasattr(self, "dataset_widget"):
            self.dataset_widget.setStyleSheet(f"""
                DatasetConfigWidget {{
                    background-color: #202020;
                }}
                DatasetConfigWidget QLineEdit {{
                    background-color: #2b2b2b;
                    border: 1px solid #4d4d4d;
                    border-radius: 3px;
                    padding: 6px;
                    color: #ffffff;
                    min-height: 20px;
                }}
                DatasetConfigWidget QLineEdit:read-only {{
                    background-color: #262626;
                    color: #cccccc;
                }}
                DatasetConfigWidget QToolButton {{
                    background-color: {accent};
                    border: 1px solid {accent};
                    border-radius: 3px;
                    color: #ffffff;
                    padding: 6px;
                    min-width: 30px;
                    min-height: 20px;
                    font-weight: bold;
                }}
                DatasetConfigWidget QToolButton:hover {{
                    background-color: {accent_hover};
                    border: 1px solid {accent_hover};
                }}
                DatasetConfigWidget QToolButton:pressed {{
                    background-color: {accent_pressed};
                    border: 1px solid {accent_pressed};
                }}
                DatasetConfigWidget QLabel {{
                    color: #ffffff;
                    background-color: transparent;
                    padding: 2px;
                }}
            """)
            if hasattr(self.dataset_widget, "commodity_combo"):
                self.dataset_widget.commodity_combo.setStyleSheet(f"""
                    QComboBox {{
                        background-color: #2b2b2b;
                        border: 2px solid {accent};
                        border-radius: 5px;
                        padding: 6px;
                        color: #ffffff;
                        min-height: 20px;
                        font-size: 11pt;
                        font-weight: 700;
                    }}
                    QComboBox:hover {{
                        border: 2px solid {accent_hover};
                    }}
                    QComboBox QAbstractItemView {{
                        background-color: #252525;
                        color: #ffffff;
                        selection-background-color: {accent};
                    }}
                """)
        if hasattr(self, "canvas"):
            if hasattr(self.canvas, "commodity"):
                self.canvas.commodity = theme["commodity"]
            if hasattr(self.canvas, "set_theme"):
                try:
                    self.canvas.set_theme(accent_hex=accent, commodity=theme["commodity"])
                except Exception:
                    pass

        base_title = f"GeoCore Analytics Studio – 3D Terrain Visualization ({getattr(self, 'viewer_type', 'Initializing')})"
        mode_title = f" [{theme['name']} MODE]"
        if getattr(self, "current_project", None):
            self.setWindowTitle(f"{base_title}{mode_title} - {self.current_project.name}")
        else:
            self.setWindowTitle(f"{base_title}{mode_title}")

        if hasattr(self, "status") and self.status is not None:
            self.status.showMessage(f"{theme['name']} mode active", 2500)

    def _init_palette(self):
        self._apply_commodity_theme("copper")

    def _create_log_dock(self):
        self.log_edit = QtWidgets.QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setFont(QtGui.QFont("Consolas", 9))
        self.log_edit.setStyleSheet(
            "QPlainTextEdit { background:#111111; color:#00ff9a; border:1px solid #333; }"
        )

        cont = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(cont)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.addWidget(self.log_edit)

        dock = QtWidgets.QDockWidget("Command Log", self)
        dock.setWidget(cont)
        dock.setAllowedAreas(Qt.BottomDockWidgetArea | Qt.LeftDockWidgetArea)
        self.addDockWidget(Qt.BottomDockWidgetArea, dock)

    def _create_project_dock(self):
        self.fs_model = QtWidgets.QFileSystemModel()
        base_path = str(Path(DEFAULT_BASE).parent)
        self.fs_model.setRootPath(base_path)

        tree = QtWidgets.QTreeView()
        tree.setModel(self.fs_model)
        tree.setRootIndex(self.fs_model.index(base_path))
        tree.setHeaderHidden(True)
        tree.setAnimated(True)

        dock = QtWidgets.QDockWidget("Project Explorer", self)
        dock.setWidget(tree)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)

    def _create_dataset_dock(self):
        self.dataset_widget = DatasetConfigWidget()
        self.dataset_widget.config_changed.connect(self._dataset_changed)
        
        # Wrap in scroll area to handle overflow
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(self.dataset_widget)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setStyleSheet("""
            QScrollArea {
                background-color: #2b2b2b;
                border: none;
            }
            QScrollBar:vertical {
                background-color: #2b2b2b;
                width: 14px;
                border: none;
            }
            QScrollBar::handle:vertical {
                background-color: #555;
                border-radius: 7px;
                min-height: 30px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #666;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)

        dock = QtWidgets.QDockWidget("Dataset Config", self)
        dock.setWidget(scroll)  # Use scroll area instead of widget directly
        dock.setAllowedAreas(Qt.RightDockWidgetArea)
        dock.setMinimumWidth(450)  # Wider minimum
        dock.setMaximumWidth(550)  # Wider maximum
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
    
    def _create_model_metrics_dock(self):
        """Create model metrics panel."""
        try:
            from model_metrics_panel import ModelMetricsPanel
            self.metrics_panel = ModelMetricsPanel()
            
            dock = QtWidgets.QDockWidget("Model Metrics", self)
            dock.setWidget(self.metrics_panel)
            dock.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)
            dock.setMinimumWidth(320)
            dock.setMaximumWidth(400)
            self.addDockWidget(Qt.RightDockWidgetArea, dock)
            
            # Hide by default to reduce clutter
            dock.hide()
            self.metrics_dock = dock
            
        except Exception as e:
            print(f"[DOCK] Model metrics dock error: {e}")
            # Create placeholder
            placeholder = QLabel("Model Metrics\n(Performance & Statistics)")
            placeholder.setAlignment(Qt.AlignCenter)
            placeholder.setStyleSheet("color: #666; padding: 20px;")
            
            dock = QtWidgets.QDockWidget("Model Metrics", self)
            dock.setWidget(placeholder)
            dock.hide()
            self.metrics_dock = dock
            self.addDockWidget(Qt.RightDockWidgetArea, dock)

    def _create_collaboration_dock(self):
        """Create collaboration dock with team features."""
        if self.user_manager.is_authenticated() and self.current_project:
            self.collaboration_panel = CollaborationPanel(
                str(self.current_project.id), 
                self.user_manager.get_current_user()
            )
        else:
            # Placeholder when not authenticated or no project
            self.collaboration_panel = QLabel("Login and open a project to access collaboration features")
            self.collaboration_panel.setAlignment(Qt.AlignCenter)
            self.collaboration_panel.setStyleSheet("color: #bbbbbb; font-style: italic; padding: 20px;")
            self.collaboration_panel.setWordWrap(True)

        dock = QtWidgets.QDockWidget("Team Collaboration", self)
        dock.setWidget(self.collaboration_panel)
        dock.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)
        dock.setMinimumWidth(320)
        dock.setMaximumWidth(400)
        
        # Hide by default to reduce clutter
        dock.hide()
        self.collaboration_dock = dock
        
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

    def _create_field_data_dock(self):
        """Create field data collection dock."""
        try:
            from mobile_integration import FieldDataWidget
            self.field_data_widget = FieldDataWidget()
            
            dock = QtWidgets.QDockWidget("Field Data Collection", self)
            dock.setWidget(self.field_data_widget)
            self.addDockWidget(Qt.RightDockWidgetArea, dock)
        except Exception as e:
            print(f"[DOCK] Field data dock error: {e}")
            # Create placeholder
            placeholder = QLabel("Field data collection\n(Mobile integration)")
            placeholder.setAlignment(Qt.AlignCenter)
            placeholder.setStyleSheet("color: #666; padding: 20px;")
            
            dock = QtWidgets.QDockWidget("Field Data Collection", self)
            dock.setWidget(placeholder)
            self.addDockWidget(Qt.RightDockWidgetArea, dock)

    def _create_security_dock(self):
        """Create security settings dock."""
        try:
            from enterprise_security import SecurityWidget
            self.security_widget = SecurityWidget()
            
            dock = QtWidgets.QDockWidget("Security Settings", self)
            dock.setWidget(self.security_widget)
            self.addDockWidget(Qt.RightDockWidgetArea, dock)
        except Exception as e:
            print(f"[DOCK] Security dock error: {e}")
            # Create placeholder
            placeholder = QLabel("Security Settings\n(2FA & Audit Logs)")
            placeholder.setAlignment(Qt.AlignCenter)
            placeholder.setStyleSheet("color: #666; padding: 20px;")
            
            dock = QtWidgets.QDockWidget("Security Settings", self)
            dock.setWidget(placeholder)
            self.addDockWidget(Qt.RightDockWidgetArea, dock)

    def _create_actions(self):
        self.act_run = QtWidgets.QAction("Run Analysis", self)
        self.act_run.triggered.connect(self.on_run)
        
        # Project management actions
        self.act_new_project = QtWidgets.QAction("New Project", self)
        self.act_new_project.setShortcut("Ctrl+N")
        self.act_new_project.triggered.connect(self.new_project)
        
        self.act_open_project_manager = QtWidgets.QAction("Project Manager", self)
        self.act_open_project_manager.setShortcut("Ctrl+P")
        self.act_open_project_manager.triggered.connect(self.open_project_manager)
        
        self.act_import_data = QtWidgets.QAction("Import Data", self)
        self.act_import_data.setShortcut("Ctrl+I")
        self.act_import_data.triggered.connect(self.open_data_import)

        # SaaS features
        self.act_login = QtWidgets.QAction("Login", self)
        self.act_login.triggered.connect(self.login_user)
        
        self.act_logout = QtWidgets.QAction("Logout", self)
        self.act_logout.triggered.connect(self.logout_user)
        
        self.act_generate_report = QtWidgets.QAction("Generate Report", self)
        self.act_generate_report.setShortcut("Ctrl+R")
        self.act_generate_report.triggered.connect(self.generate_report)
        
        self.act_cloud_sync = QtWidgets.QAction("Cloud Sync", self)
        self.act_cloud_sync.triggered.connect(self.cloud_sync)
        
        self.act_cloud_settings = QtWidgets.QAction("Cloud Settings", self)
        self.act_cloud_settings.triggered.connect(self.cloud_settings)
        
        self.act_subscription = QtWidgets.QAction("Subscription", self)
        self.act_subscription.triggered.connect(self.show_subscription)
        
        # Enterprise features
        self.act_train_model = QtWidgets.QAction("Train Custom Model", self)
        self.act_train_model.triggered.connect(self.train_custom_model)
        
        self.act_security_settings = QtWidgets.QAction("Security Settings", self)
        self.act_security_settings.triggered.connect(self.show_security_settings)

        self.act_open_results = QtWidgets.QAction("Open Results Folder", self)
        self.act_open_results.triggered.connect(self.on_open_results)

        self.act_reload_map = QtWidgets.QAction("Reload Map", self)
        self.act_reload_map.triggered.connect(self.reload_map)

        self.act_open_in_qgis = QtWidgets.QAction("Open in QGIS", self)
        self.act_open_in_qgis.triggered.connect(self.on_open_in_qgis)

        self.act_choose_python = QtWidgets.QAction("Set Python…", self)
        self.act_choose_python.triggered.connect(self.on_choose_python)

        self.act_choose_core = QtWidgets.QAction("Set Core Script…", self)
        self.act_choose_core.triggered.connect(self.on_choose_core)

        self.act_open_project = QtWidgets.QAction("Open Project…", self)
        self.act_open_project.triggered.connect(self.on_open_project)

        self.act_save_project = QtWidgets.QAction("Save Project As…", self)
        self.act_save_project.triggered.connect(self.on_save_project)

        self.act_zoom_full = QtWidgets.QAction("Reset Camera", self)
        self.act_zoom_full.triggered.connect(self.on_zoom_full)
    def _create_field_data_dock(self):
        """Create field data collection dock."""
        try:
            from mobile_integration import FieldDataWidget
            self.field_data_widget = FieldDataWidget()

            dock = QtWidgets.QDockWidget("Field Data Collection", self)
            dock.setWidget(self.field_data_widget)
            self.addDockWidget(Qt.RightDockWidgetArea, dock)
        except Exception as e:
            print(f"[DOCK] Field data dock error: {e}")
            # Create placeholder
            placeholder = QLabel("Field data collection\n(Mobile integration)")
            placeholder.setAlignment(Qt.AlignCenter)
            placeholder.setStyleSheet("color: #666; padding: 20px;")

            dock = QtWidgets.QDockWidget("Field Data Collection", self)
            dock.setWidget(placeholder)
            self.addDockWidget(Qt.RightDockWidgetArea, dock)

    def _create_web_dashboard_dock(self):
        """Create web dashboard dock."""
        try:
            from web_dashboard import WebDashboardWidget
            self.web_dashboard_widget = WebDashboardWidget()

            dock = QtWidgets.QDockWidget("Web Dashboard", self)
            dock.setWidget(self.web_dashboard_widget)
            dock.setMinimumWidth(320)
            dock.setMaximumWidth(400)
            
            # Hide by default - accessible via menu
            dock.hide()
            self.web_dashboard_dock = dock
            
            self.addDockWidget(Qt.RightDockWidgetArea, dock)
        except Exception as e:
            print(f"[DOCK] Web dashboard dock error: {e}")
            # Create placeholder
            placeholder = QLabel("Web Dashboard\n(API & Metrics)")
            placeholder.setAlignment(Qt.AlignCenter)
            placeholder.setStyleSheet("color: #bbbbbb; padding: 20px;")
            placeholder.setWordWrap(True)

            dock = QtWidgets.QDockWidget("Web Dashboard", self)
            dock.setWidget(placeholder)
            dock.setMinimumWidth(320)
            dock.setMaximumWidth(400)
            dock.hide()
            self.web_dashboard_dock = dock
            self.addDockWidget(Qt.RightDockWidgetArea, dock)

    def _create_security_dock(self):
        """Create security settings dock."""
        try:
            from enterprise_security import SecurityWidget
            self.security_widget = SecurityWidget()

            dock = QtWidgets.QDockWidget("Security Settings", self)
            dock.setWidget(self.security_widget)
            self.addDockWidget(Qt.RightDockWidgetArea, dock)
        except Exception as e:
            print(f"[DOCK] Security dock error: {e}")
            # Create placeholder
            placeholder = QLabel("Security Settings\n(2FA & Audit Logs)")
            placeholder.setAlignment(Qt.AlignCenter)
            placeholder.setStyleSheet("color: #666; padding: 20px;")

            dock = QtWidgets.QDockWidget("Security Settings", self)
            dock.setWidget(placeholder)
            self.addDockWidget(Qt.RightDockWidgetArea, dock)


    def _create_menubar(self):
        mb = self.menuBar()

        # User menu
        user_menu = mb.addMenu("&User")
        if self.user_manager.is_authenticated():
            user_info = QtWidgets.QAction(f"Logged in as: {self.user_manager.get_current_user().username}", self)
            user_info.setEnabled(False)
            user_menu.addAction(user_info)
            user_menu.addSeparator()
            user_menu.addAction(self.act_subscription)
            user_menu.addSeparator()
            user_menu.addAction(self.act_logout)
        else:
            user_menu.addAction(self.act_login)

        # Project menu
        project_menu = mb.addMenu("&Project")
        project_menu.addAction(self.act_new_project)
        project_menu.addAction(self.act_open_project_manager)
        project_menu.addSeparator()
        project_menu.addAction(self.act_import_data)
        project_menu.addSeparator()
        project_menu.addAction(self.act_open_project)
        project_menu.addAction(self.act_save_project)

        # Analysis menu
        analysis_menu = mb.addMenu("&Analysis")
        analysis_menu.addAction(self.act_run)
        analysis_menu.addAction(self.act_open_results)
        analysis_menu.addSeparator()
        analysis_menu.addAction(self.act_train_model)
        analysis_menu.addSeparator()
        analysis_menu.addAction(self.act_open_in_qgis)

        # Reports menu
        reports_menu = mb.addMenu("&Reports")
        reports_menu.addAction(self.act_generate_report)

        # Cloud menu
        cloud_menu = mb.addMenu("&Cloud")
        cloud_menu.addAction(self.act_cloud_sync)
        cloud_menu.addAction(self.act_cloud_settings)

        # View menu
        view_menu = mb.addMenu("&View")
        
        # Add Model Development action
        act_model_dev = QtWidgets.QAction("Model Development", self)
        act_model_dev.triggered.connect(self.show_model_development)
        view_menu.addAction(act_model_dev)
        
        view_menu.addSeparator()
        
        for dock in self.findChildren(QtWidgets.QDockWidget):
            view_menu.addAction(dock.toggleViewAction())
        view_menu.addSeparator()
        view_menu.addAction(self.act_zoom_full)
        view_menu.addAction(self.act_reload_map)

        # Settings menu
        settings_menu = mb.addMenu("&Settings")
        settings_menu.addAction(self.act_choose_python)
        settings_menu.addAction(self.act_choose_core)
        settings_menu.addSeparator()
        settings_menu.addAction(self.act_security_settings)

        # Help menu
        help_menu = mb.addMenu("&Help")
        about_act = QtWidgets.QAction("About GeoCore Analytics", self)
        about_act.triggered.connect(self.on_about)
        help_menu.addAction(about_act)
        
        # Exit action
        exit_act = QtWidgets.QAction("Exit", self)
        exit_act.setShortcut("Ctrl+Q")
        exit_act.triggered.connect(self.close)
        project_menu.addSeparator()
        project_menu.addAction(exit_act)

    def _create_toolbar(self):
        tb = QtWidgets.QToolBar("Main")
        tb.setMovable(False)
        tb.setIconSize(QtCore.QSize(18, 18))
        self.addToolBar(Qt.TopToolBarArea, tb)

        tb.addAction(self.act_run)
        tb.addAction(self.act_open_results)
        tb.addAction(self.act_reload_map)
        tb.addAction(self.act_open_in_qgis)
        tb.addSeparator()
        tb.addAction(self.act_zoom_full)
        tb.addSeparator()
        tb.addAction(self.act_choose_python)
        tb.addAction(self.act_choose_core)

    def append_log(self, text: str):
        self.log_edit.appendPlainText(text)
        vbar = self.log_edit.verticalScrollBar()
        vbar.setValue(vbar.maximum())

    def _dataset_changed(self):
        cfg = self.dataset_widget.get_config()
        self._apply_commodity_theme(cfg.get("commodity", "copper"))
        self.status.showMessage(
            f"Config updated ({cfg.get('commodity', 'copper').upper()} mode • results: {cfg['results']})",
            3000,
        )

    def on_zoom_full(self):
        self.canvas.reset_camera()
        self.append_log("[INFO] Reset 3D camera view")

    def on_run(self):
        if self.worker is not None and self.worker.isRunning():
            QtWidgets.QMessageBox.information(
                self, "OreInsight", "Pipeline is already running."
            )
            return

        if not os.path.exists(self.python_exe):
            QtWidgets.QMessageBox.warning(
                self, "Python not found", f"Python executable not found:\n{self.python_exe}"
            )
            return

        if not os.path.exists(self.core_script):
            QtWidgets.QMessageBox.warning(
                self, "Core script not found", f"Core script not found:\n{self.core_script}"
            )
            return
        
        cfg = self.dataset_widget.get_config()
        env = os.environ.copy()
        env["ORE_BASE_DIR"] = str(Path(cfg["results"]).parent)
        env["ORE_STACK_PATH"] = cfg["stack"]
        env["ORE_COMMODITY"] = cfg.get("commodity", "copper")
        self.append_log(f"[CONFIG] Commodity: {cfg.get('commodity', 'copper').upper()}")
        self._apply_commodity_theme(cfg.get("commodity", "copper"))
        # Set commodity on 3D viewer for click display labels
        if hasattr(self, 'canvas') and hasattr(self.canvas, 'commodity'):
            self.canvas.commodity = cfg.get("commodity", "copper")
            if hasattr(self.canvas, 'set_theme'):
                try:
                    theme = self._theme_for_commodity(cfg.get("commodity", "copper"))
                    self.canvas.set_theme(accent_hex=theme["accent"], commodity=theme["commodity"])
                except Exception:
                    pass
        env["ORE_REFERENCE_RASTER"] = cfg["ref"]  # Use the DEM selected in Dataset Config
        env["ORE_MRDS_CSV"] = cfg["mrds"]
        env["ORE_DEPOSIT_CSV"] = cfg["mrds"]  # Use the selected MRDS/REE CSV directly as deposit labels when available
        env["ORE_GEOLOGY_SHP"] = cfg["geo"]
        env["ORE_RESULTS_DIR"] = cfg["results"]
        env["ORE_ANALYSIS_CROP"] = "full" if cfg.get("commodity", "copper") == "ree" else "center"
        
        # Pass feature layers as environment variables
        feature_layers = cfg.get("feature_layers", {})
        for key, path in feature_layers.items():
            if path:  # Only set if path is not empty
                env[f"ORE_FEATURE_{key.upper()}"] = path
                self.append_log(f"[FEATURE] {key}: {path}")

        self.append_log(f"\n[DEBUG] DEFAULT_REF = {DEFAULT_REF}")
        self.append_log(f"[DEBUG] User selected REF = {cfg['ref']}")
        self.append_log(f"[DEBUG] Setting ORE_REFERENCE_RASTER to: {env['ORE_REFERENCE_RASTER']}")
        self.append_log("\n[INFO] Starting OreInsight v4 ML pipeline (Real Data, Machine Learning)…")
        self.status.showMessage("Running v4 ML analysis…")

        self.worker = PipelineWorker(self.python_exe, self.core_script, env, self)
        self.worker.log_line.connect(self.append_log)
        self.worker.finished.connect(self.on_pipeline_finished)
        self.worker.progress.connect(self._update_inline_progress)

        # Show inline progress bar and start ETA tracking
        import time
        self._analysis_start_time = time.time()
        self.inline_progress_bar.setValue(0)
        self._inline_target = 3
        self.inline_progress_container.show()
        self._inline_timer.start()
        self._update_inline_progress(3, "Starting ML analysis engine…")
        
        # Disable Run Analysis while running
        for action in self.findChildren(QtWidgets.QAction):
            if action.text() == "Run Analysis":
                action.setEnabled(False)
                self._run_action = action

        self.worker.start()

    def on_pipeline_finished(self, rc: int):
        # Stop inline progress animation
        self._inline_timer.stop()
        
        if rc == 0:
            self._update_inline_progress(100, "Analysis complete")
            # Hide progress bar after a delay
            QtCore.QTimer.singleShot(3000, self.inline_progress_container.hide)
        else:
            self.inline_progress_label.setText("Analysis failed")
            self.inline_progress_label.setStyleSheet("color: #ff4444; font-size: 10pt; background: transparent;")
            self.inline_eta_label.setText("")
            QtCore.QTimer.singleShot(5000, self.inline_progress_container.hide)
        
        # Re-enable Run Analysis button
        if hasattr(self, '_run_action'):
            self._run_action.setEnabled(True)

        if rc == 0:
            self.append_log("[INFO] Pipeline finished successfully.")
            self.status.showMessage("Pipeline completed successfully.")
        else:
            self.append_log(f"[ERROR] Pipeline exited with code {rc}.")
            self.status.showMessage(f"Pipeline error (code {rc}).")

        self.reload_map()

    def reload_map(self):
        """Load terrain into 3D viewer."""
        cfg = self.dataset_widget.get_config()
        results_dir = Path(cfg["results"])
        
        # Try v4 first (new), then fall back to v3 (old)
        prob_tif = results_dir / "oreinsight_v4_probability.tif"
        if not prob_tif.exists():
            prob_tif = results_dir / "oreinsight_v3_probability.tif"
        
        # Use cropped DEM if available (matches probability map size)
        dem_cropped = results_dir / "oreinsight_v4_dem_cropped.tif"
        print(f"[DEBUG] Checking for cropped DEM: {dem_cropped}")
        print(f"[DEBUG] Cropped DEM exists: {dem_cropped.exists()}")
        
        if dem_cropped.exists():
            ref_tif = dem_cropped
            print(f"[DEBUG] Using cropped DEM: {ref_tif}")
        else:
            ref_tif = Path(cfg["ref"])
            print(f"[DEBUG] Using full DEM: {ref_tif}")

        if not ref_tif.exists():
            self.append_log("[WARN] Reference DEM not found")
            self.map_path_label.setText("DEM not found")
            return

        # Load terrain with probability overlay
        try:
            if hasattr(self.canvas, 'load_terrain'):
                if hasattr(self.canvas, 'commodity'):
                    self.canvas.commodity = cfg.get("commodity", "copper")
                if hasattr(self.canvas, 'set_theme'):
                    try:
                        theme = self._theme_for_commodity(cfg.get("commodity", "copper"))
                        self.canvas.set_theme(accent_hex=theme["accent"], commodity=theme["commodity"])
                    except Exception:
                        pass
                if prob_tif.exists():
                    print(f"[DEBUG] Loading terrain: DEM={ref_tif}, Prob={prob_tif}")
                    self.canvas.load_terrain(str(ref_tif), str(prob_tif))
                    version = "v4" if "v4" in prob_tif.name else "v3"
                    area_note = " (analysis area)" if "cropped" in str(ref_tif) else ""
                    self.append_log(f"[3D] Loaded {version} terrain with probability overlay{area_note} ({self.viewer_type})")
                    self.map_path_label.setText(f"3D Terrain: {ref_tif.name} + {prob_tif.name}")
                    self.status.showMessage("3D terrain loaded successfully", 5000)
                    
                    # Show helpful explanation dialog (only for v4 results)
                    if version == "v4" and not self.settings.value("hide_3d_help", False):
                        self._show_3d_help_dialog()
                else:
                    self.canvas.load_terrain(str(ref_tif))
                    self.append_log(f"[3D] Loaded terrain (no probability data yet) ({self.viewer_type})")
                    self.map_path_label.setText(f"3D Terrain: {ref_tif.name}")
                    self.status.showMessage("3D terrain loaded (run analysis for probability)", 5000)
            else:
                self.append_log("[ERROR] 3D viewer not properly initialized")
                self.map_path_label.setText("3D viewer initialization failed")
        except Exception as e:
            self.append_log(f"[ERROR] Failed to load 3D terrain: {e}")
            self.map_path_label.setText("Failed to load terrain")
            import traceback
            traceback.print_exc()
            self.map_path_label.setText("Failed to load terrain")
            import traceback
            traceback.print_exc()
    
    def show_model_development(self):
        """Show the Model Development panel."""
        try:
            # First check if model file exists
            model_path = "first/results_v3/oreinsight_v4_model.pkl"
            validation_path = "first/results_v3/oreinsight_v4_validation.txt"
            
            if not os.path.exists(model_path):
                QtWidgets.QMessageBox.warning(
                    self,
                    "No Model Found",
                    "Model file not found. Please run the analysis first.\n\n"
                    "Click 'Run Analysis' to generate the model."
                )
                return
            
            # Try simple text display first (safer than loading pickle)
            if os.path.exists(validation_path):
                try:
                    with open(validation_path, 'r') as f:
                        report_text = f.read()
                    
                    # Create simple text dialog
                    dialog = QtWidgets.QDialog(self)
                    dialog.setWindowTitle("Model Development - Validation Report")
                    dialog.resize(800, 600)
                    dialog.setStyleSheet("background-color: #2d2d2d;")
                    
                    layout = QVBoxLayout(dialog)
                    
                    text_edit = QtWidgets.QTextEdit()
                    text_edit.setReadOnly(True)
                    text_edit.setPlainText(report_text)
                    text_edit.setStyleSheet("""
                        QTextEdit {
                            background-color: #1e1e1e;
                            color: #d4d4d4;
                            font-family: 'Consolas', 'Courier New', monospace;
                            font-size: 10pt;
                            border: 1px solid #555;
                            padding: 10px;
                        }
                    """)
                    
                    layout.addWidget(text_edit)
                    
                    close_btn = QtWidgets.QPushButton("Close")
                    close_btn.clicked.connect(dialog.close)
                    close_btn.setStyleSheet("""
                        QPushButton {
                            background-color: #0d7377;
                            color: white;
                            border: none;
                            padding: 8px 16px;
                            border-radius: 4px;
                        }
                        QPushButton:hover {
                            background-color: #14a085;
                        }
                    """)
                    layout.addWidget(close_btn)
                    
                    dialog.exec_()
                    return
                    
                except Exception as text_error:
                    print(f"[ERROR] Failed to show text report: {text_error}")
                    # Fall through to try the full panel
            
            # If text display failed, try the full panel (but this might crash)
            from model_development_panel import ModelDevelopmentPanel
            
            # Create dialog
            dialog = QtWidgets.QDialog(self)
            dialog.setWindowTitle("Model Development")
            dialog.resize(900, 700)
            dialog.setStyleSheet("background-color: #2d2d2d;")
            
            layout = QVBoxLayout(dialog)
            layout.setContentsMargins(0, 0, 0, 0)
            
            # Add panel
            panel = ModelDevelopmentPanel(dialog)
            layout.addWidget(panel)
            
            # Try to load model info with error handling
            try:
                panel.load_model_info()
            except Exception as load_error:
                print(f"[ERROR] Failed to load model info: {load_error}")
                import traceback
                traceback.print_exc()
                QtWidgets.QMessageBox.warning(
                    dialog,
                    "Load Error",
                    f"Model file exists but failed to load:\n\n{load_error}\n\n"
                    "The model file may be corrupted. Try running the analysis again."
                )
                dialog.close()
                return
            
            dialog.exec_()
        except Exception as e:
            print(f"[ERROR] Failed to open Model Development: {e}")
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.warning(
                self,
                "Error",
                f"Failed to open Model Development panel:\n\n{e}\n\nMake sure you've run the analysis first."
            )

    def on_open_results(self):
        cfg = self.dataset_widget.get_config()
        folder = cfg["results"]
        if os.path.exists(folder):
            os.startfile(folder)
        else:
            QtWidgets.QMessageBox.warning(
                self, "Results folder missing", f"Cannot open results folder:\n{folder}"
            )

    def on_open_in_qgis(self):
        if not os.path.exists(QGIS_EXE):
            QtWidgets.QMessageBox.warning(
                self, "QGIS not found", f"QGIS executable not found:\n{QGIS_EXE}"
            )
            return

        cfg = self.dataset_widget.get_config()
        results_dir = Path(cfg["results"])
        prob_tif = results_dir / "oreinsight_v6_prob.tif"

        if not prob_tif.exists():
            QtWidgets.QMessageBox.warning(
                self, "No probability map", f"Could not find probability GeoTIFF:\n{prob_tif}"
            )
            return

        try:
            subprocess.Popen([QGIS_EXE, str(prob_tif)])
            self.append_log(f"[INFO] Launched QGIS with {prob_tif}")
            self.status.showMessage("Opened probability map in QGIS.", 5000)
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Error launching QGIS", f"Failed to launch QGIS:\n{e}"
            )
            self.append_log(f"[ERROR] Failed to launch QGIS: {e}")

    def on_choose_python(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Python executable")
        if path:
            self.python_exe = path
            self.settings.setValue("python_exe", path)
            self.status.showMessage(f"Python set to: {path}", 4000)

    def on_choose_core(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select OreInsight core script", filter="Python files (*.py)"
        )
        if path:
            self.core_script = path
            self.settings.setValue("core_script", path)
            self.status.showMessage(f"Core script set to: {path}", 4000)

    def on_open_project(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open OreInsight Project", filter="OreInsight Project (*.oreproj)"
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to read project:\n{e}")
            return

        self.python_exe = data.get("python_exe", self.python_exe)
        self.core_script = data.get("core_script", self.core_script)
        cfg = data.get("config", {})
        self.dataset_widget.apply_config(cfg)
        self.status.showMessage(f"Loaded project: {path}", 5000)

    def on_save_project(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save OreInsight Project", filter="OreInsight Project (*.oreproj)"
        )
        if not path:
            return
        if not path.lower().endswith(".oreproj"):
            path += ".oreproj"

        data = {
            "python_exe": self.python_exe,
            "core_script": self.core_script,
            "config": self.dataset_widget.get_config(),
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save project:\n{e}")
            return

        self.status.showMessage(f"Saved project: {path}", 5000)

    def on_about(self):
        QtWidgets.QMessageBox.information(
            self,
            "About OreInsight Studio",
            "OreInsight Studio v2.0\n\n"
            "QGIS-powered desktop interface for OreInsight v6 pipeline.\n"
            "Interactive GIS map, docked layout, dataset config panel,\n"
            "project files, and background TensorFlow runs.\n\n"
            "Now with embedded QGIS for true GIS visualization!",
        )

    def closeEvent(self, event):
        self.settings.setValue("python_exe", self.python_exe)
        self.settings.setValue("core_script", self.core_script)
        super().closeEvent(event)


# =====================================================================
# Launch with QGIS initialization
# =====================================================================

def launch_main_window(app, qgs_app):
    win = GeoCoreAnalyticsMainWindow()
    win.show()
    app.main_window = win
    app.qgs_app = qgs_app
    return win


if __name__ == "__main__":
    print("[STARTUP] GeoCore Analytics Studio v2.0 - 3D Edition")
    
    # Single instance check using lock file
    import tempfile
    lock_file = os.path.join(tempfile.gettempdir(), "geocore_analytics.lock")
    
    # Try to create lock file
    try:
        if os.path.exists(lock_file):
            # Check if the process is actually running
            try:
                with open(lock_file, 'r') as f:
                    old_pid = int(f.read().strip())
                
                # Check if process exists (Windows-specific)
                import subprocess
                result = subprocess.run(['tasklist', '/FI', f'PID eq {old_pid}'], 
                                      capture_output=True, text=True)
                
                if str(old_pid) in result.stdout:
                    print(f"[ERROR] GeoCore Analytics is already running (PID: {old_pid})")
                    print("[ERROR] Please close the existing instance first")
                    
                    # Show message box
                    app = QApplication(sys.argv)
                    QMessageBox.warning(
                        None,
                        "Already Running",
                        "GeoCore Analytics is already running!\n\n"
                        "Please close the existing instance before starting a new one."
                    )
                    sys.exit(1)
                else:
                    # Stale lock file, remove it
                    os.remove(lock_file)
            except Exception as e:
                # If error checking, assume stale and remove
                print(f"[WARN] Error checking lock file: {e}")
                if os.path.exists(lock_file):
                    os.remove(lock_file)
        
        # Create lock file with current PID
        with open(lock_file, 'w') as f:
            f.write(str(os.getpid()))
        
    except Exception as e:
        print(f"[WARN] Could not create lock file: {e}")
    
    try:
        # CRITICAL: Setup QGIS environment BEFORE creating Qt application
        from qgis_init import setup_qgis_environment, initialize_qgis_application, cleanup_qgis_application
        
        print("[STEP 1/6] Setting up QGIS environment (for GeoTIFF support)...")
        setup_qgis_environment()
        
        # Create Qt application AFTER QGIS environment is set
        app = QApplication(sys.argv)

        # Show splash immediately
        splash = StartupSplash()
        splash.show()
        splash.update_progress(5, "Initializing application...")
        
        screen = app.primaryScreen()
        if screen is not None:
            geo = screen.availableGeometry()
            x = geo.center().x() - splash.width() // 2
            y = geo.center().y() - splash.height() // 2
            splash.move(x, y)
        
        splash.update_progress(25, "Loading GDAL libraries...")

        # Initialize QGIS (needed for rasterio/GDAL)
        print("[STEP 2/6] Initializing QGIS...")
        splash.update_progress(35, "Initializing QGIS core...")
        qgs_app = initialize_qgis_application(gui_flag=False)  # No GUI, just libraries
        splash.update_progress(50, "Loading raster processing modules...")
        
        splash.update_progress(60, "Initializing VTK 3D engine...")
        splash.update_progress(70, "Setting up user interface...")

        def on_splash_finished():
            # Safety check: prevent multiple calls
            if hasattr(on_splash_finished, '_called'):
                print("[WARN] on_splash_finished already called, ignoring duplicate")
                return
            on_splash_finished._called = True
            
            print("[STEP 3/6] Launching main window with 3D viewer...")
            splash.update_progress(75, "Creating main window...")
            
            main_window = launch_main_window(app, qgs_app)
            
            splash.update_progress(85, "Initializing 3D terrain viewer...")
            splash.update_progress(95, "Finalizing startup...")
            splash.update_progress(100, "Ready!")
            
            # Ensure main window is visible and raised
            main_window.show()
            main_window.raise_()
            main_window.activateWindow()
            
            # Close splash IMMEDIATELY before any dialogs
            splash.close()
            splash.deleteLater()
            
            # Show login dialog after splash is gone
            if not main_window.user_manager.is_authenticated():
                print("[AUTH] Showing login dialog...")
                try:
                    if not main_window.user_manager.login(main_window):
                        print("[AUTH] Login cancelled, continuing as guest")
                except Exception as e:
                    print(f"[AUTH] Login error: {e}")
                    print("[AUTH] Continuing in guest mode")
            
            # Ensure main window stays visible
            main_window.show()
            main_window.raise_()
            print("[MAIN] Main window should now be visible")

        splash.finished.connect(on_splash_finished)
        
        # Trigger the finished signal after a short delay to let UI update
        QTimer.singleShot(100, splash.finished.emit)

        # Run application
        print("[STEP 6/6] Running application event loop...")
        exit_code = app.exec_()

        # Cleanup QGIS
        print("[SHUTDOWN] Cleaning up QGIS...")
        cleanup_qgis_application(qgs_app)
        
        # Remove lock file
        try:
            if os.path.exists(lock_file):
                os.remove(lock_file)
                print("[SHUTDOWN] Lock file removed")
        except:
            pass

        print("[DONE] Application exited cleanly")
        sys.exit(exit_code)
        
    except Exception as e:
        print(f"[ERROR] Failed to start application: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to start without QGIS as fallback
        print("[FALLBACK] Attempting to start without QGIS...")
        try:
            app = QApplication(sys.argv)
            win = GeoCoreAnalyticsMainWindow()
            win.show()
            sys.exit(app.exec_())
        except Exception as e2:
            print(f"[FATAL] Fallback also failed: {e2}")
            sys.exit(1)


            #FIX RASTER