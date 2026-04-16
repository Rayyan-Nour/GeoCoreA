"""
3D Terrain Viewer using VTK (included with QGIS)
Simpler version that works with QGIS's Python environment.
"""

import os
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QSlider, QPushButton, QLabel

try:
    from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
    import vtkmodules.all as vtk
    VTK_AVAILABLE = True
except ImportError:
    QVTKRenderWindowInteractor = None
    vtk = None
    VTK_AVAILABLE = False
    print("[3D] VTK not available, using placeholder")


if VTK_AVAILABLE:
    _TerrainStyleBase = vtk.vtkInteractorStyleTrackballCamera
else:
    class _TerrainStyleBase(object):
        def __init__(self, *args, **kwargs):
            pass


class ImprovedTerrainStyle(_TerrainStyleBase):
    """
    Improved camera style with reliable zoom-to-cursor behavior.
    """

    def __init__(self):
        super().__init__()
        self.default_renderer = None
        self.AddObserver("InteractionEvent", self.constrain_camera)
        self.AddObserver("MouseWheelForwardEvent", self.zoom_forward)
        self.AddObserver("MouseWheelBackwardEvent", self.zoom_backward)
        self.SetMotionFactor(5.0)

    def set_default_renderer(self, renderer):
        self.default_renderer = renderer
        if hasattr(self, "SetDefaultRenderer"):
            self.SetDefaultRenderer(renderer)

    def _get_renderer(self):
        renderer = self.GetCurrentRenderer()
        if renderer is None:
            renderer = self.default_renderer
        return renderer

    def zoom_forward(self, obj, event):
        """Zoom in toward mouse position."""
        self.zoom_to_mouse(0.85)

    def zoom_backward(self, obj, event):
        """Zoom out from mouse position."""
        self.zoom_to_mouse(1.18)

    def zoom_to_mouse(self, factor):
        """Zoom toward/away from mouse cursor position."""
        renderer = self._get_renderer()
        interactor = self.GetInteractor()

        if renderer is None or interactor is None:
            return

        camera = renderer.GetActiveCamera()
        mouse_pos = interactor.GetEventPosition()

        # Use CellPicker instead of WorldPointPicker. It is much more reliable
        # for structured terrain grids and actually locks to the surface.
        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.0005)
        picker.Pick(mouse_pos[0], mouse_pos[1], 0, renderer)

        cam_pos = np.array(camera.GetPosition(), dtype=float)
        focal_point = np.array(camera.GetFocalPoint(), dtype=float)

        if picker.GetCellId() >= 0:
            target = np.array(picker.GetPickPosition(), dtype=float)
        else:
            target = focal_point

        direction = cam_pos - target
        new_pos = target + direction * factor

        # Prevent degenerate camera collapse onto the target.
        if np.linalg.norm(new_pos - target) < 1e-6:
            return

        camera.SetPosition(float(new_pos[0]), float(new_pos[1]), float(new_pos[2]))
        camera.SetFocalPoint(float(target[0]), float(target[1]), float(target[2]))
        camera.SetViewUp(0, 0, 1)

        renderer.ResetCameraClippingRange()
        interactor.Render()

    def constrain_camera(self, obj, event):
        """Constrain camera to prevent flipping upside down."""
        renderer = self._get_renderer()
        if renderer is None:
            return

        camera = renderer.GetActiveCamera()
        if camera is None:
            return

        camera.SetViewUp(0, 0, 1)

        pos = camera.GetPosition()
        focal = camera.GetFocalPoint()

        dx = pos[0] - focal[0]
        dy = pos[1] - focal[1]
        dz = pos[2] - focal[2]

        if dz < 0.1:
            distance = (dx*dx + dy*dy + dz*dz)**0.5
            new_z = focal[2] + 0.1 * max(distance, 1.0)
            camera.SetPosition(pos[0], pos[1], new_z)


class Terrain3DViewer(QtWidgets.QWidget):
    """
    3D terrain viewer widget using VTK.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Default theme state so themed startup does not crash before main window syncs mode
        self.commodity = "copper"
        self.accent_hex = "#B87333"
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        if VTK_AVAILABLE:
            # Create VTK widget
            self.vtk_widget = QVTKRenderWindowInteractor(self)
            layout.addWidget(self.vtk_widget)
            
            # Setup VTK renderer
            self.renderer = vtk.vtkRenderer()
            self.renderer.SetBackground(0.1, 0.1, 0.1)
            self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
            
            # Setup interactor with constrained camera movement
            self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
            
            # Use improved camera style with zoom-to-mouse
            self.camera_style = ImprovedTerrainStyle()
            self.camera_style.set_default_renderer(self.renderer)
            self.interactor.SetInteractorStyle(self.camera_style)
            self.interactor.Initialize()
            
            # Add click handler for terrain queries
            self.interactor.AddObserver("LeftButtonPressEvent", self.on_terrain_click)
            
            # Add mouse move handler for coordinate display
            self.interactor.AddObserver("MouseMoveEvent", self.on_mouse_move)
            
            # Create coordinate tooltip label (overlay on 3D view)
            self.coord_label = QLabel(self)
            self.coord_label.setStyleSheet("""
                QLabel {
                    background-color: rgba(0, 0, 0, 180);
                    color: #14a085;
                    border: 1px solid #14a085;
                    border-radius: 4px;
                    padding: 4px 8px;
                    font-family: 'Consolas', 'Courier New', monospace;
                    font-size: 10pt;
                }
            """)
            self.coord_label.hide()  # Hidden by default
            self.coord_label.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)  # Don't block mouse events
            
            # Set initial camera orientation for mining exploration
            camera = self.renderer.GetActiveCamera()
            camera.SetViewUp(0, 0, 1)  # Z is up (standard for terrain)
            camera.SetPosition(1, 1, 1)  # Good initial position
            camera.SetFocalPoint(0, 0, 0)
            
            # Control panel with dark theme
            controls = QHBoxLayout()
            
            # Style for buttons - dark theme
            button_style = """
                QPushButton {
                    background-color: #404040;
                    color: white;
                    border: 1px solid #606060;
                    border-radius: 4px;
                    padding: 4px 8px;
                    font-size: 10px;
                    min-width: 60px;
                }
                QPushButton:hover {
                    background-color: #505050;
                    border: 1px solid #707070;
                }
                QPushButton:pressed {
                    background-color: #303030;
                }
                QPushButton:checked {
                    background-color: #0078d4;
                    border: 1px solid #106ebe;
                }
            """
            
            # Slider style
            slider_style = """
                QSlider::groove:horizontal {
                    border: 1px solid #606060;
                    height: 6px;
                    background: #404040;
                    border-radius: 3px;
                }
                QSlider::handle:horizontal {
                    background: #0078d4;
                    border: 1px solid #106ebe;
                    width: 14px;
                    border-radius: 7px;
                    margin: -4px 0;
                }
            """
            
            label_style = "color: white; font-size: 10px;"
            
            # Vertical exaggeration
            self.exag_title_label = QLabel("Vertical Exaggeration:")
            exag_label = self.exag_title_label
            exag_label.setStyleSheet(label_style)
            controls.addWidget(exag_label)
            
            self.exaggeration_slider = QSlider(QtCore.Qt.Horizontal)
            self.exaggeration_slider.setMinimum(1)  # Start at 1x (no exaggeration)
            self.exaggeration_slider.setMaximum(50)
            self.exaggeration_slider.setValue(1)  # Default to 1x
            self.exaggeration_slider.valueChanged.connect(self.update_exaggeration)
            self.exaggeration_slider.setStyleSheet(slider_style)
            controls.addWidget(self.exaggeration_slider)
            
            self.exag_label = QLabel("1x")
            self.exag_label.setStyleSheet(label_style)
            controls.addWidget(self.exag_label)
            
            # View controls
            self.reset_btn = QPushButton("Reset View")
            reset_btn = self.reset_btn
            reset_btn.clicked.connect(self.reset_camera)
            reset_btn.setStyleSheet(button_style)
            controls.addWidget(reset_btn)
            
            self.top_btn = QPushButton("Top View")
            top_btn = self.top_btn
            top_btn.clicked.connect(self.top_view)
            top_btn.setStyleSheet(button_style)
            controls.addWidget(top_btn)
            
            self.side_btn = QPushButton("Side View")
            side_btn = self.side_btn
            side_btn.clicked.connect(self.side_view)
            side_btn.setStyleSheet(button_style)
            controls.addWidget(side_btn)
            
            layout.addLayout(controls)
            
            self.current_exaggeration = 1  # Start at 1x
            self.actor = None
            self.grid_data = None
            self.scalar_bar = None
            
            # Data for terrain queries
            self.dem_data = None
            self.prob_data = None
            self.grade_data = None
            self.transform_data = None
            
            self.set_theme(self.accent_hex, self.commodity)
            print("[3D] VTK 3D Viewer initialized")
        else:
            # Placeholder
            label = QLabel("3D Viewer requires VTK\n\nInstall with: pip install vtk")
            label.setAlignment(QtCore.Qt.AlignCenter)
            label.setStyleSheet("color: #888; font-size: 14px;")
            layout.addWidget(label)
            print("[3D] VTK not available - showing placeholder")
    
    def load_terrain(self, dem_path, probability_path=None):
        """Load DEM and probability map."""
        if not VTK_AVAILABLE:
            print("[3D] Cannot load terrain - VTK not available")
            return
        
        print(f"[3D] Loading terrain from {dem_path}")
        
        try:
            # Use GDAL (comes with QGIS) to read rasters
            from osgeo import gdal
            
            # Read DEM
            ds = gdal.Open(dem_path)
            dem = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
            transform = ds.GetGeoTransform()
            self.projection = ds.GetProjection()  # Store projection for coordinate conversion
            
            # Read probability if provided
            if probability_path:
                print(f"[3D] Loading probability from {probability_path}")
                prob_ds = gdal.Open(probability_path)
                prob = prob_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
                
                # Match DEM and probability sizes BEFORE downsampling
                if dem.shape != prob.shape:
                    min_h = min(dem.shape[0], prob.shape[0])
                    min_w = min(dem.shape[1], prob.shape[1])
                    dem = dem[:min_h, :min_w]
                    prob = prob[:min_h, :min_w]
                    print(f"[3D] Matched sizes: DEM {dem.shape}, Prob {prob.shape}")
            else:
                prob = None
            
            # Create grid
            h, w = dem.shape
            
            # Downsample if too large for performance
            if h > 300 or w > 300:
                factor = max(h // 300, w // 300)
                dem = dem[::factor, ::factor]
                if prob is not None:
                    prob = prob[::factor, ::factor]
                h, w = dem.shape
                print(f"[3D] Downsampled to {h}x{w} for performance")
                
                # Ensure exact size match after downsampling
                if prob is not None and dem.shape != prob.shape:
                    min_h = min(dem.shape[0], prob.shape[0])
                    min_w = min(dem.shape[1], prob.shape[1])
                    dem = dem[:min_h, :min_w]
                    prob = prob[:min_h, :min_w]
                    h, w = dem.shape
                    print(f"[3D] Final matched sizes: DEM {dem.shape}, Prob {prob.shape}")
            
            # Create VTK structured grid
            points = vtk.vtkPoints()
            
            for i in range(h):
                for j in range(w):
                    x = j
                    y = i
                    z = dem[i, j] * self.current_exaggeration / 100.0  # Scale down
                    points.InsertNextPoint(x, y, z)
            
            # Create grid
            grid = vtk.vtkStructuredGrid()
            grid.SetDimensions(w, h, 1)
            grid.SetPoints(points)
            
            # Add probability as scalars if available (scaled to 0-100 for display)
            if prob is not None:
                scalars = vtk.vtkFloatArray()
                scalars.SetName("Probability")
                for i in range(h):
                    for j in range(w):
                        scalars.InsertNextValue(prob[i, j] * 100)  # Scale to percentage
                grid.GetPointData().SetScalars(scalars)
            
            # Create mapper
            mapper = vtk.vtkDataSetMapper()
            mapper.SetInputData(grid)
            
            # Remove old actor and scalar bar FIRST
            if self.actor:
                self.renderer.RemoveActor(self.actor)
            
            if hasattr(self, 'scalar_bar') and self.scalar_bar:
                self.renderer.RemoveActor2D(self.scalar_bar)
                self.scalar_bar = None
            
            if prob is not None:
                # Set scalar range (now in 0-100 scale)
                prob_min = prob.min() * 100
                prob_max = prob.max() * 100
                
                # Industry standard: Only show colors for top probabilities (65-85%)
                # Everything below 60% is gray rock (background noise)
                display_min = 60.0  # Below this = gray
                display_max = 85.0  # Top of scale
                
                mapper.SetScalarRange(0, 100)  # Full range for data
                
                # Create custom color map with gray for low probabilities
                lut = vtk.vtkLookupTable()
                lut.SetNumberOfTableValues(256)
                
                # Build custom color table
                for i in range(256):
                    # Map index to probability (0-100%)
                    prob_value = (i / 255.0) * 100.0
                    
                    if prob_value < display_min:
                        # Gray rock for background (0-60%)
                        gray = 0.3  # Dark gray
                        lut.SetTableValue(i, gray, gray, gray, 1.0)
                    else:
                        # Color gradient for targets (60-85%)
                        # Normalize to 0-1 range within the display window
                        normalized = (prob_value - display_min) / (display_max - display_min)
                        normalized = max(0.0, min(1.0, normalized))  # Clamp
                        
                        # Blue -> Cyan -> Green -> Yellow -> Orange -> Red
                        if normalized < 0.2:  # 60-65%: Blue to Cyan
                            r = 0.0
                            g = normalized * 5.0  # 0 to 1
                            b = 1.0
                        elif normalized < 0.4:  # 65-70%: Cyan to Green
                            r = 0.0
                            g = 1.0
                            b = 1.0 - (normalized - 0.2) * 5.0
                        elif normalized < 0.6:  # 70-75%: Green to Yellow
                            r = (normalized - 0.4) * 5.0
                            g = 1.0
                            b = 0.0
                        elif normalized < 0.8:  # 75-80%: Yellow to Orange
                            r = 1.0
                            g = 1.0 - (normalized - 0.6) * 2.5
                            b = 0.0
                        else:  # 80-85%: Orange to Red
                            r = 1.0
                            g = 0.5 - (normalized - 0.8) * 2.5
                            b = 0.0
                        
                        lut.SetTableValue(i, r, g, b, 1.0)
                
                lut.Build()
                mapper.SetLookupTable(lut)
                
                # Scalar bar legend
                scalar_bar = vtk.vtkScalarBarActor()
                scalar_bar.SetLookupTable(lut)
                scalar_bar.SetTitle("Prospectivity")
                scalar_bar.SetNumberOfLabels(5)
                scalar_bar.SetWidth(0.08)
                scalar_bar.SetHeight(0.6)
                scalar_bar.SetPosition(0.90, 0.20)
                
                title_prop = scalar_bar.GetTitleTextProperty()
                title_prop.SetColor(0.08, 0.63, 0.52)
                title_prop.SetFontSize(14)
                title_prop.SetBold(True)
                title_prop.SetFontFamilyToArial()
                
                label_prop = scalar_bar.GetLabelTextProperty()
                label_prop.SetColor(0.85, 0.85, 0.85)
                label_prop.SetFontSize(12)
                label_prop.SetFontFamilyToArial()
                
                scalar_bar.SetLabelFormat("%.0f%%")
                scalar_bar.SetTextPositionToSucceedScalarBar()
                scalar_bar.VisibilityOn()
                
                self.renderer.AddActor2D(scalar_bar)
                self.scalar_bar = scalar_bar
                
                print(f"[3D] Legend: {prob_min:.1f}% to {prob_max:.1f}%")
            
            # Create actor
            self.actor = vtk.vtkActor()
            self.actor.SetMapper(mapper)
            self.renderer.AddActor(self.actor)
            
            # Reset camera
            self.renderer.ResetCamera()
            self.vtk_widget.GetRenderWindow().Render()
            
            # Store for updates and queries
            self.grid_data = (dem, prob, h, w)
            self.dem_data = dem
            self.prob_data = prob
            self.transform_data = transform
            
            # Try to load grade data too
            grade_path = probability_path.replace('probability', 'grade') if probability_path else None
            if grade_path and os.path.exists(grade_path):
                print(f"[3D] Loading grade data from: {grade_path}")
                grade_ds = gdal.Open(grade_path)
                grade_full = grade_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
                print(f"[3D] Grade data original shape: {grade_full.shape}")
                print(f"[3D] Grade range: {grade_full.min():.2f} - {grade_full.max():.2f}% Cu")
                
                # Match grade to DEM size BEFORE downsampling (same as DEM/prob logic)
                original_h, original_w = grade_full.shape
                
                # If we downsampled, apply the same downsampling to grade
                if h > 300 or w > 300 or original_h != h or original_w != w:
                    # Calculate the factor that was used for DEM
                    factor = max(original_h // 300, original_w // 300) if (original_h > 300 or original_w > 300) else 1
                    
                    # Apply same downsampling
                    if factor > 1:
                        self.grade_data = grade_full[::factor, ::factor]
                        print(f"[3D] Grade data downsampled by factor {factor} to: {self.grade_data.shape}")
                    else:
                        self.grade_data = grade_full
                    
                    # Ensure exact match with DEM shape
                    if self.grade_data.shape != dem.shape:
                        min_h = min(dem.shape[0], self.grade_data.shape[0])
                        min_w = min(dem.shape[1], self.grade_data.shape[1])
                        self.grade_data = self.grade_data[:min_h, :min_w]
                        print(f"[3D] Grade data cropped to match DEM: {self.grade_data.shape}")
                else:
                    self.grade_data = grade_full
                
                print(f"[3D] Final grade data shape: {self.grade_data.shape}, DEM shape: {dem.shape}")
                if self.grade_data.shape == dem.shape:
                    print(f"[3D] ✓ Grade data matches DEM shape perfectly")
                else:
                    print(f"[3D] ✗ WARNING: Shape mismatch! Grade: {self.grade_data.shape}, DEM: {dem.shape}")
            else:
                print(f"[3D] Grade file not found at: {grade_path}")
                print(f"[3D] Will estimate grade from probability")
                self.grade_data = None
            
            print("[3D] Terrain loaded successfully")
            print("[3D] ═══════════════════════════════════════════════════════")
            print("[3D] ANALYSIS AREA: 54km × 54km (center region of full DEM)")
            print("[3D] COLORED AREA: Entire analysis region (CORRECT behavior)")
            print("[3D] COLOR MEANING:")
            print("[3D]   • BLUE/GREEN (11-40%): Lower probability background")
            print("[3D]   • YELLOW/ORANGE (40-70%): Moderate probability")
            print("[3D]   • RED (70-86%): HIGH PRIORITY drilling targets")
            print("[3D] ═══════════════════════════════════════════════════════")
            print("[3D] Click on terrain to query probability and grade")
            print("[3D] The model predicts on ALL pixels - this is expected!")
            
        except Exception as e:
            print(f"[3D] Error loading terrain: {e}")
            import traceback
            traceback.print_exc()
    
    def update_exaggeration(self, value):
        """Update vertical exaggeration."""
        if not VTK_AVAILABLE or not self.grid_data:
            return
        
        self.current_exaggeration = value
        self.exag_label.setText(f"{value}x")
        
        dem, prob, h, w = self.grid_data
        
        # Update points
        points = vtk.vtkPoints()
        for i in range(h):
            for j in range(w):
                x = j
                y = i
                z = dem[i, j] * self.current_exaggeration / 100.0
                points.InsertNextPoint(x, y, z)
        
        # Update grid
        grid = vtk.vtkStructuredGrid()
        grid.SetDimensions(w, h, 1)
        grid.SetPoints(points)
        
        # Reapply probability colors (scaled to 0-100 for display)
        if prob is not None:
            scalars = vtk.vtkFloatArray()
            scalars.SetName("Probability")
            for i in range(h):
                for j in range(w):
                    scalars.InsertNextValue(prob[i, j] * 100)  # Scale to percentage
            grid.GetPointData().SetScalars(scalars)
        
        # Update mapper
        mapper = self.actor.GetMapper()
        mapper.SetInputData(grid)
        
        self.vtk_widget.GetRenderWindow().Render()
    
    def reset_camera(self):
        """Reset camera view."""
        if VTK_AVAILABLE:
            camera = self.renderer.GetActiveCamera()
            
            # Reset to a good mining exploration view
            camera.SetViewUp(0, 0, 1)  # Z is always up
            camera.SetPosition(1, 1, 0.5)  # Elevated position
            camera.SetFocalPoint(0, 0, 0)  # Look at center
            
            self.renderer.ResetCamera()
            
            # Adjust for better mining view
            camera.Elevation(30)  # 30 degrees above horizon
            camera.Azimuth(45)    # 45 degrees rotation
            
            self.vtk_widget.GetRenderWindow().Render()
    
    def top_view(self):
        """Set camera to top-down view (good for mining planning)."""
        if VTK_AVAILABLE:
            camera = self.renderer.GetActiveCamera()
            
            # Get current focal point
            focal = camera.GetFocalPoint()
            
            # Position camera directly above
            camera.SetPosition(focal[0], focal[1], focal[2] + 1000)
            camera.SetFocalPoint(focal[0], focal[1], focal[2])
            camera.SetViewUp(0, 1, 0)  # Y is up in top view
            
            self.renderer.ResetCamera()
            self.vtk_widget.GetRenderWindow().Render()
    
    def side_view(self):
        """Set camera to side view (good for geological analysis)."""
        if VTK_AVAILABLE:
            camera = self.renderer.GetActiveCamera()
            
            # Get current focal point
            focal = camera.GetFocalPoint()
            
            # Position camera to the side
            camera.SetPosition(focal[0] + 1000, focal[1], focal[2])
            camera.SetFocalPoint(focal[0], focal[1], focal[2])
            camera.SetViewUp(0, 0, 1)  # Z is up
            
            self.renderer.ResetCamera()
            self.vtk_widget.GetRenderWindow().Render()
    
    def clear(self):
        """Clear the viewer."""
        if VTK_AVAILABLE and self.actor:
            self.renderer.RemoveActor(self.actor)
            self.actor = None
            self.grid_data = None
            self.vtk_widget.GetRenderWindow().Render()
    
    def on_terrain_click(self, obj, event):
        """Handle click on terrain to show data at that point."""
        if not VTK_AVAILABLE or self.dem_data is None:
            return
        
        # Get click position
        click_pos = self.interactor.GetEventPosition()
        
        # Create picker with better tolerance
        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.005)  # More forgiving picking
        picker.Pick(click_pos[0], click_pos[1], 0, self.renderer)
        
        # Get picked position
        pos = picker.GetPickPosition()
        
        if picker.GetCellId() != -1:
            # Convert 3D position to pixel coordinates
            x, y, z = pos
            
            # Get pixel coordinates (accounting for downsampling)
            if self.grid_data:
                dem, prob, h, w = self.grid_data
                
                # x, y are in grid coordinates (0 to w-1, 0 to h-1)
                px = int(x)
                py = int(y)
                
                if 0 <= px < w and 0 <= py < h:
                    # Get values
                    elevation = dem[py, px]
                    probability = prob[py, px] if prob is not None else 0.0
                    
                    # Get grade if available
                    grade = 0.0
                    if self.grade_data is not None:
                        # Make sure grade_data has same dimensions as downsampled data
                        if self.grade_data.shape == dem.shape:
                            grade = self.grade_data[py, px]
                            print(f"[3D] Retrieved grade: {grade:.2f}% Cu at ({px}, {py})")
                        else:
                            print(f"[ERROR] Grade data shape mismatch: {self.grade_data.shape} vs {dem.shape}")
                            # Fallback: estimate from probability
                            grade = probability * 2.5
                            print(f"[3D] Using fallback grade estimate: {grade:.2f}% Cu")
                    else:
                        # Estimate grade from probability if grade data not available
                        if self.commodity == "ree":
                            grade = probability * 10.0  # Carbonatite REE: 0-10% TREO
                        else:
                            grade = probability * 2.5  # Typical porphyry copper grade range
                        print(f"[3D] No grade data, estimated from probability: {grade:.2f}% Cu")
                    
                    # Convert to real-world coordinates
                    if self.transform_data:
                        transform = self.transform_data
                        real_x = transform[0] + px * transform[1]
                        real_y = transform[3] + py * transform[5]
                        
                        # Show info dialog
                        self.show_terrain_info(real_x, real_y, elevation, probability, grade)
    
    def on_mouse_move(self, obj, event):
        """Handle mouse move to show coordinates in tooltip."""
        if not VTK_AVAILABLE or self.dem_data is None:
            return
        
        # Get mouse position
        mouse_pos = self.interactor.GetEventPosition()
        
        # Create picker
        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.01)
        picker.Pick(mouse_pos[0], mouse_pos[1], 0, self.renderer)
        
        # Get picked position
        pos = picker.GetPickPosition()
        
        if picker.GetCellId() != -1 and self.grid_data:
            # Convert 3D position to pixel coordinates
            x, y, z = pos
            dem, prob, h, w = self.grid_data
            
            px = int(x)
            py = int(y)
            
            if 0 <= px < w and 0 <= py < h:
                # Get values
                elevation = dem[py, px]
                probability = prob[py, px] if prob is not None else 0.0
                
                # Convert to real-world coordinates
                if self.transform_data:
                    transform = self.transform_data
                    real_x = transform[0] + px * transform[1]
                    real_y = transform[3] + py * transform[5]
                    
                    # Convert to Lat/Lon if possible
                    try:
                        from osgeo import osr
                        source_srs = osr.SpatialReference()
                        source_srs.ImportFromWkt(self.projection) if hasattr(self, 'projection') else source_srs.ImportFromEPSG(32612)
                        target_srs = osr.SpatialReference()
                        target_srs.ImportFromEPSG(4326)
                        coord_transform = osr.CoordinateTransformation(source_srs, target_srs)
                        lon, lat, _ = coord_transform.TransformPoint(real_x, real_y)
                        
                        # Update tooltip
                        tooltip_text = f"Lat: {lat:.6f}°  Lon: {lon:.6f}°  |  Elev: {elevation:.0f}m  |  Prob: {probability*100:.1f}%"
                    except:
                        tooltip_text = f"X: {real_x:.0f}  Y: {real_y:.0f}  |  Elev: {elevation:.0f}m  |  Prob: {probability*100:.1f}%"
                    
                    # Position tooltip near cursor
                    self.coord_label.setText(tooltip_text)
                    self.coord_label.adjustSize()
                    
                    # Position slightly offset from cursor
                    cursor_pos = self.mapFromGlobal(QtGui.QCursor.pos())
                    label_x = cursor_pos.x() + 15
                    label_y = cursor_pos.y() + 15
                    
                    # Keep within widget bounds
                    if label_x + self.coord_label.width() > self.width():
                        label_x = cursor_pos.x() - self.coord_label.width() - 5
                    if label_y + self.coord_label.height() > self.height():
                        label_y = cursor_pos.y() - self.coord_label.height() - 5
                    
                    self.coord_label.move(label_x, label_y)
                    self.coord_label.show()
                    return
        
        # Hide tooltip if not hovering over terrain
        self.coord_label.hide()
    
    def show_terrain_info(self, x, y, elevation, probability, grade):
        """Show terrain information dialog with modern styling."""
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit
        from PyQt5.QtCore import Qt
        from PyQt5.QtGui import QFont
        
        # Convert UTM to Lat/Lon if possible
        try:
            from osgeo import osr
            
            # Get the projection from the DEM
            if self.transform_data:
                # Create coordinate transformation
                source_srs = osr.SpatialReference()
                source_srs.ImportFromWkt(self.projection) if hasattr(self, 'projection') else source_srs.ImportFromEPSG(32612)  # UTM Zone 12N
                
                target_srs = osr.SpatialReference()
                target_srs.ImportFromEPSG(4326)  # WGS84 Lat/Lon
                
                transform = osr.CoordinateTransformation(source_srs, target_srs)
                
                # Transform coordinates
                lon, lat, _ = transform.TransformPoint(x, y)
                
                coord_text = f"""<b style='color: #14a085;'>📍 LOCATION</b><br>
<span style='color: #d4d4d4;'>Latitude:</span> <b>{lat:.6f}°</b><br>
<span style='color: #d4d4d4;'>Longitude:</span> <b>{lon:.6f}°</b><br>
<br>
<span style='color: #999;'>UTM Zone 12N:</span><br>
<span style='color: #d4d4d4;'>Easting:</span> {x:.2f} m<br>
<span style='color: #d4d4d4;'>Northing:</span> {y:.2f} m<br>
"""
            else:
                coord_text = f"""<b style='color: #14a085;'>📍 LOCATION</b><br>
<span style='color: #d4d4d4;'>X:</span> {x:.2f}<br>
<span style='color: #d4d4d4;'>Y:</span> {y:.2f}<br>
"""
        except Exception as e:
            print(f"[WARN] Could not convert coordinates: {e}")
            coord_text = f"""<b style='color: #14a085;'>📍 LOCATION</b><br>
<span style='color: #d4d4d4;'>X:</span> {x:.2f}<br>
<span style='color: #d4d4d4;'>Y:</span> {y:.2f}<br>
"""
        
        # Create custom dialog
        dialog = QDialog()
        dialog.setWindowTitle("🎯 Terrain Query Results")
        dialog.setMinimumWidth(500)
        dialog.setStyleSheet("""
            QDialog {
                background-color: #2d2d2d;
            }
            QLabel {
                color: #d4d4d4;
                background-color: transparent;
            }
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 10px;
                font-size: 11pt;
            }
            QPushButton {
                background-color: #0d7377;
                color: white;
                border: none;
                padding: 8px 20px;
                border-radius: 4px;
                font-weight: bold;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #14a085;
            }
        """)
        
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Probability assessment
        if probability > 0.7:
            prob_color = "#ff4444"
            prob_icon = "🔴"
            prob_text = "HIGH"
            prob_desc = "Priority target for exploration"
        elif probability > 0.4:
            prob_color = "#ffaa00"
            prob_icon = "🟡"
            prob_text = "MEDIUM"
            prob_desc = "Secondary target"
        else:
            prob_color = "#4444ff"
            prob_icon = "🔵"
            prob_text = "LOW"
            prob_desc = "Not recommended"
        
        # Grade assessment (commodity-specific thresholds)
        if self.commodity == "ree":
            if grade > 5.0:
                grade_icon, grade_text, grade_desc = "💎", "HIGH GRADE REE", "Exceptional TREO"
            elif grade > 2.0:
                grade_icon, grade_text, grade_desc = "⚡", "ECONOMIC REE", "LREE (La, Ce, Pr, Nd)"
            elif grade > 1.0:
                grade_icon, grade_text, grade_desc = "📊", "MARGINAL REE", "May be economic"
            else:
                grade_icon, grade_text, grade_desc = "⬇️", "SUB-ECONOMIC", "Below REE cutoff"
        else:
            if grade > 1.0:
                grade_icon, grade_text, grade_desc = "💎", "HIGH GRADE", "Potentially economic"
            elif grade > 0.5:
                grade_icon, grade_text, grade_desc = "⚡", "ECONOMIC GRADE", "Worth investigating"
            elif grade > 0.3:
                grade_icon, grade_text, grade_desc = "📊", "MARGINAL GRADE", "May be economic at scale"
            else:
                grade_icon, grade_text, grade_desc = "⬇️", "SUB-ECONOMIC", "Below typical cutoff"
        
        # Commodity-aware labels
        prob_label = "REE PROBABILITY" if self.commodity == "ree" else "COPPER PROBABILITY"
        grade_unit = "% TREO" if self.commodity == "ree" else "% Cu"
        
        info_html = f"""
<div style='font-family: Segoe UI, Arial; line-height: 1.6;'>
{coord_text}
<br>
<b style='color: #14a085;'>⛰️ TERRAIN</b><br>
<span style='color: #d4d4d4;'>Elevation:</span> <b>{elevation:.1f} m</b> ({elevation*3.28084:.0f} ft)<br>
<br>
<b style='color: #14a085;'>🎲 {prob_label}</b><br>
<span style='font-size: 14pt;'>{prob_icon} <b style='color: {prob_color};'>{probability*100:.1f}%</b> - {prob_text}</span><br>
<span style='color: #999;'>{prob_desc}</span><br>
<br>
<b style='color: #14a085;'>💰 ESTIMATED GRADE</b><br>
<span style='font-size: 14pt;'>{grade_icon} <b>{grade:.2f}{grade_unit}</b> - {grade_text}</span><br>
<span style='color: #999;'>{grade_desc}</span><br>
"""
        
        target_msg = "This location shows strong potential for REE (La, Ce, Pr, Nd) mineralization in carbonatite-type host rocks." if self.commodity == "ree" else "This location shows strong potential for copper mineralization. Recommended for immediate exploration."
        
        # Economic viability
        if probability > 0.7 and grade > 0.5:
            info_html += f"""<br>
<div style='background-color: #1e4d2b; border-left: 4px solid #14a085; padding: 10px; border-radius: 4px;'>
<b style='color: #14a085; font-size: 12pt;'>🎯 PRIORITY DRILLING TARGET</b><br>
<span style='color: #d4d4d4;'>{target_msg}</span>
</div>
"""
        elif probability > 0.5 and grade > 0.3:
            info_html += f"""<br>
<div style='background-color: #4d3d1e; border-left: 4px solid #ffaa00; padding: 10px; border-radius: 4px;'>
<b style='color: #ffaa00; font-size: 12pt;'>✅ RECOMMENDED FOR INVESTIGATION</b><br>
<span style='color: #d4d4d4;'>Moderate potential. Consider for follow-up sampling or geophysical surveys.</span>
</div>
"""
        
        info_html += "</div>"
        
        # Text display
        text_edit = QTextEdit()
        text_edit.setHtml(info_html)
        text_edit.setReadOnly(True)
        text_edit.setMinimumHeight(400)
        layout.addWidget(text_edit)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        copy_btn = QPushButton("📋 Copy Coordinates")
        copy_btn.clicked.connect(lambda: self._copy_coordinates(lat if 'lat' in locals() else x, 
                                                                lon if 'lon' in locals() else y))
        copy_btn.setStyleSheet("""
            QPushButton {
                background-color: #404040;
                color: white;
                border: 1px solid #555;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
            }
        """)
        button_layout.addWidget(copy_btn)
        
        button_layout.addStretch()
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
        
        dialog.exec_()
    
    def _copy_coordinates(self, lat_or_x, lon_or_y):
        """Copy coordinates to clipboard."""
        from PyQt5.QtWidgets import QApplication
        clipboard = QApplication.clipboard()
        coord_text = f"{lat_or_x:.6f}, {lon_or_y:.6f}"
        clipboard.setText(coord_text)
        print(f"[3D] Copied coordinates to clipboard: {coord_text}")
    def set_theme(self, accent_hex=None, commodity=None):
        """Apply copper/REE accent theme to the viewer controls."""
        if accent_hex:
            self.accent_hex = accent_hex
        if commodity:
            self.commodity = commodity

        accent = getattr(self, "accent_hex", "#B87333")
        if accent.lower() == "#00c9a7":
            accent_hover = "#1AD7B5"
            accent_pressed = "#009F86"
        else:
            accent_hover = "#C9864A"
            accent_pressed = "#8E5A2B"

        if hasattr(self, "coord_label"):
            self.coord_label.setStyleSheet(f"""
                QLabel {{
                    background-color: rgba(0, 0, 0, 190);
                    color: {accent};
                    border: 1px solid {accent};
                    border-radius: 4px;
                    padding: 4px 8px;
                    font-family: 'Consolas', 'Courier New', monospace;
                    font-size: 10pt;
                }}
            """)

        button_style = f"""
            QPushButton {{
                background-color: #2f2f2f;
                color: white;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 10px;
                min-width: 60px;
            }}
            QPushButton:hover {{
                background-color: {accent_hover};
                border: 1px solid {accent_hover};
            }}
            QPushButton:pressed {{
                background-color: {accent_pressed};
                border: 1px solid {accent_pressed};
            }}
            QPushButton:checked {{
                background-color: {accent};
                border: 1px solid {accent};
            }}
        """
        slider_style = f"""
            QSlider::groove:horizontal {{
                border: 1px solid #606060;
                height: 6px;
                background: #404040;
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: {accent};
                border: 1px solid {accent_pressed};
                width: 14px;
                border-radius: 7px;
                margin: -4px 0;
            }}
        """

        for attr in ("reset_btn", "top_btn", "side_btn"):
            if hasattr(self, attr):
                getattr(self, attr).setStyleSheet(button_style)

        if hasattr(self, "exaggeration_slider"):
            self.exaggeration_slider.setStyleSheet(slider_style)

        if hasattr(self, "exag_title_label"):
            self.exag_title_label.setStyleSheet(f"color: {accent}; font-size: 10px; font-weight: 600;")
        if hasattr(self, "exag_label"):
            self.exag_label.setStyleSheet(f"color: {accent}; font-size: 10px; font-weight: 600;")
