#!/usr/bin/env python3
"""
IMC .mcd File Viewer
A PyQt5-based viewer for IMC .mcd files using the readimc library.
Uses the reading method: f.read_acquisition(acquisition) with proper array transposition.
"""

import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd

# ---- UI ----
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

# ---- Plotting ----
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import gridspec

# ---- readimc (required) ----
_HAVE_READIMC = False
try:
    from readimc import MCDFile as McdFile  # pip install readimc
    _HAVE_READIMC = True
except Exception:
    _HAVE_READIMC = False


# --------------------------
# Data types / helpers
# --------------------------
@dataclass
class AcquisitionInfo:
    id: str
    name: str
    well: Optional[str]
    size: Tuple[Optional[int], Optional[int]]  # (H, W)
    channels: List[str]
    channel_metals: List[str]
    channel_labels: List[str]
    metadata: Dict


def robust_percentile_scale(arr: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    """Scale array to [0,1] using percentile normalization."""
    a = arr.astype(np.float32, copy=False)
    lo = np.percentile(a, low)
    hi = np.percentile(a, high)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = lo + 1e-6
    a = np.clip((a - lo) / (hi - lo), 0.0, 1.0)
    return a


def stack_to_rgb(stack: np.ndarray) -> np.ndarray:
    """
    Convert 1–3 channel stack to RGB for display.
    1 ch → gray ; 2 ch → (ch1, ch2, ch2) ; ≥3 ch → first 3 channels
    """
    H, W, C = stack.shape
    if C == 1:
        # Scale to 0-1 range
        g = (stack[..., 0] - np.min(stack[..., 0])) / (np.max(stack[..., 0]) - np.min(stack[..., 0]) + 1e-8)
        return np.dstack([g, g, g])
    elif C == 2:
        # Scale each channel to 0-1 range
        r = (stack[..., 0] - np.min(stack[..., 0])) / (np.max(stack[..., 0]) - np.min(stack[..., 0]) + 1e-8)
        g = (stack[..., 1] - np.min(stack[..., 1])) / (np.max(stack[..., 1]) - np.min(stack[..., 1]) + 1e-8)
        return np.dstack([r, g, g])
    else:
        # Scale each channel to 0-1 range
        r = (stack[..., 0] - np.min(stack[..., 0])) / (np.max(stack[..., 0]) - np.min(stack[..., 0]) + 1e-8)
        g = (stack[..., 1] - np.min(stack[..., 1])) / (np.max(stack[..., 1]) - np.min(stack[..., 1]) + 1e-8)
        b = (stack[..., 2] - np.min(stack[..., 2])) / (np.max(stack[..., 2]) - np.min(stack[..., 2]) + 1e-8)
        return np.dstack([r, g, b])


class MplCanvas(FigureCanvas):
    def __init__(self, width=5, height=5, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)

    def show_image(self, img: np.ndarray, title: str = "", grayscale: bool = False, show_colorbar: bool = True, raw_img: np.ndarray = None, custom_min: float = None, custom_max: float = None):
        # Clear the entire figure
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        
        im = None
        vmin, vmax = None, None  # Store the actual vmin/vmax used
        
        if img.ndim == 2:
            if custom_min is not None and custom_max is not None:
                vmin, vmax = custom_min, custom_max
            else:
                vmin, vmax = np.min(img), np.max(img)
            
            if grayscale:
                im = self.ax.imshow(img, interpolation="nearest", cmap='gray', vmin=vmin, vmax=vmax)
            else:
                im = self.ax.imshow(img, interpolation="nearest", cmap='viridis', vmin=vmin, vmax=vmax)
        elif img.ndim == 3:
            if grayscale:
                # Show first channel in grayscale
                if custom_min is not None and custom_max is not None:
                    vmin, vmax = custom_min, custom_max
                else:
                    vmin, vmax = np.min(img[..., 0]), np.max(img[..., 0])
                im = self.ax.imshow(img[..., 0], interpolation="nearest", cmap='gray', vmin=vmin, vmax=vmax)
            else:
                if img.shape[-1] <= 3:
                    im = self.ax.imshow(stack_to_rgb(img), interpolation="nearest")
                else:
                    im = self.ax.imshow(stack_to_rgb(img[..., :3]), interpolation="nearest")
        else:
            self.ax.text(0.5, 0.5, "Unsupported image shape", ha="center", va="center")
        
        self.ax.set_title(title)
        self.ax.axis("off")
        
        # Add colorbar if requested and image is valid
        if show_colorbar and im is not None and img.ndim == 2 and vmin is not None and vmax is not None:
            cbar = self.fig.colorbar(im, ax=self.ax, shrink=0.8, aspect=20)
            # The colorbar automatically shows the vmin/vmax range from imshow
            # We just need to format the tick labels nicely
            cbar.set_ticks([vmin, vmax])
            cbar.set_ticklabels([f'{vmin:.1f}', f'{vmax:.1f}'])
        
        self.draw()

    def show_grid(self, images: List[np.ndarray], titles: List[str], grayscale: bool = False, raw_images: List[np.ndarray] = None, custom_min: float = None, custom_max: float = None, channel_names: List[str] = None, channel_scaling: dict = None, custom_scaling_enabled: bool = False):
        """Show multiple images in a grid layout."""
        n_images = len(images)
        if n_images == 0:
            return
        
        # Calculate grid dimensions
        cols = min(3, n_images)  # Max 3 columns
        rows = (n_images + cols - 1) // cols
        
        # Clear and create subplots
        self.fig.clear()
        
        for i, (img, title) in enumerate(zip(images, titles)):
            ax = self.fig.add_subplot(rows, cols, i + 1)
            im = None
            vmin, vmax = None, None  # Store the actual vmin/vmax used
            
            # Get per-channel scaling if available
            channel_min = None
            channel_max = None
            if custom_scaling_enabled and channel_names and i < len(channel_names) and channel_scaling:
                channel_name = channel_names[i]
                if channel_name in channel_scaling:
                    channel_min = channel_scaling[channel_name]['min']
                    channel_max = channel_scaling[channel_name]['max']
            
            if img.ndim == 2:
                if channel_min is not None and channel_max is not None:
                    vmin, vmax = channel_min, channel_max
                elif custom_min is not None and custom_max is not None:
                    vmin, vmax = custom_min, custom_max
                else:
                    vmin, vmax = np.min(img), np.max(img)
                
                if grayscale:
                    im = ax.imshow(img, interpolation="nearest", cmap='gray', vmin=vmin, vmax=vmax)
                else:
                    im = ax.imshow(img, interpolation="nearest", cmap='viridis', vmin=vmin, vmax=vmax)
            elif img.ndim == 3 and not grayscale:
                if img.shape[-1] <= 3:
                    im = ax.imshow(stack_to_rgb(img), interpolation="nearest")
                else:
                    im = ax.imshow(stack_to_rgb(img[..., :3]), interpolation="nearest")
            else:
                # Grayscale for multi-channel
                if channel_min is not None and channel_max is not None:
                    vmin, vmax = channel_min, channel_max
                elif custom_min is not None and custom_max is not None:
                    vmin, vmax = custom_min, custom_max
                else:
                    vmin, vmax = np.min(img[..., 0]), np.max(img[..., 0])
                im = ax.imshow(img[..., 0], interpolation="nearest", cmap='gray', vmin=vmin, vmax=vmax)
            
            ax.set_title(title, fontsize=10)
            ax.axis("off")
            
            # Add colorbar for each subplot
            if im is not None and img.ndim == 2 and vmin is not None and vmax is not None:
                cbar = self.fig.colorbar(im, ax=ax, shrink=0.8, aspect=20)
                # The colorbar automatically shows the vmin/vmax range from imshow
                # We just need to format the tick labels nicely
                cbar.set_ticks([vmin, vmax])
                cbar.set_ticklabels([f'{vmin:.1f}', f'{vmax:.1f}'])
        
        self.fig.tight_layout()
        self.draw()


# --------------------------
# IMC loader using readimc
# --------------------------
class MCDLoader:
    """Loader for IMC .mcd files using the readimc library with f.read_acquisition() method."""
    
    def __init__(self):
        if not _HAVE_READIMC:
            raise RuntimeError(
                "readimc is not installed. Run: pip install readimc"
            )
        self.mcd: Optional[McdFile] = None
        self._acq_map: Dict[str, object] = {}
        self._acq_channels: Dict[str, List[str]] = {}
        self._acq_channel_metals: Dict[str, List[str]] = {}
        self._acq_channel_labels: Dict[str, List[str]] = {}
        self._acq_size: Dict[str, Tuple[Optional[int], Optional[int]]] = {}
        self._acq_name: Dict[str, str] = {}
        self._acq_well: Dict[str, Optional[str]] = {}
        self._acq_metadata: Dict[str, Dict] = {}

    def open(self, path: str):
        """Open an .mcd file."""
        self.mcd = McdFile(path)
        if hasattr(self.mcd, "open"):
            self.mcd.open()
        self._index()

    def _index(self):
        """Index all acquisitions in the .mcd file."""
        self._acq_map.clear()
        self._acq_channels.clear()
        self._acq_channel_metals.clear()
        self._acq_channel_labels.clear()
        self._acq_size.clear()
        self._acq_name.clear()
        self._acq_well.clear()
        self._acq_metadata.clear()

        acq_counter = 0
        slides = getattr(self.mcd, "slides", [])
        
        if slides:
            for slide_idx, slide in enumerate(slides):
                for acq_idx, acq in enumerate(getattr(slide, "acquisitions", [])):
                    acq_id = f"slide_{slide_idx}_acq_{acq_idx}"
                    acq_counter += 1
                    
                    # Get acquisition name
                    name = getattr(acq, "name", f"Slide {slide_idx + 1} Acquisition {acq_idx + 1}")
                    
                    # Get well information
                    well = getattr(acq, "well", getattr(slide, "well", None))
                    if well is None and hasattr(acq, "metadata"):
                        metadata = acq.metadata
                        if isinstance(metadata, dict) and 'Description' in metadata:
                            well = metadata['Description']
                    
                    # Get channel information
                    channel_metals = getattr(acq, "channel_names", [])
                    channel_labels = getattr(acq, "channel_labels", [])
                    
                    # Create display names for channels
                    channels = []
                    for i, (metal, label) in enumerate(zip(channel_metals, channel_labels)):
                        if label and metal:
                            channels.append(f"{label}_{metal}")
                        elif label:
                            channels.append(label)
                        elif metal:
                            channels.append(metal)
                        else:
                            channels.append(f"Channel_{i+1}")
                    
                    # Get image dimensions
                    try:
                        # Try to get dimensions from acquisition metadata
                        H = getattr(acq, "height", None) or getattr(acq, "rows", None)
                        W = getattr(acq, "width", None) or getattr(acq, "cols", None)
                        size = (int(H), int(W)) if H and W else (None, None)
                    except Exception:
                        size = (None, None)
                    
                    # Get metadata
                    metadata = getattr(acq, "metadata", {})
                    if not isinstance(metadata, dict):
                        metadata = {}
                    
                    # Store acquisition info
                    self._acq_map[acq_id] = acq
                    self._acq_channels[acq_id] = channels
                    self._acq_channel_metals[acq_id] = channel_metals
                    self._acq_channel_labels[acq_id] = channel_labels
                    self._acq_size[acq_id] = size
                    self._acq_name[acq_id] = name
                    self._acq_well[acq_id] = well
                    self._acq_metadata[acq_id] = metadata

        if not self._acq_map:
            raise RuntimeError("No acquisitions found in this .mcd file.")

    def list_acquisitions(self) -> List[AcquisitionInfo]:
        """List all acquisitions in the .mcd file."""
        infos: List[AcquisitionInfo] = []
        for acq_id in self._acq_map:
            infos.append(
                AcquisitionInfo(
                    id=acq_id,
                    name=self._acq_name.get(acq_id, acq_id),
                    well=self._acq_well.get(acq_id),
                    size=self._acq_size.get(acq_id, (None, None)),
                    channels=self._acq_channels.get(acq_id, []),
                    channel_metals=self._acq_channel_metals.get(acq_id, []),
                    channel_labels=self._acq_channel_labels.get(acq_id, []),
                    metadata=self._acq_metadata.get(acq_id, {}),
                )
            )
        return infos

    def get_channels(self, acq_id: str) -> List[str]:
        """Get channel names for a specific acquisition."""
        return self._acq_channels[acq_id]

    def get_image(self, acq_id: str, channel: str) -> np.ndarray:
        """Get image data for a specific acquisition and channel."""
        acq = self._acq_map[acq_id]
        channels = self._acq_channels[acq_id]
        
        if channel not in channels:
            raise ValueError(f"Channel '{channel}' not found in acquisition {acq_id}.")
        
        ch_idx = channels.index(channel)
        
        # Use the reading method from the notebook: f.read_acquisition(acquisition)
        with self.mcd as f:
            img = f.read_acquisition(acq)  # array, shape: (c, y, x), dtype: float32
            
            # Reorder to height, width, channel as shown in the notebook
            img = np.transpose(img, (1, 2, 0))
            
            # Return the specific channel
            return img[..., ch_idx]

    def get_all_channels(self, acq_id: str) -> np.ndarray:
        """Get all channels for a specific acquisition as a 3D array (H, W, C)."""
        acq = self._acq_map[acq_id]
        
        # Use the reading method from the notebook: f.read_acquisition(acquisition)
        with self.mcd as f:
            img = f.read_acquisition(acq)  # array, shape: (c, y, x), dtype: float32
            
            # Reorder to height, width, channel as shown in the notebook
            img = np.transpose(img, (1, 2, 0))
            
            return img

    def close(self):
        """Close the .mcd file."""
        if self.mcd and hasattr(self.mcd, "close"):
            self.mcd.close()


# --------------------------
# Dynamic Comparison dialog
# --------------------------
class DynamicComparisonDialog(QtWidgets.QDialog):
    def __init__(self, acqs: List[AcquisitionInfo], loader: MCDLoader, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Dynamic Comparison Mode")
        
        # Set dialog size to 80% of screen size
        screen = QtWidgets.QApplication.desktop().screenGeometry()
        self.resize(int(screen.width() * 0.8), int(screen.height() * 0.8))
        
        self.acqs = acqs
        self.loader = loader
        self.selected_acquisitions = []
        self.current_channel = None
        
        # Cache for loaded images to avoid reloading
        self.image_cache = {}  # {(acquisition_id, channel): image}
        self.max_cache_size = 50  # Limit cache size to prevent memory issues
        
        # Store last selected channel for auto-selection
        self.last_selected_channel: Optional[str] = None
        
        # Store per-image scaling values for individual scaling
        self.image_scaling = {}  # {acquisition_id: {'min': value, 'max': value}}
        
        self.setMinimumSize(1000, 700)
        
        # Create UI
        self._create_ui()
        
        # Start with empty selection - user must select acquisitions of interest

    def _create_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        
        # Control panel
        control_panel = QtWidgets.QWidget()
        control_layout = QtWidgets.QHBoxLayout(control_panel)
        
        # Acquisition selection
        acq_group = QtWidgets.QGroupBox("Acquisitions")
        acq_layout = QtWidgets.QVBoxLayout(acq_group)
        
        acq_layout.addWidget(QtWidgets.QLabel("Available:"))
        self.available_acq_list = QtWidgets.QListWidget()
        self.available_acq_list.setMaximumHeight(150)
        for ai in self.acqs:
            label = f"{ai.name}" + (f" ({ai.well})" if ai.well else "")
            item = QtWidgets.QListWidgetItem(label)
            item.setData(Qt.UserRole, ai.id)
            self.available_acq_list.addItem(item)
        
        acq_buttons = QtWidgets.QHBoxLayout()
        self.add_acq_btn = QtWidgets.QPushButton("Add →")
        self.remove_acq_btn = QtWidgets.QPushButton("← Remove")
        acq_buttons.addWidget(self.add_acq_btn)
        acq_buttons.addWidget(self.remove_acq_btn)
        
        acq_layout.addWidget(self.available_acq_list)
        acq_layout.addLayout(acq_buttons)
        
        acq_layout.addWidget(QtWidgets.QLabel("Selected:"))
        self.acq_list = QtWidgets.QListWidget()
        self.acq_list.setMaximumHeight(150)
        
        acq_layout.addWidget(self.acq_list)
        control_layout.addWidget(acq_group)
        
        # Channel selection
        channel_group = QtWidgets.QGroupBox("Channel")
        channel_layout = QtWidgets.QVBoxLayout(channel_group)
        
        self.channel_combo = QtWidgets.QComboBox()
        channel_layout.addWidget(QtWidgets.QLabel("Marker channel:"))
        channel_layout.addWidget(self.channel_combo)
        
        # Display options
        options_group = QtWidgets.QGroupBox("Display Options")
        options_layout = QtWidgets.QVBoxLayout(options_group)
        
        self.link_chk = QtWidgets.QCheckBox("Linked scaling (shared min/max)")
        self.link_chk.setChecked(True)
        self.grayscale_chk = QtWidgets.QCheckBox("Grayscale mode")
        
        options_layout.addWidget(self.link_chk)
        options_layout.addWidget(self.grayscale_chk)
        
        # Custom scaling controls for comparison mode
        self.custom_scaling_chk = QtWidgets.QCheckBox("Custom scaling")
        self.custom_scaling_chk.toggled.connect(self._on_comparison_scaling_toggled)
        options_layout.addWidget(self.custom_scaling_chk)
        
        self.scaling_frame = QtWidgets.QFrame()
        self.scaling_frame.setFrameStyle(QtWidgets.QFrame.Box)
        scaling_layout = QtWidgets.QVBoxLayout(self.scaling_frame)
        scaling_layout.addWidget(QtWidgets.QLabel("Custom Intensity Range:"))
        
        # Image selection for individual scaling
        image_selection_layout = QtWidgets.QHBoxLayout()
        image_selection_layout.addWidget(QtWidgets.QLabel("Image:"))
        self.image_combo = QtWidgets.QComboBox()
        self.image_combo.currentTextChanged.connect(self._on_image_selection_changed)
        image_selection_layout.addWidget(self.image_combo)
        image_selection_layout.addStretch()
        scaling_layout.addLayout(image_selection_layout)
        
        # Min/Max controls
        minmax_layout = QtWidgets.QHBoxLayout()
        minmax_layout.addWidget(QtWidgets.QLabel("Min:"))
        self.min_spinbox = QtWidgets.QDoubleSpinBox()
        self.min_spinbox.setRange(-999999, 999999)
        self.min_spinbox.setDecimals(3)
        self.min_spinbox.setValue(0.0)
        minmax_layout.addWidget(self.min_spinbox)
        
        minmax_layout.addWidget(QtWidgets.QLabel("Max:"))
        self.max_spinbox = QtWidgets.QDoubleSpinBox()
        self.max_spinbox.setRange(-999999, 999999)
        self.max_spinbox.setDecimals(3)
        self.max_spinbox.setValue(100.0)
        minmax_layout.addWidget(self.max_spinbox)
        
        scaling_layout.addLayout(minmax_layout)
        
        # Control buttons
        button_layout = QtWidgets.QHBoxLayout()
        self.auto_contrast_btn = QtWidgets.QPushButton("Auto Contrast")
        self.auto_contrast_btn.clicked.connect(self._comparison_auto_contrast)
        button_layout.addWidget(self.auto_contrast_btn)
        
        self.default_range_btn = QtWidgets.QPushButton("Original Range")
        self.default_range_btn.clicked.connect(self._comparison_default_range)
        button_layout.addWidget(self.default_range_btn)
        
        self.apply_scaling_btn = QtWidgets.QPushButton("Apply")
        self.apply_scaling_btn.clicked.connect(self._apply_comparison_scaling)
        self.apply_scaling_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        button_layout.addWidget(self.apply_scaling_btn)
        
        scaling_layout.addLayout(button_layout)
        self.scaling_frame.setVisible(False)
        
        options_layout.addWidget(self.scaling_frame)
        
        control_layout.addWidget(channel_group)
        control_layout.addWidget(options_group)
        
        layout.addWidget(control_panel)
        
        # Image display area
        self.image_scroll = QtWidgets.QScrollArea()
        self.image_widget = QtWidgets.QWidget()
        self.image_layout = QtWidgets.QGridLayout(self.image_widget)
        self.image_scroll.setWidget(self.image_widget)
        self.image_scroll.setWidgetResizable(True)
        layout.addWidget(self.image_scroll)
        
        # Connect signals
        self.add_acq_btn.clicked.connect(self._add_acquisition)
        self.remove_acq_btn.clicked.connect(self._remove_acquisition)
        self.channel_combo.currentTextChanged.connect(self._update_display)
        self.link_chk.toggled.connect(self._on_link_contrast_toggled)
        self.grayscale_chk.toggled.connect(self._update_display)
        self.acq_list.itemSelectionChanged.connect(self._update_display)
        
        # Initialize channel combo when acquisitions are added
        self._update_channel_combo()

    def _add_acquisition(self):
        current_item = self.available_acq_list.currentItem()
        if current_item:
            acq_id = current_item.data(Qt.UserRole)
            if acq_id not in self.selected_acquisitions:
                # Store current channel before updating
                current_channel = self.channel_combo.currentText()
                
                self.selected_acquisitions.append(acq_id)
                # Create a new item with the same data
                new_item = QtWidgets.QListWidgetItem(current_item.text())
                new_item.setData(Qt.UserRole, acq_id)
                self.acq_list.addItem(new_item)
                
                # Update channel combo (will preserve current channel if possible)
                self._update_channel_combo()
                
                # Preload images for this acquisition in background
                self._preload_images(acq_id)

    def _remove_acquisition(self):
        current_item = self.acq_list.currentItem()
        if current_item:
            acq_id = current_item.data(Qt.UserRole)
            self.selected_acquisitions.remove(acq_id)
            self.acq_list.takeItem(self.acq_list.row(current_item))
            self._update_channel_combo()

    def _update_display(self):
        # Clear existing images
        for i in reversed(range(self.image_layout.count())):
            self.image_layout.itemAt(i).widget().setParent(None)
        
        if not self.selected_acquisitions or not self.channel_combo.currentText():
            return
        
        channel = self.channel_combo.currentText()
        self.last_selected_channel = channel
        grayscale = self.grayscale_chk.isChecked()
        link_contrast = self.link_chk.isChecked()
        
        # Load images
        images = []
        titles = []
        for acq_id in self.selected_acquisitions:
            try:
                img = self.loader.get_image(acq_id, channel)
                images.append(img)
                titles.append(self._get_acquisition_subtitle(acq_id))
            except Exception as e:
                print(f"Error loading image for {acq_id}: {e}")
                continue
        
        if not images:
            return
        
        # Calculate grid layout
        n_images = len(images)
        cols = min(3, n_images)
        rows = (n_images + cols - 1) // cols
        
        # Calculate scaling based on custom scaling and link contrast settings
        custom_scaling_enabled = self.custom_scaling_chk.isChecked()
        custom_min = self.min_spinbox.value() if custom_scaling_enabled else None
        custom_max = self.max_spinbox.value() if custom_scaling_enabled else None
        
        # Calculate global min/max if link contrast is enabled
        vmin = None
        vmax = None
        if link_contrast and len(images) > 1:
            if custom_scaling_enabled:
                # Use custom scaling for linked contrast
                vmin = custom_min
                vmax = custom_max
            else:
                # Use global min/max for fair comparison
                vmin = min(np.min(img) for img in images)
                vmax = max(np.max(img) for img in images)
        elif not link_contrast and custom_scaling_enabled:
            # For individual scaling with custom scaling enabled,
            # each image can have its own custom scaling range
            # We'll handle this per-image in the display loop
            pass
        
        # Display images
        for i, (img, title) in enumerate(zip(images, titles)):
            row = i // cols
            col = i % cols
            
            # Create canvas
            canvas = MplCanvas(width=4, height=4, dpi=100)
            
            # Determine scaling for this image
            img_vmin = vmin
            img_vmax = vmax
            
            if img_vmin is None or img_vmax is None:
                # Individual scaling - check for per-image custom scaling
                acq_id = self.selected_acquisitions[i]
                if custom_scaling_enabled and not link_contrast and acq_id in self.image_scaling:
                    # Use custom scaling for this specific image
                    img_vmin = self.image_scaling[acq_id]['min']
                    img_vmax = self.image_scaling[acq_id]['max']
                else:
                    # Use image's own min/max range
                    img_vmin = np.min(img)
                    img_vmax = np.max(img)
            
            # Display image with appropriate scaling
            im = canvas.ax.imshow(img, interpolation="nearest", 
                                cmap='gray' if grayscale else 'viridis',
                                vmin=img_vmin, vmax=img_vmax)
            
            canvas.ax.set_title(title, fontsize=10)
            canvas.ax.axis("off")
            
            # Add colorbar
            cbar = canvas.fig.colorbar(im, ax=canvas.ax, shrink=0.8, aspect=20)
            
            self.image_layout.addWidget(canvas, row, col)
        
    def _update_channel_combo(self):
        """Update the channel combo box based on selected acquisitions."""
        old_channel = self.channel_combo.currentText()
        self.channel_combo.clear()
        if self.selected_acquisitions:
            # Get channels from first selected acquisition
            ai = next(a for a in self.acqs if a.id == self.selected_acquisitions[0])
            self.channel_combo.addItems(ai.channels)
            
            # Pre-select the last selected channel if it exists in this acquisition
            if self.last_selected_channel and self.last_selected_channel in ai.channels:
                index = ai.channels.index(self.last_selected_channel)
                self.channel_combo.setCurrentIndex(index)
            elif old_channel and old_channel in ai.channels:
                # Try to preserve the previously selected channel if it exists
                index = ai.channels.index(old_channel)
                self.channel_combo.setCurrentIndex(index)
                self.last_selected_channel = old_channel
            elif ai.channels:
                # Select first channel by default if no previous selection
                self.channel_combo.setCurrentIndex(0)
                self.last_selected_channel = ai.channels[0]
            
            # Clear cache if channel changed
            new_channel = self.channel_combo.currentText()
            if old_channel and old_channel != new_channel:
                self._clear_cache()
            
            # Auto-load images if channel is selected
            if self.channel_combo.currentText():
                self._update_display()
        else:
            self._clear_cache()

    def _clear_cache(self):
        """Clear the image cache."""
        self.image_cache.clear()
    
    def _manage_cache_size(self):
        """Remove oldest entries if cache exceeds max size."""
        if len(self.image_cache) > self.max_cache_size:
            # Remove oldest entries (simple FIFO)
            keys_to_remove = list(self.image_cache.keys())[:len(self.image_cache) - self.max_cache_size]
            for key in keys_to_remove:
                del self.image_cache[key]
    
    def _get_acquisition_subtitle(self, acq_id: str) -> str:
        """Get acquisition subtitle showing well/description instead of acquisition number."""
        acq_info = next((ai for ai in self.acqs if ai.id == acq_id), None)
        if not acq_info:
            return "Unknown"
        
        # Use well if available, otherwise use name (which might be more descriptive)
        if acq_info.well:
            return f"{acq_info.well}"
        else:
            return acq_info.name

    def _on_comparison_scaling_toggled(self):
        """Handle custom scaling checkbox toggle in comparison mode."""
        self.scaling_frame.setVisible(self.custom_scaling_chk.isChecked())
        if self.custom_scaling_chk.isChecked():
            if self.link_chk.isChecked():
                self._comparison_auto_range()
            else:
                self._update_image_combo()
        
        # Update control states based on link contrast
        self._on_link_contrast_toggled()

    def _comparison_auto_range(self):
        """Set min/max values to the current images' range in comparison mode."""
        if not self.selected_acquisitions or not self.channel_combo.currentText():
            return
        
        channel = self.channel_combo.currentText()
        try:
            # Get all images to determine range
            images = []
            for acq_id in self.selected_acquisitions:
                img = self.loader.get_image(acq_id, channel)
                images.append(img)
            
            if images:
                # Calculate global range across all images
                global_min = min(np.min(img) for img in images)
                global_max = max(np.max(img) for img in images)
                
                self.min_spinbox.setValue(float(global_min))
                self.max_spinbox.setValue(float(global_max))
        except Exception as e:
            print(f"Error getting comparison range: {e}")

    def _comparison_auto_contrast(self):
        """Set scaling to maximize contrast using percentile-based scaling in comparison mode."""
        if not self.selected_acquisitions or not self.channel_combo.currentText():
            return
        
        channel = self.channel_combo.currentText()
        
        if self.link_chk.isChecked():
            # Linked scaling - use global percentiles across all images
            try:
                images = []
                for acq_id in self.selected_acquisitions:
                    img = self.loader.get_image(acq_id, channel)
                    images.append(img)
                
                if images:
                    all_pixels = np.concatenate([img.flatten() for img in images])
                    min_val = float(np.percentile(all_pixels, 1))
                    max_val = float(np.percentile(all_pixels, 99))
                    
                    self.min_spinbox.setValue(min_val)
                    self.max_spinbox.setValue(max_val)
            except Exception as e:
                print(f"Error in comparison auto contrast: {e}")
        else:
            # Individual scaling - use percentiles for selected image only
            current_acq_id = self.image_combo.currentData()
            if not current_acq_id:
                return
            
            try:
                img = self.loader.get_image(current_acq_id, channel)
                min_val = float(np.percentile(img, 1))
                max_val = float(np.percentile(img, 99))
                
                self.min_spinbox.setValue(min_val)
                self.max_spinbox.setValue(max_val)
            except Exception as e:
                print(f"Error in individual auto contrast: {e}")

    def _comparison_default_range(self):
        """Set scaling to the images' actual min/max range in comparison mode."""
        if not self.selected_acquisitions or not self.channel_combo.currentText():
            return
        
        channel = self.channel_combo.currentText()
        
        if self.link_chk.isChecked():
            # Linked scaling - use global range across all images
            try:
                images = []
                for acq_id in self.selected_acquisitions:
                    img = self.loader.get_image(acq_id, channel)
                    images.append(img)
                
                if images:
                    global_min = min(np.min(img) for img in images)
                    global_max = max(np.max(img) for img in images)
                    
                    self.min_spinbox.setValue(float(global_min))
                    self.max_spinbox.setValue(float(global_max))
            except Exception as e:
                print(f"Error in comparison default range: {e}")
        else:
            # Individual scaling - use range for selected image only
            current_acq_id = self.image_combo.currentData()
            if not current_acq_id:
                return
            
            try:
                img = self.loader.get_image(current_acq_id, channel)
                min_val = float(np.min(img))
                max_val = float(np.max(img))
                
                self.min_spinbox.setValue(min_val)
                self.max_spinbox.setValue(max_val)
            except Exception as e:
                print(f"Error in individual default range: {e}")

    def _apply_comparison_scaling(self):
        """Apply the current scaling settings and refresh display in comparison mode."""
        if not self.link_chk.isChecked():
            # For individual scaling, save the scaling for the selected image
            self._save_image_scaling()
        self._update_display()

    def _on_link_contrast_toggled(self):
        """Handle link contrast checkbox toggle in comparison mode."""
        # Update custom scaling controls based on link contrast setting
        if self.link_chk.isChecked():
            # Link contrast ON - custom scaling controls are enabled
            self.min_spinbox.setEnabled(True)
            self.max_spinbox.setEnabled(True)
            self.auto_contrast_btn.setEnabled(True)
            self.default_range_btn.setEnabled(True)
            self.apply_scaling_btn.setEnabled(True)
            self.image_combo.setEnabled(False)  # No image selection needed for linked scaling
        else:
            # Link contrast OFF - individual scaling, enable image selection
            self.min_spinbox.setEnabled(True)
            self.max_spinbox.setEnabled(True)
            self.auto_contrast_btn.setEnabled(True)
            self.default_range_btn.setEnabled(True)
            self.apply_scaling_btn.setEnabled(True)
            self.image_combo.setEnabled(True)  # Enable image selection for individual scaling
            self._update_image_combo()
        
        # Refresh display
        self._update_display()

    def _update_image_combo(self):
        """Update the image selection combo box with selected acquisitions."""
        self.image_combo.clear()
        if not self.selected_acquisitions or not self.channel_combo.currentText():
            return
        
        channel = self.channel_combo.currentText()
        for acq_id in self.selected_acquisitions:
            acq_info = next((ai for ai in self.acqs if ai.id == acq_id), None)
            if acq_info:
                label = f"{acq_info.name}" + (f" ({acq_info.well})" if acq_info.well else "")
                self.image_combo.addItem(label, acq_id)
        
        # Select first image if available
        if self.image_combo.count() > 0:
            self.image_combo.setCurrentIndex(0)
            self._load_image_scaling()

    def _on_image_selection_changed(self):
        """Handle changes to the image selection for individual scaling."""
        if not self.link_chk.isChecked() and self.custom_scaling_chk.isChecked():
            self._load_image_scaling()

    def _load_image_scaling(self):
        """Load scaling values for the currently selected image."""
        current_acq_id = self.image_combo.currentData()
        if not current_acq_id or not self.channel_combo.currentText():
            return
        
        if current_acq_id in self.image_scaling:
            # Load saved values
            min_val = self.image_scaling[current_acq_id]['min']
            max_val = self.image_scaling[current_acq_id]['max']
        else:
            # Use default range (image's own min/max)
            try:
                channel = self.channel_combo.currentText()
                img = self.loader.get_image(current_acq_id, channel)
                min_val = float(np.min(img))
                max_val = float(np.max(img))
            except Exception as e:
                print(f"Error loading image scaling: {e}")
                return
        
        # Update spinboxes
        self.min_spinbox.setValue(min_val)
        self.max_spinbox.setValue(max_val)

    def _save_image_scaling(self):
        """Save current scaling values for the selected image."""
        current_acq_id = self.image_combo.currentData()
        if not current_acq_id:
            return
        
        min_val = self.min_spinbox.value()
        max_val = self.max_spinbox.value()
        
        self.image_scaling[current_acq_id] = {'min': min_val, 'max': max_val}

    def _preload_images(self, acq_id):
        """Preload images for an acquisition in the background."""
        if not self.channel_combo.currentText():
            return
            
        channel = self.channel_combo.currentText()
        cache_key = (acq_id, channel)
        
        if cache_key not in self.image_cache:
            try:
                img = self.loader.get_image(acq_id, channel)
                self.image_cache[cache_key] = img
                self._manage_cache_size()
            except Exception as e:
                print(f"Error preloading image for {acq_id}: {e}")

    def _refresh_markers(self):
        """Legacy method - redirect to _update_channel_combo."""
        self._update_channel_combo()


# --------------------------
# Main Window
# --------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IMC .mcd File Viewer")
        
        # Set window size to full screen
        screen = QtWidgets.QApplication.desktop().screenGeometry()
        self.resize(screen.width(), screen.height())

        # State
        self.loader: Optional[MCDLoader] = None
        self.current_path: Optional[str] = None
        self.acquisitions: List[AcquisitionInfo] = []
        self.current_acq_id: Optional[str] = None

        # Per (acq_id, channel) annotations → label
        self.annotations: Dict[Tuple[str, str], str] = {}
        self.annotation_labels = ["Unlabeled", "High-quality", "Low-quality", "Artifact/Exclude"]
        
        # Store last selected channels for auto-selection
        self.last_selected_channels: List[str] = []

        # Widgets
        self.canvas = MplCanvas(width=6, height=6, dpi=100)
        self.open_btn = QtWidgets.QPushButton("Open .mcd")
        self.acq_combo = QtWidgets.QComboBox()
        self.channel_list = QtWidgets.QListWidget()
        self.channel_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.deselect_all_btn = QtWidgets.QPushButton("Deselect all")
        self.view_btn = QtWidgets.QPushButton("View selected")
        self.comparison_btn = QtWidgets.QPushButton("Comparison mode")
        
        # Visualization options
        self.grayscale_chk = QtWidgets.QCheckBox("Grayscale mode")
        self.grid_view_chk = QtWidgets.QCheckBox("Grid view for multiple channels")
        self.grid_view_chk.setChecked(True)
        
        # Custom scaling controls
        self.custom_scaling_chk = QtWidgets.QCheckBox("Custom scaling")
        self.custom_scaling_chk.toggled.connect(self._on_custom_scaling_toggled)
        
        self.scaling_frame = QtWidgets.QFrame()
        self.scaling_frame.setFrameStyle(QtWidgets.QFrame.Box)
        scaling_layout = QtWidgets.QVBoxLayout(self.scaling_frame)
        scaling_layout.addWidget(QtWidgets.QLabel("Custom Intensity Range:"))
        
        # Channel selection for per-channel scaling
        channel_row = QtWidgets.QHBoxLayout()
        channel_row.addWidget(QtWidgets.QLabel("Channel:"))
        self.scaling_channel_combo = QtWidgets.QComboBox()
        self.scaling_channel_combo.currentTextChanged.connect(self._on_scaling_channel_changed)
        channel_row.addWidget(self.scaling_channel_combo)
        channel_row.addStretch()
        scaling_layout.addLayout(channel_row)
        
        # Slider controls
        slider_layout = QtWidgets.QVBoxLayout()
        
        # Min slider
        min_row = QtWidgets.QHBoxLayout()
        min_row.addWidget(QtWidgets.QLabel("Min:"))
        self.min_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.min_slider.setRange(0, 1000)
        self.min_slider.setValue(0)
        self.min_slider.valueChanged.connect(self._on_slider_changed)
        min_row.addWidget(self.min_slider)
        self.min_label = QtWidgets.QLabel("0.000")
        self.min_label.setMinimumWidth(60)
        min_row.addWidget(self.min_label)
        slider_layout.addLayout(min_row)
        
        # Max slider
        max_row = QtWidgets.QHBoxLayout()
        max_row.addWidget(QtWidgets.QLabel("Max:"))
        self.max_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.max_slider.setRange(0, 1000)
        self.max_slider.setValue(1000)
        self.max_slider.valueChanged.connect(self._on_slider_changed)
        max_row.addWidget(self.max_slider)
        self.max_label = QtWidgets.QLabel("1000.000")
        self.max_label.setMinimumWidth(60)
        max_row.addWidget(self.max_label)
        slider_layout.addLayout(max_row)
        
        scaling_layout.addLayout(slider_layout)
        
        # Control buttons
        button_row = QtWidgets.QHBoxLayout()
        self.auto_contrast_btn = QtWidgets.QPushButton("Auto Contrast")
        self.auto_contrast_btn.clicked.connect(self._auto_contrast)
        button_row.addWidget(self.auto_contrast_btn)
        
        self.percentile_btn = QtWidgets.QPushButton("Percentile Scaling")
        self.percentile_btn.clicked.connect(self._percentile_scaling)
        button_row.addWidget(self.percentile_btn)
        
        self.default_range_btn = QtWidgets.QPushButton("Default Range")
        self.default_range_btn.clicked.connect(self._default_range)
        button_row.addWidget(self.default_range_btn)
        
        self.apply_btn = QtWidgets.QPushButton("Apply")
        self.apply_btn.clicked.connect(self._apply_scaling)
        self.apply_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        button_row.addWidget(self.apply_btn)
        
        scaling_layout.addLayout(button_row)
        self.scaling_frame.setVisible(False)
        
        # Store per-channel scaling values
        self.channel_scaling = {}  # {channel_name: {'min': value, 'max': value}}
        
        # Color assignment for RGB composite
        self.color_assignment_frame = QtWidgets.QFrame()
        self.color_assignment_frame.setFrameStyle(QtWidgets.QFrame.Box)
        color_layout = QtWidgets.QVBoxLayout(self.color_assignment_frame)
        color_layout.addWidget(QtWidgets.QLabel("Color Assignment (for RGB composite):"))
        
        color_row_layout = QtWidgets.QHBoxLayout()
        color_row_layout.addWidget(QtWidgets.QLabel("Red:"))
        self.red_combo = QtWidgets.QComboBox()
        self.red_combo.setMaximumWidth(120)
        color_row_layout.addWidget(self.red_combo)
        
        color_row_layout.addWidget(QtWidgets.QLabel("Green:"))
        self.green_combo = QtWidgets.QComboBox()
        self.green_combo.setMaximumWidth(120)
        color_row_layout.addWidget(self.green_combo)
        
        color_row_layout.addWidget(QtWidgets.QLabel("Blue:"))
        self.blue_combo = QtWidgets.QComboBox()
        self.blue_combo.setMaximumWidth(120)
        color_row_layout.addWidget(self.blue_combo)
        
        color_layout.addLayout(color_row_layout)

        self.ann_combo = QtWidgets.QComboBox()
        self.ann_combo.addItems(self.annotation_labels)
        self.ann_apply_btn = QtWidgets.QPushButton("Apply label")
        self.ann_save_btn = QtWidgets.QPushButton("Save annotations CSV")
        self.ann_load_btn = QtWidgets.QPushButton("Load annotations CSV")

        # Metadata display
        self.metadata_text = QtWidgets.QTextEdit()
        self.metadata_text.setMaximumHeight(150)
        self.metadata_text.setReadOnly(True)

        # Left panel layout
        controls = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(controls)
        v.addWidget(self.open_btn)

        v.addWidget(QtWidgets.QLabel("Acquisition:"))
        v.addWidget(self.acq_combo)

        v.addWidget(QtWidgets.QLabel("Channels:"))
        v.addWidget(self.channel_list, 1)
        
        # Channel control buttons
        channel_btn_row = QtWidgets.QHBoxLayout()
        channel_btn_row.addWidget(self.deselect_all_btn)
        channel_btn_row.addStretch()
        v.addLayout(channel_btn_row)
        
        # Visualization options
        v.addWidget(self.grayscale_chk)
        v.addWidget(self.grid_view_chk)
        v.addWidget(self.custom_scaling_chk)
        v.addWidget(self.scaling_frame)
        v.addWidget(self.color_assignment_frame)

        ann_row = QtWidgets.QHBoxLayout()
        ann_row.addWidget(QtWidgets.QLabel("Annotation:"))
        ann_row.addWidget(self.ann_combo, 1)
        v.addLayout(ann_row)
        v.addWidget(self.ann_apply_btn)

        v.addSpacing(8)
        v.addWidget(self.view_btn)
        v.addWidget(self.comparison_btn)
        v.addSpacing(8)
        v.addWidget(self.ann_save_btn)
        v.addWidget(self.ann_load_btn)
        v.addSpacing(8)
        
        v.addWidget(QtWidgets.QLabel("Metadata:"))
        v.addWidget(self.metadata_text)
        v.addStretch(1)

        # Splitter
        splitter = QtWidgets.QSplitter(Qt.Horizontal)
        leftw = QtWidgets.QWidget()
        leftw.setLayout(v)
        splitter.addWidget(leftw)
        splitter.addWidget(self.canvas)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

        # Menu
        file_menu = self.menuBar().addMenu("&File")
        act_open = file_menu.addAction("Open .mcd…")
        act_open.triggered.connect(self._open_dialog)
        file_menu.addSeparator()
        act_quit = file_menu.addAction("Quit")
        act_quit.triggered.connect(self.close)

        # Signals
        self.open_btn.clicked.connect(self._open_dialog)
        self.acq_combo.currentIndexChanged.connect(self._on_acq_changed)
        self.deselect_all_btn.clicked.connect(self._deselect_all_channels)
        self.channel_list.itemChanged.connect(self._on_channel_selection_changed)
        self.view_btn.clicked.connect(self._view_selected)
        self.comparison_btn.clicked.connect(self._comparison)
        self.ann_apply_btn.clicked.connect(self._apply_annotation)
        self.ann_save_btn.clicked.connect(self._save_annotations)
        self.ann_load_btn.clicked.connect(self._load_annotations)

        # Loader
        try:
            self.loader = MCDLoader()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Missing dependency", str(e))

    # ---------- File open ----------
    def _open_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open IMC .mcd file", "", "IMC MCD files (*.mcd);;All files (*.*)"
        )
        if path:
            self._load_mcd(path)

    def _load_mcd(self, path: str):
        if self.loader is None:
            try:
                self.loader = MCDLoader()
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Dependency error", str(e))
                return
        try:
            self.loader.open(path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Open failed", f"Failed to open {path}\n\n{e}")
            return
        self.current_path = path
        self.acquisitions = self.loader.list_acquisitions()
        self.acq_combo.clear()
        for ai in self.acquisitions:
            label = ai.name + (f"({ai.well})" if ai.well else "")
            self.acq_combo.addItem(label, ai.id)
        if self.acquisitions:
            self._populate_channels(self.acquisitions[0].id)

    # ---------- Acquisition / channels ----------
    def _on_acq_changed(self, idx: int):
        acq_id = self.acq_combo.itemData(idx)
        if acq_id:
            self._populate_channels(acq_id)
            # Update scaling channel combo when acquisition changes
            if self.custom_scaling_chk.isChecked():
                self._update_scaling_channel_combo()

    def _populate_channels(self, acq_id: str):
        self.current_acq_id = acq_id
        self.channel_list.clear()
        try:
            chans = self.loader.get_channels(acq_id)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Channels error", str(e))
            return
        
        # Pre-select channels that were selected in the previous acquisition
        selected_channels = []
        for ch in chans:
            item = QtWidgets.QListWidgetItem(ch)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable | Qt.ItemIsEnabled)
            
            # Check if this channel was selected in the previous acquisition
            if ch in self.last_selected_channels:
                item.setCheckState(Qt.Checked)
                selected_channels.append(ch)
            else:
                item.setCheckState(Qt.Unchecked)
            
            self.channel_list.addItem(item)
        
        # Update color assignment dropdowns
        self._populate_color_assignments(chans)
        
        # Auto-load image if channels were pre-selected
        if selected_channels:
            self._auto_load_image(selected_channels)
        
        # Update metadata display
        ai = next(a for a in self.acquisitions if a.id == acq_id)
        metadata_text = f"Acquisition: {ai.name}\n"
        if ai.well:
            metadata_text += f"{ai.well}\n"
        if ai.size[0] and ai.size[1]:
            metadata_text += f"Size: {ai.size[1]} x {ai.size[0]} pixels\n"
        metadata_text += f"Channels: {len(ai.channels)}\n\n"
        
        if ai.metadata:
            metadata_text += "Metadata:\n"
            for key, value in ai.metadata.items():
                metadata_text += f"  {key}: {value}\n"
        
        self.metadata_text.setPlainText(metadata_text)

    def _on_channel_selection_changed(self):
        """Update color assignment dropdowns when channel selection changes."""
        selected_channels = self._selected_channels()
        self._populate_color_assignments(selected_channels)
        
        # Clear any color assignments that are no longer in the selected channels
        self._clear_invalid_color_assignments(selected_channels)

    def _populate_color_assignments(self, channels: List[str]):
        """Populate the color assignment dropdowns with selected channels only."""
        # Clear existing items
        self.red_combo.clear()
        self.green_combo.clear()
        self.blue_combo.clear()
        
        # Add "None" option
        self.red_combo.addItem("None", None)
        self.green_combo.addItem("None", None)
        self.blue_combo.addItem("None", None)
        
        # Add only selected channels
        for ch in channels:
            self.red_combo.addItem(ch, ch)
            self.green_combo.addItem(ch, ch)
            self.blue_combo.addItem(ch, ch)
        
        # Set default assignments to None
        self.red_combo.setCurrentIndex(0)  # "None" is at index 0
        self.green_combo.setCurrentIndex(0)  # "None" is at index 0
        self.blue_combo.setCurrentIndex(0)  # "None" is at index 0

    def _clear_invalid_color_assignments(self, selected_channels: List[str]):
        """Clear color assignments that are no longer in the selected channels."""
        # Check each color assignment and clear if not in selected channels
        red_channel = self.red_combo.currentData()
        green_channel = self.green_combo.currentData()
        blue_channel = self.blue_combo.currentData()
        
        if red_channel and red_channel not in selected_channels:
            # Find "None" option and select it
            for i in range(self.red_combo.count()):
                if self.red_combo.itemData(i) is None:
                    self.red_combo.setCurrentIndex(i)
                    break
        
        if green_channel and green_channel not in selected_channels:
            # Find "None" option and select it
            for i in range(self.green_combo.count()):
                if self.green_combo.itemData(i) is None:
                    self.green_combo.setCurrentIndex(i)
                    break
        
        if blue_channel and blue_channel not in selected_channels:
            # Find "None" option and select it
            for i in range(self.blue_combo.count()):
                if self.blue_combo.itemData(i) is None:
                    self.blue_combo.setCurrentIndex(i)
                    break

    def _deselect_all_channels(self):
        """Deselect all channels in the channel list."""
        for i in range(self.channel_list.count()):
            item = self.channel_list.item(i)
            item.setCheckState(Qt.Unchecked)
        self.channel_list.clearSelection()
        
        # Clear all color assignments when deselecting all channels
        self._populate_color_assignments([])

    def _selected_channels(self) -> List[str]:
        chans: List[str] = []
        for i in range(self.channel_list.count()):
            it = self.channel_list.item(i)
            if it.checkState() == Qt.Checked or it.isSelected():
                chans.append(it.text())
        # unique, preserve order
        seen = set()
        uniq = []
        for c in chans:
            if c not in seen:
                uniq.append(c)
                seen.add(c)
        return uniq

    def _get_acquisition_subtitle(self, acq_id: str) -> str:
        """Get acquisition subtitle showing well/description instead of acquisition number."""
        acq_info = next((ai for ai in self.acquisitions if ai.id == acq_id), None)
        if not acq_info:
            return "Unknown"
        
        # Use well if available, otherwise use name (which might be more descriptive)
        if acq_info.well:
            return f"{acq_info.well}"
        else:
            return acq_info.name

    def _on_custom_scaling_toggled(self):
        """Handle custom scaling checkbox toggle."""
        self.scaling_frame.setVisible(self.custom_scaling_chk.isChecked())
        if self.custom_scaling_chk.isChecked():
            self._update_scaling_channel_combo()
            self._load_channel_scaling()

    def _update_scaling_channel_combo(self):
        """Update the scaling channel combo box with available channels."""
        self.scaling_channel_combo.clear()
        if self.current_acq_id is None:
            return
        
        # Get all available channels for current acquisition
        if hasattr(self, 'acquisitions') and self.acquisitions:
            acq_info = next((ai for ai in self.acquisitions if ai.id == self.current_acq_id), None)
            if acq_info and hasattr(acq_info, 'channels'):
                for channel in acq_info.channels:
                    self.scaling_channel_combo.addItem(channel)
        
        # Select first channel if available
        if self.scaling_channel_combo.count() > 0:
            self.scaling_channel_combo.setCurrentIndex(0)
            self._load_channel_scaling()

    def _on_scaling_channel_changed(self):
        """Handle changes to the scaling channel selection."""
        if self.custom_scaling_chk.isChecked():
            self._load_channel_scaling()

    def _on_slider_changed(self):
        """Handle changes to the min/max sliders."""
        if self.custom_scaling_chk.isChecked():
            self._update_slider_labels()
            # Don't auto-refresh display - user must click Apply

    def _update_slider_labels(self):
        """Update the min/max labels based on slider values."""
        current_channel = self.scaling_channel_combo.currentText()
        if not current_channel or self.current_acq_id is None:
            return
        
        try:
            img = self.loader.get_image(self.current_acq_id, current_channel)
            img_min, img_max = float(np.min(img)), float(np.max(img))
            
            # Convert slider values (0-1000) to actual image range
            min_val = img_min + (self.min_slider.value() / 1000.0) * (img_max - img_min)
            max_val = img_min + (self.max_slider.value() / 1000.0) * (img_max - img_min)
            
            self.min_label.setText(f"{min_val:.3f}")
            self.max_label.setText(f"{max_val:.3f}")
        except Exception as e:
            print(f"Error updating slider labels: {e}")

    def _load_channel_scaling(self):
        """Load scaling values for the currently selected channel."""
        current_channel = self.scaling_channel_combo.currentText()
        if not current_channel:
            return
        
        if current_channel in self.channel_scaling:
            # Load saved values
            min_val = self.channel_scaling[current_channel]['min']
            max_val = self.channel_scaling[current_channel]['max']
        else:
            # Use default range (full image range)
            if self.current_acq_id is None:
                return
            try:
                img = self.loader.get_image(self.current_acq_id, current_channel)
                min_val = float(np.min(img))
                max_val = float(np.max(img))
            except Exception as e:
                print(f"Error loading channel scaling: {e}")
                return
        
        # Update sliders based on actual values
        self._update_sliders_from_values(min_val, max_val)

    def _save_channel_scaling(self):
        """Save current scaling values for the selected channel."""
        current_channel = self.scaling_channel_combo.currentText()
        if not current_channel or self.current_acq_id is None:
            return
        
        try:
            img = self.loader.get_image(self.current_acq_id, current_channel)
            img_min, img_max = float(np.min(img)), float(np.max(img))
            
            # Convert slider values to actual image range
            min_val = img_min + (self.min_slider.value() / 1000.0) * (img_max - img_min)
            max_val = img_min + (self.max_slider.value() / 1000.0) * (img_max - img_min)
            
            self.channel_scaling[current_channel] = {'min': min_val, 'max': max_val}
        except Exception as e:
            print(f"Error saving channel scaling: {e}")

    def _update_sliders_from_values(self, min_val, max_val):
        """Update sliders based on actual min/max values."""
        if self.current_acq_id is None:
            return
        
        current_channel = self.scaling_channel_combo.currentText()
        if not current_channel:
            return
        
        try:
            img = self.loader.get_image(self.current_acq_id, current_channel)
            img_min, img_max = float(np.min(img)), float(np.max(img))
            
            # Convert actual values to slider positions (0-1000)
            if img_max > img_min:
                min_slider_val = int(((min_val - img_min) / (img_max - img_min)) * 1000)
                max_slider_val = int(((max_val - img_min) / (img_max - img_min)) * 1000)
            else:
                min_slider_val = 0
                max_slider_val = 1000
            
            # Update sliders without triggering valueChanged
            self.min_slider.blockSignals(True)
            self.max_slider.blockSignals(True)
            self.min_slider.setValue(min_slider_val)
            self.max_slider.setValue(max_slider_val)
            self.min_slider.blockSignals(False)
            self.max_slider.blockSignals(False)
            
            # Update labels
            self._update_slider_labels()
        except Exception as e:
            print(f"Error updating sliders from values: {e}")

    def _auto_contrast(self):
        """Set scaling to maximize contrast using percentile-based scaling."""
        if self.current_acq_id is None:
            return
        
        current_channel = self.scaling_channel_combo.currentText()
        if not current_channel:
            return
        
        try:
            img = self.loader.get_image(self.current_acq_id, current_channel)
            # Use 1st and 99th percentiles for better contrast
            min_val = float(np.percentile(img, 1))
            max_val = float(np.percentile(img, 99))
            
            self._update_sliders_from_values(min_val, max_val)
            # Don't auto-apply - user must click Apply button
        except Exception as e:
            print(f"Error in auto contrast: {e}")

    def _percentile_scaling(self):
        """Set scaling using robust percentile scaling (1st-99th percentiles)."""
        if self.current_acq_id is None:
            return
        
        current_channel = self.scaling_channel_combo.currentText()
        if not current_channel:
            return
        
        try:
            img = self.loader.get_image(self.current_acq_id, current_channel)
            # Use robust percentile scaling function
            scaled_img = robust_percentile_scale(img, low=1.0, high=99.0)
            
            # Get the actual min/max values that were used for scaling
            min_val = float(np.percentile(img, 1))
            max_val = float(np.percentile(img, 99))
            
            self._update_sliders_from_values(min_val, max_val)
            # Don't auto-apply - user must click Apply button
        except Exception as e:
            print(f"Error in percentile scaling: {e}")

    def _default_range(self):
        """Set scaling to the image's actual min/max range."""
        if self.current_acq_id is None:
            return
        
        current_channel = self.scaling_channel_combo.currentText()
        if not current_channel:
            return
        
        try:
            img = self.loader.get_image(self.current_acq_id, current_channel)
            min_val = float(np.min(img))
            max_val = float(np.max(img))
            
            self._update_sliders_from_values(min_val, max_val)
            # Don't auto-apply - user must click Apply button
        except Exception as e:
            print(f"Error in default range: {e}")

    def _apply_scaling(self):
        """Apply the current scaling settings to the selected channel and refresh display."""
        if self.current_acq_id is None:
            return
        
        current_channel = self.scaling_channel_combo.currentText()
        if not current_channel:
            return
        
        # Save current scaling values
        self._save_channel_scaling()
        
        # Refresh display
        self._view_selected()


    # ---------- View ----------
    def _view_selected(self):
        if self.current_acq_id is None:
            QtWidgets.QMessageBox.information(self, "No acquisition", "Open a .mcd and select an acquisition.")
            return
        chans = self._selected_channels()
        if not chans:
            QtWidgets.QMessageBox.information(self, "No channels", "Select one or more channels.")
            return
        
        # Store selected channels for auto-selection in next acquisition
        self.last_selected_channels = chans.copy()
        
        grayscale = self.grayscale_chk.isChecked()
        grid_view = self.grid_view_chk.isChecked()
        
        # Get custom scaling values if enabled
        # For single channel view, use that channel's scaling
        # For RGB/grid view, we'll handle per-channel scaling in the display methods
        custom_min = None
        custom_max = None
        if self.custom_scaling_chk.isChecked() and len(chans) == 1:
            # For single channel, use the scaling for that specific channel
            channel = chans[0]
            if channel in self.channel_scaling:
                custom_min = self.channel_scaling[channel]['min']
                custom_max = self.channel_scaling[channel]['max']
        
        try:
            if len(chans) == 1 and not grid_view:
                # Single channel view
                img = self.loader.get_image(self.current_acq_id, chans[0])
                # Get acquisition subtitle
                acq_subtitle = self._get_acquisition_subtitle(self.current_acq_id)
                title = f"{chans[0]}\n{acq_subtitle}"
                self.canvas.show_image(img, title, grayscale=grayscale, raw_img=img, custom_min=custom_min, custom_max=custom_max)
            elif len(chans) <= 3 and not grid_view:
                # RGB composite view using user-selected color assignments
                self._show_rgb_composite(chans, grayscale)
            else:
                # Grid view for multiple channels (when grid_view is True or >3 channels)
                images = [self.loader.get_image(self.current_acq_id, c) for c in chans]
                # Get acquisition subtitle
                acq_subtitle = self._get_acquisition_subtitle(self.current_acq_id)
                # Add acquisition subtitle to each channel title
                titles = [f"{ch}\n{acq_subtitle}" for ch in chans]
                self.canvas.show_grid(images, titles, grayscale=grayscale, raw_images=images, 
                                    channel_names=chans, channel_scaling=self.channel_scaling, 
                                    custom_scaling_enabled=self.custom_scaling_chk.isChecked())
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "View error", str(e))

    def _auto_load_image(self, selected_channels: List[str]):
        """Automatically load and display image for pre-selected channels."""
        try:
            grayscale = self.grayscale_chk.isChecked()
            grid_view = self.grid_view_chk.isChecked()
            
            # Get custom scaling values if enabled
            custom_min = None
            custom_max = None
            if self.custom_scaling_chk.isChecked() and len(selected_channels) == 1:
                # For single channel, use the scaling for that specific channel
                channel = selected_channels[0]
                if channel in self.channel_scaling:
                    custom_min = self.channel_scaling[channel]['min']
                    custom_max = self.channel_scaling[channel]['max']
            
            if len(selected_channels) == 1 and not grid_view:
                # Single channel view
                img = self.loader.get_image(self.current_acq_id, selected_channels[0])
                # Get acquisition subtitle
                acq_subtitle = self._get_acquisition_subtitle(self.current_acq_id)
                title = f"{selected_channels[0]}\n{acq_subtitle}"
                self.canvas.show_image(img, title, grayscale=grayscale, raw_img=img, custom_min=custom_min, custom_max=custom_max)
            elif len(selected_channels) <= 3 and not grid_view:
                # RGB composite view using user-selected color assignments
                self._show_rgb_composite(selected_channels, grayscale)
            else:
                # Grid view for multiple channels (when grid_view is True or >3 channels)
                images = [self.loader.get_image(self.current_acq_id, c) for c in selected_channels]
                # Get acquisition subtitle
                acq_subtitle = self._get_acquisition_subtitle(self.current_acq_id)
                # Add acquisition subtitle to each channel title
                titles = [f"{ch}\n{acq_subtitle}" for ch in selected_channels]
                self.canvas.show_grid(images, titles, grayscale=grayscale, raw_images=images, 
                                    channel_names=selected_channels, channel_scaling=self.channel_scaling, 
                                    custom_scaling_enabled=self.custom_scaling_chk.isChecked())
        except Exception as e:
            print(f"Auto-load error: {e}")

    def _show_rgb_composite(self, selected_channels: List[str], grayscale: bool):
        """Show RGB composite using user-selected color assignments."""
        # Get user-selected color assignments
        red_channel = self.red_combo.currentData()
        green_channel = self.green_combo.currentData()
        blue_channel = self.blue_combo.currentData()
        
        # If currentData() returns None, try currentText()
        if red_channel is None:
            red_channel = self.red_combo.currentText() if self.red_combo.currentText() != "None" else None
        if green_channel is None:
            green_channel = self.green_combo.currentText() if self.green_combo.currentText() != "None" else None
        if blue_channel is None:
            blue_channel = self.blue_combo.currentText() if self.blue_combo.currentText() != "None" else None
        
        # If no color assignments are made, use the first 1-3 selected channels as default
        if not red_channel and not green_channel and not blue_channel:
            if len(selected_channels) >= 1:
                red_channel = selected_channels[0]
            if len(selected_channels) >= 2:
                green_channel = selected_channels[1]
            if len(selected_channels) >= 3:
                blue_channel = selected_channels[2]
        
        # Create RGB stack based on user selections
        rgb_channels = []
        rgb_titles = []
        raw_channels = []  # Store raw images for colorbar
        
        # Get the first selected channel to determine image size
        first_img = None
        if selected_channels:
            first_img = self.loader.get_image(self.current_acq_id, selected_channels[0])
        
        if first_img is None:
            QtWidgets.QMessageBox.information(self, "No RGB channels", "Please select at least one channel for RGB composite.")
            return
        
        # Always create 3 channels (R, G, B) even if some are empty
        for color, channel, color_name in [(red_channel, red_channel, 'Red'), (green_channel, green_channel, 'Green'), (blue_channel, blue_channel, 'Blue')]:
            # Convert both to strings for comparison to handle any type mismatches
            channel_str = str(channel) if channel else None
            selected_channels_str = [str(c) for c in selected_channels]
            
            if channel and channel_str in selected_channels_str:
                img = self.loader.get_image(self.current_acq_id, channel)
                rgb_channels.append(img)
                raw_channels.append(img)
                rgb_titles.append(f"{channel} ({color_name})")
            else:
                # Add zero-filled channel if not selected
                rgb_channels.append(np.zeros_like(first_img))
                raw_channels.append(np.zeros_like(first_img))
                rgb_titles.append(f"None ({color_name})")
        
        # Ensure we have exactly 3 channels
        while len(rgb_channels) < 3:
            rgb_channels.append(np.zeros_like(first_img))
            raw_channels.append(np.zeros_like(first_img))
            rgb_titles.append(f"None ({['Red', 'Green', 'Blue'][len(rgb_channels)-1]})")
        
        # Stack channels
        stack = np.dstack(rgb_channels)
        raw_stack = np.dstack(raw_channels)
        
        # Get acquisition subtitle
        acq_subtitle = self._get_acquisition_subtitle(self.current_acq_id)
        title = " + ".join(rgb_titles) + f"\n{acq_subtitle}"
        
        # Clear canvas and show RGB composite with individual colorbars
        self.canvas.fig.clear()
        
        if grayscale:
            # Single grayscale image with colorbar
            ax = self.canvas.fig.add_subplot(111)
            
            # Get per-channel scaling for the first channel
            vmin, vmax = None, None
            if self.custom_scaling_chk.isChecked() and selected_channels and len(selected_channels) > 0:
                first_channel = selected_channels[0]
                if first_channel in self.channel_scaling:
                    vmin = self.channel_scaling[first_channel]['min']
                    vmax = self.channel_scaling[first_channel]['max']
            
            if vmin is None or vmax is None:
                vmin, vmax = np.min(stack[..., 0]), np.max(stack[..., 0])
            
            im = ax.imshow(stack[..., 0], interpolation="nearest", cmap='gray', vmin=vmin, vmax=vmax)
            # Add colorbar for grayscale
            cbar = self.canvas.fig.colorbar(im, ax=ax, shrink=0.8, aspect=20)
            cbar.set_ticks([vmin, vmax])
            cbar.set_ticklabels([f'{vmin:.1f}', f'{vmax:.1f}'])
            ax.set_title(title)
            ax.axis("off")
        else:
            # RGB composite with individual channel colorbars
            # Create a 2x2 grid: main image (top), colorbars (bottom)
            gs = self.canvas.fig.add_gridspec(2, 3, height_ratios=[3, 1], hspace=0.3, wspace=0.3)
            
            # Main RGB composite image (spans top row)
            ax_main = self.canvas.fig.add_subplot(gs[0, :])
            im = ax_main.imshow(stack_to_rgb(stack), interpolation="nearest")
            ax_main.set_title(title)
            ax_main.axis("off")
            
            # Individual channel colorbars (bottom row)
            for i, (channel_name, color_name) in enumerate(zip(rgb_titles, ['Red', 'Green', 'Blue'])):
                ax_cbar = self.canvas.fig.add_subplot(gs[1, i])
                
                # Create a colorbar for this channel
                if i < len(raw_channels) and raw_channels[i] is not None and not np.all(raw_channels[i] == 0):
                    # This channel has data
                    raw_min, raw_max = np.min(raw_channels[i]), np.max(raw_channels[i])
                    
                    # Check for per-channel scaling
                    if self.custom_scaling_chk.isChecked() and selected_channels and i < len(selected_channels):
                        channel = selected_channels[i]
                        if channel in self.channel_scaling:
                            raw_min = self.channel_scaling[channel]['min']
                            raw_max = self.channel_scaling[channel]['max']
                    
                    if raw_max > raw_min:  # Valid data range
                        # Create a gradient for the colorbar
                        gradient = np.linspace(0, 1, 256).reshape(1, -1)
                        ax_cbar.imshow(gradient, aspect='auto', cmap='gray' if grayscale else ['Reds', 'Greens', 'Blues'][i])
                        ax_cbar.set_xticks([0, 255])
                        ax_cbar.set_xticklabels([f'{raw_min:.1f}', f'{raw_max:.1f}'])
                        ax_cbar.set_yticks([])
                        ax_cbar.set_title(f"{channel_name}", fontsize=10)
                    else:
                        # No data
                        ax_cbar.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax_cbar.transAxes)
                        ax_cbar.set_title(f"{channel_name}", fontsize=10)
                else:
                    # No data for this channel
                    ax_cbar.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax_cbar.transAxes)
                    ax_cbar.set_title(f"{channel_name}", fontsize=10)
                
                ax_cbar.set_xlim(0, 255)
        
        self.canvas.draw()


    # ---------- Annotations ----------
    def _apply_annotation(self):
        if self.current_acq_id is None:
            QtWidgets.QMessageBox.information(self, "No acquisition", "Select an acquisition first.")
            return
        chans = self._selected_channels()
        if not chans:
            QtWidgets.QMessageBox.information(self, "No channels", "Select channel(s) to annotate.")
            return
        label = self.ann_combo.currentText()
        for ch in chans:
            self.annotations[(self.current_acq_id, ch)] = label
        QtWidgets.QMessageBox.information(self, "Annotated", f"Labeled {len(chans)} channel(s) as '{label}'.")

    def _save_annotations(self):
        if not self.annotations:
            QtWidgets.QMessageBox.information(self, "Nothing to save", "No annotations yet.")
            return
        rows = [
            {"acquisition_id": a, "channel": c, "label": lab}
            for (a, c), lab in self.annotations.items()
        ]
        df = pd.DataFrame(rows).sort_values(["acquisition_id", "channel"])
        base = "channel_annotations.csv"
        if self.current_path:
            stem = os.path.splitext(os.path.basename(self.current_path))[0]
            base = f"{stem}_annotations.csv"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save annotations CSV", base, "CSV (*.csv)")
        if not path:
            return
        try:
            df.to_csv(path, index=False)
            QtWidgets.QMessageBox.information(self, "Saved", f"Annotations saved to:\n{path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save failed", str(e))

    def _load_annotations(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load annotations CSV", "", "CSV (*.csv)")
        if not path:
            return
        try:
            df = pd.read_csv(path)
            if not {"acquisition_id", "channel", "label"}.issubset(df.columns):
                raise ValueError("CSV must have columns: acquisition_id, channel, label")
            for _, row in df.iterrows():
                self.annotations[(str(row["acquisition_id"]), str(row["channel"]))] = str(row["label"])
            QtWidgets.QMessageBox.information(self, "Loaded", f"Loaded {len(df)} annotations.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load failed", str(e))

    # ---------- Comparison ----------
    def _comparison(self):
        if not self.acquisitions:
            QtWidgets.QMessageBox.information(self, "No acquisitions", "Open a .mcd first.")
            return
        
        # Open the dynamic comparison dialog
        dlg = DynamicComparisonDialog(self.acquisitions, self.loader, self)
        dlg.exec_()

    def closeEvent(self, event):
        """Clean up when closing the application."""
        if self.loader:
            self.loader.close()
        event.accept()


# --------------------------
# Entrypoint
# --------------------------
def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
