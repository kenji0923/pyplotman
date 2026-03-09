from __future__ import annotations
import os
import sys
import types
import io
import numpy as np
import h5py
from typing import Any
from matplotlib import font_manager
import matplotlib.pyplot as real_plt
from matplotlib.figure import Figure
from matplotlib.patches import StepPatch
from cycler import cycler

def _save_to_h5_file(fig: Figure, h5: h5py.File, plot_name: str, groups: dict):
    """Internal helper to save image and data using specific group names."""
    script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    if not script_name or script_name == '-c': script_name = 'analysis'

    # --- PART A: Save Image (Byte Array) ---
    res_plots_group = h5.require_group(groups['plots'])
    buf = io.BytesIO()
    fig._original_savefig(buf, format='png', dpi=fig.canvas.get_renderer().dpi)
    img_bytes = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    if plot_name in res_plots_group: del res_plots_group[plot_name]
    res_plots_group.create_dataset(plot_name, data=img_bytes)

    # --- PART B: Save as HDF5 Image (HDFView compatible) ---
    img_group = h5.require_group(groups['images'])
    fig.canvas.draw()
    rgba_buf = fig.canvas.buffer_rgba()
    w, h = fig.canvas.get_width_height()
    pixel_data = np.frombuffer(rgba_buf, dtype=np.uint8).reshape((int(h), int(w), 4))
    rgb_data = np.array(pixel_data[:, :, :3], copy=True)
    if plot_name in img_group: del img_group[plot_name]
    ds = img_group.create_dataset(plot_name, data=rgb_data, compression="gzip")
    
    ds.attrs['CLASS'] = np.bytes_('IMAGE')
    ds.attrs['IMAGE_VERSION'] = np.bytes_('1.2')
    ds.attrs['IMAGE_SUBCLASS'] = np.bytes_('IMAGE_TRUECOLOR')
    ds.attrs['IMAGE_COLORMODEL'] = np.bytes_('RGB')
    ds.attrs['INTERLACE_MODE'] = np.bytes_('INTERLACE_PIXEL')
    ds.attrs['DISPLAY_ORIGIN'] = np.bytes_('UPPER_LEFT')
    ds.attrs['IMAGE_MINMAXRANGE'] = np.array([0, 255], dtype=np.uint8)

    # --- PART C: Save Raw Plotting Data ---
    data_root = h5.require_group(f"{groups['data']}/{plot_name}")
    for i, ax in enumerate(fig.axes):
        ax_label = ax.get_label()
        ax_id = getattr(ax, 'name', None)
        if not ax_id: ax_id = ax_label if ax_label and not ax_label.startswith('axes') else f'axis{i}'
        ax_group = data_root.require_group(ax_id)
        
        ax_group.attrs['xlabel'] = np.bytes_(ax.get_xlabel())
        ax_group.attrs['ylabel'] = np.bytes_(ax.get_ylabel())
        ax_group.attrs['xlim'] = np.asarray(ax.get_xlim())
        ax_group.attrs['ylim'] = np.asarray(ax.get_ylim())
        ax_group.attrs['xscale'] = np.bytes_(ax.get_xscale())
        ax_group.attrs['yscale'] = np.bytes_(ax.get_yscale())
        
        lines = ax.get_lines()
        if lines:
            lines_group = ax_group.require_group('lines')
            for j, line in enumerate(lines):
                label = line.get_label()
                line_id = label if label and not label.startswith('_') else f'line{j}'
                lg = lines_group.require_group(line_id)
                x, y = line.get_xdata(), line.get_ydata()
                if 'x' in lg: del lg['x']
                if 'y' in lg: del lg['y']
                lg.create_dataset('x', data=np.asarray(x))
                lg.create_dataset('y', data=np.asarray(y))
                lg.attrs['label'] = np.bytes_(str(label))

        if ax.collections:
            coll_group = ax_group.require_group('collections')
            for k, coll in enumerate(ax.collections):
                label = coll.get_label()
                coll_id = label if label and not label.startswith('_') else f'collection{k}'
                cg = coll_group.require_group(coll_id)
                offsets = coll.get_offsets()
                if len(offsets) > 0:
                    if 'offsets' in cg: del cg['offsets']
                    cg.create_dataset('offsets', data=np.asarray(offsets))
                    cg.attrs['label'] = np.bytes_(str(label))

        patches = [p for p in ax.patches if isinstance(p, StepPatch)]
        if patches:
            stairs_group = ax_group.require_group('stairs')
            for m, patch in enumerate(patches):
                label = patch.get_label()
                stairs_id = label if label and not label.startswith('_') else f'stairs{m}'
                sg = stairs_group.require_group(stairs_id)
                st_data = patch.get_data()
                if 'values' in sg: del sg['values']
                if 'edges' in sg: del sg['edges']
                sg.create_dataset('values', data=np.asarray(st_data.values))
                sg.create_dataset('edges', data=np.asarray(st_data.edges))
                sg.attrs['label'] = np.bytes_(str(label))

def _custom_figure_savefig(self, fname: str | os.PathLike, **kwargs):
    """Monkeypatched savefig that handles local multi-format and HDF5 archival.
    
    This function is automatically attached to all Figure objects. 
    It ensures that a PDF and a PNG (in png/ subdir) are created locally,
    and that data is mirrored to all registered HDF5 destinations.
    """
    target_path = os.path.abspath(fname)
    target_dir = os.path.dirname(target_path)
    base_name, ext = os.path.splitext(os.path.basename(target_path))
    
    # 1. Local Saves
    self._original_savefig(fname, **kwargs)
    if ext.lower() != '.pdf':
        self._original_savefig(os.path.join(target_dir, f"{base_name}.pdf"), **kwargs)
    
    png_dir = os.path.join(target_dir, "png")
    os.makedirs(png_dir, exist_ok=True)
    png_path = os.path.abspath(os.path.join(png_dir, f"{base_name}.png"))
    if target_path != png_path:
        self._original_savefig(png_path, **kwargs)

    # 2. Multi-HDF5 Saves
    for dest_config in plt.hdf5_destinations:
        try:
            dest = dest_config['path']
            groups = dest_config['groups']
            if isinstance(dest, (str, os.PathLike)):
                with h5py.File(dest, 'a') as h5:
                    _save_to_h5_file(self, h5, base_name, groups)
            elif isinstance(dest, h5py.File):
                _save_to_h5_file(self, dest, base_name, groups)
        except Exception:
            pass

# Monkeypatch Figure globally
if not hasattr(Figure, '_original_savefig'):
    Figure._original_savefig = Figure.savefig
    Figure.savefig = _custom_figure_savefig

class CustomPLT:
    """Wrapper for matplotlib.pyplot with custom lab defaults and HDF5 tracking.
    
    This class mimics the pyplot interface via __getattr__ while providing
    automated archival and consistent styling.
    
    Attributes:
        hdf5_destinations (list[dict]): List of registered HDF5 save targets.
        group_images (str): Default HDF5 group for Image API datasets.
        group_plots (str): Default HDF5 group for PNG byte-arrays.
        group_data (str): Default HDF5 group for raw numerical arrays.
    """
    
    default_figsize = (3.386 * 3, 2.418 * 3)

    def __init__(self):
        self._plt: types.ModuleType = real_plt
        self.hdf5_destinations: list[dict] = []
        
        # Global Fallback Defaults
        self.group_images = "images"
        self.group_plots = "plots"
        self.group_data = "data"
        
        self.set_defaults()

    def add_hdf5_dest(self, path: str | os.PathLike | h5py.File, 
                      group_images: str | None = None, 
                      group_plots: str | None = None, 
                      group_data: str | None = None):
        """Registers an HDF5 file for automatic plot archival.
        
        Args:
            path: File path or opened h5py.File handle.
            group_images: Custom group for images (defaults to self.group_images).
            group_plots: Custom group for byte-arrays (defaults to self.group_plots).
            group_data: Custom group for raw data (defaults to self.group_data).
        """
        self.hdf5_destinations.append({
            'path': path,
            'groups': {
                'images': group_images or self.group_images,
                'plots': group_plots or self.group_plots,
                'data': group_data or self.group_data
            }
        })

    def reset_hdf5_dests(self):
        """Clears all registered HDF5 destinations."""
        self.hdf5_destinations = []

    def set_defaults(self):
        """Configures matplotlib rcParams with lab-standard styles.
        
        Applies Liberation Sans font, custom color cycle, constrained layout,
        and high-resolution save settings.
        """
        font_path = '/usr/share/fonts/liberation-sans/LiberationSans-Regular.ttf'
        if os.path.exists(font_path):
            try: font_manager.fontManager.addfont(font_path)
            except Exception: pass

        self._plt.rcParams['figure.constrained_layout.use'] = True
        self._plt.rcParams['figure.constrained_layout.w_pad'] = 0.125
        self._plt.rcParams['figure.constrained_layout.h_pad'] = 0.15
        self._plt.rcParams['figure.constrained_layout.wspace'] = 0
        self._plt.rcParams['figure.constrained_layout.hspace'] = 0
        self._plt.rcParams['figure.figsize'] = self.default_figsize
        self._plt.rcParams['axes.labelpad'] = 10
        self._plt.rcParams['font.size'] = 8 * 3 
        self._plt.rcParams['font.family'] = 'sans-serif'
        self._plt.rcParams['font.sans-serif'] = ["Helvetica", "Arial", 'Liberation Sans', 'DejaVu Sans']
        self._plt.rcParams['pdf.fonttype'] = 42
        self._plt.rcParams['savefig.dpi'] = 150
        self._plt.rcParams["axes.xmargin"] = 0.01
        self._plt.rcParams["legend.frameon"] = False
        
        # Color cycle: Colorblind-friendly sequence
        colors = ["#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6", "#a96b59", "#e76300", "#b9ac70", "#717581", "#92dadd"]
        self._plt.rcParams['axes.prop_cycle'] = cycler(color=colors)

    def subplots(self, nrows: int = 1, ncols: int = 1, figsize: tuple[float, float] | None = None, **kwargs) -> Any:
        """Wrapper for plt.subplots with automatic figsize scaling."""
        if figsize is None:
            figsize = (self.default_figsize[0] * ncols, self.default_figsize[1] * nrows)
        return self._plt.subplots(nrows, ncols, figsize=figsize, **kwargs)

    def fill_range(self, ax, xmin, xmax, color='gray', alpha=0.3, **kwargs):
        """Draws a vertical span (axvspan) to highlight a range."""
        return ax.axvspan(xmin, xmax, color=color, alpha=alpha, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._plt, name)

plt = CustomPLT()
