# pyplotman

Custom plot manager for Python as a `matplotlib.pyplot` wrapper with automatic multi-format and HDF5 saving.

## Features
- **Automatic Dual-Save**: Every `savefig` call generates both a PDF and a PNG (in a `png/` subdirectory) by default.
- **HDF5 Archiving**: Automatically mirrors images and raw plotting data (lines, scatters, stairs) into multiple HDF5 files.
- **Axis Metadata**: Automatically archives `xlabel`, `ylabel`, `xlim`, `ylim`, and scaling info as HDF5 attributes.
- **Lab Defaults**: Pre-configured with high-quality fonts (Helvetica, Arial, Liberation Sans), colorblind-friendly cycle, and constrained layout.
- **LSP Friendly**: Full type hinting for better autocompletion in VS Code/Vim.

## Installation

```bash
cd pyplotman
pip install -e .
```

## Usage

### Basic Usage
```python
from pyplotman import plt

# Plot like normal
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [4, 5, 6], label="Signal")

# Saves to plot.pdf and png/plot.png
plt.savefig("plot.pdf")
```

### HDF5 Archival
```python
# Add a destination with default groups ('images', 'plots', 'data')
plt.add_hdf5_dest("summary.h5")

# Add another with custom Lyse-style paths
plt.add_hdf5_dest("shot.h5", 
                  group_images="images/analysis/myscript",
                  group_plots="results/myscript/plots")

# This one call now saves to 2 local files AND 2 HDF5 files
plt.savefig("analysis")
```

## Configuration
You can change the global fallback group names:
```python
plt.group_images = "archived_plots"
plt.group_plots  = "previews"
plt.group_data   = "raw_arrays"
```
