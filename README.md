# mrfmsim Examples

This repository contains Jupyter notebooks and supporting code for reproducing the simulations and figures presented in the mrfmsim paper. The examples demonstrate the capabilities of the [mrfmsim](https://github.com/Marohn-Group/mrfmsim) package for simulating Magnetic Resonance Force Microscopy (MRFM) experiments.

## Overview

The repository includes three main case studies that compare experimental MRFM data with simulations:

- **Case Study A (Longenecker 2012)**: IBM-style cyclic-inversion MRFM, experiment data and lineshape analysis
- **Case Study B (Moore 2009)**: Time-dependent CERMIT (Cantilever-Enabled Readout of Magnetization Inversion Transients) experiment, experiment data and lineshape analysis
- **Appendix**: Comparison of CERMIT saturation algorithms (Moore's vs. Isaac's methods (Isaac 2018))

### Installation

The run the examples, use the specific package versions.

Install the mrfmsim (version 0.4.1) and mrfmsim-plot (version 0.1.1) using pip:

```bash
pip install mrfmsim==0.4.1
pip install git+https://github.com/Marohn-Group/mrfmsim-plot.git@v0.1.1

```

A "figures" folder needs to be created under the /example directory to properly save the figures generated in this notebook.

## References

1. **Moore et al. (2009)**: "Magnetic resonance force microscopy with a nickel tip and a near-surface ensemble of electron spins" *Proc. Natl. Acad. Sci. USA* **106**, 22251-22256
2. **Longenecker et al. (2012)**: "High-gradient nanomagnets on cantilevers for sensitive detection of nuclear magnetic resonance" *ACS Nano* **6**, 9637-9645
3. **Isaac et al. (2018)**: "Harnessing Electron Spin Labels for Single Molecule Magnetic Resonance Imaging", *Thesis*, Cornell University, 2018
