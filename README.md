# MSCNet

MSCNet (Multi-Scale Network with Convolutions) is a deep learning framework for **long-term cloud workload prediction**. It is designed to capture **complex temporal dynamics** in cloud environments by combining **multi-scale patch representations**, **transformer-based sequence modeling**, and **multi-scale convolutional feature extraction**.

At its core, this project demonstrates how **multi-resolution temporal modeling** improves forecasting accuracy and stability for highly variable, long-horizon cloud workloads.

---

## System Architecture

The MSCNet workflow follows a sequential deep learning pipeline:

`Time-Series Input → Multi-Scale Patch Extraction → Transformer Encoder → Multi-Scale Convolution → Trend Prediction → Output Forecast`

1. **Input Module**  
   Accepts univariate or multivariate cloud workload time-series (e.g., CPU utilization, memory usage, request rate). Performs normalization and embedding.

2. **Multi-Scale Patch Block**  
   Decomposes the input time-series into multiple temporal resolutions to capture short-term variations and long-term periodic patterns.

3. **Transformer Encoder**  
   Models long-range temporal dependencies, enabling effective long-term forecasting.

4. **Multi-Scale Convolution Block**  
   Applies parallel convolution kernels of different sizes to extract local, mid-range, and global temporal features.

5. **Trend Prediction Block**  
   Explicitly models global workload trends to improve robustness against workload drift.

6. **Output Layer**  
   Generates the final long-term workload prediction with a configurable forecast horizon.

---

## Installation & Setup

This project is provided as **Jupyter Notebook (`.ipynb`) implementations**, and all experiments have **already been executed**.

- No separate installation or training setup is required.
- The notebooks include:
  - Data preprocessing
  - Model implementation
  - Training and evaluation
  - Result visualization

All experimental **results are available directly inside the notebooks**, where **MSCNet achieves the lowest mean error** compared to baseline models.

To reproduce or inspect the results:

```bash
# Clone the repository
git clone https://github.com/<your-username>/MSCNet.git
cd MSCNet
