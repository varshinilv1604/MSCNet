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

MSCNet runs on standard Python environments and does not require GPU support for basic experimentation.

```bash
# Clone the repository
git clone https://github.com/<your-username>/MSCNet.git
cd MSCNet

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
