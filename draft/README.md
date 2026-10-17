## 📁 Project Structure & Core Files

This repository contains the full production pipeline for the automated algorithmic trading bot, broken down into modular components:

* **`live_bot.py`**
    The main live execution engine. It runs an infinite loop that connects to the Upbit API via `ccxt`, pulls real-time order books, calculates rolling mathematical features on the fly, feeds them into the neural network, and executes live market orders. It includes safety guardrails for asset balance checks and `OMP` thread duplicate bypasses.

* **`test_trade.py`**
    The local sandbox script used to test API connectivity, verify authentication keys, simulate feature calculation, and safely dry-run orders before moving the system to full live-production status.

* **`quant_model.pth`**
    The compiled binary file containing the PyTorch neural network state dictionary (trained model weights). It is loaded dynamically by the execution scripts to generate real-time market prediction scores.

## 🛠️ Getting Started

1. **Install dependencies:**
   ```bash
   pip install ccxt torch numpy pandas joblib scikit-learn

1. **Run the live engine:**
   ```bash
   python live_bot.py
