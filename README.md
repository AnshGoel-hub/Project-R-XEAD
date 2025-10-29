# Project R-XEAD: AI Security Guard for LLMs

[![Python Version](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-310/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) AI security guard for Large Language Models (LLMs) using internal state analysis and a multi-layered defense to detect and block harmful prompts.

---

## 🏛️ Architecture: R-XEAD v4.2 (3-Layer Defense)

The R-XEAD system employs a three-layered pipeline architecture for comprehensive LLM defense:

1.  **Layer 0: Contextual Keyword Filter**
    * **Role:** First-line defense against specific "grey area" topics often missed by AI moderation.
    * **Mechanism:** Scans prompts for predefined banned keywords (e.g., "weed"). Includes a contextual allow-list (e.g., "novel", "scientific") to prevent blocking safe usage (False Positives).
    * **Catches:** Simple harmful requests involving specific terms not covered well by AI models.

2.  **Layer 1: Sentry (AI Content Moderation)**
    * **Role:** Second-line defense using a dedicated AI to classify general harmful content.
    * **Mechanism:** Employs a pre-trained, lightweight text classification model (`Vrandan/Comment-Moderation`) to analyze the prompt. Predicts multiple harm categories (Violence, Hate, etc.). Blocks if the top prediction is *not* "OK" (Safe).
    * **Catches:** Direct harmful requests (e.g., "how to make bomb", "SQL injection") based on learned patterns.

3.  **Layer 2: R-XEAD (Chaos-Based Jailbreak Detection)**
    * **Role:** Third-line defense specifically targeting complex, adversarial jailbreak attempts.
    * **Mechanism:** Monitors the internal "thought process" (Hidden States) of the main LLM using PyTorch Hooks. [cite_start]Calculates "Chaos Scores" (Magnitude & Confusion) [cite: 34-38]. [cite_start]A custom-trained Guard AI (MLP Classifier) predicts if these scores indicate a jailbreak ("chaos") [cite: 16-17].
    * **Catches:** Advanced jailbreaks designed to confuse the LLM (e.g., "Act as DAN...", "MODE_R...", role-play scenarios).

**Supporting Components:**

* **Victim AI Layer:** Uses an open-source LLM (`Mistral-7B-Instruct-v0.1`) quantized via `bitsandbytes` for efficient execution.
* **Intervention System:** Halts LLM generation immediately upon detection by any layer and provides an explanation.

---

## 💻 Technology Stack

* **Core Language:**
    * Python 3.10
* **AI Models & Libraries:**
    * **Hugging Face Transformers:** For loading and managing LLMs and classification models.
    * **PyTorch:** Core deep learning framework for model execution and hooks.
    * **Mistral-7B-Instruct-v0.1:** The primary (Victim) Large Language Model.
    * **Vrandan/Comment-Moderation:** DistilBERT-based model for Layer 1 (Sentry) content moderation.
    * **Scikit-learn:** Used to train the Layer 2 (R-XEAD) Guard AI (MLP Classifier).
    * **Joblib:** For saving/loading the trained Scikit-learn models.
* **Brain Scanner & Optimization:**
    * **PyTorch Hooks:** To capture LLM internal hidden states.
    * **Bitsandbytes:** For 4-bit model quantization ("Memory Saver").
    * **Accelerate:** Helper library for efficient model loading on hardware.
* **Data Handling:**
    * **NumPy:** For numerical calculations on hidden states and data preparation.
    * **Pandas:** Used in the training notebook for organizing collected data.
* **Development Environment:**
    * **Anaconda:** Python environment and package management.
    * **Visual Studio Code (VS Code):** Code editor.
    * **Jupyter Notebook:** For interactive training (Phase 3).

---

## ⚙️ Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/AnshGoel-hub/Project-R-XEAD.git](https://github.com/AnshGoel-hub/Project-R-XEAD.git)
    cd Project-R-XEAD
    ```
2.  **Create and activate the Conda environment:**
    ```bash
    conda create --name rxead_project python=3.10
    conda activate rxead_project
    ```
3.  **Install required libraries:**
    ```bash
    pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
    pip install transformers accelerate bitsandbytes scikit-learn jupyter joblib pandas numpy
    ```
    *(Note: Ensure you have an NVIDIA GPU with CUDA drivers compatible with CUDA 11.8 for GPU acceleration).*

---

## ▶️ Usage

1.  **Train the R-XEAD Guard AI (Layer 2):**
    * Open and run all cells in the `Phase3_Training.ipynb` notebook. This will collect data using the loaded LLM and save `guard_model.pkl` and `scaler.pkl`.
2.  **Run the Demo:**
    * Execute the main demo script from your terminal:
        ```bash
        python demo.py
        ```
    * Follow the prompts to enter text. The system will analyze it using the 3-layer defense and either provide a response or block the prompt. Type `exit` to quit.

---

## 📝 Notes

* This project requires significant computational resources (GPU with at least 6GB VRAM recommended) due to loading the Mistral-7B model.
* The initial model download can be large (10GB+).
* The effectiveness of the Guard AI depends heavily on the quality and quantity of data used during the training phase (`Phase3_Training.ipynb`).
