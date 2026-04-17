# 📊 Corporate Life-Cycle & Capital Structure Strategist

This repository contains a Python-based interactive web application designed for corporate strategists, CFOs, and financial analysts. It operationalizes academic research on the determinants of capital structure over corporate life stages.

## The Core Concept
Traditional analysis often relies on "firm age" to benchmark leverage. This tool utilizes the **Dickinson (2011) cash flow pattern methodology**, proving that life stage is dictated by the behavioral footprint of Operating, Investing, and Financing Cash Flows (NCFO, NCFI, NCFF), rather than the year of incorporation.

## Key Features
* **Life-Stage Diagnostic:** Automatically classifies firms into 8 distinct life stages (Startup, Growth, Maturity, Shakeout, Decline, Decay) based on cash flow combinations.
* **Leverage Benchmarking:** Visually compares a target firm's debt load strictly against peers in the exact same life cycle phase.
* **Capital Structure Stickiness:** Plots lagged leverage ($t$ vs $t-1$) to reveal the dynamic impact of past debt constraints on current capacity.
* **The Quant Engine:** A built-in econometric sandbox running Fixed Effects (FE) and Random Effects (RE) panel regressions, complete with an automated Hausman test to determine the statistically valid model for any selected variables.

## Tech Stack
* **Frontend/Framework:** Streamlit
* **Data Processing:** Pandas, NumPy
* **Econometrics:** linearmodels (PanelOLS, RandomEffects), Statsmodels, SciPy
* **Visualization:** Plotly Express

## How to Run Locally
1. Clone this repository.
2. Ensure you have the required dataset (`.dta` format) in the root folder.
3. Install the required libraries:
   `pip install streamlit pandas plotly statsmodels linearmodels scipy`
4. Run the app:
   `streamlit run app.py`
