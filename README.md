# Target Reply – RUL Estimation of Turbofan Engines

**Competition:** September Data Science Training Camp @ Sapienza University  
**Organizer:** Target Reply Roma

## Overview

This repository contains the code and methodology for the "Target Reply – RUL Estimation of Turbofan Engines" challenge. The competition focuses on predicting the Remaining Useful Life (RUL) of aircraft engines using time-series sensor data from NASA's C-MAPSS simulator. Each engine runs until failure, generating multivariate sensor readings at each cycle. The goal: estimate how many cycles remain before an engine fails, supporting smarter predictive maintenance.

## Challenge & Dataset

- **Training:** Full run-to-failure sensor trajectories for many engines. RUL for each cycle is known.
- **Test:** Truncated trajectories; you only see up to a certain cycle and must predict RUL at that point.
- **Sensors:** 21 sensor measurements + 3 operational settings per cycle.
- **Submission:** For each test engine, predict RUL at its last observed cycle.
- **Evaluation:** Mean Squared Error (MSE) between predicted and actual RUL.

## Solution Approach

Our solution evolved from deep feature engineering to a streamlined, robust approach that generalized best on unseen data:

### 1. Data Exploration & Key Considerations

- **RUL Clipping:** RUL labels were capped at 125 cycles during training to focus the model on the critical degradation period and avoid chasing unreliable, high RUL values.
- **Engine-wise Splitting:** Training/validation splits were done by engine ID to avoid leakage; models were always validated on engines they never saw in training.
- **Sequence Features vs. Snapshots:** After exploring complex per-engine degradation "onset" detection and rich feature sets, we adopted a sliding window approach for both simplicity and improved generalization.

### 2. Feature Engineering

**Sliding Window Segmentation:**
- Each engine is segmented into overlapping windows (50 cycles, stride 5), generating thousands of samples from each run.

**Per-Window Features (for each sensor):**
- **Mean:** Average sensor value in the window.
- **Std:** Standard deviation (volatility) in the window.
- **Last Value:** Sensor value at the window's end (most recent).
- **Slope:** Linear trend fitted to the window's readings.

**Additional Features:**
- **Operational Settings:** Last value for each of the three settings in the window.
- **Operating Regime:** One-hot encoded regime indicators (6 regimes discovered via clustering settings).
- **Cycle Index:** Engine's age at the end of the window.

This approach replaces earlier, more complex features (onset detection, change since onset, etc.) with robust, general statistics and recent trends—letting the model infer degradation implicitly.

### 3. Preprocessing

- **Standardization:** All features are scaled (mean-centered, unit variance) using Scikit-Learn's `StandardScaler` in a Pipeline.
- **No Per-Regime Scaling:** Regime effects are captured via one-hot flags rather than separate normalization.
- **Missing Values:** Not an issue; simulated data is complete.

### 4. Model Choice & Training

- **Ridge Regression:** Chosen for its robustness to multicollinearity and ability to balance many correlated features. Linear statistics (means, slopes, etc.) map well to a regularized linear model.
- **Regularization (α):** Thoroughly tuned via cross-validation (GroupKFold by engine), with the best result at α ≈ 5.3e-4.
- **Cross-Validation:** Always validated on unseen engines to ensure generalization.

### 5. Why Simpler Features Worked Best

- **Generalization:** Simple statistical features are less noisy than hand-crafted onset metrics and better capture fundamental degradation patterns.
- **Efficiency:** Sliding window sampling yields thousands of training samples (vs. one per engine), improving learning and generalization.
- **Robustness:** Ridge regression balances the influence of correlated features, reducing overfitting and exploiting the rich feature set.
- **Competition Performance:** Achieved 3rd place with validation RMSE ≈ 32.26 and MAE ≈ 24.1, proving the approach generalizes well to unseen engines.

## License & Contact

For questions or collaborations, contact [emanueleiacca](https://github.com/emanueleiacca).

---

**Summary:**  
This repo demonstrates how careful feature engineering, data splitting, and a simple, regularized model can outperform more complex pipelines in RUL prediction challenges. The final approach uses sliding window statistics, regime flags, and Ridge regression for robust, interpretable, and accurate engine health forecasting.
