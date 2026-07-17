# ROGII - Wellbore Geology Prediction: AeroRidge Blueprint

Welcome to the repository for the **ROGII Wellbore Geology Prediction** challenge.

This project focuses on automating drilling operations in the oil and gas industry by predicting the **True Vertical Thickness (TVT)** along horizontal wellbores.

## Current State & Achievement
The base repository implementation starts with a validation baseline:
- **Validation RMSE**: `18.6589` (utilizing simple Physics-Constrained Kalman Filter Smoothing).

To optimize this submission to a world-class level that trumps the current top-rated entry (`7.022` Public RMSE), we have designed and architected a comprehensive hybrid sequence-alignment solution.

## Elite Blueprint: AeroRidge Triple-Track Engine
The complete, publication-grade architectural blueprint is documented in:
👉 **[SOLUTION_BLUEPRINT.md](SOLUTION_BLUEPRINT.md)**

### Key Highlights of the AeroRidge Architecture:
1. **Decoupling the TVT-Z Paradox**: Separates long-range cross-well structural trends from high-frequency lateral dip variations.
2. **Track 1: Stochastic Particle Filter + Beam Search**: A physical state-space tracker maintaining continuous structural dip hypotheses.
3. **Track 2: LightGBM Gradient Booster**: Exploits multi-scale Normalized Cross-Correlation (NCC) against typewells and Q-3D wellbore tortuosity indices.
4. **Track 3: Global-Optimal Dynamic Programming (DP)**: Adaption of Hale's Dynamic Image Warping to find globally minimal alignment paths, preventing local tracking collapse.
5. **Real-time Prefix Calibration**: Evaluates track performance in-distribution on the visible heel prefix to dynamically tune track blend weights per well.

For implementation templates and complete mathematical formulations, please refer to the [SOLUTION_BLUEPRINT.md](SOLUTION_BLUEPRINT.md).
