# Highly Imbalanced Regression with Tabular Data in SEP and Other Applications

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)

## Abstract

We investigate imbalanced regression with tabular data that have an imbalance ratio larger than 1,000 ("highly imbalanced"). Accurately estimating the target values of rare instances is important in applications such as forecasting the intensity of rare harmful Solar Energetic Particle (SEP) events. For regression, the MSE loss does not consider the correlation between predicted and actual values. Typical inverse importance functions allow only convex functions. Uniform sampling might yield mini-batches that do not have rare instances. We propose CISIR that incorporates correlation, Monotonically Decreasing Involution (MDI) importance, and stratified sampling. Based on five datasets, our experimental results indicate that CISIR can achieve lower error and higher correlation than some recent methods. Also, adding our correlation component to other recent methods can improve their performance. Lastly, MDI importance can outperform other importance functions.

## Overview

This repository contains the official implementation of **CISIR** (Correlation, Involution, Stratified Importance Regression), a novel method for highly imbalanced regression with tabular data. The method is particularly applicable to SEP (Solar Energetic Particle) forecasting research for NASA and other domains requiring accurate prediction of rare, high-impact events.

### Key Contributions

- **Correlation-aware loss function** that considers the correlation between predicted and actual values
- **Monotonically Decreasing Involution (MDI) importance** weighting that outperforms traditional convex importance functions
- **Stratified sampling strategy** that ensures rare instances are included in mini-batches
- **Comprehensive evaluation** on five highly imbalanced datasets with imbalance ratios > 1,000

## Datasets

We evaluate our method on five highly imbalanced datasets:

### SEP Datasets
- **SEP-EC**: Forecasts the change (delta) in proton intensity based on features from electron intensity and CMEs (Coronal Mass Ejections)
- **SEP-C**: Forecasts peak proton intensity based on CME characteristics

### Other Datasets
- **SARCOS**: Estimates the torque vector based on joint-state inputs for a 7-DOF robot arm
- **Blog Feedback (BF)**: Forecasts the number of comments based on textual, temporal, and engagement features  
- **Online News Popularity (ONP)**: Estimates the number of shares of an article based on content, topic, and sentiment attributes

### Data Availability
https://huggingface.co/datasets/Machine-Earning/CISIR-datasets/resolve/main/CISIR-data.zip

All datasets exhibit high imbalance ratios (ρ > 1,000), making them ideal for evaluating highly imbalanced regression methods.

<!-- ## Installation

```bash
git clone https://github.com/Machine-Earning/CISIR.git
cd CISIR
pip install -r requirements.txt
```

## Usage

```bash
# Basic usage example
./run.sh
```

For detailed usage instructions and examples, please refer to the source code documentation. -->

## Authors

**Josias K. Moukpe**¹ · **Philip K. Chan**¹ · **Ming Zhang**²

¹Department of Electrical Engineering and Computer Science  
²Department of Aerospace, Physics and Space Sciences  
Florida Institute of Technology, Melbourne, FL, USA

**Contact**: jmoukpe2016@my.fit.edu, {pkc, mzhang}@fit.edu

---

## Citation

If you find this repository useful in your research, please consider giving a star ⭐️ and a citation:

```bibtex
@inproceedings{moukpe2024cisir,
  title={Highly Imbalanced Regression with Tabular Data in SEP and Other Applications},
  author={Moukpe, Josias K. and Chan, Philip K. and Zhang, Ming},
  booktitle={Proceedings of the IEEE International Conference on Machine Learning and Applications (ICMLA)},
  year={2024},
  organization={IEEE}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Keywords

`regression` `tabular-data` `imbalanced-learning` `SEP-forecasting` `solar-physics` `machine-learning` `deep-learning`