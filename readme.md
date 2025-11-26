# Plastic Classification using UV-IR-Visible Spectroscopy

Machine learning system for automated plastic polymer classification using spectroscopic data. Identifies different plastic types (HDPE, LDPE, PET, PP, PS, PVC) through spectral analysis with Support Vector Machines and Random Forest classifiers.

## 🎯 Project Overview

This project uses machine learning to classify six different types of plastics based on their unique UV-IR-Visible spectroscopy signatures. Spectroscopy measures how materials interact with electromagnetic radiation across different wavelengths, creating distinctive "fingerprints" for each plastic polymer.

### Why This Matters
- **Environmental Impact**: Proper plastic sorting is crucial for effective recycling and reducing environmental pollution
- **Automation**: Traditional manual sorting is slow, expensive, and error-prone
- **High Accuracy**: ML models achieve 92-94% accuracy in identifying plastic types
- **Scalability**: Can process thousands of samples quickly in industrial settings

## 📊 Dataset

The dataset contains spectroscopic measurements across UV, Infrared, and Visible wavelengths for different plastic samples.

**Plastic Types Classified:**
1. **HDPE** (High-Density Polyethylene) - Milk jugs, detergent bottles
2. **LDPE** (Low-Density Polyethylene) - Plastic bags, wraps, squeeze bottles
3. **PET** (Polyethylene Terephthalate) - Water bottles, food containers
4. **PP** (Polypropylene) - Food containers, automotive parts, bottle caps
5. **PS** (Polystyrene) - Foam cups, packaging, disposable cutlery
6. **PVC** (Polyvinyl Chloride) - Pipes, window frames, medical tubing

**Dataset Characteristics:**
- **Measurement Technique**: FTIR (Fourier Transform Infrared Spectroscopy)
- **Features**: Spectral data across wavelength range (399-4001 wavenumbers)
- **Format**: CSV with polymer type, technique, sample ID, and intensity measurements
- **Total Samples**: Multiple samples per plastic type for robust training

## 🔬 Methodology

### 1. Data Preprocessing
- Load spectroscopic data from CSV
- Extract features (wavelength intensities) and labels (plastic types)
- Split data into training (80%) and testing (20%) sets
- Standardize features using StandardScaler for optimal model performance

### 2. Machine Learning Models

#### Support Vector Machine (SVM)
- **Algorithm**: SVM with RBF (Radial Basis Function) kernel
- **Strengths**: Effective in high-dimensional spaces, robust to outliers
- **Use Case**: Complex non-linear classification boundaries

#### Random Forest Classifier
- **Algorithm**: Ensemble of decision trees
- **Strengths**: Handles high-dimensional data, reduces overfitting
- **Use Case**: Robust feature importance and prediction stability

### 3. Model Evaluation
- Accuracy score on test set
- Detailed classification report (precision, recall, F1-score)
- Confusion matrix visualization
- Per-class performance analysis

## 📈 Results

### Support Vector Machine (SVM)
**Overall Accuracy: 92.44%**

| Plastic Type | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| HDPE         | 0.89      | 0.91   | 0.90     | 149     |
| LDPE         | 0.95      | 0.97   | 0.96     | 151     |
| PET          | 0.95      | 0.93   | 0.94     | 152     |
| PP           | 0.91      | 0.93   | 0.92     | 148     |
| PS           | 0.90      | 0.89   | 0.90     | 148     |
| PVC          | 0.93      | 0.91   | 0.92     | 152     |

**Key Insights:**
- LDPE and PET show highest precision (95%)
- Balanced performance across all plastic types
- Strong recall for LDPE (97%) - minimal false negatives

### Random Forest Classifier
**Overall Accuracy: 93.89%** ⭐ **Best Model**

| Plastic Type | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| HDPE         | 0.88      | 0.93   | 0.90     | 149     |
| LDPE         | 0.95      | 0.95   | 0.95     | 151     |
| PET          | 0.98      | 0.93   | 0.96     | 152     |
| PP           | 0.97      | 0.94   | 0.96     | 148     |
| PS           | 0.92      | 0.95   | 0.93     | 148     |
| PVC          | 0.93      | 0.94   | 0.94     | 152     |

**Key Insights:**
- **1.45% improvement** over SVM
- PET shows exceptional precision (98%)
- PP achieves 97% precision - highly reliable predictions
- Consistent performance with F1-scores above 0.90 for all classes

### Model Comparison

| Metric           | SVM    | Random Forest | Winner         |
|------------------|--------|---------------|----------------|
| Accuracy         | 92.44% | 93.89%        | Random Forest  |
| Avg Precision    | 0.92   | 0.94          | Random Forest  |
| Avg Recall       | 0.92   | 0.94          | Random Forest  |
| Training Speed   | Fast   | Moderate      | SVM            |
| Interpretability | Low    | High          | Random Forest  |

## 🚀 Installation & Usage

### Prerequisites
```bash
Python 3.8+
pandas
numpy
scikit-learn
matplotlib
seaborn
plotly (optional for interactive visualizations)
```

### Installation
```bash
# Clone the repository
git clone https://github.com/aditya-singh2005/Plastic-Classification---Machine-Learning-.git
cd Plastic-Classification---Machine-Learning-

# Install required packages
pip install pandas numpy scikit-learn matplotlib seaborn plotly
```

### Run the Project
```bash
# Open Jupyter Notebook
jupyter notebook "Al-IR spectroscopy project.ipynb"

# Run all cells to:
# 1. Load and preprocess data
# 2. Train both models
# 3. Generate predictions
# 4. View performance metrics and visualizations
```

## 📁 Project Structure
```
Plastic-Classification---Machine-Learning-/
│
├── Al-IR spectroscopy project.ipynb    # Main Jupyter notebook
├── plastic_data.csv                     # Spectroscopy dataset
├── README.md                           # This file
└── requirements.txt                    # Python dependencies
```

## 🔍 Key Features

-✅ **Dual Model Approach**: Compare SVM vs Random Forest performance
-✅ **High Accuracy**: 93.89% classification accuracy
-✅ **Comprehensive Metrics**: Precision, recall, F1-score for each plastic type
-✅ **Visual Analysis**: Confusion matrices and performance charts
-✅ **Production Ready**: Standardized preprocessing pipeline
-✅ **Reproducible**: Fixed random seeds for consistent results

## 🎓 Applications

- **Recycling Facilities**: Automated sorting of mixed plastic waste
- **Quality Control**: Verify plastic composition in manufacturing
- **Environmental Monitoring**: Identify plastic pollution types
- **Research**: Study plastic degradation and composition changes
- **Supply Chain**: Authenticate plastic materials in logistics

## 🧪 Technical Details

**Spectroscopy Range**: 399-4001 cm⁻¹ (wavenumbers)
**Feature Dimension**: High-dimensional spectral data
**Preprocessing**: StandardScaler for feature normalization
**Train-Test Split**: 80-20 ratio with stratified sampling
**Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix

## 🤝 Contributing

Contributions are welcome! Here are some ways to improve the project:
- Add more plastic types (e.g., ABS, PC, PMMA)
- Implement deep learning models (CNN, LSTM)
- Add real-time prediction interface
- Optimize hyperparameters with Grid/Random Search
- Create web API for deployment


## 📧 Contact

**Aditya Singh**
- GitHub: [@aditya-singh2005](https://github.com/aditya-singh2005)
- LinkedIn: [Aditya Singh](https://www.linkedin.com/in/aditya-singh-7658a1291/)
- Email: job.singhaditya00005@gmail.com

## 🙏 Acknowledgments

- Spectroscopy data collected using FTIR technique
- Inspired by real-world recycling and environmental challenges
- Built with scikit-learn and Python data science ecosystem

---

⭐ **Star this repo** if you find it helpful! Feel free to fork and contribute.