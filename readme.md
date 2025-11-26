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

### 🤖 Machine Learning Models

✅ **Dual Model Approach**: Comparative analysis using two powerful supervised learning algorithms

   - **Support Vector Machine (SVM)**: Uses RBF (Radial Basis Function) kernel to find the optimal hyperplane that maximally separates plastic types in high-dimensional spectral feature space. Maps data into higher dimensions to handle complex non-linear classification boundaries. Effective for high-dimensional data and robust to outliers.
   
   - **Random Forest Classifier**: Ensemble learning method that constructs 100 decision trees during training and outputs the class that is the mode of the predictions. Each tree votes on the classification, and the majority vote determines the final prediction. Reduces overfitting through bootstrap aggregating (bagging) and provides stable, robust predictions.

### 📊 Performance Metrics Explained

✅ **High Classification Accuracy**: 93.89% overall accuracy achieved through optimized feature engineering, standardized preprocessing pipeline, and ensemble learning techniques

✅ **Comprehensive Multi-Metric Evaluation**: Detailed performance assessment for each of the 6 plastic types

   - **Precision**: Answers "Of all samples we predicted as this plastic type, what percentage are actually correct?" 
     - Example: 98% precision for PET means that when the model predicts PET, it's correct 98% of the time
     - High precision = fewer false positives = less contamination in sorted batches
   
   - **Recall (Sensitivity)**: Answers "Of all actual samples of this plastic type, what percentage did we correctly identify?"
     - Example: 97% recall for LDPE means we successfully identified 97% of all LDPE samples
     - High recall = fewer false negatives = less material missed during sorting
   
   - **F1-Score**: Harmonic mean of precision and recall (2 × precision × recall / (precision + recall))
     - Provides a single balanced measure when you need both precision and recall to be high
     - Perfect score = 1.0, ranges from 0 to 1
     - More useful than simple accuracy when dealing with class imbalances

### 🎯 Advanced Spectroscopic Analysis

✅ **High-Dimensional Feature Space**: Processes spectral measurements across 399-4001 cm⁻¹ wavenumber range, capturing unique molecular "fingerprints" for each plastic polymer. Each wavelength represents a feature, creating hundreds of dimensions for classification.

✅ **Visual Analysis Tools**: 
   - Confusion matrices showing true vs predicted classifications
   - Performance comparison charts between SVM and Random Forest
   - Per-class accuracy visualizations
   - Feature importance rankings (Random Forest)

### 🔧 Production-Ready Pipeline

✅ **Standardized Preprocessing**: Complete StandardScaler pipeline ensures:
   - Zero mean and unit variance for all features
   - Consistent feature normalization for deployment
   - Prevents features with larger scales from dominating the model
   - Essential for SVM performance

✅ **Reproducible Results**: 
   - Fixed random seeds (random_state) for train-test splits ensure same data distribution across runs
   - Model initialization seeds guarantee consistent training behavior
   - Enables reliable model comparison and debugging

✅ **Stratified Data Splitting**: 80-20 train-test split maintains class proportions, ensuring each plastic type is represented fairly in both training and testing sets

## 🎓 Applications

- **Recycling Facilities**: Automated sorting of mixed plastic waste streams at industrial scale
- **Quality Control**: Verify plastic composition in manufacturing processes
- **Environmental Monitoring**: Identify plastic pollution types in marine and terrestrial ecosystems
- **Research**: Study plastic degradation patterns and composition changes over time
- **Supply Chain**: Authenticate plastic materials in logistics and verify material specifications

## 🧪 Technical Details

**Spectroscopy Range**: 399-4001 cm⁻¹ (wavenumbers)

**Feature Dimension**: High-dimensional spectral data (hundreds of wavelength measurements)

**Preprocessing**: StandardScaler for feature normalization (zero mean, unit variance)

**Train-Test Split**: 80-20 ratio with stratified sampling to maintain class balance

**Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix

**Model Parameters**: 
- SVM: RBF kernel with default gamma and C parameters
- Random Forest: 100 estimators with bootstrap sampling

## 🤝 Contributing

Contributions are welcome! Here are some ways to improve the project:
- Add more plastic types (e.g., ABS, PC, PMMA)
- Implement deep learning models (CNN, LSTM for spectral sequences)
- Add real-time prediction interface with webcam integration
- Optimize hyperparameters with Grid Search or Bayesian Optimization
- Create REST API for deployment
- Add cross-validation for more robust evaluation
- Implement feature selection techniques

## 📧 Contact

**Aditya Singh**
- GitHub: [@aditya-singh2005](https://github.com/aditya-singh2005)
- LinkedIn: [Aditya Singh](https://www.linkedin.com/in/aditya-singh-7658a1291/)
- Email: job.singhaditya00005@gmail.com

## 🙏 Acknowledgments

- Spectroscopy data collected using FTIR (Fourier Transform Infrared Spectroscopy) technique
- Inspired by real-world recycling and environmental sustainability challenges
- Built with scikit-learn and Python data science ecosystem
- Thanks to the open-source community for machine learning tools and libraries

---

⭐ **Star this repo** if you find it helpful! Feel free to fork and contribute to improve plastic waste management through AI.