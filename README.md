# Breast Cancer Diagnosis Predictor

A machine learning-powered diagnostic tool that assists medical professionals in classifying breast masses as benign or malignant using cytological measurements.

## 🔬 Overview

This application leverages machine learning algorithms to predict breast cancer diagnosis based on quantitative measurements of cell nuclei characteristics. The tool provides:

- **Real-time predictions** with probability scores for benign/malignant classification
- **Interactive radar chart visualization** of input measurements
- **User-friendly interface** for manual data entry
- **Extensible architecture** for potential integration with cytology lab equipment

### Key Features
- Built using the Wisconsin Breast Cancer Diagnostic Dataset
- Interactive web interface powered by Streamlit
- Visual data representation with radar charts
- Probability-based predictions with confidence scores
- Export functionality for further analysis

## 🚀 Live Demo

Experience the application: [Streamlit Community Cloud](https://alejandro-ao-streamlit-cancer-predict-appmain-uitjy1.streamlit.app/)

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Machine Learning**: scikit-learn
- **Data Processing**: pandas, numpy
- **Visualization**: plotly, matplotlib

## 📋 Prerequisites

- Python 3.10 or higher
- conda (recommended) or pip

## 🔧 Installation

### Option 1: Using Conda (Recommended)

1. **Create a virtual environment**:
   ```bash
   conda create -n breast-cancer-diagnosis python=3.10
   ```

2. **Activate the environment**:
   ```bash
   conda activate breast-cancer-diagnosis
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Option 2: Using pip

1. **Create a virtual environment**:
   ```bash
   python -m venv breast-cancer-diagnosis
   ```

2. **Activate the environment**:
   - Windows: `breast-cancer-diagnosis\Scripts\activate`
   - macOS/Linux: `source breast-cancer-diagnosis/bin/activate`

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

1. **Start the application**:
   ```bash
   streamlit run app/main.py
   ```

2. **Access the web interface**:
   - Open your browser and navigate to `http://localhost:8501`

3. **Using the application**:
   - Input cell measurement values manually
   - View real-time predictions and probability scores
   - Analyze results using the interactive radar chart
   - Export measurements to CSV for further analysis

## 📊 Input Parameters

The model requires the following cytological measurements:
- Cell nucleus radius, texture, perimeter, area, smoothness
- Compactness, concavity, concave points, symmetry, fractal dimension
- Mean, standard error, and worst values for each parameter

## 🎯 Model Performance

The machine learning model was trained on the Wisconsin Breast Cancer Diagnostic Dataset and achieves:
- High accuracy in distinguishing between benign and malignant cases
- Probability-based predictions for clinical decision support
- Visual confidence indicators through radar chart analysis

## ⚠️ Important Disclaimer

**This application is developed for educational and research purposes only.** It is not intended for clinical use or as a substitute for professional medical diagnosis. Always consult qualified healthcare professionals for medical decisions.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Dataset Attribution

This project uses the [Breast Cancer Wisconsin (Diagnostic) Data Set](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data) available on Kaggle.

## 🔮 Future Enhancements

- Integration with laboratory equipment APIs
- Enhanced visualization options
- Model performance metrics dashboard
- Batch processing capabilities
- Multi-language support

## 📧 Contact

For questions or collaboration opportunities, please open an issue or reach out through GitHub.

---

*Built with ❤️ for the machine learning and healthcare communities*
