# Employee Attrition Prediction Project

This project aims to predict employee attrition using various machine learning techniques and provides an interactive web application for data analysis and visualization. The application is built using Streamlit for the front-end and Scikit-learn for the machine learning models.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Modeling](#modeling)
- [Data Visualization](#data-visualization)
- [Contributing](#contributing)
- [License](#license)

## Overview

Employee attrition is a significant concern for many organizations. Predicting which employees are likely to leave can help companies take proactive measures to retain valuable staff. This project uses historical employee data to build machine learning models that predict whether an employee will leave the company.

## Features

- **Interactive Web Application**: Built with Streamlit, allowing users to explore data, visualize trends, and make predictions.
- **Machine Learning Models**: Implemented using Scikit-learn, including data preprocessing, model training, and evaluation.
- **Data Visualization**: Various plots and charts to understand the factors affecting employee attrition.
- **User Input**: Users can input new employee data to predict attrition risk.

## Installation

To run this project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/drukachi/employee_attrition.git
   cd employee-attrition
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit application**:
   ```bash
   streamlit run app.py
   ```

## Usage

Once the application is running, open your web browser and navigate to `http://localhost:8501` to access the Streamlit interface.

### Streamlit Interface

- **Home**: Overview of the project and navigation.
- **Data**: Input the Employee information.
- **Visualization**: Interactive charts and graphs to visualize data trends, by clicking on the Explore Page
- **Prediction**: Input employee attributes to predict attrition risk using trained models.

## Modeling

### Data Preprocessing

The data preprocessing steps include handling missing values, encoding categorical variables, feature scaling, and splitting the dataset into training and testing sets.

### Model Training

- A Clustering analysis was done on the data to determine distinct clusters, after which two machine learning algorithms were trained on the these clusters
  to produce different models for these clusters. So based on the cluster an employee belongs to the corresponding cluster model will perform its classification.

- Although two algorithms were evaluated after tuning the parameters i.e.
1. Random Forest
2. XGBoost

- The XGBoost model performed better than the RF counter part and its being used to perform the prediction


### Model Evaluation

The models were evaluated using a test set to ensure they generalize well to unseen data. Cross-validation was also used to assess the model's performance.

## Data Visualization

The Streamlit application includes several data visualization features:
- **Distribution Plots**: Visualize the distribution of continuous variables.
- **Box Plots**: Compare the distribution of a variable across different categories.
- **Bar Charts**: Show the frequency of categorical variables.
- **Correlation Heatmaps**: Display the correlation between different features.

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Thank you for checking out this project! If you have any questions or suggestions, feel free to open an issue or contact the project maintainers.
