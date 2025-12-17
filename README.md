Berlin Airbnb Pricing Studio

A machine learning–based pricing and investment analysis application for Airbnb listings in Berlin.

Live App:

http://3.72.159.159:8501/

Project Overview

The Berlin Airbnb market is highly competitive, making accurate pricing a critical factor for both hosts and property investors.
This project provides a data-driven pricing solution that predicts optimal nightly prices based on property characteristics and location.

The application is designed for:

•	Hosts, who want to price their listings competitively and create effective descriptions

•	Investors, who want to evaluate potential rental income and profitability before purchasing a property

Dataset

•	Source: Kaggle
https://www.kaggle.com/datasets/thedevastator/berlin-airbnb-ratings-and-reviews-overview

•	Dataset: Airbnb Berlin.csv

•	Scope: ~456,000 listings with location, property attributes, and booking information

Modeling Approach

The full data science workflow was implemented in a Jupyter Notebook:

•	Data exploration and cleaning

•	Feature engineering (categorical encoding, geographic features)

•	Model training and evaluation

Models Used

•	RandomForestRegressor

•	GradientBoostingRegressor

Random Forest was selected as the final model due to superior performance (higher R² and lower error metrics).

Model notebook:

M516_Business_Project_in_Big_Data_&_AI.ipynb

Application Features

•	Host Mode:

Price recommendation, explainable pricing insights, Berlin price heatmap, and an AI-powered listing description assistant.

•	Investor Mode:

Revenue and yield estimation under different occupancy scenarios with optional manual price override.

Deployment

•	Platform: AWS EC2 (Ubuntu)

•	Framework: Streamlit

•	Environment: Python venv

•	Service Management: systemd (auto-start on reboot)

•	Access: Public Elastic IP

Technologies

Python, Pandas, NumPy, Scikit-learn, Streamlit, AWS EC2, GitHub
