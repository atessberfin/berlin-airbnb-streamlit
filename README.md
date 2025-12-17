Berlin Airbnb Pricing Studio

A machine learning–powered web application for estimating Airbnb listing prices in Berlin.
The project integrates a trained ML model with an interactive Streamlit interface and is deployed on AWS EC2.

URL:

http://3.72.159.159:8501
The application is deployed on an AWS EC2 instance with a static Elastic IP and runs continuously using systemd.

Project Overview

This project aims to support Airbnb hosts and property investors by providing data-driven price estimations based on listing characteristics such as location, room type, and accommodation capacity.
The application offers two main user flows:
Host Mode: Price recommendation for Airbnb hosts
Investor Mode: Market insights and pricing analysis for potential investors

Machine Learning

Trained regression-based model using historical Airbnb listing data
Model serialized using joblib
Deployed model size: ~769 MB
Prediction executed in real time within the Streamlit application

Tech Stack

Python
Streamlit (Frontend & App logic)
scikit-learn (Machine Learning)
pandas / numpy (Data processing)
Plotly (Visualizations)
AWS EC2 (Cloud deployment)

Deployment Details

Cloud Provider: AWS EC2
Instance: t3.small (with swap memory enabled)
Process Management: systemd (auto-restart & reboot persistence)
Static IP: AWS Elastic IP
Access: Public HTTP endpoint
The application is configured to automatically start after instance reboots and remains accessible without manual intervention.


GitHub + Git LFS (Version control for large model files)
