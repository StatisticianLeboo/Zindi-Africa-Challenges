# AI4EAC Health Practice Challenge

This repository contains my work for the AI4EAC Health Practice Challenge hosted on Zindi (zindi.africa). The competition focuses on leveraging machine learning to improve health practices across East Africa by analyzing survey and behavioral data.

## Overview
Can you predict who is suffering from depression based on routine survey data?
The World Health Organization estimates that 1.3 million Kenyans suffer from untreated major depressive disorder (MDD; commonly known as depression) every year, and that sub-Saharan Africa has the highest prevalence of the illness of any region in the world. Yet mental health treatment in Kenya suffers from a lack of resources and stigmatization. There are only two certified psychiatrists per million people in Kenya. Few facilities exist outside of urban areas and people are unlikely to know about or access them.

Can machine learning help alleviate this problem? Smart targeting of potential cases could help enable scarce resources to reach those who most need it and improve or save an untold number of lives.

With this in mind, the Busara Center for Behavioral Economics has decided to challenge the data science community in Nairobi, across Africa, and around the world to predict depression cases from routine survey data.

A model which accurately predicts which individuals are likely to be depressed from the survey data could be used by mental health providers such as local clinics, NGOs or community health volunteers to reach out to those at risk. We look forward to your solutions to this pressing problem.

This competition is hosted by the Busara Center for Behavioral Economics in partnership with AI Kenya, Women in Machine Learning and Data Science, Tulaa and ALX Launchpad.



## ⚙️ Methodology
- Data Preprocessing
- Handle missing values.
- Encode categorical variables.
- Apply feature scaling where necessary.
- Explore class balance and apply resampling if required.
- Model Development
- Baseline models: Logistic Regression, Random Forest, XGBoost.
- Advanced models: Neural networks (MLP), ensemble methods.
- Hyperparameter tuning with GridSearchCV or Bayesian optimization.
- Evaluation
- Use cross-validation to ensure robustness.
- Track metrics such as precision, recall, F1-score, and confusion matrix.
- Compare models and select the best-performing one for submission.
- Submission
- Generate predictions on the test set.
- Format according to Zindi’s submission requirements (CSV file with ID and target columns).
- Upload to the competition portal.

## 📊 Project Structure
├── data/                # Raw and processed datasets
├── notebooks/           # Jupyter notebooks for exploration and modeling
├── src/                 # Python scripts for preprocessing and training
├── models/              # Saved trained models
├── submissions/         # Ready for Zindi submission
└── README.md            # Project documentation
└── Report.md            # Mini Report
