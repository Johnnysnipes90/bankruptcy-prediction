# # Poland Bankruptcy Prediction

## 📌 Project Overview
This project aims to predict bankruptcy in Polish companies using financial indicators. The dataset contains 64 financial features and a target variable indicating bankruptcy status.


# Poland Bankruptcy Data

Below is a summary of the features from the Poland bankruptcy dataset.

## 📊 Feature Descriptions

| Feature  | Description |
|----------|------------|
| **1**  | Net profit / total assets |
| **Attr2**  | Total liabilities / total assets |
| **Attr3**  | Working capital / total assets |
| **Attr4**  | Current assets / short-term liabilities |
| **Attr5**  | [(Cash + short-term securities + receivables - short-term liabilities) / (Operating expenses - depreciation)] * 365 |
| **Attr6**  | Retained earnings / total assets |
| **Attr7**  | EBIT / total assets |
| **Attr8**  | Book value of equity / total liabilities |
| **Attr9**  | Sales / total assets |
| **Attr10** | Equity / total assets |
| **Attr11** | (Gross profit + extraordinary items + financial expenses) / total assets |
| **Attr12** | Gross profit / short-term liabilities |
| **Attr13** | (Gross profit + depreciation) / sales |
| **Attr14** | (Gross profit + interest) / total assets |
| **Attr15** | (Total liabilities * 365) / (Gross profit + depreciation) |
| **Attr16** | (Gross profit + depreciation) / total liabilities |
| **Attr17** | Total assets / total liabilities |
| **Attr18** | Gross profit / total assets |
| **Attr19** | Gross profit / sales |
| **Attr20** | (Inventory * 365) / sales |
| **Attr21** | Sales (n) / Sales (n-1) |
| **Attr22** | Profit on operating activities / total assets |
| **Attr23** | Net profit / sales |
| **Attr24** | Gross profit (in 3 years) / total assets |
| **Attr25** | (Equity - share capital) / total assets |
| **Attr26** | (Net profit + depreciation) / total liabilities |
| **Attr27** | Profit on operating activities / financial expenses |
| **Attr28** | Working capital / fixed assets |
| **Attr29** | Logarithm of total assets |
| **Attr30** | (Total liabilities - cash) / sales |
| **Attr31** | (Gross profit + interest) / sales |
| **Attr32** | (Current liabilities * 365) / cost of products sold |
| **Attr33** | Operating expenses / short-term liabilities |
| **Attr34** | Operating expenses / total liabilities |
| **Attr35** | Profit on sales / total assets |
| **Attr36** | Total sales / total assets |
| **Attr37** | (Current assets - inventories) / long-term liabilities |
| **Attr38** | Constant capital / total assets |
| **Attr39** | Profit on sales / sales |
| **Attr40** | (Current assets - inventory - receivables) / short-term liabilities |
| **Attr41** | Total liabilities / [(Profit on operating activities + depreciation) * (12/365)] |
| **Attr42** | Profit on operating activities / sales |
| **Attr43** | Rotation receivables + inventory turnover in days |
| **Attr44** | (Receivables * 365) / sales |
| **Attr45** | Net profit / inventory |
| **Attr46** | (Current assets - inventory) / short-term liabilities |
| **Attr47** | (Inventory * 365) / cost of products sold |
| **Attr48** | EBITDA (Profit on operating activities - depreciation) / total assets |
| **Attr49** | EBITDA (Profit on operating activities - depreciation) / sales |
| **Attr50** | Current assets / total liabilities |
| **Attr51** | Short-term liabilities / total assets |
| **Attr52** | (Short-term liabilities * 365) / cost of products sold |
| **Attr53** | Equity / fixed assets |
| **Attr54** | Constant capital / fixed assets |
| **Attr55** | Working capital |
| **Attr56** | (Sales - cost of products sold) / sales |
| **Attr57** | (Current assets - inventory - short-term liabilities) / (Sales - gross profit - depreciation) |
| **Attr58** | Total costs / total sales |
| **Attr59** | Long-term liabilities / equity |
| **Attr60** | Sales / inventory |
| **Attr61** | Sales / receivables |
| **Attr62** | (Short-term liabilities * 365) / sales |
| **Attr63** | Sales / short-term liabilities |
| **Attr64** | Sales / fixed assets |
| **bankrupt** | Whether the company went bankrupt at the end of the forecasting period (2013) |



## 🏗️ Project Structure
#### 📂 Poland-Bankruptcy-Prediction │── 📂 data 
#### Folder for datasets │ ├── raw 
#### Raw data files │ ├── processed 
#### Cleaned and preprocessed data │── 📂 notebooks 
#### Jupyter notebooks for EDA and modeling │── 📂 src 
#### Source code directory │ ├── 📂 data_preprocessing 
#### Scripts for data cleaning & feature engineering │ ├── 📂 models 
#### Machine learning models │ ├── 📂 utils # Helper functions │── 📂 reports 
#### Generated reports and results │── 📂 config # Configuration files │── 📂 logs 
#### Log files for tracking experiments │── 📂 models 
#### Saved trained models │── 📂 requirements 
#### Dependencies and environment setup │── .gitignore 
#### Ignoring unnecessary files │── README.md 
#### Project documentation │── requirements.txt 
#### List of dependencies │── setup.py # Script to install package │── main.py 
#### Main execution script


## 🚀 Setup Instructions (Windows)
Follow these steps to set up the project:

### 1️⃣ Clone the GitHub Repository
```bash
git clone https://github.com/yourusername/Poland-Bankruptcy-Prediction.git
cd Poland-Bankruptcy-Prediction 
```
### 2️⃣ Create a Virtual Environment
python -m venv venv

### 3️⃣ Activate Virtual Environment
source venv\Scripts\activate

### 4️⃣ Install Dependencies
pip install -r requirements.txt

### 5️⃣ Run Jupyter Notebook (For EDA & Modeling)
jupyter notebook

### 6️⃣ Start the API Server (FastAPI)
uvicorn api:app --reload

### 7️⃣ Run the Streamlit App
streamlit run streamlit.py


