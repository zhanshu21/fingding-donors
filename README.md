
# Supervised Learning
## Project: Finding Donors for CharityML

### Install

This project requires **Python 3.x** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)

You will also need to have software installed to run and execute an [iPython Notebook](http://ipython.org/notebook.html)


### Run

In a terminal or command window, navigate to the top-level project directory `finding_donors/` (that contains this README) and run one of the following commands:

```bash
ipython notebook kaggle-competetion.ipynb
```  
or
```bash
jupyter notebook kaggle-competetion.ipynb
```

This will open the iPython Notebook software and project file in your browser.

### Data Analysis and Preprocessing

**Data Exploration and Analysis**

An in-depth data analysis was conducted to understand the distribution and characteristics of each feature. Key insights included examining relationships between features, identifying skewness in numerical data, and calculating correlations with the target variable (```income```). This analysis helped guide the preprocessing steps.

**Data Preprocessing**
To prepare the data for model training, several preprocessing steps were applied:

1. Encoding Categorical Variables: All categorical features were encoded to numerical values, including both one-hot encoding and frequency encoding where appropriate.
2. Handling Missing Values: The training set was checked for any missing or inconsistent data. In the test dataset, missing values were handled through a combination of mean, median, and mode imputation strategies to ensure no data was left incomplete.
3. Feature Scaling: Numerical features were standardized using standard scaling,  to improve model convergence and performance.
   
**Model Training and Evaluation**

Various machine learning models, including a neural network implemented in PyTorch, were trained and evaluated on this dataset. The neural network underwent tuning of layers and dropout rates to optimize performance. Additionally, traditional models like Decision Tree, Random Forest, and AdaBoost were also applied

**Kaggle Submission**
The preprocessed data and optimized models were used to submit predictions to the Kaggle competition, aiming to maximize accuracy in identifying high-income individuals for targeted donor outreach.

### Data

The modified census dataset consists of approximately 32,000 data points, with each datapoint having 13 features. This dataset is a modified version of the dataset published in the paper *"Scaling Up the Accuracy of Naive-Bayes Classifiers: a Decision-Tree Hybrid",* by Ron Kohavi. You may find this paper [online](https://www.aaai.org/Papers/KDD/1996/KDD96-033.pdf), with the original dataset hosted on [UCI](https://archive.ics.uci.edu/ml/datasets/Census+Income).

**Features**
- `age`: Age
- `workclass`: Working Class (Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked)
- `education_level`: Level of Education (Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool)
- `education-num`: Number of educational years completed
- `marital-status`: Marital status (Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse)
- `occupation`: Work Occupation (Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces)
- `relationship`: Relationship Status (Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried)
- `race`: Race (White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black)
- `sex`: Sex (Female, Male)
- `capital-gain`: Monetary Capital Gains
- `capital-loss`: Monetary Capital Losses
- `hours-per-week`: Average Hours Per Week Worked
- `native-country`: Native Country (United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands)

**Target Variable**
- `income`: Income Class (<=50K, >50K)

