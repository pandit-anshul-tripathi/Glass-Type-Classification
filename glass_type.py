import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 

# ML classifier Python modules
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
# Loading the dataset.
@st.cache()
def load_data():
    file_path = "glass-types.csv"
    df = pd.read_csv(file_path, header = None)
    # Dropping the 0th column as it contains only the serial numbers.
    df.drop(columns = 0, inplace = True)
    column_headers = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType']
    columns_dict = {}
    # Renaming columns with suitable column headers.
    for i in df.columns:
        columns_dict[i] = column_headers[i - 1]
        # Rename the columns.
        df.rename(columns_dict, axis = 1, inplace = True)
    return df

glass_df = load_data() 

# Creating the features data-frame holding all the columns except the last column.
X = glass_df.iloc[:, :-1]

# Creating the target series that holds last column.
y = glass_df['GlassType']

# Spliting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
@st.cache()
def prediction(model,ri, na, mg, al, si, k, ca, ba, fe):
  glass_type = model.predict([[ri, na, mg, al, si, k, ca, ba, fe]])
  glass_type = glass_type[0]
  if glass_type == 1:
    return "building windows float processed"
  elif glass_type == 2:
    return "building windows non float processed"
  elif glass_type == 3:
    return "vehicle windows float processed"
  elif glass_type == 4:
    return "vehicle windows non float processed"
  elif glass_type == 5:
    return "containers"
  elif glass_type == 6:
    return "tableware"
  else:
    return "headlamp"
st.title("Glass Type Predictor")
st.sidebar.title('Explanatory Data Analysis')
if st.sidebar.checkbox("Show raw data"):
  st.subheader('Full Dataset')
  st.dataframe(glass_df)
  # Add a subheader in the sidebar with label "Scatter Plot".
st.sidebar.subheader("Scatter Plot")
# Choosing x-axis values for the scatter plot.
# Add a multiselect in the sidebar with the 'Select the x-axis values:' label
# and pass all the 9 features as a tuple i.e. ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe') as options.
# Store the current value of this widget in the 'features_list' variable.
features_list = st.sidebar.multiselect("Select X axis values", ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))
st.set_option('deprecation.showPyplotGlobalUse', False)
for i in features_list:
  st.subheader(f'Scatter plot between {i} and Glass Type')
  plt.figure(figsize=(12,6))
  sns.scatterplot(x = i, y = 'GlassType', data = glass_df)
  st.pyplot()
st.sidebar.subheader('Visualisation Selector')
plot_types = st.sidebar.multiselect('Choose the type of the plot', ('Histogram', 'Boxplot', 'Count Plot', 'Pie Chart', 'Coorelation Heatmap', 'Pair Plot'))
if 'Histogram' in plot_types:
  st.subheader("Histogram")
  columns = st.sidebar.selectbox("Select Feature to create histogram", ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))
  plt.figure(figsize=(12,6))
  plt.hist(glass_df[columns], bins = 'sturges', edgecolor = 'black')
  st.pyplot()
if 'Boxplot' in plot_types:
  st.subheader('Boxplot')
  columns = st.sidebar.selectbox("Select Features to create Boxplot", ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))
  plt.figure(figsize=(12,6))
  sns.boxplot(glass_df[columns])
  st.pyplot()
if 'Count Plot' in plot_types:
  st.subheader('Count Plot')
  plt.figure(figsize = (12,6))
  sns.countplot(glass_df['GlassType'])
  st.pyplot()
if 'Pie Chart' in plot_types:
  st.subheader('Pie Chart')
  glass_type_count = glass_df['GlassType'].value_counts()
  plt.figure(figsize = (12,6))
  plt.pie(glass_type_count, labels = glass_type_count.index, autopct = '%1.2f%%', startangle = 30)
  st.pyplot()
if 'Coorelation Heatmap' in plot_types:
  st.subheader('Coorelation Heatmap')
  plt.figure(figsize = (12,6))
  ax = sns.heatmap(glass_df.corr(), annot  = True)
  bottom,top = ax.get_ylim()
  ax.set_ylim(bottom+0.5, top-0.5)
  st.pyplot()
if 'Pair Plot' in plot_types:
  st.subheader('Pair Plot')
  plt.figure(figsize = (12,6))
  sns.pairplot(glass_df)
  st.pyplot()

st.sidebar.subheader('Select the values')
ri = st.sidebar.slider('Input RI', float(glass_df['RI'].min()), float(glass_df['RI'].max()))
na = st.sidebar.slider('Input Na', float(glass_df['Na'].min()), float(glass_df['Na'].max()))
mg = st.sidebar.slider('Input Mg', float(glass_df['Mg'].min()), float(glass_df['Mg'].max()))
al = st.sidebar.slider('Input Al', float(glass_df['Al'].min()), float(glass_df['Al'].max()))
si = st.sidebar.slider('Input Si', float(glass_df['Si'].min()), float(glass_df['Si'].max()))
k = st.sidebar.slider('Input K', float(glass_df['K'].min()), float(glass_df['K'].max()))
ca = st.sidebar.slider('Input Ca', float(glass_df['Ca'].min()), float(glass_df['Ca'].max()))
ba = st.sidebar.slider('Input Ba', float(glass_df['Ba'].min()), float(glass_df['Ba'].max()))
fe = st.sidebar.slider('Input Fe', float(glass_df['Fe'].min()), float(glass_df['Fe'].max()))
st.sidebar.subheader('Choose classifier')
classifier = st.sidebar.selectbox('Classifier', ('Support Vector Machines', 'Random Forest Classifier', 'LogisticRegression'))
if classifier == 'Support Vector Machines':
  st.sidebar.subheader('Model Hyper Parameters')
  c_value = st.sidebar.number_input('C', 1,100,step = 1)
  kernel_input = st.sidebar.radio('Kernel', ('linear', 'rbf', 'poly'))
  gamma_input = st.sidebar.number_input('Gamma', 1,100,step =1)
  if st.sidebar.button('Classify'):
    st.subheader('Support Vector Machines')
    svc_model = SVC(kernel  = kernel_input, C = c_value, gamma = gamma_input)
    svc_model.fit(X_train, y_train)
    y_pred = svc_model.predict(X_test)
    accuracy = svc_model.score(X_train, y_train)
    glass_type = prediction(svc_model, ri, na, mg, al, si, k, ca, ba, fe)
    st.write('The type of glass predicted is :', glass_type)
    st.write('accuracy:', accuracy.round(2))
    plot_confusion_matrix(svc_model, X_test, y_test)
    st.pyplot()
if classifier == 'Random Forest Classifier':
  st.sidebar.subheader('Model Hyper Parameters')
  n_estimators_input = st.sidebar.number_input('Number of trees', 100,5000,step = 10)
  max_depth_input = st.sidebar.number_input('Max Depth', 1,100,step =1)
  if st.sidebar.button('Classify'):
    st.subheader('Random Forest Classifier')
    rfc_model = RandomForestClassifier(n_estimators = n_estimators_input, max_depth = max_depth_input, n_jobs = -1)
    rfc_model.fit(X_train, y_train)
    y_pred = rfc_model.predict(X_test)
    accuracy = rfc_model.score(X_train, y_train)
    glass_type = prediction(rfc_model, ri, na, mg, al, si, k, ca, ba, fe)
    st.write('The type of glass predicted is :', glass_type)
    st.write('accuracy:', accuracy.round(2))
    plot_confusion_matrix(rfc_model, X_test, y_test)
    st.pyplot()
if classifier == 'Logistic Regression':
  st.sidebar.subheader('Model Hyper Parameters')
  c_value = st.sidebar.number_input('c', 1, 100, step = 1)
  max_iter_input = st.sidebar.number_input('Maximum Iterations', 10, 1000, step = 10)
  if st.sidebar.button('Classify'):
    st.subheader('Logistic Regression')
    log_reg = LogisticRegression(C = c_value, max_iter = max_iter_input)
    log_reg.fit(X_train, y_train)
    y_pred = log_reg.predict(X_test)
    accuracy = log_reg.score(X_train, y_train)
    glass_type = prediction(log_reg,ri, na, mg, al, si, k, ca, ba, fe)
    st.write('The type of glass predicted is :', glass_type)
    st.write('accuracy:', accuracy.round(2))
    plot_confusion_matrix(log_reg, X_test, y_test)
    st.pyplot()
