# Predictive-Pulse-Harnessing-Machine-Learning-For-Blood-Pressure-Analysis
#Libraries required:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")

#reading the dataframe

df = pd.read_csv('/Insurance_Data_Project.csv')
df
df.shape

#checking if any null values


df.isnull().sum()
print("The Missing value percentage of classif column is:",(df.classif.isnull().sum()/df.classif.shape[0]*100)
)

#replaced the missing column values with using Mode (Central Tendency Measure)

df['classif'].fillna(df['classif'].mode()[0], inplace=True)

#No null values

df.isnull().sum()

#duplicate records

df.duplicated().sum()
#keeping the first duplicates and dropping the next occuring duplicate record

df.drop_duplicates(keep='first',inplace=True)
#information of the columns

df.info()
#statistical description of the numerical data

df.describe()

#created a function to divide the categorical and numerical values

from tabulate import tabulate
def dtype(col):
    cat = []
    con = []
    for col in df:
        if(df[col].dtypes == 'object'):
            cat.append(col)
        else:
            con.append(col)
    return cat, con
cat, con = dtype(df)
table = [cat, con]
print(tabulate({"Categorical": cat, "Continuous": con}, headers=['Category', "Continuous"]))

#created a function for category column info

def cat_info(col):
    print(f'Unique values in {col} :{df[col].nunique()}')
    print(f'Missing values count for {col} : {df[col].isnull().sum()}')
    print(f'Mode of {col} : {df[col].mode()[0]}')
    value_count=df[col].value_counts()
    value_per=df[col].value_counts(normalize=True)*100

#created a function for category column visualization

def visual(col):
    value_count=df[col].value_counts()
    value_per=df[col].value_counts(normalize=True)*100
    plt.figure(figsize=(5,3))
    plt.bar(value_count.index,value_count.values)
    plt.title(f'Value count of {col}')
    plt.figure(figsize=(5,3))
    plt.pie(value_per.values,labels=value_count.index,autopct='%1.1f%%')
    plt.title(f'Percentage of {col}')

def main_cat(col):
    cat_info(col)
    visual(col)
  main_cat('gender')
 main_cat('classif')
main_cat('smoker')
main_cat('region')

#created a function for numerical column info
def num_info(col):
    print(col)
    print("Mean", df[col].mean())
    print(f'skewness {df[col].skew()}')
    print(f'kurtos {df[col].kurt()}')

#created a function for numerical column visualization

def num_visual(col):
    fig,ax=plt.subplots(2,1,figsize=(6,7))
    ax[0].hist(df[col])
    ax[0].set_xlabel(col)
    ax[0].set_ylabel('count')
    sns.boxplot(y=df[col],ax=ax[1])
    ax[1].set_xlabel(col)
    ax[1].set_ylabel('count')
def main_num(col):
    num_info(col)
    num_visual(col)

main_num('age')

a = df[(df["age"] > 0) | (df["age"] < 100)].index
for i in a:
  df.loc[i, "age"] = df["age"].mean()


main_num('age')

df["age"][(df["age"]<5)|(df["age"]>100)]=round(df["age"].mean(),0)
df['bmi'] = df['bmi'].clip(upper = df['bmi'].quantile(0.95))

main_num('bmi')
main_num('children')
main_num('charges')

#Analysd categorical values based on the 'charges'

fig,axs = plt.subplots(2,2,figsize = (15,6))
axs = axs.ravel()
j = 0
for i in cat:
    sns.boxplot(y="charges", x =i, data = df, ax=axs[j])
    j+=1

plt.show()

#Visualized numerical values based on the 'charges'

fig,axs = plt.subplots(2,2,figsize = (15,6))
axs = axs.ravel()
j = 0
for i in con:
    sns.scatterplot(y="charges", x =i, data = df, ax=axs[j])
    j+=1

plt.show()

#linear regression plot between 'Bmi' & 'Charges'

sns.regplot(data = df,x = 'charges', y = 'bmi')
plt.title('bmi Vs  charges')
plt.show()

 sns.pairplot(df)
 plt.show()

 cond1 = df[(df['age'] > 30) & (df['gender'] == 'male') & (df['bmi'] >40)]
fig, ax=plt.subplots(1,2,figsize=(15,5))
sns.countplot(x='gender',hue='smoker',data=cond1,ax=ax[0])
sns.boxplot(x='gender',y='charges',data=cond1,ax=ax[1])
plt.show()

cond2 = df[(df['region'] != 'southwest' ) & (df['children'] >=2) & (df['smoker'] == 'yes')]
fig, ax=plt.subplots(1,2,figsize=(15,5))
sns.countplot(x='classif',hue='region',data=cond2,ax=ax[0])
sns.boxplot(x='bmi',y='smoker',data=cond2,ax=ax[1])
plt.show()

#converting all categorical features (columns) into numerical features

label_encoder = preprocessing.LabelEncoder()
for i in cat:
    df[i] = label_encoder.fit_transform(df[i])
df.info()

#scaling all the numerical values under one scale using MinMaxScaler()

num_cat = ['int32', 'float64', 'int64']
num_df  = df.select_dtypes(include=num_cat)
scaler = MinMaxScaler()
scale_df = pd.DataFrame(scaler.fit_transform(num_df), columns=num_df.columns)
scale_df

#correlation between each features

corel_matrix = scale_df.corr()
corel_matrix

#heat map
plt.figure(figsize=(10,7))
sns.heatmap(corel_matrix, annot = True, cmap = 'Oranges')
plt.show()

#threshold correlation
def correlation(df,threshold):
    col_corr=set()
    corel_matrix=df.corr()
    for i in range(len(corel_matrix.columns)):
        for j in range(i):
            if(abs(corel_matrix.iloc[i,j]))>threshold:
                col_name=corel_matrix.columns[i]
                col_corr.add(col_name)
    return col_corr

  sel_features=correlation(scale_df,0.5)
len(set(sel_features))

#here smoker hue is 0 & 1
sns.pairplot(scale_df, hue= 'gender')
plt.show()
    
sel_features
y =df['charges']
X = scale_df.drop(['charges'], axis = 1)
X.shape
y.shape

#splitting the data into training nd testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
print(X_train.shape)
print(y_test.shape)

linear = LinearRegression()
lr_model = linear.fit(X_train, y_train)
y_pred = lr_model.predict(X_test)

r2_train = lr_model.score(X_train, y_train)
intercept = lr_model.intercept_
slope= lr_model.coef_

lr_rmse = mean_squared_error(y_test, y_pred)
lr_mse = mean_squared_error(y_test, y_pred)
lr_mae = mean_absolute_error(y_test, y_pred)
lr_mape = mean_absolute_percentage_error(y_test, y_pred)

r2_test = lr_model.score(X_test, y_test)
r2_test

print('R_square:', r2_train)
print('Intercept:', intercept)
print('Slope:',slope)

r2_score_train=[]
r2_score_test=[]
rmse=[]
mse=[]
mae=[]
mape=[]

print("RMSE:", lr_rmse)
rmse.append(lr_rmse)
print("MSE:", lr_mse)
mse.append(lr_mse)
print("MAE:", lr_mae )
mae.append(lr_mae)
print("MAPE:", lr_mape)
mape.append(lr_mape)

f = 7 #features count
print("tr:",r2_train)
print("te:",r2_test)

 def adjusted_r2_score(model,x,y):
     r2_scores=model.score(x,y)
     n=x.count()
     x=len(X_test.columns)-1
     adjusted_r2_score=1-((1-r2_scores)*(n-1)/(n-x-1))
     return adjusted_r2_score[0]

adjusted_train_r2 = 1-((1-0.64)*(935-1)/(935-f-1))
adjusted_test_r2 = 1-((1-0.70)*(402-1)/(402-f-1))
print("Adjusted Train R2:",adjusted_train_r2)
r2_score_train.append(adjusted_train_r2)
print("Adjusted Test R2:", adjusted_test_r2)
r2_score_test.append(adjusted_test_r2)
#clear descp

plt.scatter(y_test,y_pred)
plt.show()

from sklearn import tree

kf=KFold(n_splits=5, shuffle=True, random_state=42)

dt = DecisionTreeRegressor()

param_grid2={"min_samples_split":np.arange(10,51,10),
            "min_samples_leaf":np.arange(10,101,20),
            "max_depth":np.arange(3,15)}

grid_cv2 = GridSearchCV(dt,param_grid2,cv=kf,scoring="r2")

grid_cv2.fit(X_train,y_train)

grid_cv2.best_score_
grid_cv2.best_params_
grid_cv2.best_estimator_
dt_model = DecisionTreeRegressor(max_depth=4, min_samples_leaf=10, min_samples_split=30)
dt_model.fit(X_train,y_train)

dt_r2train = dt_model.score(X_train,y_train)

r2_score_train.append(dt_r2train)
dt_r2train
dt_r2test = dt_model.score(X_test, y_test)
r2_score_test.append(dt_r2test)
dt_r2test

y_pred1=dt_model.predict(X_test)

dt_rmse = mean_squared_error(y_test, y_pred1)
dt_r2 = r2_score(y_test,y_pred1)
dt_mae = mean_absolute_error(y_test, y_pred1)
dt_mape = mean_absolute_percentage_error(y_test, y_pred1)

print("rmse",dt_rmse)
rmse.append(dt_rmse)
print("r2:",dt_r2)
mse.append(dt_r2)
print("Mae:", dt_mae)
mae.append(dt_mae)
print("MAPE:", dt_mape)
mape.append(dt_mape)

plt.scatter(y_test,y_pred1)
plt.show()

plt.scatter(y_test,y_pred1)
plt.show()
plt.figure(figsize=(20,7))
tree.plot_tree(dt_model,feature_names=X.columns,filled=True,fontsize=5)
plt.show()

param_grid={"n_estimators":np.arange(20,201,20),
            "min_samples_split":np.arange(10,51,10),
            "min_samples_leaf":np.arange(10,101,20),
            "max_depth":np.arange(3,15)}

rf = RandomForestRegressor()
grid_cv1=GridSearchCV(rf,param_grid,cv=kf,scoring="r2")
# grid_cv1.fit(X_train,y_train)

# grid_cv1.fit(X_train,y_train)
# grid_cv1.best_score_
# grid_cv1.best_params_
# grid_cv1.best_estimator_

rf_model = RandomForestRegressor(max_depth=4, min_samples_leaf=40, min_samples_split=10,n_estimators=10)
rf_model.fit(X_train,y_train)

rf_r2_train = rf_model.score(X_train,y_train)
r2_score_train.append(rf_r2_train)
rf_r2_train

rf_r2_test = rf_model.score(X_test, y_test)
r2_score_test.append(rf_r2_test)
rf_r2_test

y_pred2=rf_model.predict(X_test)


rf_rmse = mean_squared_error(y_test, y_pred2)
rf_r2 = r2_score(y_test,y_pred2)
rf_mae =  mean_absolute_error(y_test, y_pred2)
rf_mape = mean_absolute_percentage_error(y_test, y_pred2)

print("rmse",rf_rmse)
rmse.append(rf_rmse)
print("r2:",rf_r2)
mse.append(rf_r2)
print("Mae:",rf_mae)
mae.append(rf_mae)
print("MAPE:", rf_mape)
mape.append(rf_mape)

plt.scatter(y_test,y_pred2)
plt.show()
ada = AdaBoostRegressor()
kf=KFold(n_splits=5)
param_grid={"n_estimators":np.arange(10,101,10),
            "learning_rate":np.arange(0.05,1,0.05),
}

grid_cv = GridSearchCV(ada,param_grid,cv=kf,scoring="r2")

# grid_cv.fit(X_train,y_train)
# grid_cv.best_params_
# grid_cv.best_estimator_
# grid_cv.best_score_  #accuracy
# ada_model = AdaBoostRegressor(learning_rate=0.05, n_estimators=20,random_state=42)

# # ada_model.fit(X_train,y_train)
# adr2_train = ada_model.score(X_train,y_train)
# r2_score_train.append(adr2_train)
# adr2_train
# y_pred3 = ada_model.predict(X_test)# ad_rmse = mean_squared_error(y_test, y_pred3, squared=False)
# ad_r2 = r2_score(y_test,y_pred3)
# ad_mae =  mean_absolute_error(y_test, y_pred3)
# ad_mape = mean_absolute_percentage_error(y_test, y_pred3)

# print("rmse",ad_rmse)
# rmse.append(ad_rmse)
# print("r2:",ad_r2)
# mse.append(ad_r2)
# print("Mae:", ad_mae)
# mae.append(ad_mae)
# print("MAPE:",ad_mape)
# mape.append(ad_mape)
# plt.scatter(y_test,y_pred3)
# plt.show()
## KNN
r2_scores=[]
for k in range(2,25):
    knn_score=cross_val_score(KNeighborsRegressor(k),X_train,y_train,scoring="r2",cv=kf)
    r2_scores.append(np.mean(knn_score))

for k in range(2,25):
    print("number of neighbors:",k,":",r2_scores[k-2])

plt.figure(figsize=(9,5))
plt.plot(range(2,25),r2_scores,marker="o")
plt.ylabel("r2_scores")
plt.xlabel("k_values")
plt.title("r2_scores in different k values")
plt.xticks(range(0,25,3))
plt.grid()
plt.show()

k= 9
kn_model = KNeighborsRegressor(k).fit(X_train, y_train)
y_pred_4 = kn_model.predict(X_test)

knr2_train = kn_model.score(X_train, y_train)
r2_score_train.append(knr2_train)
knr2_test = kn_model.score(X_test, y_test)
r2_score_test.append(knr2_test)
# print("Model adjusted r2 score on training data :",adjusted_r2_score(knn_model,x_train,y_train))
# print("Model adjusted r2 score on test data :",adjusted_r2_score(knn_model,x_test,y_test))
# print()

print("accuracy_train:",knr2_train)
print("accuracy_test:",knr2_test)

kn_rmse = mean_squared_error(y_test, y_pred_4)
kn_r2 = r2_score(y_test,y_pred_4)
kn_mae = mean_absolute_error(y_test, y_pred_4)
kn_mape = mean_absolute_percentage_error(y_test, y_pred_4)

print("rmse",kn_rmse)
rmse.append(kn_rmse)
print("r2:",kn_r2)
mse.append(kn_r2)
print("Mae:", kn_mae)
mae.append(kn_mae)
print("Mape:", kn_mape)
mape.append(kn_mape)
plt.scatter(y_test,y_pred_4)
plt.show()
level1=[]
level1.append(("lr",lr_model))
level1.append(("knn",kn_model))
level1.append(("svr",SVR()))
level1.append(("dt",dt_model))
# level1.append(("rnd",rf_model))
# level1.append(("ada", ada_model))
level2=LinearRegression()
stack_model=StackingRegressor(estimators=level1,final_estimator=level2,cv=kf)

level1

st_model =stack_model.fit(X_train, y_train)
y_pred_st = st_model.predict(X_test)

score=cross_val_score(stack_model,X_train,y_train,scoring="r2",cv=kf)
print(score)
print("Rscore:",np.mean(score))

str2_train = st_model.score(X_train, y_train)
r2_score_train.append(str2_train)
str2_test = st_model.score(X_test, y_test)
r2_score_test.append(str2_test)

print("R-square train data:",str2_train )
print("R-square test data:",str2_test )
st_rmse = mean_squared_error(y_test, y_pred_st)
st_r2 = r2_score(y_test,y_pred_st)
st_mae = mean_absolute_error(y_test, y_pred_st)
st_mape = mean_absolute_percentage_error(y_test, y_pred_st)

print("rmse",st_rmse)
rmse.append(st_rmse)
print("r2:",st_r2)
mse.append(st_r2)
print("Mae:",st_mae)
mae.append(st_mae)
print("MAPE:", st_mape)
mape.append(st_mape)

plt.scatter(y_test,y_pred_st)
plt.show()

model_list=["Linear Regression","Decision Tree Regression","KNN Regression","Stacked Regression"]
metric_list=["Models","r2 Score(Train)","r2 Score(Test)","RMSE","MSE","MAE","MAPE"]

mse 
rmse
mae
final_results=pd.DataFrame()
for i in range(0,len(model_list)):
    ab=[[model_list[i],r2_score_train[i],r2_score_test[i],rmse[i],mse[i],mae[i],mape[i]]]
    new=pd.DataFrame(ab)
    final_results=pd.concat([final_results,new],axis=0)
final_results.columns=metric_list
final_results=final_results.reset_index(drop=True)
final_resultsrvse_list = list(final_results['Models'])
rvse_list

rvse_list

models = final_results['Models']
test_score = final_results['r2 Score(Test)']
# ab = round(test_score.reverse(),4)

# Explicitly set x and y arguments
sns.barplot(x=test_score, y=models, orient='h', order=rvse_list)
plt.xlabel('r2_Scores')
# for i, v in enumerate(test_score):
#     plt.text(v, i, str(v), ha='left', va='center')
plt.title('Models')
plt.show()

import pickle

# Find the best model based on the R2 score on the test set
best_model_index = final_results['r2 Score(Test)'].idxmax()
best_model_name = final_results.loc[best_model_index, 'Models']

# Get the actual best model object
if best_model_name == "Linear Regression":
    best_model = lr_model
elif best_model_name == "Decision Tree Regression":
    best_model = dt_model
elif best_model_name == "Random Forest Regressor":
     best_model = rf_model
elif best_model_name == "Ada Booster":
     best_model = ada_model
elif best_model_name == "KNN Regression":
    best_model = kn_model
elif best_model_name == "Stacked Regression":
    best_model = st_model
else:
    best_model = None
    print(f"Could not find model object for {best_model_name}")


if best_model is not None:
    # Pickle the best model
    filename = f'{best_model_name.replace(" ", "_").lower()}_model.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(best_model, f)

   print(f"Best model '{best_model_name}' pickled successfully as '{filename}'")

  # You can also download the pickled file
  try:
        files.download(filename)
    except NameError:
        print("google.colab.files not imported. Skipping download.")

!pip install flask pandas scikit-learn pyngrok

import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, jsonify, render_template_string
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import os
from pyngrok import ngrok

# Define HTML template for input form
HTML_FORM = """
<!doctype html>
<title>Insurance Charge Prediction</title>
<h2>Predict Insurance Charges</h2>
<form method="post">
    Age: <input type="text" name="age"><br>
    Gender:
    <select name="gender">
        <option value="male">male</option>
        <option value="female">female</option>
    </select><br>
    BMI: <input type="text" name="bmi"><br>
    Children: <input type="text" name="children"><br>
    Smoker:
    <select name="smoker">
        <option value="yes">yes</option>
        <option value="no">no</option>
    </select><br>
    Region:
    <select name="region">
        <option value="southwest">southwest</option>
        <option value="southeast">southeast</option>
        <option value="northwest">northwest</option>
        <option value="northeast">northeast</option>
    </select><br>
    Classif:
    <select name="classif">
        <option value="Overweight">Overweight</option>
        <option value="Obesity Stage 1">Obesity Stage 1</option>
        <option value="Pre-Obesity">Pre-Obesity</option>
        <option value="Normal Weight">Normal Weight</option>
        <option value="Obesity Stage 2">Obesity Stage 2</option>
        <option value="Underweight">Underweight</option>
        <option value="Obesity Stage 3">Obesity Stage 3</option>
    </select><br>
    <input type="submit" value="Predict">
</form>
{% if prediction %}
    <h3>Predicted Insurance Charge: {{ prediction }}</h3>
{% endif %}
"""

# Load the pre-trained model (assuming stacked_regression_model.pkl was saved from the previous code)
try:
    with open('stacked_regression_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Stacked Regression model loaded successfully.")
except FileNotFoundError:
    print("Error: stacked_regression_model.pkl not found. Please ensure the model is pickled and saved.")
    # As a fallback, try other models if available (less ideal)
    try:
        with open('random_forest_regressor_model.pkl', 'rb') as f:
            model = pickle.load(f)
        print("Random Forest Regressor model loaded successfully as fallback.")
    except FileNotFoundError:
         print("Error: random_forest_regressor_model.pkl also not found. Please run the preceding code to train and save a model.")
         model = None # Set model to None if no model is found

# Data preparation (same as in the preceding code, needed for scaling/encoding)
df = pd.read_csv('/Insurance_Data_Project.csv')
df['classif'].fillna(df['classif'].mode()[0], inplace=True)
df.drop_duplicates(keep='first',inplace=True)
df["age"][(df["age"]<5)|(df["age"]>100)]=round(df["age"].mean(),0)
df['bmi'] = df['bmi'].clip(upper = df['bmi'].quantile(0.95))

cat_cols = ['gender', 'classif', 'smoker', 'region']
for i in cat_cols:
    le = LabelEncoder() # Create a new LabelEncoder for each column
    # Fit on the training data column to ensure all possible categories are known
    le.fit(df[i])
    df[i] = le.transform(df[i]) # Transform the original df

num_cat = ['int32', 'float64', 'int64']
num_df  = df.select_dtypes(include=num_cat)
scaler = MinMaxScaler()
scaler.fit(num_df.drop('charges', axis=1)) # Fit scaler on features


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        try:
            # Get input data from form
            age = float(request.form['age'])
            gender = request.form['gender']
            bmi = float(request.form['bmi'])
            children = int(request.form['children'])
            smoker = request.form['smoker']
            region = request.form['region']
            classif = request.form['classif']

            # Create a DataFrame from the input
            input_data = pd.DataFrame([[age, gender, bmi, children, smoker, region, classif]],
                                      columns=['age', 'gender', 'bmi', 'children', 'smoker', 'region', 'classif'])

            # Apply the same preprocessing as the training data
            # Label encode categorical features
            for i in cat_cols:
                le = LabelEncoder()
                # Fit on the training data column to ensure all possible categories are known
                le.fit(df[i]) # Fit on the original df column
                input_data[i] = le.transform(input_data[i])

            # Scale numerical features
            input_scaled = scaler.transform(input_data)

            # Make prediction
            if model is not None:
                prediction = model.predict(input_scaled)[0]
                prediction = f"${prediction:,.2f}" # Format as currency
            else:
                prediction = "Model not loaded. Cannot predict."

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template_string(HTML_FORM, prediction=prediction)

if __name__ == '__main__':
    if model is not None:
        # Authenticate ngrok (replace with your actual authtoken from https://dashboard.ngrok.com/get-started/your-authtoken)
        # You can add your authtoken as a Colab secret and access it like:
        # from google.colab import userdata
        # ngrok.set_auth_token(userdata.get('NGROK_AUTH_TOKEN'))
        # For a free account, authtoken might not be strictly necessary for basic http tunnels, but it's good practice.
        # ngrok.set_auth_token("YOUR_NGROK_AUTHTOKEN")


       # Establish a public tunnel
        public_url = ngrok.connect(5000).public_url
        print(f" * Ngrok Tunnel established at: {public_url}")

        # Run the Flask app
        app.run()
    else:
        print("Flask app not started because the model could not be loaded.")
