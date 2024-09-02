#!/usr/bin/env python
# coding: utf-8

# In[198]:


import numpy as np # linear algebra
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import joblib


# In[199]:


dataset = pd.read_csv(r"C:/Users/srupa/Downloads/unified mentor/1st project/Crop Production data(1).csv")


# In[3]:


dataset


# Data Description:
# State_Name: Name of the state where the crop was grown.
# 
# District_Name: Name of the district within the state.
# 
# Crop_Year: Year in which the crop was grown.
# 
# Season: Season during which the crop was grown (e.g., 
# 
# Kharif, Rabi, Whole Year).
# 
# Crop: Type of crop grown.
# 
# Area: Area of land used for cultivating the crop (in hectares).
# 
# Production: Quantity of crop produced (in units specified for each crop).
# 
# 

# #  Data preprocessing and cleaning 

# In[6]:


dataset.shape


# In[6]:


dataset.describe()


# In[7]:


dataset.info()


# we can see there are some null values in the production column 

# In[8]:


## checking for the null values
dataset.isnull().sum()


# In[200]:


## there are 3730 null values . imputing the null values with median

median_production  = dataset["Production"].median()
dataset["Production"].fillna(median_production, inplace = True)


# In[5]:


## checking whether the null values are imputed or not
dataset['Production'].isnull().sum()


# In[11]:


## checking for the duplicate values 
dataset.duplicated().sum()


# In[12]:


## checking the distinct value count of states
dataset["State_Name"].value_counts()


# In[13]:


print(dataset['State_Name'].nunique())
print(dataset['District_Name'].nunique())


# There is 33 districts and 646 districts name 

# In[17]:


print(dataset["Season"].value_counts())
dataset["Season"].value_counts().plot()


# In[18]:


## chcking the correlation 
dataset.dtypes


# # crop with highest production

# In[19]:


# Grouping and aggregation
grouped_data = dataset.groupby('Crop')['Production'].mean()
max_production = grouped_data.max()
crop_with_highest_production = grouped_data.idxmax()

# Printing the crop name and the production
print(f"Crop with the highest mean production: {crop_with_highest_production}")
print(f"Mean production: {max_production}")


# # crop grown in highest area 

# In[20]:


crop_area_data = dataset.groupby('Crop')['Area'].sum()

# Finding the crop with the highest area
max_area_crop = crop_area_data.idxmax()
max_area = crop_area_data.max()

# Printing the crop name and the area
print(f"Crop grown in the highest area: {max_area_crop}")
print(f"Total area: {max_area}")


# # year with highest production

# In[23]:


yearly_production = dataset.groupby('Crop_Year')['Production'].sum()

# Finding the year with the highest production
year_highest_production = yearly_production.idxmax()
highest_production_value = yearly_production.max()

# Printing the year with the highest production
print(f"Year with highest production: {year_highest_production}")
print(f"Highest production value: {highest_production_value}")


# # Total production in different year

# In[24]:


yearly_production = dataset.groupby('Crop_Year')['Production'].sum()

# Printing the total production for each year
print("Total production in different years:")
print(yearly_production)


# # state with higest production according to area and production

# In[25]:


state_crop_data = dataset.groupby(['State_Name', 'Crop']).agg({'Area': 'sum', 'Production': 'sum'}).reset_index()

# Finding the state with the highest production for each crop
highest_production_states = state_crop_data.loc[state_crop_data.groupby('Crop')['Production'].idxmax()]

# Finding the state with the highest area for each crop
highest_area_states = state_crop_data.loc[state_crop_data.groupby('Crop')['Area'].idxmax()]

# Printing the results
print("States with the highest production for each crop:")
print(highest_production_states)

print("\nStates with the highest area for each crop:")
print(highest_area_states)


# # Total production of each crop in diffferent year 

# In[26]:


crop_yearly_production = dataset.groupby(['Crop', 'Crop_Year'])['Production'].sum().reset_index()

# Printing the total production for each crop in different years
print("Total production of different crops in different years:")
print(crop_yearly_production.head(20))


# # Data Visualization 

# #### univarient vizualization

# In[28]:


## count plot for categorical data
sns.countplot(dataset["Season"])


# In[31]:


sns.countplot(dataset["State_Name"])


# In[32]:


## piechart
dataset['Season'].value_counts().plot(kind='pie',autopct='%.2f')


# In[33]:


plt.figure(figsize=(7,5),dpi=100)
sns.barplot(data=dataset,x='Season',y='Production');


# whole year seems to have yeilded more crops compared to other seasons in a year

# In[42]:


## year in which most of the crop were cultivated
plt.figure(figsize=(9,7),dpi=100)
sns.countplot(data=dataset,y='Crop_Year')


# 2003 was the year when most of the crops were cultivated in india

# ## bivarient visualization

# In[36]:


## state with highest production
state_production = dataset.groupby("State_Name")["Production"].sum().reset_index().sort_values(by='Production', ascending=False)

# Plot the bar graph
plt.figure(figsize=(12, 8))
plt.bar(state_production['State_Name'], state_production['Production'])
plt.xlabel('State Name')
plt.ylabel('Total Production')
plt.title('Total Crop Production by State')
plt.xticks(rotation=90)
plt.show()


# we can see that Kerela has thehighest productio, also we can see top 3 states with highest production over the year are from south

# In[37]:


state_production 


# In[24]:


# scatter plot on area and prodution 
plt.figure(figsize=(10, 6))
sns.scatterplot(data=dataset, x='Area', y='Production')

# Adding labels and title
plt.xlabel('Area')
plt.ylabel('Production')
plt.title('Scatter Plot of Area vs Production')

# Displaying the plot
plt.show()


# In[41]:


yearly_production = dataset.groupby('Crop_Year')['Production'].sum()

# Plotting the total production for each year using a bar plot
plt.figure(figsize=(10, 6))
plt.bar(yearly_production.index, yearly_production.values, color='skyblue')

# Adding labels and title
plt.xlabel('Year')
plt.ylabel('Total Production')
plt.title('Total Production of Crops in Different Years')

# Displaying the plot
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout to prevent cutting off labels
plt.show()


# # 2011 is the year with highest production 

# In[50]:


dataset['Crop'].value_counts().reset_index()


# over the years Rice, Maize, Moong have been cultivated more accross all the states in India

# In[51]:


# top 20 worst performing crops
dataset.groupby('Crop').sum()['Production'].nsmallest(20).reset_index() 



# ## performing one hot encoding in the categorical columns as all are nominal data 
# 

# In[129]:


from sklearn.preprocessing import OneHotEncoder 


# ## we have large number of state_name, district name , and crops so we will apply one hot encoding by putting the threshold to the value 
# 

# In[152]:


counts = dataset["State_Name"].value_counts(ascending  = False)
counts


# In[153]:


dataset["State_Name"].nunique()


# In[163]:


dataset["District_Name"].value_counts(ascending = False)


# In[164]:


dataset["District_Name"].nunique()


# In[165]:


count = dataset["Crop"].value_counts(ascending=False).head(20)
print(count)


# In[166]:


dataset["Crop"].nunique()


# In[167]:


dataset["Season"].nunique()


# In[201]:


dataset.drop(columns = ["District_Name"], inplace = True)


# 
# The reason here I dropped districts is that we have the info on states and districts is a subset of states and also there 646 districts name in this dataset and maybe using them has negative results- this is my assumption.

# In[202]:


# Replace states with counts less than 8000 with 'other_state'
state_threshold = 8000
state_counts = dataset['State_Name'].value_counts()
state_repl = state_counts[state_counts < state_threshold].index
dataset['State_Name'] = dataset['State_Name'].replace(state_repl, 'other_state')

# Replace crops with counts less than 6500 with 'other_crops'
crop_threshold = 6500
crop_counts = dataset['Crop'].value_counts()
crop_repl = crop_counts[crop_counts < crop_threshold].index
dataset['Crop'] = dataset['Crop'].replace(crop_repl, 'other_crops')

# Apply one-hot encoding to 'State_Name', 'Crop', and 'Season'
state_dummies = pd.get_dummies(dataset['State_Name'], prefix='State', drop_first=False, dtype=int)
crop_dummies = pd.get_dummies(dataset['Crop'], prefix='Crop', drop_first=False, dtype=int)
season_dummies = pd.get_dummies(dataset['Season'], prefix='Season', drop_first=True, dtype=int)

# Concatenate the dummy variables with the original dataset
dataset = pd.concat([dataset, state_dummies, crop_dummies, season_dummies], axis=1)

# Drop the original 'State_Name', 'Crop', and 'Season' columns
dataset.drop(['State_Name', 'Crop', 'Season'], axis=1, inplace=True)

# Check the updated dataset
print(dataset.head())


# In[ ]:





# # Model Building ******
# 

# ## splitting data into dependent and independent variable
# 

# In[203]:


x = dataset.drop(columns = ["Production"])
x


# In[204]:


y = dataset["Production"]
y


# In[205]:


# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(x, y, test_size=0.4, random_state=1)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=1)


# In[206]:


X_train.shape


# In[207]:


X_test.shape


# In[208]:


X_valid.shape


# In[212]:


# Save the columns for later use in prediction
columns_to_save = {
    'State_Name': list(dataset['State_Name'].unique()),
    'Season': list(dataset['Season'].unique()),
    'Crop': list(dataset['Crop'].unique()),
    'encoded': list(X_train.columns)
}
joblib.dump(columns_to_save, 'columns.pkl')


# ## Gradient BOOSTING 

# In[191]:


from sklearn.ensemble import GradientBoostingRegressor
# Define the Gradient Boosting model
model = GradientBoostingRegressor(
    n_estimators=1000,  # Set a high number to allow early stopping to work
    learning_rate=0.1,
    max_depth=3,
    random_state=1
)

# Fit the model with manual early stopping
best_iteration = 0
best_val_score = float('inf')
early_stopping_rounds = 10
no_improvement_count = 0

for i in range(1, model.n_estimators + 1):
    model.n_estimators = i
    model.fit(X_train, y_train)
    
    y_val_pred = model.predict(X_valid)
    val_score = mean_squared_error(y_valid, y_val_pred)
    
    if val_score < best_val_score:
        best_val_score = val_score
        best_iteration = i
        no_improvement_count = 0
    else:
        no_improvement_count += 1
    
    if no_improvement_count >= early_stopping_rounds:
        break

# Set the best number of estimators and refit the model
model.n_estimators = best_iteration
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'best_model.pkl')

# Evaluate the model
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_valid)
y_test_pred = model.predict(X_test)

mse_train = mean_squared_error(y_train, y_train_pred)
mse_val = mean_squared_error(y_valid, y_val_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

r2_train = r2_score(y_train, y_train_pred)
r2_val = r2_score(y_valid, y_val_pred)
r2_test = r2_score(y_test, y_test_pred)

print(f"MSE train: {mse_train:.3f}, val: {mse_val:.3f}, test: {mse_test:.3f}")
print(f"R^2 train: {r2_train:.3f}, val: {r2_val:.3f}, test: {r2_test:.3f}")


# ## xgboost

# In[192]:


import xgboost as xgb


# In[193]:


model = xgb.XGBRegressor(
    n_estimators=2000,
    learning_rate=0.01,
    max_depth=8,
    min_child_weight=3,
    gamma=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror',
    nthread=4,
    scale_pos_weight=1,
    seed=1
    
    
)

model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    early_stopping_rounds=50,
    verbose=10
)

# Save the model
joblib.dump(model, 'best_model.pkl')

# Evaluate the model
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)


# In[194]:


print(f"MSE train: {mse_train:.3f}, test: {mse_test:.3f}")
print(f"R^2 train: {r2_train:.3f}, test: {r2_test:.3f}")


# # Random forest

# In[210]:


from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
param_dist = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 30, None],
    'min_samples_split': [ 2, 5, 10],
    'min_samples_leaf': [ 2, 4, 6],
    'bootstrap': [True, False]
}

# Create a RandomForestRegressor model
rf = RandomForestRegressor()

# Use RandomizedSearchCV for hyperparameter tuning
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=25,
    cv=3,
    verbose=2,
    random_state=1,
    n_jobs=-1
)

# Fit the random search model
random_search.fit(X_train, y_train)

# Print the best parameters
print(f"Best parameters found: {random_search.best_params_}")

# Train the model with the best parameters
best_rf = random_search.best_estimator_

# Evaluate the model
y_train_pred = best_rf.predict(X_train)
y_valid_pred = best_rf.predict(X_valid)
y_test_pred = best_rf.predict(X_test)

mse_train = mean_squared_error(y_train, y_train_pred)
mse_valid = mean_squared_error(y_valid, y_valid_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

r2_train = r2_score(y_train, y_train_pred)
r2_valid = r2_score(y_valid, y_valid_pred)
r2_test = r2_score(y_test, y_test_pred)

print(f"MSE train: {mse_train}, validation: {mse_valid}, test: {mse_test}")
print(f"R^2 train: {r2_train}, validation: {r2_valid}, test: {r2_test}")

# Save the model
joblib.dump(best_rf, 'best_rf_model.pkl')


# In[214]:


dataset.columns


# # Random forest is good model so using that for the model building 
