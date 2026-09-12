%pip install seaborn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df1=pd.read_csv("test[1].csv")
df2=pd.read_csv("train[1].csv")

df1.head()

df2.head()

df1.describe()

df2.describe()

df1.shape

df2.shape

df1.info()

df2.info()

from sklearn.impute import SimpleImputer
median=["LotFrontage","MasVnrArea","BsmtFinSF1","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF","BsmtFullBath","BsmtHalfBath","GarageYrBlt","GarageCars","GarageArea"]
imputer1=SimpleImputer(strategy="median")
df2[median]=imputer1.fit_transform(df2[median])
df1[median]=imputer1.transform(df1[median])

none2=["Alley","MasVnrType","BsmtFinType2","Electrical","FireplaceQu","GarageType","GarageFinish","GarageQual","GarageCond","PoolQC","Fence","MiscFeature","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1"]
none1=["MSZoning","Alley","Utilities","Exterior1st","Exterior2nd","MasVnrType","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2","KitchenQual","Functional","FireplaceQu","GarageType","GarageFinish","GarageQual","GarageCond","PoolQC","Fence","MiscFeature","SaleType"]
df2[none2]=df2[none2].fillna("None")
df1[none1]=df1[none1].fillna("None")

num2=["MSZoning","Street","Alley","LotShape","LandContour","Utilities","LandSlope","Neighborhood","Condition1","BldgType","HouseStyle","RoofStyle","RoofMatl","Exterior1st","Exterior2nd","MasVnrType","ExterQual","ExterCond","Foundation","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2","Heating","HeatingQC","CentralAir","Electrical","KitchenQual","Functional","FireplaceQu","GarageType","GarageFinish","GarageQual","GarageCond","PavedDrive","PoolQC","Fence","MiscFeature","SaleType","SaleCondition","LotConfig","Condition2"]
num1=["MSZoning","Street","Alley","LotShape","LandContour","Utilities","LandSlope","Neighborhood","Condition1","BldgType","HouseStyle","RoofStyle","RoofMatl","Exterior1st","Exterior2nd","MasVnrType","ExterQual","ExterCond","Foundation","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2","Heating","HeatingQC","CentralAir","Electrical","KitchenQual","Functional","FireplaceQu","GarageType","GarageFinish","GarageQual","GarageCond","PavedDrive","PoolQC","Fence","MiscFeature","SaleType","SaleCondition","LotConfig","Condition2"]
encoder2=pd.get_dummies(df2,columns=num2,dtype=int)
encoder1=pd.get_dummies(df1,columns=num1,dtype=int)
encoder2,encoder1=encoder2.align(encoder1,join="left",axis=1,fill_value=0)

encoder2.info()

encoder1.info()

X=encoder2.drop("SalePrice",axis=1)
y=encoder2["SalePrice"]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train_scaler=scaler.fit_transform(X_train)
X_test_scaler=scaler.transform(X_test)

y_train_log=np.log1p(y_train)

from sklearn.linear_model import RidgeCV
ridge=RidgeCV(alphas=np.logspace(-3,3,100))
ridge.fit(X_train_scaler,y_train_log)
y_pred_ridge=ridge.predict(X_test_scaler)
ridge_preds_dollars=np.expm1(ridge.predict(X_test_scaler))


from sklearn.linear_model import LassoCV
lasso=LassoCV(max_iter=10000,tol=0.01,random_state=42)
lasso.fit(X_train_scaler,y_train_log)
y_pred_lasso=lasso.predict(X_test_scaler)
lasso_preds_dollars=np.expm1(lasso.predict(X_test_scaler))

from sklearn.linear_model import ElasticNetCV
en=ElasticNetCV(max_iter=10000,tol=0.01,random_state=42)
en.fit(X_train_scaler,y_train_log)
y_pred_en=en.predict(X_test_scaler)
preds_dollars=np.expm1(y_pred_en)
en_preds_dollars = np.expm1(en.predict(X_test_scaler))

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
mse=mean_squared_error(y_test,y_pred_ridge)
rmse=np.sqrt(mse)
r2=r2_score(y_test,y_pred_ridge)
mae=mean_absolute_error(y_test,y_pred_ridge)

models={"RIDGE":ridge_preds_dollars,"LASSO":lasso_preds_dollars,"ELASTIC NET":en_preds_dollars}
          
for name,preds in models.items():
    mse=mean_squared_error(y_test,preds)
    rmse=np.sqrt(mse)
    mae=mean_absolute_error(y_test,preds)
    r2=r2_score(y_test,preds)
    
    print(f"{name}")
    print(f"MSE: {mse:,.2f}")
    print(f"RMSE: ${rmse:,.2f}")
    print(f"MAE: ${mae:,.2f}")
    print(f"R2: {r2:.4f}\n")
