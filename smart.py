import streamlit as st
import pandas as pd
import numpy as np
import sckit
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error,root_mean_squared_error

st.set_page_config(page_title="MY SMART ML DASHBOARD", layout='wide')
st.title("MY SMART ML DASHBOARD")

#---side bar controls---
st.sidebar.header("Dashboard controls")
uploaded_file=st.sidebar.file_uploader("Upload a CSV Dataset", type=['csv'])
cleaning_enable=st.sidebar.checkbox("Enable cleaning panel")
encoding_method=st.sidebar.selectbox("Encoding Method", ['Label encoding',"one hot encoding"])
scaling_method=st.sidebar.selectbox("Scaling method", ["None", "Standard scaler", "MinMaxScaler"])
model_choice=st.sidebar.selectbox("Choose Regression Model",["Linear Regression","Ridge Regression","Lasso Regression","Decission Tree","Random forest"])

if uploaded_file:
    df=pd.read_csv(uploaded_file)
    #---auto encoder other non numeric columns--
    non_numeric_cols=df.select_dtypes(include="object").columns.to_list

    for col in non_numeric_cols():
        unique_vals=df[col].nunique()
        if encoding_method=="One hot encoding" or unique_vals<=5:
            df=pd.get_dummies(df, columns=[col])
        elif encoding_method=="Label Encoding":
            le=LabelEncoder()
            df[col]=le.fit_transform(df[col].astype(str))

    #---cleanliness check---
    st.header("Data cleaning summary")
    missing=df.isnull().sum()
    duplicate=df.duplicated().sum()
    st.write("Missing Values:")
    st.write("missing[missing>0]")
    st.write(f"Duplicate rows: {duplicate}")

    #---cleaning pannel---
    if cleaning_enable and (missing.sum()>0 or duplicate>0):
        st.subheader("Cleaning pannel")
        for col in df.columns[df.isnull().any()]:
            method=st.selectbox(f"fill missing values in '{col}' with:",["Drop", "Mean", "Mode"], key=col)
            if method=='Drop':
                df=df.dropna(subset=[col])
            elif method=='Mean':
                df[col]=df[col].fillna(df[col].mean[0])
            elif method=='Mode':
                df[col]=df[col].fillna(df[col].mode[0])
        if st.checkbox("Remove duplicated rows"):
            df=df.drop_duplicates()
        st.success("Cleaning applied")

    #---encoding---
    st.subheader("Encoding categorical features")
    cat_cols=df.select_dtypes(include="object").columns.tolist()
    if cat_cols:
        if encoding_method=="Label Encoding":
            for col in cat_cols:
                le=LabelEncoder()
                df[col]=le.fit_transform(df[col])
        elif encoding_method==" one hot encoding":
            df=pd.get_dummies(df, columns=cat_cols)

    #---scaling---
    st.subheader("Feature scaling")
    scale_cols=st.multiselect("select columns to scale", df.select_dtypes(include=np.number).columns.tolist())
    if scaling_method!="None" and scale_cols:
        scaler=StandardScaler() if scaling_method =="StandardScaler" else MinMaxScaler()
        df[scale_cols]=scaler.fit_transform(df[scale_cols])
        st.success("Scaling applied")

    #---model training---
    st.subheader("Train selected model")
    target=st.selectbox("Select target column",df.columns)
    featues=st.multiselect("Select feature columns", [col for col in df.columns if col!=target])

    if featues and target:
        X=df[featues]
        y=df[target]
        X_train, X_test, y_train, Y_test=train_test_split(X, y, test_size=0.2, random_state=42)

    #---model selection---
    if model_choice=="Linear Regression":
        model=LinearRegression()
    elif model_choice=="Ridge Regression":
        model=Ridge()
    elif model_choice=="Lasso Regression":
        model=Lasso()
    elif model_choice=="Decision tree":
        model=DecisionTreeRegressor()
    elif model_choice=="Random Forest":
        model=RandomForestRegressor()
    
    model.fit(X_train, y_train)
    y_pred=model.predict(X_test)

    st.write(f"**Model used:** {model_choice}")
    st.write(f"R2 score: {r2_score(Y_test,y_pred):.4f}")
    st.write(f"mean Absolute Error: {mean_absolute_error(Y_test,y_pred):.4f}")
    st.write(f"RMSE: {root_mean_squared_error(Y_test,y_pred):.4f}")

    st.line_chart(pd.DataFrame({"Actual": Y_test, "Predicted": y_pred}).reset_index(drop=True))
else:
    st.info("Upload a csv file to begin.")