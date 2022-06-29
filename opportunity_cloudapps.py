import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import xgboost as xgb
from xgboost import DMatrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score
import pandas as pd
from PIL import Image
import pickle
from collections import Counter

def importance(x_train, x_test, y_train, y_test, file_name, cat_model):
    xgb_model = pickle.load(open(file_name, "rb"))
    cat_model = pickle.load(open(cat_model, "rb"))
    train_dmatrix = DMatrix(x_train, y_train)
    test_dmatrix = DMatrix(x_test)

    params = {'objective': 'binary:logistic', 'eta': 0.1, 'gamma': 1.0,
              'min_child_weight': 0.1, 'max_depth': 6}
    # xgb_model = xgb.train(params, train_dmatrix, num_boost_round=4,)
    cat_pred = cat_model.predict(x_test)
    df_temp = pd.DataFrame([accuracy_score(y_test, cat_pred.round()), precision_score(y_test, cat_pred.round()),
                            recall_score(y_test, cat_pred.round()), f1_score(y_test, cat_pred.round())],
                           index=["Accuracy", "Precision", "Recall", "F1 Score"])

    df_temp.columns = ['Scores']
    st.dataframe(df_temp)
    dict_val = xgb_model.get_score(importance_type='gain')
    value = dict_val.values()
    min_value = min(value)
    max_value = max(value)
    dict_val_={k:(v-min_value)/(max_value - min_value) for k,v in dict_val.items()}
    count = 1
    graph_visualize(dict_val_,count)  ## call functions for importance graphs

    return dict_val_

def graph_visualize(dict_val,count):

    top_n = st.selectbox("Choose # of top Important Features", list(range(2, len(dict_val))),key= count)
    new_dict_val = dict(Counter(dict_val).most_common(top_n))

    objects = list(new_dict_val.keys())
    # y_pos = np.arange(len(objects))
    performance = new_dict_val.values()
    plt.rcParams.update({'font.size': 35})
    width = st.sidebar.slider("Importance width", 1, 25, 25)
    height = st.sidebar.slider("Importance height", 1, 25, 12)
    plt.figure(figsize=(width, height))
    plt.barh(objects, performance, align='center')
    # plt.xticks(y_pos, objects, fontsize=16)
    # plt.setp(ax.get_xticklabels(), rotation='vertical', fontsize=14)

    plt.ylabel("Feature", fontsize=40)
    plt.xlabel('Importance', fontsize=40)
    plt.title('Features Importance', fontsize=50)

    # plt.show()
    st.pyplot(plt)
    return dict_val

def graph_contribution(dict_val,count):

    top_n = st.selectbox("Choose # of top features Contributers", list(range(2, len(dict_val))),key= count)
    new_dict_val = dict(Counter(dict_val).most_common(top_n))

    objects = list(new_dict_val.keys())
    # y_pos = np.arange(len(objects))
    performance = new_dict_val.values()
    plt.rcParams.update({'font.size': 35})
    width = st.sidebar.slider("Contribution width", 1, 25, 25)
    height = st.sidebar.slider("Contribution height", 1, 25, 12)

    plt.figure(figsize=(width, height))
    plt.barh(objects, performance, align='center')
    # plt.xticks(y_pos, objects, fontsize=16)
    # plt.setp(ax.get_xticklabels(), rotation='vertical', fontsize=14)

    plt.ylabel("Feature", fontsize=40)
    plt.xlabel('Contribution', fontsize=40)
    plt.title('Feature Contribution', fontsize=50)

    # plt.show()
    st.pyplot(plt)
    return dict_val


def onehotEncode(one_hot_df, dict_val):
    new_keys = list(dict_val.keys())
    new_keys.append('IsWon')
    original_df = one_hot_df
    selected_df = original_df[new_keys]
    cols = selected_df.columns
    num_cols = selected_df._get_numeric_data().columns
    catagorical_df = selected_df[list(set(cols) - set(num_cols))]
    one_hot_df = pd.get_dummies(selected_df, columns=catagorical_df.columns.to_list())
    X = one_hot_df.drop(columns=['IsWon'])
    y = one_hot_df['IsWon']
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=4)
    ###################################
    ##### xgboost part ####
    ####################################

    train_dmatrix = DMatrix(x_train, y_train)
    test_dmatrix = DMatrix(x_test)
    params = {'objective': 'binary:logistic', 'eta': 0.1, 'gamma': 1.0,
              'min_child_weight': 0.1, 'max_depth': 6}
    xgb_model = xgb.train(params, train_dmatrix, num_boost_round=4, )
    dict_val = xgb_model.get_score(importance_type='gain')
    value = dict_val.values()
    min_value = min(value)
    max_value = max(value)
    dict_val_={k:(v-min_value)/(max_value - min_value) for k,v in dict_val.items()}
    count = 2
    graph_contribution(dict_val_,count)


if __name__ == "__main__":
    file_name = r"C:\Users\nilanjan.das\xgb_model_file"
    cat_model = r"C:\Users\nilanjan.das\cat_model_oppor"
    image = Image.open(r"C:\Users\nilanjan.das\Downloads\Diebold_Nixdorf_logo_2018.jpg")
    st.sidebar.image(image)
    st.sidebar.header("Competitive Intelligence")
    st.title("Deals Win-Loss Prediction")
    st.cache(allow_output_mutation=True)
    uploaded_file = st.file_uploader("Choose a file")  # read_dataset()
    if uploaded_file is not None:
        updated_df = pd.read_csv(uploaded_file)
        X = updated_df.drop(columns=['IsWon'])
        y = updated_df['IsWon']
        X = np.nan_to_num(X)
        y = np.nan_to_num(y)
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=4)
        dict_val = importance(x_train, x_test, y_train, y_test, file_name, cat_model)  ## variable importance

        st.cache(allow_output_mutation=True)
        uploaded_file1 = st.file_uploader("Choose a file",key=1)
        if uploaded_file1 is not None:
            one_hot_df = pd.read_csv(uploaded_file1)
            onehotEncode(one_hot_df, dict_val)