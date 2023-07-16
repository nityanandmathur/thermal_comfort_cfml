import os
import pickle
import random

import dice_ml
import gradio as gr
import pandas as pd
from sklearn.model_selection import train_test_split


def read_csv(dataset):
    path = '../data/' + dataset + '.csv'
    data = pd.read_csv(path)
    return data.head(10)

def train_model(input_df, target, test_size, model_name, features_to_drop):

    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    path = '../data/' + input_df + '.csv'

    data = pd.read_csv(path, index_col=0)
    data = data.dropna()
    data = data.drop(features_to_drop, axis=1)

    metrics = ['TSV','TPV','TCV','TSL']
    metrics.remove(target[0])

    data = data.drop(metrics, axis=1)
    features = data.drop(target, axis=1).columns.to_list()
    target_f = data[target[0]]

    datasetX = data.drop(target, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(datasetX, target_f, 
                                                        test_size=test_size, random_state=42)
    
    categorical_features = X_train.columns.difference(features)

    # We create the preprocessing pipelines for both numeric and categorical data.
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    transformations = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, features),
            ('cat', categorical_transformer, categorical_features)])

    regr = Pipeline(steps=[('preprocessor', transformations),
                            ('regressor', RandomForestRegressor())])
    model = regr.fit(X_train, y_train)
    pickle.dump(model, open('../models/' + model_name + '.pkl', 'wb'))  
    return 'Model Saved'

def generate_cfs_total(input_df_T, target_T, radio_T, predefined_T, custom_T, dropped_features_T, freeze_features_T, model_T):
    path = '../data/' + input_df_T + '.csv'
    data = pd.read_csv(path)
    data = data.dropna()
    model = pickle.load(open('../models/' + model_T + '.pkl', 'rb'))
    data = data.drop(dropped_features_T, axis=1)
    metrics = ['TSV','TPV','TCV','TSL']
    metrics.remove(target_T[0])

    data = data.drop(metrics, axis=1)
    features = data.drop(target_T[0], axis=1).columns.tolist()
    target = data[target_T[0]]
    datasetX = data.drop('TSV', axis=1)
    x_train, x_test, y_train, y_test = train_test_split(datasetX,
                                                    target,
                                                    test_size=0.2,
                                                    random_state=0)

    always_immutable = ['AvgMaxDailyTemp','AvgMinDailyTemp','School','StartTime']
    freezed = always_immutable + freeze_features_T + [target_T[0]]

    features_to_vary = data.columns.difference(freezed).to_list()

    d = dice_ml.Data(dataframe=data, continuous_features=features, outcome_name=target_T[0])
    m = dice_ml.Model(model=model, backend='sklearn', model_type='regressor')

    exp = dice_ml.Dice(d, m, method='random')

    if radio_T == 'Predefined':
        random_index = random.randint(0, len(x_train-2))
        print(int(predefined_T))
        query_instances = x_test[random_index:random_index+int(predefined_T)]

    elif radio_T == 'Custom':
        query_instances = custom_T
    
    dice_exp = exp.generate_counterfactuals(query_instances, total_CFs=4, desired_range=[0.0, 2.0], features_to_vary=features_to_vary)
    return dice_exp.visualize_as_dataframe(show_only_changes=True)


def generate_cfs_individual(input_df_I, target_I, radio_I, predefined_I, custom_I, dropped_features_I, freeze_features_I, model_I):
    pass

with gr.Blocks() as demo:
    with gr.Tab('Dataset'):
        gr.Markdown('Visualize the dataset to apply CFML')
        df_vis = gr.Dropdown(['IndividualClothingBinary','IndividualClothingBinary+3Binary',
                            'IndividualClothingValue','IndividualClothingValue+3Binary','Multi_TotalCLO_w_Chair',
                            'Summer','TotalClothingValue','TotalClothingValue+3Binary'], label='Dataset')
        output_vis = gr.DataFrame()
        button_vis = gr.Button(label="Run")

    with gr.Tab('Model'):
        gr.Markdown('Choose the features to apply CFML')
        input_df = gr.Dropdown(['IndividualClothingBinary','IndividualClothingBinary+3Binary',
                            'IndividualClothingValue','IndividualClothingValue+3Binary','Multi_TotalCLO_w_Chair',
                            'Summer','TotalClothingValue','TotalClothingValue+3Binary'], label='Dataset')
        target = gr.CheckboxGroup(['TSV','TPV','TCV','TSL'], label='Target Metric', info='Please select only one')
        test_size = gr.Slider(minimum=0.1, maximum=0.5, step=0.05, value=0.2, label='Test Size', interactive=True)
        model_name = gr.Textbox(label='Model Name', placeholder='Enter the model name')
        features_to_drop = gr.CheckboxGroup(['SwC', 'MC', 'Grade', 'Age', 'Gender'], 
                                            label='Features to Drop', info='Select the features to drop')
        model_output = gr.Textbox(label='Status')
        button_model = gr.Button(label="Train Model")

    #list add .pkl files from models folder
    models = []
    for file in os.listdir('../models/'):
        if file.endswith('.pkl'):
            models.append(file.split('.')[0])

    with gr.Tab('Counterfactuals-Total'):
        gr.Markdown('Generate Counterfactuals for Total CLO Dataset')
        input_df_T = gr.Dropdown(['Multi_TotalCLO_w_Chair','Summer',
                                  'TotalClothingValue','TotalClothingValue+3Binary'], label='Dataset')
        target_T = gr.CheckboxGroup(['TSV','TPV','TCV','TSL'], label='Target Metric', info='Please select only one')
        #target_T_range = gr.Textbox(label='Target Range', placeholder='Enter the target range [start,end]')
        radio_T = gr.Radio(['Predefined', 'Custom'], label='Type of Input')
        predefined_T = gr.Number(default=0, label='Number of inputs to provide')
        custom_T = gr.Dataframe(
            headers=['DAY','School','SchoolType','StartTime','AvgMaxDailyTemp','AvgMinDailyTemp','AvgIndoorRelativeHumidity',
                     'IndoorTempDuringSurvey','Grade','Age','Gender','FormalClothing','TotalCLOwithChair'],
                     row_count=(2, 'dynamic')
        )

        dropped_features_T = gr.CheckboxGroup(['SwC', 'MC', 'Grade', 'Age', 'Gender'],
                                            label='Features to Drop', info='Select the features that are dropped from feature set')
        
        freeze_features_T = gr.CheckboxGroup(['AvgIndoorRelativeHumidity',
                     'IndoorTempDuringSurvey','Grade','Age','Gender','FormalClothing','TotalCLOwithChair'],
                     info = 'Select the features to be freezed to generate CFs')
        model_T = gr.Dropdown(models, label='Model', info='Select the model to generate CFs')
        output_T = gr.DataFrame()
        button_cf_T = gr.Button(label="Generate CFs")

    with gr.Tab('Counterfactuals-Individual'):
        gr.Markdown('Generate Counterfactuals for Individual Clothing Dataset')
        input_df_I = gr.Dropdown(['IndividualClothingBinary','IndividualClothingBinary+3Binary',
                            'IndividualClothingValue','IndividualClothingValue+3Binary'], label='Dataset')
        target_I = gr.CheckboxGroup(['TSV','TPV','TCV','TSL'], label='Target Metric', info='Please select only one')
        radio_I = gr.Radio(['Predefined', 'Custom'], label='Type of Input')
        predefined_I = gr.Number(default=0, label='Number of inputs to provide')
        custom_I = gr.Dataframe(
            headers=['DAY','School','SchoolType','StartTime','AvgMaxDailyTemp','AvgMinDailyTemp','AvgIndoorRelativeHumidity','IndoorTempDuringSurvey',
                     'Grade','Age','Gender','FormalClothing','Pant','Trackpant','Halfshirt','Blazer','Jacket','Skirt',
                     'FullShirt','HalfSweater','Tshirt','Socks','Thermal','Vest','FullSweater','SwC','MC'],
            row_count=(2, 'dynamic')
        )
        dropped_features_I = gr.CheckboxGroup(['SwC', 'MC', 'Grade', 'Age', 'Gender'], 
                                            label='Features to Drop', info='Select the features that are dropped from feature set')
        
        freeze_features_I = gr.CheckboxGroup(['AvgIndoorRelativeHumidity','IndoorTempDuringSurvey',
                                            'Grade','Age','Gender', 'FormalClothing','Pant','Trackpant','Halfshirt','Blazer','Jacket','Skirt',
                                            'FullShirt','HalfSweater','Tshirt','Socks','Thermal','Vest','FullSweater','SwC','MC'],
                                            info='Select the features to be freezed to generate CFs')
        

        model_I = gr.Dropdown(models, label='Model', info='Select the model to generate CFs')
        button_cf_I = gr.Button(label="Generate CFs")

    button_vis.click(read_csv, df_vis, outputs=output_vis)
    button_model.click(train_model, [input_df, target, test_size, model_name, features_to_drop], outputs=model_output)
    button_cf_T.click(generate_cfs_total, [input_df_T, target_T, radio_T, predefined_T, 
                                           custom_T, dropped_features_T, freeze_features_T, model_T],
                                           outputs=output_T)

demo.launch(share=True)