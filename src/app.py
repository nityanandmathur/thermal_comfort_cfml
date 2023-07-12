import gradio as gr
import pandas as pd


def read_csv(dataset):
    path = '../data/' + dataset + '.csv'
    data = pd.read_csv(path)
    return data.head(10)

def train_model(input_df, target, test_size, model_name, features_to_drop):
    import pickle

    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    path = '../data/' + input_df + '.csv'

    data = pd.read_csv(path, index_col=0)
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

    button_vis.click(read_csv, df_vis, outputs=output_vis)
    button_model.click(train_model, [input_df, target, test_size, model_name, features_to_drop], outputs=model_output)


demo.launch(share=True)