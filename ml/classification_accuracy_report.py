import secrets
import mysql.connector
from datetime import datetime
import pandas as pd
import json
import torch
from sentence_transformers import util


def get_classification_scores(model, X_train, X_test, y_train, y_test, threshold=0.5,
                              data_info=True, detailed=True, class_prob_distplot=True, 
                              positive_class="Churned", round_to=4, save_json=False, save_folder=None,
                              model_name=None):
    y_train_prob = model.predict_proba(X_train)[:,1]
    y_test_prob = model.predict_proba(X_test)[:,1]
    y_train_class = y_train_prob > threshold
    y_test_class = y_test_prob > threshold
    
    
    metrics = {
        "datetime": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model': model, 
        "model_name": model.__class__.__name__,
        "Accuracy train:":accuracy_score(y_train, y_train_class),
        "Accuracy test:":accuracy_score(y_test, y_test_class),
        "Average Precision train:": average_precision_score(y_train, y_train_class),
        "Average Precision test:": average_precision_score(y_test, y_test_class),
        "ROC AUC train:":roc_auc_score(y_train, y_train_class),
        "ROC AUC test:":roc_auc_score(y_test, y_test_class),
    }
    
    metrics["Overfitting"] = metrics["ROC AUC train:"] - metrics["ROC AUC test:"]
    
    if detailed:
        metrics.update({
        "F1 train:":f1_score(y_train, y_train_class),
        "F1 test:":f1_score(y_test, y_test_class),
        "Precision train:":precision_score(y_train, y_train_class),
        "Precision test:":precision_score(y_test, y_test_class),
        "Recall train:":recall_score(y_train, y_train_class),
        "Recall test:":recall_score(y_test, y_test_class)
        })
    
        
    if data_info:
        metrics.update({
            "Train rows": len(y_train),
            "Columns": X_train.shape[1],
            "Test rows": len(y_test),
            "Train positive class %": y_train.mean(),
            "Test positive class %": y_test.mean()
        })
    
    if class_prob_distplot:
        fig = plt.figure(figsize=(6,3))
        plt.title(f'Model: {model.__class__.__name__} - Probability Distribution')
        sns.distplot(y_test_prob[y_test==0],label=f'not {positive_class}', color='green')
        sns.distplot(y_test_prob[y_test==1],label=positive_class)
        plt.legend()    
        plt.show()
        metrics["plot"] = fig
        
    if round_to:
        metrics = {k: round(v, round_to) if isinstance(v, (int, float)) else v for k, v in metrics.items()}

    if save_json:
        keys_to_ignore = ["model", "plot"]
        
        path = f'{metrics["model_name"]}_metrics.json'
        if save_folder:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            path = f'{save_folder}/{path}' if save_folder else path

        with open(path, 'w') as f:
            json.dump({k: v for k, v in metrics.items() if k not in keys_to_ignore}, f)

    if model_name:
        metrics["model_name"] = model_name

    return metrics

def compare_classification(res_1, res_2, round_to=3):
    comparison = {}
    for k, v in res_1.items():
        if isinstance(v, (int, float)):
            comparison[f"{k}" ] = v - res_2[k]
        
    
    if round_to:
        comparison = {k: round(v, round_to) if isinstance(v, (int, float)) else v for k, v in comparison.items()}    
    
    if res_1["model_name"] == res_2["model_name"]:
        res_1["model_name"] = res_1["model_name"] + " (1st)"
        res_2["model_name"] = res_2["model_name"] + " (2nd)"
    
    res_lst = []
    for c in comparison:
        res_lst.append(
            {
                "metric": c, 
                res_1["model_name"]: res_1.get(c, None),
                res_2["model_name"]: res_2.get(c, None),
                "diff 1st and 2nd": comparison[c]
            }
        )
    
    df = pd.DataFrame(res_lst)
    
    return df

def compute_accuracy(df, category, prediction):
    """
    computes the accuracy(adjusted, en category_2n el enq vercnum)
    
    Args:
        category - category column name
        category_2 - category_2 column name
        prediction - prediction column name
    
    Returns:
        float
    """
    import numpy as np
    category = df[category].values
    prediction = df[prediction].values
    
    correct_categ = (category==prediction)
    
    num_correct = sum([correct_categ[i] for i in range(len(category))])
    
    return np.round(num_correct / len(category) * 100, 2)

def compute_accuracy_per_category(df, category, prediction):
    """Returns sorted dataframe with model's accuracy for each category
    
    Args:
        df
        category (str)- name of the column with correct category
        prediction (str) 
        
    Return:
        df
    """
    
    df['correct'] = df['prediction'] == df['CATEGORY_MAIN']
    res = pd.DataFrame(df.groupby(category)['correct'].sum() \
                     / df.groupby(category)['correct'].count())
    
    res_2 = pd.DataFrame(df.groupby(category)['correct'].sum())

    count_categ = pd.DataFrame(df[category].value_counts())

    res = pd.merge(res, count_categ, left_index=True, right_index=True)
    res = pd.merge(res, res_2, left_index=True, right_index=True)
    
    res = res.sort_values('correct_x', ascending=False)

    return res

def compute_accuracy_per_prediction_method(df, category, prediction):
    """Returns sorted dataframe with model's accuracy for each category
    
    Note:
        Assumes column with prediction method is called 'how_predicted'
    
    Args:
        df
        category (str)- name of the column with correct category
        prediction (str) 
        
    Return:
        df
    """
    df['correct'] = df['prediction'] == df['CATEGORY_MAIN']
    res = pd.DataFrame(df.groupby('how_predicted')['correct'].sum() \
                     / df.groupby('how_predicted')['correct'].count())
    res_2 = pd.DataFrame(df.groupby('how_predicted')['correct'].sum())
    
    
    count_how_pred = pd.DataFrame(df['how_predicted'].value_counts())#['how_predicted'], res
    res = pd.merge(res, count_how_pred, left_index=True, right_index=True)
    res = pd.merge(res, res_2, left_index=True, right_index=True)
    res = res.sort_values('correct_x', ascending=False)

    return res

def find_out_what_category_confuses_with(df, category, prediction):
    from scipy import stats
    df['correct'] = df[category]== df[prediction]

    df_mistakes = df[~df.correct]
    
    return pd.DataFrame(df_mistakes.groupby(category)[prediction].agg(lambda x: pd.Series.mode(x)[0]))



def calculate_classification_report(df, category, prediction):
    from sklearn.metrics import classification_report
    return pd.DataFrame(classification_report(df[prediction].values,  \
                        df[category].values, output_dict=True)).transpose().sort_values(by='f1-score', ascending=False)

def count_frequent_confusions(df, category, prediction):
    res = find_out_what_category_confuses_with(df, category, prediction)
    most_commons = res[prediction].value_counts().to_frame()
    return most_commons

def get_common_confusion_by_prediction_method(df, category, prediction, pred_method):
    df['correct'] = df[prediction] == df[category]
    df_corrects = df[~df.correct]
    res = df_corrects[df_corrects.how_predicted == pred_method].prediction.value_counts().to_frame()
    return res





def evaluate_model(df, category, prediction, save_path=f'report_{datetime.now()}.html', print_res=False):
    """Geneartes html report with models accuracy report
    
    Args:
        df
        category (str) - name of the column with category
        prediction (str)
        save_path (str) - defaults to 'report_{todays date}'.html
        print_res (bool) - whether to output (default False)
    """
    from datetime import datetime
    pd.options.display.float_format = '{:,.3f}'.format

    acc = compute_accuracy(df, category, prediction)
    df_per_categ = compute_accuracy_per_category(df, category, prediction)
    df_per_method = compute_accuracy_per_prediction_method(df, category, prediction)
    df_classification_report = calculate_classification_report(df, category, prediction)
    df_confusions = find_out_what_category_confuses_with(df, category, prediction)
    df_frequent_confusions = count_frequent_confusions(df, category, prediction)
    
    if print_res:
        print (f'Models Accuracy is: {acc}%')
        print()
        print (f'Accuracies per category are {df_per_categ}')
        print()
        print (f'Accuracies per prediction method are {df_per_method}')
        print()
    
    pred_methods = ['META', "META and one main model", "FINAL"]
    
    text_file = open(save_path, "w")
    text_file.write(f'<h3>Report generated - {datetime.now()}</h3>')
    text_file.write(f'<h1> Models accuracy is <i>{acc}</i>%</h1>')
    text_file.write('<h2>Accuracies per category are </h2>')
    text_file.write(df_per_categ.to_html())
    text_file.write('<h2>Accuracies per prediction method are </h2>')
    text_file.write(df_per_method.to_html())
    text_file.write('<h2>Classification Report</h2>')
    text_file.write(df_classification_report.to_html())
    text_file.write('<h2>Confusions</h2>')
    text_file.write(df_confusions.to_html())
    text_file.write('<h2>Frequent Confusions</h2>')
    text_file.write(df_frequent_confusions.to_html())
    
    for predictor in pred_methods:
        text_file.write(f'<h2>Confusions for {predictor}</h2>')
        text_file.write(get_common_confusion_by_prediction_method(df, category, prediction, predictor).to_html())
    
    text_file.close()
    

    if __name__=='__main__':
        # clean_db()
        pass



