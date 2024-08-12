def evaluate_model(actual, predictions, model_name=None,  filename=None, notes=None, return_metrics=False, 
                   show_plots=False, show_metrics=True):
    """
    Function to evaluate regression model

    Warning:
        Assumes that scipy, sklearn, matplotlib are present in your environment

    This function:
        - prints R2, MAE, MSE, RMSE metrics
        - prints Kendall's tau, Pearson's R, Spearman's rho correlation metrics
        - plots actual vs predicted values
        - plots residuals vs predicted values
        - plot distribution of residuals
        - plot predicted vs actual distribution
        - saves results to file (if specified)
        - returns metrics as dictionary (if specified)

    Args:
        actual (np.array): actual values
        predictions (np.array): model predictions 
        model_name (optional): name of the model 
        filename (optional): if specified, results will be saved to file
        notes (optional): additional notes to be saved to file
        return_metrics (optional): if True, returns metrics as dictionary
        show_plots (optional): if True, shows plots
        show_metrics (optional): if True, shows metrics

    Returns:
        dict or None
    """
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error   
    from scipy.stats import kendalltau, pearsonr, spearmanr
    import matplotlib.pyplot as plt
    from datetime import datetime    
    from io import BytesIO
    import base64
#     import statsmodels.api as sm # for qq plot

    def save_figure_to_file(fig):
        # we need this for saving figures to html file
        tmpfile = BytesIO()
        fig.savefig(tmpfile, format='png')
        encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
        return encoded

    round_digits = 3
    # calculate metrics
    r2 = r2_score(actual, predictions)
    mae = mean_absolute_error(actual, predictions)
    mse = mean_squared_error(actual, predictions)
    rmse = np.sqrt(mean_squared_error(actual, predictions))

    # print metrics
    if show_metrics:
        print(f"Model: {model_name}")
        print(f"R2: {round(r2, round_digits)}")
        print(f"MAE: {round(mae, round_digits)}")
        print(f"MSE: {round(mse, round_digits)}")
        print(f"RMSE: {round(rmse, round_digits)}")

    # calculate correlations
    pearson = pearsonr(actual, predictions)[0]
    kendall = kendalltau(actual, predictions)[0]
    spearman = spearmanr(actual, predictions)[0]
    
    if show_metrics:
        print(f"Pearson Correlation: {round(pearson, round_digits)}")
        print(f"Spearman Rho: {round(spearman, round_digits)}") 
        print(f"Kendall Tau: {round(kendall, round_digits)}")
              
        
    # plot predictions vs actual
    fig1 = plt.figure(1)
    # plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'k--', lw=4)
    plt.scatter(actual, predictions)    
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted vs Actual")
    if show_plots: 
        plt.show()
    prediction_vs_actual = save_figure_to_file(fig1)
    plt.close(fig1)

    # plot residuals vs Predicted
    fig2 = plt.figure(2)
    plt.scatter(actual, actual - predictions)
    plt.xlabel("Predicted")
    plt.ylabel("Residual")
    plt.title("Residuals vs Predicted")
    if show_plots:
        plt.show()
    residuals_vs_predicted = save_figure_to_file(fig2)
    plt.close(fig2)

    # plot distribution of residuals
    fig3 = plt.figure(3)
    plt.hist(actual - predictions)
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.title("Distribution of Residuals")
    if show_plots:
        plt.show()
    residuals_distribution = save_figure_to_file(fig3)
    plt.close(fig3)

    # plot predicted vs actual distribution
    fig4 = plt.figure(4) 
    plt.hist(actual, alpha=0.5, label="Actual")
    plt.hist(predictions, alpha=0.5, label="Predicted")
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.title("Distribution of Predicted vs Actual")    
    plt.legend()
    if show_plots:
        plt.show()
    predicted_vs_actual_distribution = save_figure_to_file(fig4)
    plt.close(fig4)

#     fig5 = sm.qqplot(actual - predictions, line ='45', fit=True)
#     plt.show()
#     qqplot = save_figure_to_file(fig5)
#     plt.close(fig5)


    # save to file
    if filename:    
        with open(filename, "w") as f:
            f.write(f"Geport generated: {datetime.now()}")
            if model_name: f.write(f"<h1> Model Name: {model_name} </h2>")
            if notes: 
                f.write(f"<h2> Notes:</h2>")
                f.write(f"{notes}")

            f.write(f"<h2> Metrics </h2>")
            f.write(f"<b> R2: {round(r2, round_digits)} </b> <br>")
            f.write(f"MAE: {round(mae, round_digits)} <br>")
            f.write(f"MSE: {round(mse, round_digits)} <br>")
            f.write(f"RMSE: {round(rmse,round_digits)} <br>")

            f.write(f"<h2> Correlations </h2>")
            f.write(f"Pearson Correlation: {pearson.round(round_digits)} <br>")
            f.write(f"Kendall Tau: {kendall.round(round_digits)} <br>")
            f.write(f"Spearman Rho: {spearman.round(round_digits)} <br>")

            f.write(f"<h2> Plots </h2>")
            f.write(f'<img src="data:image/png;base64,{predicted_vs_actual_distribution}">')
            f.write(f'<img src="data:image/png;base64,{prediction_vs_actual}">')
            f.write(f'<img src="data:image/png;base64,{residuals_vs_predicted}">')
            f.write(f'<img src="data:image/png;base64,{residuals_distribution}">')
            # f.write(f'<img src="data:image/png;base64,{qqplot}">')
            
    if return_metrics:
        metric_dict = {
            "name": model_name, "notes": notes,
            "r2": r2,
            "mae": mae, "mse": mse, "rmse": rmse,
            "pearson": pearson, "kendall": kendall, "spearman": spearman
        }
        return metric_dict
