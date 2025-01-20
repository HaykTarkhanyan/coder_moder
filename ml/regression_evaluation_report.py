def evaluate_regression(actual, predictions,
    model_name=None, filename=None, notes=None,
    return_metrics=False, show_plots=False,
    show_metrics=True, plots=False, round_digits=3):
    """
    Function to evaluate a regression model.

    .. warning::
        Assumes that ``scipy``, ``sklearn``, and ``matplotlib`` are installed
        in your environment.

    This function:
        - Prints R2, MAE, MSE, RMSE metrics.
        - Prints Kendall's tau, Pearson's R, Spearman's rho correlation metrics.
        - Plots actual vs. predicted values.
        - Plots residuals vs. predicted values.
        - Plots distribution of residuals.
        - Plots predicted vs. actual distribution.
        - Saves results to file (if specified).
        - Returns metrics as a dictionary (if specified).
    Args:
        actual (array-like): Ground-truth target values.
        predictions (array-like): Model predictions.
        model_name (str, optional): Name of the model (for display/record-keeping).
        filename (str, optional): Path to an HTML file to save the results.
        notes (str, optional): Additional notes to include in the saved file (if `filename` is provided).
        return_metrics (bool, optional): If True, returns a dictionary of metrics. Defaults to False.
        show_plots (bool, optional): If True, calls `plt.show()` for each figure. Defaults to False.
        show_metrics (bool, optional): If True, prints the metrics and correlations to stdout. Defaults to True.
        plots (bool, optional): If True, generates plots. Defaults to False.
        round_digits (int, optional): Number of digits to round the metrics. Defaults to 3.

    Returns:
        dict or None: 
            A dictionary of computed metrics if `return_metrics=True`, otherwise None.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
    from scipy.stats import kendalltau, pearsonr, spearmanr
    from datetime import datetime
    from io import BytesIO
    import base64
    # Ensure inputs are NumPy arrays
    actual = np.array(actual)
    predictions = np.array(predictions)

    def save_figure_to_file(fig):
        """
        Helper function:
        Convert a Matplotlib figure to a base64-encoded PNG for embedding in HTML.
        """
        tmpfile = BytesIO()
        fig.savefig(tmpfile, format='png')
        encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
        return encoded

    # 1. Calculate regression metrics
    r2 = round(r2_score(actual, predictions), round_digits)
    mae = round(mean_absolute_error(actual, predictions), round_digits)
    mape = round(mean_absolute_percentage_error(actual, predictions), round_digits)
    mse = round(mean_squared_error(actual, predictions), round_digits)
    rmse = round(np.sqrt(mean_squared_error(actual, predictions)), round_digits)

    # 2. Calculate correlation metrics
    pearson = round(pearsonr(actual, predictions)[0], round_digits)
    spearman = round(spearmanr(actual, predictions)[0], round_digits)
    kendall = round(kendalltau(actual, predictions)[0], round_digits)

    # 3. Print metrics if needed
    if show_metrics:
        print(f"Model: {model_name or 'N/A'}")
        print(f"R2: {r2}")
        print(f"MAE: {mae}")
        print(f"MAPE: {mape}")
        print(f"MSE: {mse}")
        print(f"RMSE: {rmse}")
        print(f"Pearson Correlation: {pearson}")
        print(f"Spearman Rho: {spearman}")
        print(f"Kendall Tau: {kendall}")

    # 4. Generate plots if requested
    if plots:
        residuals = actual - predictions

        # (a) Predicted vs. Actual
        fig1 = plt.figure()
        plt.scatter(actual, predictions, edgecolor='k', alpha=0.7)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Predicted vs. Actual")
        # add a diagonal line
        plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'k--', lw=2)
        if show_plots:
            plt.show()
        prediction_vs_actual = save_figure_to_file(fig1)
        plt.close(fig1)

        # (b) Residuals vs. Predicted
        fig2 = plt.figure()
        plt.scatter(predictions, residuals, edgecolor='k', alpha=0.7)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel("Predicted")
        plt.ylabel("Residual")
        plt.title("Residuals vs. Predicted")
        if show_plots:
            plt.show()
        residuals_vs_predicted = save_figure_to_file(fig2)
        plt.close(fig2)

        # (c) Distribution of Residuals
        fig3 = plt.figure()
        plt.hist(residuals, bins=30, edgecolor='k', alpha=0.7)
        plt.xlabel("Residual")
        plt.ylabel("Count")
        plt.title("Distribution of Residuals")
        if show_plots:
            plt.show()
        residuals_distribution = save_figure_to_file(fig3)
        plt.close(fig3)

        # (d) Distribution of Predicted vs. Actual
        fig4 = plt.figure()
        plt.hist(actual, bins=30, alpha=0.5, label="Actual", edgecolor='k')
        plt.hist(predictions, bins=30, alpha=0.5, label="Predicted", edgecolor='k')
        plt.xlabel("Value")
        plt.ylabel("Count")
        plt.title("Distribution of Predicted vs. Actual")
        plt.legend()
        if show_plots:
            plt.show()
        predicted_vs_actual_distribution = save_figure_to_file(fig4)
        plt.close(fig4)

    # 5. Save results to file (HTML) if requested
    if filename:
        with open(filename, "w") as f:
            f.write(f"<html><body>\n")
            f.write(f"<h2>Report generated: {datetime.now()}</h2>\n")
            if model_name:
                f.write(f"<h2>Model Name: {model_name}</h2>\n")

            if notes:
                f.write(f"<h3>Notes:</h3>\n<p>{notes}</p>\n")

            f.write("<h3>Metrics</h3>\n")
            f.write(f"<b>R2:</b> {r2} <br>\n")
            f.write(f"<b>MAE:</b> {mae} <br>\n")
            f.write(f"<b>MAPE:</b> {mape} <br>\n")
            f.write(f"<b>MSE:</b> {mse} <br>\n")
            f.write(f"<b>RMSE:</b> {rmse} <br>\n")

            f.write("<h3>Correlations</h3>\n")
            f.write(f"Pearson: {pearson} <br>\n")
            f.write(f"Spearman: {spearman} <br>\n")
            f.write(f"Kendall Tau: {kendall} <br>\n")

            if plots:
                f.write("<h3>Plots</h3>\n")
                f.write(f'<img src="data:image/png;base64,{prediction_vs_actual}"><br><br>\n')
                f.write(f'<img src="data:image/png;base64,{residuals_vs_predicted}"><br><br>\n')
                f.write(f'<img src="data:image/png;base64,{residuals_distribution}"><br><br>\n')
                f.write(f'<img src="data:image/png;base64,{predicted_vs_actual_distribution}"><br><br>\n')

            f.write("</body></html>\n")

    # 6. Optionally return a dictionary of metrics
    if return_metrics:
        return {
            "model_name": model_name,
            "notes": notes,
            "r2": r2,
            "mae": mae,
            "mape": mape,
            "mse": mse,
            "rmse": rmse,
            "pearson": pearson,
            "spearman": spearman,
            "kendall": kendall
        }
