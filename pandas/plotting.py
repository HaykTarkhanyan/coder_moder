import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_data_distribution_by_month(data, date_column):
    # Ensure the date column is in datetime format
    data[date_column] = pd.to_datetime(data[date_column])
    
    # Extract month and year from the date column
    data['Month'] = data[date_column].dt.to_period('M')
    
    # Count the occurrences per month
    monthly_counts = data['Month'].value_counts().sort_index().reset_index()
    monthly_counts.columns = ['Month', 'Count']
    
    # Calculate percentage distribution
    monthly_counts['Percentage'] = 100 * monthly_counts['Count'] / monthly_counts['Count'].sum()
    
    # Calculate CDF
    monthly_counts['CDF'] = monthly_counts['Percentage'].cumsum()
    
    # Create a subplot with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add bar trace for absolute counts
    fig.add_trace(
        go.Bar(x=monthly_counts['Month'].astype(str), y=monthly_counts['Count'], name='Absolute Counts'),
        secondary_y=False,
    )

    # Add line trace for percentage distribution
    fig.add_trace(
        go.Scatter(x=monthly_counts['Month'].astype(str), y=monthly_counts['Percentage'], name='Percentage', mode='lines+markers'),
        secondary_y=True,
    )
    
    # Add line trace for CDF
    fig.add_trace(
        go.Scatter(x=monthly_counts['Month'].astype(str), y=monthly_counts['CDF'], name='CDF', mode='lines+markers', line=dict(dash='dot')),
        secondary_y=True,
    )

    # Set titles and labels
    fig.update_layout(
        title_text="Data Distribution by Month (Absolute, Percentage, and CDF)",
        xaxis_title="Month",
    )

    fig.update_yaxes(title_text="Absolute Counts", secondary_y=False)
    fig.update_yaxes(title_text="Percentage / CDF", secondary_y=True)

    # Show the plot
    fig.show()

# Example usage with a sample DataFrame
data = pd.DataFrame({
    'date': pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
})

plot_data_distribution_by_month(data, 'date')
