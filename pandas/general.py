"""
Table of content:
1. get_datetime_features
2. value_count_the_value_count
3. get_sessions_occuring_multiple_times
4. calculate_event_duration

date added here 26.03.22
"""

def get_datetime_features(df):
    df['ds'] = pd.to_datetime(df.ds)
    df['month'] = df.ds.dt.month
    import calendar
    df['month_english'] = df['month'].apply(lambda x: calendar.month_abbr[x])
    df['day_of_month'] = df.ds.dt.day
    df['year'] = df.ds.dt.year

    # https://pypi.org/project/holidays/
    # https://stackoverflow.com/questions/29688899/pandas-checking-if-a-date-is-a-holiday-and-assigning-boolean-value
    cal = calend()
    holidays = cal.holidays(start=df.ds.min(), end=df.ds.max())
    print(holidays)
    df['holiday'] = df['ds'].isin(holidays)
    
    df['day_of_week'] = df.ds.apply(lambda x: x.dayofweek)
    df['day_of_week_english'] = df['day_of_week'].map({0.0: 'Monday', 1.0: 'Tuesday', 2.0: 'Wednesday', \
                                                       3.0: 'Thursday', 4.0: 'Friday', 5.0:'Saturday', 6.0: "Sunday"})

    return df

def value_count_the_value_count(df, col_name="sessionId", percentages_just_occurance=False, percentages_overall=False):
    """Computes number of times values occur x times 
    
    Args:
        df
        col_name (str): default is `sessionId`
        percentages_just_occurance (bool): will understand from example, hopefully
        percentages_overall (bool): will understand from example, hopefully
    
    
    Example:
        ['a', 'a', 'a', 'a', 'b', 'b', 'c', 'c'] -> '2': 2, '4':1}
        if percentages_overall is True '2'-> 25, '4'->12.5
        if percentages_just_occurance is True '2'->66.6. '4'->33.3
        
    """
    res = df[col_name].value_counts().to_frame().value_counts()
    
    if percentages_just_occurance and percentages_overall:
        raise ValueError('both percantage_just_occurance and percentage_overall are True, specify one of them')
    
    if percentages_overall:
        res = res / len(df) * 100
    if percentages_just_occurance:
        res = res / res.sum() * 100
    
    return res

def get_sessions_occuring_multiple_times(df, col_name='sessionId', output=False):
    """
    Given a dataframe returns all the rows with sessions that
    have more than 1 record
    
    Args:
        df (pd.DataFrame): generally must have column `sessionId`
        col_name (string): for whih column to filter the data 
        output (bool): whether or not to print logs in the process
        
    Returns:
        df 
    """
    
    counts = df[col_name].value_counts()
    df_only_multiples = df[df[col_name].isin(counts.index[counts > 1])]
    
    if output:
        print(f'inital df had {len(df)} rows, now has {len(df_only_multiples)}')
        print(f'only {len(df_only_multiples) / len(df) * 100:.2f}%')
    
    return df_only_multiples

def calculate_event_duration(df, date_column='date', event_column='sessionId', event_values='all'):
    """
    Function calculates how long the event was open
    
    Args:
        df (pd.DataFrame): 
        date_column (str) : default is `date`
        event_values (list) : list of event values to filter df by, defaults in `all`
        event_column (str): anme of the column to calc duration of, default is `sessionId`
    
    Returns:
        pd.DataFrame
    
    """
    if event_values != 'all':
        df = df[df[event_column].isin(event_values)]
    
    group_by_event = df.groupby(event_column)
    durations = group_by_event[date_column].max() - group_by_event[date_column].min()
    
    return durations.to_frame().reset_index()