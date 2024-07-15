
def split_data_by_date(df, col_to_split_by, train_size, validation_size, logs=True):
    df[col_to_split_by] = pd.to_datetime(df[col_to_split_by])

    train = users[users[col_to_split_by] < users[col_to_split_by].quantile(train_size)]
    validation = users[(users[col_to_split_by] >= users[col_to_split_by].quantile(train_size)) & 
                       (users[col_to_split_by] < users[col_to_split_by].quantile(train_size + validation_size))]
    test = users[users[col_to_split_by] >= users[col_to_split_by].quantile(train_size + validation_size)]

    train_start = train[col_to_split_by].min().date()
    train_end = train[col_to_split_by].max().date()
    validation_start = validation[col_to_split_by].min().date()
    validation_end = validation[col_to_split_by].max().date()
    test_start = test[col_to_split_by].min().date()
    test_end = test[col_to_split_by].max().date()

    train_duration = (train_end - train_start).days
    validation_duration = (validation_end - validation_start).days
    test_duration = (test_end - test_start).days

    if logs:
        print(f"Train period: {train_start} - {train_end} ({train_duration} days) {len(train)} rows")
        print(f"Val   period: {validation_start} - {validation_end} ({validation_duration} days) {len(validation)} rows")
        print(f"Test  period: {test_start} - {test_end} ({test_duration} days) {len(test)} rows")
        
    return train, validation, test
