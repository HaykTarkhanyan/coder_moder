"""
Table of content:
1. execute_given_query
2. execute_basic_query

date added here 26.03.22
"""

def execute_given_query(q, limit=20, output=True):
    """
    Runs a given query
    
    Args:
        q (str): query to execute
        limit (int): default is 20
        output (bool): wheteher to have logging in process or not
    
    Note:
        1. expects to have `cursor` already declared
        2. written for mysql

    Raises:
        ValueError if query returned no results
        
    Returns:
        pd.DataFrame
       
    """
    from time import time
    st = time()
    
    if output:
        print('executing following query \n')
        print(q)
    
    cursor.execute(q)
    results = cursor.fetchall()
    columns = [i[0] for i in cursor.description]
    df = pd.DataFrame(results, columns = columns)
    
    if limit:
        q += f" limit {limit}"
    
    
    if output:
        print(f"\nfetched {len(df)} rows, df has {df.shape[1]} rows")

        print(f"\nquery ran for {time() - st:.2f} seconds({(time() - st) / 60:.2f} minutes)")
    
    
    if df.empty:
        raise ValueError(f'df for \n {q} \n is empty')
    
    return df

def execute_basic_query(table_name, columns='*', additional_filter=None, 
                        date_col_name='date', start_date=None, end_date=None, 
                        order_by_column=None, order_type="DESC", 
                        limit = 20, output=True):
    """
    Runs a basic 
    
    select {columns} from {table_name} where {date_col_name} between {start_date} and {end_date} {additional_filter} 
    order by {order_by_column} {order_type}  limit {limit}
     
    and fetches column names to return a clear pd.DataFrame

    Args:
        table_name (str):
        columns (list of str): by default is *
        additional_filter (str): custom where clause like `merchant_id=509`, (default None) 
        date_col_name (str): name of the timestamp column (default is 'date')
        start_date (str): option, default is None
        end_date (str): option, default is None
        order_by_column (str): column to sort by it (default None)
        order_type (str): default is DESC, either "DESC" or ""
        limit (int): default is None
        output (bool): wheteher to have logging in process or not
    
    Raises:
        ValueError if query returned no results
        
    Returns:
        pd.DataFrame
    
    """
    from time import time
    st = time()

    q = f"""select {', '.join(columns)} from {table_name}""" 
    
    if start_date and end_date:
        q += f' where {date_col_name} between "{start_date}" and "{end_date}"'

        if additional_filter:
            q += f" and {additional_filter}"
    else:
        if additional_filter:
            q += f" where {additional_filter}"
    
    if order_by_column:
        q += f"ORDER BY {order_by_column} {order_type}"
    
    if limit:
        q += f" limit {limit}"
    
    
    if output:
        print('executing following query \n')
        print(q)
   
    
    cursor.execute(q)
    results = cursor.fetchall()
    columns = [i[0] for i in cursor.description]
    df = pd.DataFrame(results, columns = columns)
    
    if output:
        print(f"\nfetched {len(df)} rows")
        print(f"query ran for {time() - st:.2f} seconds({(time() - st) / 60:.2f} minutes)")
    if df.empty:
        raise ValueError(f'df for \n {q} \n is empty')
    
    return df
