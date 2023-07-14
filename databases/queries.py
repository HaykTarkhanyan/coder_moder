"""
Table of content:
1. execute_given_query (added 26.03.22)
2. execute_basic_query (added 26.03.22)
3. get_all_tables (added 14.07.23)
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

def get_all_tables(cursor, schema, save_to_csv=False, output=False):
    """
    Returns a list of all table names in a given schema.
    
    Args:
        cursor: a cursor object to execute queries
        schema (str): the name of the schema to get tables from
        save_to_csv (bool): whether to save the list of tables to a csv file (default False)
        output (bool): whether to print progress messages (default True)
        
    Returns:
        A list of all table names in the given schema.
    """
        
    if output:  print(f"getting all tables from {schema} schema")
    
    cursor.execute(f"SELECT table_name FROM information_schema.tables WHERE table_schema = '{schema}'")

    # Fetch all table names
    tables = cursor.fetchall()

    # Print all table names
    all_tables = sorted([table[0] for table in tables])
    
    if output:  print(f"there are {len(all_tables)} tables in {schema}")
    
    if save_to_csv:
        if output:  print(f"saving all tables to 'all_tables_{schema}.csv'")
        pd.DataFrame(all_tables).to_csv('all_tables_{schema}.csv', index=False)
    
    return all_tables

