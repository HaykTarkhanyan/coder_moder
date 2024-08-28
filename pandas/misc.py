def extract_data_from_sql(file_content):
    """
    Extracts the table name and data from a SQL insert statement.
    
    Example:
    insert into dbname.table_name (id, name)
    values  (1, 'panir'),
            (5, 'hndkahav');	
        
    Note:
    - this has the problem of not including the last column of the last row
    """
    # Extract table name and columns
    table_name_match = re.search(r"insert into ([\w\.]+) \((.*?)\)\nvalues", file_content, re.IGNORECASE)
    if not table_name_match:
        raise ValueError("Table name or columns not found in the provided SQL insert statement.")
    
    table_name = table_name_match.group(1).split(".")[1]
    columns = table_name_match.group(2).split(", ")
    # print(columns)
    # Extract the values block and replace the delimiters for parsing
    values_block = file_content.split('values')[1].strip()
    values_block = values_block.replace('),\n(', '|\n|')
    values_block = values_block.replace('(', '').replace(')', '').replace('|\n|', '\n')
    
    # Use the csv module to parse the values
    data = []
    csv_reader = csv.reader(StringIO(values_block), quotechar="'", skipinitialspace=True)
    for row in csv_reader:
        data.append(row[:-1])

    df = pd.DataFrame(data, columns=columns)
    
    return table_name, df

# not tested
def extract_data_from_sql_for_folder(folder_name, output_folder_name, save_to_parquet=False):
    if not os.path.exists(folder_name):
        raise ValueError(f"Folder {folder_name} not found")
    
    output_folder_name_csv = output_folder_name + "_csv"
    if save_to_parquet:
        output_folder_name_parquet = output_folder_name + "_parquet"
        if not os.path.exists(output_folder_name_parquet):
            os.makedirs(output_folder_name_parquet)
    
    if not os.path.exists(output_folder_name_csv):
        os.makedirs(output_folder_name_csv)
        
    for i in tqdm(sql_files):
        try:
            file_path = os.path.join(output_folder_name_csv, i)
            with open(file_path, 'r', encoding="utf8") as f:
                data = f.read()

            table_name, df = extract_data_from_sql(data)
            
            df.to_csv(f"output_folder_name_csv/{table_name}.csv", index=False)
            if save_to_parquet:
                df.to_parquet(f"{output_folder_name_parquet}/{table_name}.parquet")            
            
        except Exception as e:
            print(f"Error: {e}", i, table_name)
            continue
    
