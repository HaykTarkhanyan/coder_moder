def extract_data_from_sql(file_content):
    """
    Works for this struture
    
    insert into db_name.subjects (id, title)
    values  (1, 'asdasdaad'),
            (2, 'asdasdadas')

    Note:
        Fails to include the last value
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

    # print(data[1])
    # Create a DataFrame
    df = pd.DataFrame(data, columns=columns)
    
    

    return table_name, df