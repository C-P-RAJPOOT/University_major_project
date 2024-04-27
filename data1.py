import mysql.connector
import pandas as pd

def data():
    # Replace these values with your actual MySQL database connection details
    db_config = {
        'host': '127.0.0.1',
        'user': 'root',
        'password': '',
        'database': 'cpdatabase',
    }

    # Connect to the MySQL database
    conn = mysql.connector.connect(**db_config)

    # Replace 'your_table' with the actual table name in your database
    query = "SELECT * FROM studentinfo"

    # Use pandas to read the data from the database
    df = pd.read_sql_query(query, conn)

    # Close the database connection
    conn.close()

    # Save the data to a CSV file
    df.to_csv('output.csv', index=False)

    print("CSV file created successfully.")

# Call the function to execute the code
data()
