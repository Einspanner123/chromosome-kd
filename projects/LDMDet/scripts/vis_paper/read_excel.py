import pandas as pd
import sys

def read_excel_data(file_path):
    try:
        # 读取 Excel 文件
        df = pd.read_excel(file_path)
        
        # 打印基本信息
        print("--- Column Names ---")
        print(df.columns.tolist())
        
        print("\n--- Data Preview (First 10 rows) ---")
        print(df.head(10).to_string())
        
        print("\n--- Summary Statistics ---")
        print(df.describe().to_string())
        
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    file_path = "/home/linkst/workplace/chromo/chromosome-kd/compare_class.XLSX"
    read_excel_data(file_path)
