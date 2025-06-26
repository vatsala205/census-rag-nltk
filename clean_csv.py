import pandas as pd


def clean_csv_file(input_file, output_file=None):
    """Clean the CSV file by fixing column names and data issues"""

    if output_file is None:
        output_file = input_file.replace('.csv', '_cleaned.csv')

    # Read the CSV file
    df = pd.read_csv(input_file)

    # Clean column names - remove line breaks and extra spaces
    df.columns = df.columns.str.replace('\n', '').str.replace('\r', '').str.strip()

    # Fix the specific column name issue
    df.columns = df.columns.str.replace('Sub District Cod\ne', 'Sub District Code')

    # Display the cleaned column names
    print("Cleaned column names:")
    for i, col in enumerate(df.columns):
        print(f"{i + 1}. '{col}'")

    # Basic data info
    print(f"\nDataset shape: {df.shape}")
    print(f"Sample of first few rows:")
    print(df.head(3))

    # Check for any data quality issues
    print(f"\nData quality check:")
    print(f"- Missing values per column:")
    missing_counts = df.isnull().sum()
    for col, count in missing_counts.items():
        if count > 0:
            print(f"  {col}: {count} missing values")

    # Save the cleaned file
    df.to_csv(output_file, index=False)
    print(f"\nCleaned data saved to: {output_file}")

    return df


if __name__ == "__main__":
    # Clean your CSV file
    cleaned_df = clean_csv_file('feature_engineered.csv')

    # Show some sample location names
    print("\nSample location names:")
    sample_names = cleaned_df['Name'].dropna().head(10)
    for name in sample_names:
        print(f"- {name}")