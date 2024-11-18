
import dask.dataframe as dd

input_file = 'large_input.csv'
output_file = 'deduplicated_output_dask.csv'

# Read the CSV with all columns as object
df = dd.read_csv(input_file, assume_missing=True, dtype='object')

# Drop duplicates based on 'place_id'
df_unique = df.drop_duplicates(subset=['place_id'])

# Write to a single CSV file
df_unique.to_csv(output_file, single_file=True, index=False)

print('Deduplication complete. Output saved to', output_file)
