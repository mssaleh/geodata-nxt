import pandas as pd

# Define the input and output file paths
input_file = 'large_input.csv'
output_file = 'deduplicated_output_pd.csv'

# Initialize an empty set to keep track of seen place_ids
seen_place_ids = set()

# Define the chunk size (number of rows per chunk)
chunk_size = 10 ** 6  # Adjust based on your memory capacity

# Open the output file in write mode
with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
    # Iterate over the CSV file in chunks
    for i, chunk in enumerate(pd.read_csv(input_file, chunksize=chunk_size)):
        print(f'Processing chunk {i+1}')
        
        # Drop duplicates within the chunk based on 'place_id'
        chunk_unique = chunk.drop_duplicates(subset=['place_id'])
        
        # Filter out rows with place_ids already seen
        mask = ~chunk_unique['place_id'].isin(seen_place_ids)
        unique_new = chunk_unique[mask]
        
        # Update the set of seen place_ids
        seen_place_ids.update(unique_new['place_id'].tolist())
        
        # Write the unique rows to the output file
        if i == 0:
            # Write header for the first chunk
            unique_new.to_csv(outfile, index=False)
        else:
            # Append without writing the header
            unique_new.to_csv(outfile, index=False, header=False)
        
print('Deduplication complete. Output saved to', output_file)
