# Read terms from terms.txt
with open("terms.txt", "r") as terms_file:
    terms = [line.strip() for line in terms_file.readlines()]

# Read places from places.txt
with open("places.txt", "r") as places_file:
    places = [line.strip() for line in places_file.readlines()]

# Create the combined list
output = []
for term in terms:
    for place in places:
        output.append(f"{term} in {place}")

# Save the output to a new file
with open("output.txt", "w") as output_file:
    for line in output:
        output_file.write(line + "\n")

print("The output has been saved to output.txt")
