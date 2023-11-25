input_file_path = 'stuff.txt'
csv_data = []

with open(input_file_path, 'r') as input_file:
    lines = [line.rstrip(')\n') for line in input_file.readlines()]

for line in lines:
    line = line.replace(".jpg", ".png")
    parts = line.strip().split(',', 1)
    
    # Merge consecutive repeating characters in the second part
    part1, part2 = parts
    new_part2 = []
    prev_char = ''
    for char in part2:
        if char != prev_char:
            new_part2.append(char)
        prev_char = char
    new_part2 = ''.join(new_part2)
    
    # Create a CSV row with the modified second part
    csv_row = part1 + ',' + new_part2
    csv_data.append(csv_row)

output_csv_path = 'output_file1.csv'
with open(output_csv_path, 'w') as output_csv_file:
    for row in csv_data:
        output_csv_file.write("%s\n" % row)

print("Data converted to CSV format and saved in output_file.csv.")

