import csv
import ast
import pandas as pd

# input_file = 'data/anime.csv'  # Name of the input CSV file
# output_file = 'data/anime1.csv'  # Name of the output CSV file
# column_to_delete = "['Japanese name','English name','Ranked','Favorites','Watching','Completed','On-Hold','Dropped','Plan to Watch','Score-10','Score-9','Score-8','Score-7','Score-6','Score-5','Score-4','Score-3','Score-2','Score-1']"
# columns_to_delete = ast.literal_eval(column_to_delete)
#
# with open(input_file, 'r', encoding='utf-8', newline='') as infile, open(output_file, 'w', encoding='utf-8',
#                                                                          newline='') as outfile:
#     reader = csv.reader(infile)
#     writer = csv.writer(outfile)
#
#     # Read the header and determine the indices of the columns to delete
#     header = next(reader)
#     indices_to_delete = [header.index(col) for col in columns_to_delete if col in header]
#
#     if not indices_to_delete:
#         print(f"None of the specified columns were found in the header")
#     else:
#         # Write the header to the output file excluding the columns to delete
#         new_header = [col for i, col in enumerate(header) if i not in indices_to_delete]
#         writer.writerow(new_header)
#
#         # Write the remaining rows excluding the columns to delete
#         for row in reader:
#             new_row = [value for i, value in enumerate(row) if i not in indices_to_delete]
#             writer.writerow(new_row)


input_file = 'data/CleanFinal.csv'  # Name of the input CSV file
output_file = 'data/Final.csv'  # Name of the output CSV file

# Function to modify the first value of each row based on loop iteration number
def modify_first_value(row, iteration_number):
    row[0] = str(iteration_number)
    return row

# Read the input CSV file and write to the output CSV file with modified first values
with open(input_file, 'r', encoding='utf-8', errors='replace', newline='') as infile, \
     open(output_file, 'w', encoding='utf-8', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    for iteration_number, row in enumerate(reader, start=0):
        if iteration_number == 0:
            modified_row = modify_first_value(row, "Index")
        else:
            modified_row = modify_first_value(row, iteration_number)
        writer.writerow(modified_row)

print("CSV file has been processed and saved as", output_file)




# input_file = 'data/Final.csv'
# output_file = 'data/CleanFinal.csv'
# column_index = 3  # Index of the column to check for the value 'unknown'
#
# # Read the input CSV file and write to the output CSV file with 'unknown' replaced by ''
# with open(input_file, 'r', newline='', encoding='utf-8') as infile, open(output_file, 'w', newline='', encoding='utf-8') as outfile:
#     reader = csv.reader(infile)
#     writer = csv.writer(outfile)
#
#     for row in reader:
#         if "Hentai" not in row[column_index]:
#             if "Ecchi" not in row[column_index]:
#                 if "Yaoi" not in row[column_index]:
#                     writer.writerow(row)
#
#     # for row in reader:
#     #     if row[column_index] == 'Unknown':
#     #         row[column_index] = ''  # Replace 'unknown' with an empty string
#     #     writer.writerow(row)
#
# print("CSV file has been processed and saved as", output_file)