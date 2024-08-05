# # import pandas as pd
# # import matplotlib.pyplot as plt
# # import numpy as np

# # def read_dataset(sheet, start_row, name):
# #     print(f"\nLooking for dataset: {name} starting from row {start_row}")
# #     # Find the header row
# #     header_row = start_row
# #     while header_row < len(sheet) and (pd.isna(sheet.iloc[header_row, 0]) or sheet.iloc[header_row, 0] != 'LLMs'):
# #         header_row += 1
    
# #     if header_row >= len(sheet):
# #         print(f"Could not find 'LLMs' header for {name} dataset")
# #         return None

# #     print(f"Found 'LLMs' header at row {header_row}")

# #     # Find the end of the dataset
# #     end_row = header_row + 1
# #     while end_row < len(sheet) and pd.notna(sheet.iloc[end_row, 0]):
# #         end_row += 1
    
# #     # Extract the dataset
# #     dataset = sheet.iloc[header_row:end_row].copy()
# #     dataset.columns = dataset.iloc[0]
# #     dataset = dataset.iloc[1:]
    
# #     # Set the first column as index
# #     dataset = dataset.set_index('LLMs')
    
# #     # Convert to float, excluding 'Sentiment Match' column
# #     for col in dataset.columns:
# #         if col != 'Sentiment Match':
# #             dataset[col] = pd.to_numeric(dataset[col], errors='coerce')
    
# #     print(f"\nDataset: {name}")
# #     print(dataset)
# #     return dataset, end_row

# # # Read the Excel file
# # excel_file = 'Evaluation Metrics of LLMs for Collabcoder.xlsx'
# # sheet = pd.read_excel(excel_file, header=None)

# # # Extract datasets
# # sutd_data, end_row = read_dataset(sheet, 0, "SUTD")
# # positive_data, end_row = read_dataset(sheet, end_row, "Positive")
# # negative_data, _ = read_dataset(sheet, end_row, "Negative")

# # # Function to calculate average score
# # def calculate_average_score(row):
# #     return row.drop('Sentiment Match', errors='ignore').mean()

# # # Calculate average scores for each dataset
# # sutd_scores = sutd_data.apply(calculate_average_score, axis=1)
# # positive_scores = positive_data.apply(calculate_average_score, axis=1)
# # negative_scores = negative_data.apply(calculate_average_score, axis=1)

# # # Combine scores into a single dataframe
# # results = pd.DataFrame({
# #     'LLMs': sutd_scores.index,
# #     'SUTD': sutd_scores.values,
# #     'Positive': positive_scores.values,
# #     'Negative': negative_scores.values
# # })

# # print("\nResults DataFrame:")
# # print(results)

# # # Set up the plot
# # fig, ax = plt.subplots(figsize=(12, 6))

# # # Create the grouped bar chart
# # bar_width = 0.25
# # index = np.arange(len(results))

# # ax.bar(index, results['SUTD'], bar_width, label='SUTD Data', color='b')
# # ax.bar(index + bar_width, results['Positive'], bar_width, label='Positive Reviews', color='g')
# # ax.bar(index + 2 * bar_width, results['Negative'], bar_width, label='Negative Reviews', color='r')

# # # Customize the plot
# # ax.set_ylabel('Average Score')
# # ax.set_title('LLM Performance Across Three Datasets')
# # ax.set_xticks(index + bar_width)
# # ax.set_xticklabels(results['LLMs'], rotation=45, ha='right')
# # ax.legend()

# # plt.tight_layout()

# # # Save the plot
# # plt.savefig('llm_performance_comparison.png')

# # # Print the results
# # print("\nFinal Results:")
# # print(results)

# # # Identify the best performing LLM
# # best_llm = results.iloc[:, 1:].mean(axis=1).idxmax()
# # print(f"\nBest performing LLM overall: {results.loc[best_llm, 'LLMs']}")

# # # Print performance by dataset
# # for dataset in ['SUTD', 'Positive', 'Negative']:
# #     best_for_dataset = results[dataset].idxmax()
# #     print(f"Best for {dataset} data: {results.loc[best_for_dataset, 'LLMs']} (Score: {results.loc[best_for_dataset, dataset]:.4f})")

# # # Calculate and print the overall average score for each LLM
# # results['Overall Average'] = results[['SUTD', 'Positive', 'Negative']].mean(axis=1)
# # print("\nOverall Average Scores:")
# # print(results[['LLMs', 'Overall Average']].sort_values('Overall Average', ascending=False))

# # Overall LLM

# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# def read_dataset(sheet, start_row, name):
#     print(f"\nLooking for dataset: {name} starting from row {start_row}")
#     # Find the header row
#     header_row = start_row
#     while header_row < len(sheet) and (pd.isna(sheet.iloc[header_row, 0]) or sheet.iloc[header_row, 0] != 'LLMs'):
#         header_row += 1
    
#     if header_row >= len(sheet):
#         print(f"Could not find 'LLMs' header for {name} dataset")
#         return None, header_row

#     print(f"Found 'LLMs' header at row {header_row}")

#     # Find the end of the dataset
#     end_row = header_row + 1
#     while end_row < len(sheet) and pd.notna(sheet.iloc[end_row, 0]):
#         end_row += 1
    
#     # Extract the dataset
#     dataset = sheet.iloc[header_row:end_row].copy()
#     dataset.columns = dataset.iloc[0]
#     dataset = dataset.iloc[1:]
    
#     # Set the first column as index
#     dataset = dataset.set_index('LLMs')
    
#     # Convert to float, excluding 'Sentiment Match' column
#     for col in dataset.columns:
#         if col != 'Sentiment Match':
#             dataset[col] = pd.to_numeric(dataset[col], errors='coerce')
    
#     print(f"\nDataset: {name}")
#     print(dataset)
#     return dataset, end_row

# # Read the Excel file
# excel_file = 'Evaluation Metrics of LLMs for Collabcoder.xlsx'
# sheet = pd.read_excel(excel_file, header=None)

# # Extract datasets
# sutd_data, end_row = read_dataset(sheet, 0, "SUTD")
# positive_data, end_row = read_dataset(sheet, end_row, "Positive")
# negative_data, _ = read_dataset(sheet, end_row, "Negative")

# # Function to calculate average score
# def calculate_average_score(row):
#     return row.drop('Sentiment Match', errors='ignore').mean()

# # Calculate average scores for each dataset
# sutd_scores = sutd_data.apply(calculate_average_score, axis=1)
# positive_scores = positive_data.apply(calculate_average_score, axis=1)
# negative_scores = negative_data.apply(calculate_average_score, axis=1)

# # Combine scores into a single dataframe
# results = pd.DataFrame({
#     'LLMs': sutd_scores.index,
#     'SUTD': sutd_scores.values,
#     'Positive': positive_scores.values,
#     'Negative': negative_scores.values
# })

# print("\nResults DataFrame:")
# print(results)

# # Calculate overall average score
# results['Overall Average'] = results[['SUTD', 'Positive', 'Negative']].mean(axis=1)

# # Sort results by overall average
# results_sorted = results.sort_values('Overall Average', ascending=False)

# print("\nRanked Overall Performance:")
# print(results_sorted[['LLMs', 'Overall Average']])

# # Identify the best performing LLM
# best_llm = results_sorted.iloc[0]['LLMs']
# best_score = results_sorted.iloc[0]['Overall Average']

# print(f"\nBest performing LLM: {best_llm} with an average score of {best_score:.4f}")

# # Create a bar plot for dataset comparison
# fig, ax = plt.subplots(figsize=(12, 6))
# bar_width = 0.25
# index = np.arange(len(results))

# ax.bar(index, results['SUTD'], bar_width, label='SUTD Data', color='b')
# ax.bar(index + bar_width, results['Positive'], bar_width, label='Positive Reviews', color='g')
# ax.bar(index + 2 * bar_width, results['Negative'], bar_width, label='Negative Reviews', color='r')

# ax.set_ylabel('Average Score')
# ax.set_title('LLM Performance Across Three Datasets')
# ax.set_xticks(index + bar_width)
# ax.set_xticklabels(results['LLMs'], rotation=45, ha='right')
# ax.legend()

# plt.tight_layout()
# plt.savefig('llm_performance_comparison.png')

# # Create a bar plot for overall performance
# plt.figure(figsize=(10, 6))
# bars = plt.bar(results_sorted['LLMs'], results_sorted['Overall Average'], color='skyblue')
# plt.title('Overall LLM Performance')
# plt.xlabel('LLMs')
# plt.ylabel('Average Score')
# plt.xticks(rotation=45, ha='right')

# # Highlight the best performing LLM
# bars[0].set_color('gold')

# # Add value labels on top of each bar
# for bar in bars:
#     height = bar.get_height()
#     plt.text(bar.get_x() + bar.get_width()/2., height,
#              f'{height:.4f}',
#              ha='center', va='bottom')

# plt.tight_layout()
# plt.savefig('overall_llm_performance.png')

# # Create a summary table
# summary_table = results_sorted[['LLMs', 'SUTD', 'Positive', 'Negative', 'Overall Average']]
# summary_table = summary_table.round(4)
# summary_table['Rank'] = range(1, len(summary_table) + 1)
# summary_table = summary_table.set_index('Rank')

# print("\nSummary Table:")
# print(summary_table)

# # Save the summary table to a CSV file
# summary_table.to_csv('llm_performance_summary.csv')

# print("\nAnalysis complete. Check 'llm_performance_comparison.png' and 'overall_llm_performance.png' for visualizations, and 'llm_performance_summary.csv' for the detailed summary.")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def read_dataset(sheet, start_row, name):
    print(f"\nLooking for dataset: {name} starting from row {start_row}")
    header_row = start_row
    while header_row < len(sheet) and (pd.isna(sheet.iloc[header_row, 0]) or sheet.iloc[header_row, 0] != 'LLMs'):
        header_row += 1
    
    if header_row >= len(sheet):
        print(f"Could not find 'LLMs' header for {name} dataset")
        return None, header_row

    print(f"Found 'LLMs' header at row {header_row}")

    end_row = header_row + 1
    while end_row < len(sheet) and pd.notna(sheet.iloc[end_row, 0]):
        end_row += 1
    
    dataset = sheet.iloc[header_row:end_row].copy()
    dataset.columns = dataset.iloc[0]
    dataset = dataset.iloc[1:]
    dataset = dataset.set_index('LLMs')
    
    for col in dataset.columns:
        dataset[col] = pd.to_numeric(dataset[col], errors='coerce')
    
    print(f"\nDataset: {name}")
    print(dataset)
    return dataset, end_row

def create_bar_plot(data, title, filename):
    plt.figure(figsize=(10, 6))
    bars = plt.bar(data.index, data.values, color='skyblue')
    plt.title(title)
    plt.xlabel('LLMs')
    plt.ylabel('Score')
    plt.xticks(rotation=45, ha='right')

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.4f}',
                 ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Read the Excel file
excel_file = 'Evaluation Metrics of LLMs for Collabcoder.xlsx'
sheet = pd.read_excel(excel_file, sheet_name=1, header=None)  # Reading the second sheet

# Extract datasets
sutd_data, end_row = read_dataset(sheet, 0, "SUTD")
positive_data, end_row = read_dataset(sheet, end_row, "Positive")
negative_data, _ = read_dataset(sheet, end_row, "Negative")

# Calculate average scores for each dataset
results = pd.DataFrame({
    'SUTD': sutd_data.mean(axis=1),
    'Positive': positive_data.mean(axis=1),
    'Negative': negative_data.mean(axis=1)
})

results['Overall Average'] = results.mean(axis=1)
results_sorted = results.sort_values('Overall Average', ascending=False)

print("\nResults:")
print(results_sorted)

# Create four plots
create_bar_plot(results_sorted['Overall Average'], 'Overall LLM Performance', 'overall_performance.png')
create_bar_plot(results_sorted['SUTD'], 'LLM Performance on SUTD Data', 'sutd_performance.png')
create_bar_plot(results_sorted['Positive'], 'LLM Performance on Positive Reviews', 'positive_performance.png')
create_bar_plot(results_sorted['Negative'], 'LLM Performance on Negative Reviews', 'negative_performance.png')

# Identify the best performing LLM
best_llm = results_sorted.index[0]
best_score = results_sorted['Overall Average'].iloc[0]

print(f"\nBest performing LLM: {best_llm} with an average score of {best_score:.4f}")

# Create a summary table
summary_table = results_sorted.round(4)
summary_table['Rank'] = range(1, len(summary_table) + 1)
summary_table = summary_table.reset_index()
summary_table = summary_table.set_index('Rank')

print("\nSummary Table:")
print(summary_table)

# Save the summary table to a CSV file
summary_table.to_csv('llm_performance_summary_v2.csv')

print("\nAnalysis complete. Check the generated PNG files for visualizations, and 'llm_performance_summary_v2.csv' for the detailed summary.")