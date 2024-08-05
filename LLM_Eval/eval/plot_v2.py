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

# Read the Excel file
excel_file = 'Evaluation Metrics of LLMs for Collabcoder.xlsx'
sheet = pd.read_excel(excel_file, sheet_name=1, header=None)  # Reading the second sheet (Version 2)

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

# Plot 1: Comparison across datasets
fig, ax = plt.subplots(figsize=(12, 6))
bar_width = 0.25
index = np.arange(len(results_sorted))

ax.bar(index, results_sorted['SUTD'], bar_width, label='SUTD Data', color='b')
ax.bar(index + bar_width, results_sorted['Positive'], bar_width, label='Positive Reviews', color='g')
ax.bar(index + 2*bar_width, results_sorted['Negative'], bar_width, label='Negative Reviews', color='r')

ax.set_ylabel('Average Score')
ax.set_title('LLM Performance Across Datasets (Version 2)')
ax.set_xticks(index + bar_width)
ax.set_xticklabels(results_sorted.index, rotation=45, ha='right')
ax.legend()

plt.tight_layout()
plt.savefig('v2_llm_performance_comparison.png')
plt.close()

# Plot 2: Overall performance
plt.figure(figsize=(10, 6))
bars = plt.bar(results_sorted.index, results_sorted['Overall Average'], color='skyblue')
plt.title('Overall LLM Performance (Version 2)')
plt.xlabel('LLMs')
plt.ylabel('Average Score')
plt.xticks(rotation=45, ha='right')

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}',
             ha='center', va='bottom')

plt.tight_layout()
plt.savefig('v2_overall_llm_performance.png')
plt.close()

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

print("\nAnalysis complete. Check 'v2_llm_performance_comparison.png' and 'v2_overall_llm_performance.png' for visualizations, and 'llm_performance_summary_v2.csv' for the detailed summary.")



