
import matplotlib.pyplot as plt

# Assuming get_data returns Full_data


def plot_entire_candidatety(candidate_index, time_window, Full_data):
    candidate_data = Full_data[candidate_index]

    # Assuming the time information is in the 'T' column
    time_column = 'T'
    time_values = candidate_data[time_column]

    # Select columns for plotting (replace with your actual column names)
    columns_to_plot = ['LX', 'LY', 'RX', 'RY']

    # Plot each selected column
    for column in columns_to_plot:
        plt.figure(figsize=(8, 6))
        plt.plot(time_values, candidate_data[column], label=column)
        plt.xlabel('Time (seconds)')
        plt.ylabel(column)
        plt.title(f'Entire Candidate {candidate_index} - {column}')
        plt.legend()
        plt.show()

# Example usage

