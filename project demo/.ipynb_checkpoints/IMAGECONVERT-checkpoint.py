import os
import matplotlib.pyplot as plt
from PIL import Image

def save_and_convert_to_gray(candidate_data, candidate_index, column, save_path):
    # Create a directory to save the plots
    os.makedirs(save_path, exist_ok=True)

    # Assuming the time information is in the 'T' column
    time_column = 'T'
    time_values = candidate_data[time_column]

    # Plot the column
    plt.figure(figsize=(8, 6))
    plt.plot(time_values, candidate_data[column], label=column)
    plt.xlabel('Time (seconds)')
    plt.ylabel(column)
    plt.title(f'Entire Candidate {candidate_index} - {column}')
    plt.legend()

    # Save the plot as an image
    save_filename = os.path.join(save_path, f"candidate_{candidate_index}_{column}.png")
    plt.savefig(save_filename)
    plt.close()

    # Convert the saved image to grayscale
    img = Image.open(save_filename)
    img_gray = img.convert('L')

    # Save the grayscale image
    gray_save_filename = os.path.join(save_path, f"candidate_{candidate_index}_{column}_gray.png")
    img_gray.save(gray_save_filename)

    # Display the grayscale image
    plt.figure(figsize=(8, 6))
    plt.imshow(img_gray, cmap='gray')
    plt.axis('off')
    plt.title(f'Grayscale Image - Candidate {candidate_index} - {column}')
    plt.show()

    

def main():
    import extarction as dl
    # Assuming get_data returns Full_data
    Full_data = dl.get_data()
    # Set the path to save pictures
    save_path = r'C:\Users\SUMANTHA\Pictures\Saved Pictures'
    
    # Example usage
    for candidate_index in range(len(Full_data)):
        for column in ['LX', 'LY', 'RX', 'RY']:
            save_and_convert_to_gray(Full_data[candidate_index], candidate_index, column, save_path)

if __name__ == "__main__":
    main()
