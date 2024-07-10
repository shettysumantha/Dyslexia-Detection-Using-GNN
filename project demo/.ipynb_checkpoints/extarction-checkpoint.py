import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image
def preprocess_data(df):
    # Ensure that 'LX', 'LY', 'RX', 'RY' columns are present
    if {'LX', 'LY', 'RX', 'RY'}.issubset(df.columns):
        # Remove rows with any NaN values in 'LX', 'LY', 'RX', 'RY'
        df = df.dropna(subset=['LX', 'LY', 'RX', 'RY'])

        # Calculate 'avg_x' and 'avg_y'
        df['avg_x'] = (df['LX'] + df['RX']) / 2
        df['avg_y'] = (df['LY'] + df['RY']) / 2

        return df[['T', 'avg_x', 'avg_y']]  # Return only relevant columns

    else:
        raise ValueError("Columns 'LX', 'LY', 'RX', 'RY' are missing in the DataFrame")

def plot_scatter(df, label, color):
    plt.plot(df['avg_x'], df['avg_y'], color=color, alpha=0.2, marker='o', label=label)
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.title(f'Scatter Plot for {label}')
    plt.legend()
    plt.show()

def save_and_convert_to_gray_scatter(df, label, color, save_path):
    # Create a directory to save the plots
    os.makedirs(save_path, exist_ok=True)

    # Scatter plot
    plt.plot(df['avg_x'], df['avg_y'], color=color, alpha=0.2, marker='o', label=label)
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.title(f'Scatter Plot for {label}')
    plt.legend()

    # Save the scatter plot as an image
    scatter_save_filename = os.path.join(save_path, f"scatter_{label}.png")
    plt.savefig(scatter_save_filename)
    plt.close()

    # Convert the saved scatter plot image to grayscale
    scatter_img = Image.open(scatter_save_filename)
    scatter_img_gray = scatter_img.convert('L')

    # Save the grayscale scatter plot image
    gray_scatter_save_filename = os.path.join(save_path, f"scatter_{label}_gray.png")
    scatter_img_gray.save(gray_scatter_save_filename)

    # Display the grayscale scatter plot image
    plt.figure(figsize=(8, 6))
    plt.imshow(scatter_img_gray, cmap='gray')
    plt.axis('off')
    plt.title(f'Grayscale Scatter Plot - {label}')
    plt.show()