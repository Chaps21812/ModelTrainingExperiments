import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import os
import re
from astropy.visualization import ZScaleInterval
import matplotlib.gridspec as gridspec
from .preprocess_functions import _iqr_log
from matplotlib.animation import FuncAnimation
from IPython.display import display
import matplotlib.pyplot as plt
from IPython.display import display
# from .raw_datset import raw_dataset, satsim_path_loader

#This code was AI generated. I would love to spend time meticulously making plots, but I think my time is better spent analyzing them rather than adjusting details on a plot. 

def detect_column_type(series: pd.Series) -> str:
    """
    Detect whether a pandas Series (column) is categorical or numerical.

    Returns:
        "categorical" or "numerical"
    """
    # Drop NaNs for analysis
    s = series.dropna()

    if "time" in s.name.lower():
        return "time"
    if "file" in s.name.lower():
        return "file"
    if "path" in s.name.lower():
        return "file"

    # # If dtype is object or category, assume categorical
    # if pd.api.types.is_object_dtype(s) or pd.api.types.is_categorical_dtype(s):
    #     return "categorical"
    
    # If dtype is numeric, further check the number of unique values
    unique_vals = s.unique()
    num_unique = len(unique_vals)
    total = len(s)
    
    # Heuristic: if very few unique values relative to total, it's probably categorical
    if num_unique < 30:
        return "categorical"
    elif pd.api.types.is_numeric_dtype(s):
        return "numerical"
    # For boolean
    elif pd.api.types.is_bool_dtype(s):
        return "categorical"
    else:
        return "categorical"
                
def plot_categorical_column(series: pd.Series, filepath: str=None, dpi: int = 300 ):
    """
    Plot a bar chart for a categorical pandas Series with the mode, counts, and category name.
    """
    # Drop missing values
    s = series.dropna()

    # Count values
    value_counts = s.value_counts()
    mode_val = s.mode().iloc[0] if not s.mode().empty else "N/A"
    col_name = series.name if series.name else "Unnamed Column"

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(value_counts.index.astype(str), value_counts.values, color='teal', edgecolor='black')

    # Annotate bars with counts
    for bar, count in zip(bars, value_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), str(count),
                ha='center', va='bottom', fontsize=10)

    # Add title with mode and column name
    ax.set_title(f"Distribution of '{col_name}' (Mode: {mode_val})", fontsize=14)
    ax.set_ylabel("Count")
    ax.set_xlabel("Category")

    # Rotate x-axis labels if needed
    plt.xticks(rotation=45, ha='right')

    # Create legend with counts
    legend_labels = [f"{cat}: {cnt}" for cat, cnt in value_counts.items()]
    ax.legend(legend_labels, title="Category Counts", loc='upper right')

    plt.tight_layout()
    if filepath is not None: 
        os.makedirs(filepath, exist_ok=True)
        
        # Sanitize the attribute name to create a safe filename
        safe_name = re.sub(r'[^\w\-_.]', '_', col_name.strip())
        filename = f"{safe_name}.png"
        full_path = os.path.join(filepath, filename)
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(full_path, dpi=dpi, bbox_inches='tight')
        # print(f"Plot saved to: {full_path}")
    plt.show()
    
def plot_numerical_column(series: pd.Series, bins: int = 30, filepath: str=None, dpi: int = 300 ):
    """
    Plot a histogram for a numerical pandas Series with key statistics and column name.
    """
    # Drop missing values
    s = series.dropna()

    # Compute statistics
    mean_val = s.mean()
    median_val = s.median()
    std_val = s.std()
    min_val = s.min()
    max_val = s.max()
    col_name = series.name if series.name else "Unnamed Column"

    # Create histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(s, bins=bins, color='teal', edgecolor='black')

    # Add vertical lines for mean and median
    ax.axvline(mean_val, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {mean_val:.2f}')
    ax.axvline(median_val, color='blue', linestyle='dashed', linewidth=1.5, label=f'Median: {median_val:.2f}')

    # Set title and labels
    ax.set_title(f"Distribution of '{col_name}'", fontsize=14)
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")

    # Create text box with stats
    stats_text = (f"Mean: {mean_val:.2f}\n"
                  f"Median: {median_val:.2f}\n"
                  f"Std Dev: {std_val:.2f}\n"
                  f"Min: {min_val:.2f}\n"
                  f"Max: {max_val:.2f}")
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.98, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    # Add legend for mean and median lines
    ax.legend(loc='upper left')

    plt.tight_layout()
    if filepath is not None: 
        os.makedirs(filepath, exist_ok=True)
        
        # Sanitize the attribute name to create a safe filename
        safe_name = re.sub(r'[^\w\-_.]', '_', col_name.strip())
        filename = f"{safe_name}.png"
        full_path = os.path.join(filepath, filename)
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(full_path, dpi=dpi, bbox_inches='tight')
        # print(f"Plot saved to: {full_path}")
    plt.show()

def plot_scatter(x: pd.Series, y: pd.Series, alpha: float = 0.2, point_size: int = 40, color: str = 'teal', filepath: str=None, dpi: int = 300 ):
    """
    Create a scatter plot for two pandas Series with partial transparency to show point density.
    """
    # Align and drop missing values
    df = pd.concat([x, y], axis=1).dropna()
    x_vals = df.iloc[:, 0]
    y_vals = df.iloc[:, 1]
    x_name = x_vals.name if x_vals.name else "X"
    y_name = y_vals.name if y_vals.name else "Y"

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(x_vals, y_vals, alpha=alpha, s=point_size, color=color, edgecolor='black', linewidth=0.5)

    # Set labels and title
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title(f"Scatter Plot of '{x_name}' vs '{y_name}'", fontsize=14)
    # Axis limits from 0 to 1
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Add grid and layout
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    if filepath is not None: 
        os.makedirs(filepath, exist_ok=True)
        
        # Sanitize the attribute name to create a safe filename
        safe_name = "Scatter_Plot"
        filename = f"{safe_name}.png"
        full_path = os.path.join(filepath, filename)
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(full_path, dpi=dpi, bbox_inches='tight')
        # print(f"Plot saved to: {full_path}")
    plt.show()
    
def plot_lines(x1: pd.Series, y1: pd.Series, x2: pd.Series, y2: pd.Series,
               alpha: float = 0.2, color: str = 'teal', linewidth: float = 1.5, filepath: str=None, dpi: int = 300 ):
    """
    Plot line segments from (x1, y1) to (x2, y2) with transparency to visualize density.
    """
    # Combine into DataFrame and drop rows with missing values
    df = pd.concat([x1, y1, x2, y2], axis=1).dropna()
    col_names = df.columns.tolist()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for idx, row in df.iterrows():
        ax.plot([row[0], row[2]], [row[1], row[3]], color=color,
                alpha=alpha, linewidth=linewidth)

    # Axis labeling with fallback names
    x_label = f"{col_names[0]} → {col_names[2]}" if all(col_names) else "X"
    y_label = f"{col_names[1]} → {col_names[3]}" if all(col_names) else "Y"

    ax.set_title(f"Line Segments from ({col_names[0]}, {col_names[1]}) to ({col_names[2]}, {col_names[3]})", fontsize=14)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, linestyle='--', alpha=0.5)
    # Axis limits from 0 to 1
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    if filepath is not None: 
        os.makedirs(filepath, exist_ok=True)
        
        # Sanitize the attribute name to create a safe filename
        safe_name = "Line_Scatter_Plot"
        filename = f"{safe_name}.png"
        full_path = os.path.join(filepath, filename)
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(full_path, dpi=dpi, bbox_inches='tight')
        # print(f"Plot saved to: {full_path}")
    plt.show()

def plot_time_column(series: pd.Series, bins: int = 30, filepath: str=None, dpi: int = 300 ):
    """
    Plot a histogram for a pandas Series of time data, showing key statistics such as mean, median, 
    std dev, min, and max.
    """
    # Ensure the series is datetime
    s = pd.to_datetime(series.dropna(), errors='coerce')
    
    # Compute statistics
    mean_val = s.mean()
    median_val = s.median()
    std_val = s.std()
    min_val = s.min()
    max_val = s.max()
    col_name = series.name if series.name else "Unnamed Column"

    # Create histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(s, bins=bins, color='teal', edgecolor='black')

    # Add vertical lines for mean and median
    ax.axvline(mean_val, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {mean_val}')
    ax.axvline(median_val, color='blue', linestyle='dashed', linewidth=1.5, label=f'Median: {median_val}')

    # Set title and labels
    ax.set_title(f"Distribution of '{col_name}'", fontsize=14)
    ax.set_xlabel("Time")
    ax.set_ylabel("Frequency")

    # Format x-axis to show time properly
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M:%S'))

    # Create text box with stats
    stats_text = (f"Mean: {mean_val}\n"
                  f"Median: {median_val}\n"
                  f"Std Dev: {std_val}\n"
                  f"Min: {min_val}\n"
                  f"Max: {max_val}")
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.98, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    # Add legend for mean and median lines
    ax.legend(loc='upper left')

    plt.tight_layout()
    if filepath is not None: 
        os.makedirs(filepath, exist_ok=True)
        
        # Sanitize the attribute name to create a safe filename
        safe_name = "times_Plot"
        filename = f"{safe_name}.png"
        full_path = os.path.join(filepath, filename)
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(full_path, dpi=dpi, bbox_inches='tight')
        # print(f"Plot saved to: {full_path}")
    plt.show()

def plot_image_with_bbox(image: np.ndarray, x: int, y: int, size: int, object_n:int, full_file_path:str=None, dpi: int = 300, snr:tuple=None, alpha=.3, show:bool=False):
    """
    Plots an image with a square annotation and a padding of 100 pixels on all sides.

    Args:
        image (np.ndarray): The image to display.
        x (int): X-coordinate of the top-left corner of the annotation.
        y (int): Y-coordinate of the top-left corner of the annotation.
        size (int): Width/height of the square annotation.
    """
    # Image dimensions
    img_h, img_w = image.shape[:2]

    # Calculate padded annotation bounds
    pad = 100
    # Plot image
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image, cmap='gray')

    # Draw the padded rectangle
    rect = patches.Rectangle(
        (x-size/2, y-size/2),
        size,
        size,
        linewidth=2,
        edgecolor='red',
        facecolor='none', 
        alpha=alpha
    )
    ax.add_patch(rect)
    # ax.text(snr[0], snr[1], snr[2], fontsize=12, color='red', ha='right', va='bottom', alpha=alpha)
    plt.xlim(x-size-50,x+size+50)
    plt.ylim(y-size-50,y+size+50)

    # Turn off axes and show
    ax.set_axis_off()
    ax.set_title("Bounding Box Annotation")
    ax.set_axis_off()
    plt.tight_layout()
    if full_file_path is not None: 
        file = os.path.basename(full_file_path)
        direc = os.path.dirname(full_file_path)
        direc = os.path.dirname(direc)
        direc = os.path.join(direc, "annotation_view")
        os.makedirs(direc, exist_ok=True)
        
        # Sanitize the attribute name to create a safe filename
        safe_name = re.sub(r'[^\w\-_.]', '_', file.strip()).replace(".json","")
        filename = f"{safe_name}_{object_n}.png"
        full_path = os.path.join(direc, filename)
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(full_path, dpi=dpi, bbox_inches='tight')
    if show: plt.show()
    plt.close()
        # print(f"Plot saved to: {full_path}")

def plot_image_with_line(image: np.ndarray, x1: int, y1: int, x2: int, y2: int, object_n:int, full_file_path:str=None, dpi: int = 300, snr:tuple=None, alpha=.3, show:bool=False):
    """
    Plots an image with a line (x1, y1) -> (x2, y2) and a 100-pixel padded bounding box around the line.

    Args:
        image (np.ndarray): The image to display.
        x1, y1, x2, y2 (int): Coordinates of the line.
    """
    img_h, img_w = image.shape[:2]
    pad = 100

    # Get bounding box around the line
    min_x = max(0, min(x1, x2) - pad)
    max_x = min(img_w, max(x1, x2) + pad)
    min_y = max(0, min(y1, y2) - pad)
    max_y = min(img_h, max(y1, y2) + pad)

    # Compute width and height of padded box
    width = max_x - min_x
    height = max_y - min_y


    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image, cmap='gray')
    ax.text(snr[0], snr[1], snr[2], fontsize=12, color='red', ha='right', va='bottom', alpha=alpha)
    plt.plot(snr[0], snr[1], 'ro', markersize=5, alpha=alpha)


    # Draw the line
    # ax.plot([x1, x2], [y1, y2], color='red', linewidth=2, label='Line', alpha=.2)
    plt.arrow(x1, y1, x2-x1, y2-y1, head_width=2, head_length=2, fc='red', ec='red', alpha=alpha)

    plt.xlim(min_x-width-50,max_x+width+50)
    plt.ylim(min_y-height-50,max_y+height+50)


    # Style
    ax.set_title("Line Segment Annotation")
    ax.set_axis_off()
    plt.tight_layout()
    if full_file_path is not None: 
        file = os.path.basename(full_file_path)
        direc = os.path.dirname(full_file_path)
        direc = os.path.dirname(direc)
        direc = os.path.join(direc, "annotation_view")
        os.makedirs(direc, exist_ok=True)
        
        # Sanitize the attribute name to create a safe filename
        safe_name = re.sub(r'[^\w\-_.]', '_', file.strip()).replace(".json","")
        filename = f"{safe_name}_{object_n}.png"
        full_path = os.path.join(direc, filename)
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(full_path, dpi=dpi, bbox_inches='tight')
        # print(f"Plot saved to: {full_path}")
    if show: plt.show()
    plt.close()

def plot_image(image: np.ndarray, full_file_path:str=None, dpi: int = 300):
    """
    Plots an image with a line (x1, y1) -> (x2, y2) and a 100-pixel padded bounding box around the line.

    Args:
        image (np.ndarray): The image to display.
        x1, y1, x2, y2 (int): Coordinates of the line.
    """

    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image, cmap='gray')

    # Style
    ax.set_title("Base Image")
    ax.set_axis_off()
    ax.invert_yaxis()  # Invert y-axis to match the original image orientation
    plt.tight_layout()
    if full_file_path is not None: 
        file = os.path.basename(full_file_path)
        direc = os.path.dirname(full_file_path)
        direc = os.path.dirname(direc)
        direc = os.path.join(direc, "annotation_view")
        
        
        # Sanitize the attribute name to create a safe filename
        safe_name = re.sub(r'[^\w\-_.]', '_', file.strip()).replace(".json","")
        filename = f"{safe_name}_full.png"
        full_path = os.path.join(direc, filename)
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(full_path, dpi=dpi, bbox_inches='tight')
        # print(f"Plot saved to: {full_path}")
    plt.close()

def plot_all_annotations(image: np.ndarray, annotations: list, img_size: tuple, full_file_path:str=None, dpi: int = 300):
    """
    Plots an image with all annotations.

    Args:
        image (np.ndarray): The image to display.
        annotations (list): List of annotation dictionaries.
    """
    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image, cmap='gray')

    # Draw all annotations
    for annotation in annotations:
        if annotation['class_name'] == "Satellite":
            x = annotation['x_center']*img_size[0]
            y = annotation['y_center']*img_size[1]
            size = annotation['x_max']*img_size[0] - annotation['x_min']*img_size[0]
            rect = patches.Rectangle(
                (x-size/2, y-size/2),
                size,
                size,
                linewidth=2,
                edgecolor='red',
                facecolor='none'
            )
            ax.add_patch(rect)
        elif annotation['class_name'] == "Star":
            if "x_start" in annotation.keys():
                x1 = annotation['x_start']*img_size[0]
                y1 = annotation['y_start']*img_size[1]
                x2 = annotation['x_end']*img_size[0]
                y2 = annotation['y_end']*img_size[1]
            else:
                x1 = annotation['x1']*img_size[0]
                y1 = annotation['y1']*img_size[1]
                x2 = annotation['x2']*img_size[0]
                y2 = annotation['y2']*img_size[1]
            plt.arrow(x1, y1, x2-x1, y2-y1, head_width=2, head_length=2, fc='red', ec='red', alpha=.2)
            # ax.plot([x1, x2], [y1, y2], color='red', linewidth=2, alpha=.2)

    # Style
    ax.set_title("Base Image with All Annotations")
    ax.set_axis_off()
    ax.invert_yaxis()  # Invert y-axis to match the original image orientation
    plt.tight_layout()
    if full_file_path is not None: 
        file = os.path.basename(full_file_path)
        direc = os.path.dirname(full_file_path)
        direc = os.path.dirname(direc)
        direc = os.path.join(direc, "annotation_view")
        os.makedirs(direc, exist_ok=True)
        
        # Sanitize the attribute name to create a safe filename
        safe_name = re.sub(r'[^\w\-_.]', '_', file.strip()).replace(".json","")
        filename = f"{safe_name}_all_annotations.png"
        full_path = os.path.join(direc, filename)
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(full_path, dpi=dpi, bbox_inches='tight')
        # print(f"Plot saved to: {full_path}")
    plt.close()

def plot_single_annotation(image: np.ndarray, bbox_old:tuple, bbox_new:tuple, title:str):
    """
    Plots an image with all annotations.

    Args:
        image (np.ndarray): The image to display.
        annotations (list): List of annotation dictionaries.
    """
    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    # ax.imshow(_iqr_log(image), cmap='gray')
    ax.imshow(image, cmap='gray')
    padding = 20

    # Draw all annotations
    rect = patches.Rectangle(
        (bbox_new[0], bbox_new[1]),
        bbox_new[2],
        bbox_new[3],
        linewidth=2,
        edgecolor='green',
        facecolor='none'
    )
    ax.add_patch(rect)
    ax.plot(bbox_new[0]+bbox_new[2]/2, bbox_new[1]+bbox_new[3]/2, '.', color="green", alpha=.5)
    # Draw all annotations
    rect = patches.Rectangle(
        (bbox_old[0], bbox_old[1]),
        bbox_old[2],
        bbox_old[3],
        linewidth=2,
        edgecolor='red',
        facecolor='none'
    )
    ax.add_patch(rect)
    ax.plot(bbox_old[0]+bbox_old[2]/2, bbox_old[1]+bbox_old[3]/2, '.', color="red", alpha=.5)
    # Style
    ax.set_title(f"Adjusted Annotation: {title}")
    ax.set_axis_off()
    # ax.invert_yaxis()  # Invert y-axis to match the original image orientation
    plt.xlim(bbox_old[0]-padding, bbox_old[0]+bbox_old[2]+padding)
    plt.ylim(bbox_old[1]-padding, bbox_old[1]+bbox_old[3]+padding)
    plt.tight_layout()
    plt.show()

def plot_error_evaluator(image: np.ndarray, bboxes:tuple, index:int, attributes:dict):
    """
    Plots an image with all annotations.

    Args:
        image (np.ndarray): The image to display.
        annotations (list): List of annotation dictionaries.
    """

    fig = plt.figure(figsize=(12, 12))
    gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1])  # 2 rows, 2 cols

    ax_top = fig.add_subplot(gs[0, :])
    ax_bl = fig.add_subplot(gs[1, 0])
    ax_br = fig.add_subplot(gs[1, 1])

    # Plot
    fig.suptitle("Error Evaluation")

    # Plot the first image
    ax_top.imshow(_iqr_log(image), cmap='gray')
    ax_top.set_title(f'Image {attributes["fits_file"]}')

    # Plot the second image
    ax_bl.imshow(_iqr_log(image), cmap='gray')
    ax_bl.set_title(f'Annotation')

    # Add text in the third subplot
    ax_br.axis('off')
    text = ""
    for key,value in attributes.items():
        text = text+f"{key}: {value}\n"
    ax_br.text(0.01, 0.5, text, va='center', fontsize=12)

    if bboxes:
        spotlite_box = bboxes[index]
        rect = patches.Rectangle(
            (spotlite_box[0], spotlite_box[1]),
            spotlite_box[2],
            spotlite_box[3],
            linewidth=2,
            edgecolor='red',
            facecolor='none'
        )
        ax_bl.add_patch(rect)
        ax_bl.plot(spotlite_box[0]+spotlite_box[2]/2, spotlite_box[1]+spotlite_box[3]/2, '.', color="red", alpha=.5)
        ax_bl.set_xlim(spotlite_box[0]-20, spotlite_box[0]+spotlite_box[2]+20)
        ax_bl.set_ylim(spotlite_box[1]-20, spotlite_box[1]+spotlite_box[3]+20)

        for bbox in bboxes:
            # Draw all annotations
            rect = patches.Rectangle(
                (bbox[0], bbox[1]),
                bbox[2],
                bbox[3],
                linewidth=2,
                edgecolor='red',
                facecolor='none'
            )
            ax_top.add_patch(rect)
            ax_top.plot(bbox[0]+bbox[2]/2, bbox[1]+bbox[3]/2, '.', color="red", alpha=.5)
    plt.tight_layout()
    plt.show()

def z_scale_image(image:np.ndarray) -> np.ndarray:
    norm = ZScaleInterval(contrast=0.2)
    zscaled_data = norm(image)
    return zscaled_data

def plot_stacked_errors_by_date(df):
    """
    Creates a stacked bar chart of error types by string-formatted date.

    Parameters:
    - df: Pandas DataFrame with 'created' (string) and 'error_type' (categorical string)
    """
    # Group and count errors by 'created' date and 'error_type'
    grouped = df.groupby(['created', 'error_type_str']).size().unstack(fill_value=0)

    # Sort rows and columns for consistent plotting
    grouped = grouped.sort_index(axis=0).sort_index(axis=1)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    bottom = None
    colors = plt.cm.tab20.colors  # Up to 20 distinct colors
    error_types = grouped.columns.tolist()

    for i, error_type in enumerate(error_types):
        counts = grouped[error_type]
        ax.bar(grouped.index, counts, bottom=bottom, 
               label=error_type, color=colors[i % len(colors)])
        bottom = counts if bottom is None else bottom + counts

    ax.set_xlabel("Date")
    ax.set_ylabel("Error Count")
    ax.set_title("Stacked Error Types by Date")
    ax.legend(title="Error Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_stacked_errors_with_percent_legend_by_annotator_id(df):
    """
    Plots a stacked bar chart of error types by date string.
    Adds count and percentage in the legend for each error_type.

    Parameters:
    - df: Pandas DataFrame with 'created' and 'error_type' columns (both as strings)
    """
    # Group and count errors
    grouped = df.groupby(['labeler_id', 'error_type_str']).size().unstack(fill_value=0)
    grouped = grouped.sort_index(axis=0).sort_index(axis=1)

    # Total number of errors (all types)
    total_samples = grouped.sum().sum()

    # Plot setup
    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = [0] * len(grouped)
    colors = plt.cm.tab20.colors
    error_types = grouped.columns.tolist()

    for i, error_type in enumerate(error_types):
        counts = grouped[error_type]
        total_for_type = counts.sum()
        percent = (total_for_type / total_samples) * 100

        label = f"{error_type} ({total_for_type}, {percent:.1f}%)"

        ax.bar(grouped.index, counts, bottom=bottom,
               label=label, color=colors[i % len(colors)])

        # Update bottom for next stack
        bottom = [btm + val for btm, val in zip(bottom, counts)]

    # Labels and formatting
    ax.set_xlabel("Annotator")
    ax.set_ylabel("Error Count")
    ax.set_title("Stacked Error Types by Annotator")
    ax.legend(title="Error Type (Count, % of total)", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_stacked_errors_with_percent_legend(df):
    """
    Plots a stacked bar chart of error types by date string.
    Adds count and percentage in the legend for each error_type.

    Parameters:
    - df: Pandas DataFrame with 'created' and 'error_type' columns (both as strings)
    """
    # Group and count errors
    grouped = df.groupby(['created', 'error_type_str']).size().unstack(fill_value=0)
    grouped = grouped.sort_index(axis=0).sort_index(axis=1)

    # Total number of errors (all types)
    total_samples = grouped.sum().sum()

    # Plot setup
    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = [0] * len(grouped)
    colors = plt.cm.tab20.colors
    error_types = grouped.columns.tolist()

    for i, error_type in enumerate(error_types):
        counts = grouped[error_type]
        total_for_type = counts.sum()
        percent = (total_for_type / total_samples) * 100

        label = f"{error_type} ({total_for_type}, {percent:.1f}%)"

        ax.bar(grouped.index, counts, bottom=bottom,
               label=label, color=colors[i % len(colors)])

        # Update bottom for next stack
        bottom = [btm + val for btm, val in zip(bottom, counts)]

    # Labels and formatting
    ax.set_xlabel("Date")
    ax.set_ylabel("Error Count")
    ax.set_title("Stacked Error Types by Date")
    ax.legend(title="Error Type (Count, % of total)", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_animated_collect(images_list: list, bboxes_list:list, attributes_list:dict):
    """
    Plots an image with all annotations.

    Args:
        image (np.ndarray): The image to display.
        annotations (list): List of annotation dictionaries.
    """
    mpl_images = []
    mpl_bboxes = []
    mpl_texts = []

    for image,bboxes,attributes in zip(images_list, bboxes_list, attributes_list):
        mpl_images.append(_iqr_log(image))
        text = ""
        for key,value in attributes.items():
            text = text+f"{key}: {value}\n"
        mpl_texts.append(text)
        rects = []
        for bbox in bboxes:
            # Draw all annotations
            rect = patches.Rectangle(
                (bbox[0], bbox[1]),
                bbox[2],
                bbox[3],
                linewidth=2,
                edgecolor='red',
                facecolor='none'
            )
            rects.append(rect)
            # ax_top.add_patch(rect)
            # ax_top.plot(bbox[0]+bbox[2]/2, bbox[1]+bbox[3]/2, '.', color="red", alpha=.5)
        mpl_bboxes.append(rects)

    fig = plt.figure(figsize=(16, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.5, 1])  # 2 rows, 2 cols

    ax_top = fig.add_subplot(gs[0, 0])
    ax_br = fig.add_subplot(gs[0, 1])

    # Plot
    fig.suptitle("Collect View")

    # Plot the first image
    im = ax_top.imshow(mpl_images[0], cmap='gray')
    ax_top.set_title(f'Image {attributes["fits_file"]}')

    # Add text in the third subplot
    ax_br.axis('off')
    txt = ax_br.text(0.01, 0.5, mpl_texts[0], va='center', fontsize=12)

    def update(frame):
        # Update the image in the top axis
        im.set_array(mpl_images[frame])
        
        # Update the line in the bottom-right axis
        txt.set_text(mpl_texts[frame])
        
        return [im, txt]  # Return all changed artists

    ani = FuncAnimation(fig, update, frames=len(images_list), interval=200, blit=True)
    return fig, ani

def plot_star_selection(image: np.ndarray, bboxes:tuple, index:int, attributes:dict):
    """
    Plots an image with all annotations.

    Args:
        image (np.ndarray): The image to display.
        annotations (list): List of annotation dictionaries.
    """
    fig = plt.figure(figsize=(6, 6))
    fig, ax_top = plt.subplots()

    # ax.imshow(_iqr_log(image), cmap='gray')
    ax_top.imshow(_iqr_log(image), cmap='gray')

    # List to store points
    points = []

    def onclick(event):
        if event.xdata is not None and event.ydata is not None:
            x, y = event.xdata, event.ydata
            points.append((x, y))
            ax_top.plot(x, y, 'ro')  # mark the clicked point
            fig.canvas.draw()
            print(f"Clicked at: ({x:.2f}, {y:.2f})")

    # Plot
    fig.suptitle("Plate Solver")

    # Plot the first image
    ax_top.imshow(_iqr_log(image), cmap='gray')
    ax_top.set_title(f'Image {attributes["fits_file"]}')


    if bboxes:
        spotlite_box = bboxes[index]
        rect = patches.Rectangle(
            (spotlite_box[0], spotlite_box[1]),
            spotlite_box[2],
            spotlite_box[3],
            linewidth=2,
            edgecolor='red',
            facecolor='none'
        )

        for bbox in bboxes:
            # Draw all annotations
            rect = patches.Rectangle(
                (bbox[0], bbox[1]),
                bbox[2],
                bbox[3],
                linewidth=2,
                edgecolor='red',
                facecolor='none'
            )
            ax_top.add_patch(rect)
            ax_top.plot(bbox[0]+bbox[2]/2, bbox[1]+bbox[3]/2, '.', color="red", alpha=.5)


    # Output widget for matplotlib figure
    out = Output()

    # Function to handle click events
    def on_click(event):
        if event.xdata is not None and event.ydata is not None:
            x, y = event.xdata, event.ydata
            points.append((x, y))
            # run your custom code here
            print(f"Dot added at: ({x:.1f}, {y:.1f})")
            # update plot
            ax_top.plot(x, y, 'ro')
            fig.canvas.draw()

    # Text box to indicate done
    done_text = Text(
        placeholder='Type "done" when finished',
        description='Status:'
    )

    def on_done_change(change):
        if change['new'].lower() == 'done':
            print("Annotation complete!")
            print("All points:", points)

    done_text.observe(on_done_change, names='value')
    cid = fig.canvas.mpl_connect('button_press_event', on_click)

    # Combine widgets
    ui = VBox([out, done_text])

    with out:
        display(fig)

    display(ui)

    return points



# def plot_annotation_subset(pandas_library:pd.DataFrame, loader, view_satellite:bool=False, view_star:bool=False, view_image:bool=True):
#     """
#     Plots annotations from a dataset.

#     Parameters:
#     dataset_path (str): The path to the dataset directory.
#     view_satellite (bool): Whether to plot satellite annotations.
#     view_star (bool): Whether to plot star annotations.
#     view_image (bool): Whether to plot the image with annotations.
#     """

#     dataset_path = loader.directory
#     annotation_view_path = os.path.join(dataset_path, "annotation_view")
#     os.makedirs(annotation_view_path, exist_ok=True)



#     for json_path in tqdm(pandas_library["json_path"].unique(), desc="Plotting annotations", unit="file"):
#         fits_path = loader.annotation_to_fits[json_path]
#         with open(json_path, 'r') as file:
#             annotation = json.load(file)
#         fits_file = fits.open(fits_path)
#         hdu = fits_file[0].header
#         raw_data = fits_file[0].data

#         if "data" in annotation.keys():
#             annotation = annotation["data"]

#         #The XY coordinates are reverse intentionally. Beware!
#         y_res = hdu["NAXIS2"]
#         x_res = hdu["NAXIS1"]

#         noise = np.std(raw_data)
#         median_pixel = np.median(raw_data)

#         data = z_scale_image(raw_data)
#         if view_image: plot_all_annotations(data, annotation["objects"], (x_res,y_res), json_path, dpi=500)
#         for index,object in enumerate(annotation["objects"]):
#             x_cord= object["x_center"]*x_res
#             y_cord= object["y_center"]*y_res

#             if x_cord < 0 or y_cord < 0 or x_cord > data.shape[1] or y_cord > data.shape[0]:
#                 continue

#             half = 50
#             x_start = max(0, x_cord - half)
#             x_end   = min(data.shape[1], x_cord + half)
#             y_start = max(0, y_cord - half)
#             y_end   = min(data.shape[0], y_cord + half)

#             window = raw_data[int(y_start):int(y_end), int(x_start):int(x_end)]
#             local_minimum = np.min(window)
#             local_std = np.std(window)
#             signal = raw_data[int(y_cord), int(x_cord)]
#             snr_tuple = (object["x_center"]*x_res, object["y_center"]*y_res,"Prom: {}".format((signal-local_minimum)/local_std))

#             if object['class_name']=="Satellite": 
#                 if view_satellite: plot_image_with_bbox(data,object['x_center']*x_res,object['y_center']*y_res,object['x_max']*x_res-object['x_min']*x_res,index, json_path, dpi=500, snr=snr_tuple)

#             if object['class_name']=="Star": 
#                 if "x_start" in object.keys():
#                     x1 = object['x_start']
#                     y1 = object['y_start']
#                     x2 = object['x_end']
#                     y2 = object['y_end']
#                 else:
#                     x1 = object['x1']
#                     y1 = object['y1']
#                     x2 = object['x2']
#                     y2 = object['y2']
#                 if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0 or x1 > data.shape[1] or y1 > data.shape[0] or x2 > data.shape[1] or y2 > data.shape[0]:
#                     continue
#                 if view_star: plot_image_with_line(data,x1*x_res,y1*y_res,x2*x_res,y2*y_res,index, json_path, dpi=500, snr=snr_tuple)

# def plot_annotations(dataset_path:str, view_satellite:bool=False, view_star:bool=False, view_image:bool=True):
#     """
#     Plots annotations from a dataset.

#     Parameters:
#     dataset_path (str): The path to the dataset directory.
#     view_satellite (bool): Whether to plot satellite annotations.
#     view_star (bool): Whether to plot star annotations.
#     view_image (bool): Whether to plot the image with annotations.
#     """

#     loader = raw_dataset(dataset_path)
#     annotation_view_path = os.path.join(dataset_path, "annotation_view")
#     os.makedirs(annotation_view_path, exist_ok=True)

#     for json_path,fits_path in tqdm(loader.annotation_to_fits.items(), desc="Plotting annotations", unit="file"):
#         with open(json_path, 'r') as file:
#             annotation = json.load(file)
#         fits_file = fits.open(fits_path)
#         hdu = fits_file[0].header
#         data = fits_file[0].data

#         #The XY coordinates are reverse intentionally. Beware!
#         x_res = hdu["NAXIS2"]
#         y_res = hdu["NAXIS1"]

#         data = z_scale_image(data)
#         if view_image: plot_all_annotations(data, annotation["objects"], (x_res,y_res), json_path, dpi=500)
#         for index,object in enumerate(annotation["objects"]):
#             if object['class_name']=="Satellite": 
#                 if view_satellite: plot_image_with_bbox(data,object['x_center']*x_res,object['y_center']*y_res,object['x_max']*x_res-object['x_min']*x_res,index, json_path, dpi=500)

#             if object['class_name']=="Star": 
#                 if view_star: plot_image_with_line(data,object['x1']*x_res,object['y1']*y_res,object['x2']*x_res,object['y2']*y_res,index, json_path, dpi=500)

