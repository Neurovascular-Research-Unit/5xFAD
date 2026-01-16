import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, Button
import tkinter as tk
from tkinter import filedialog
import tifffile as tiff
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class EraserTool:
    def __init__(self, image_array, input_type, maxxed_fov=None):
        self.image_array = image_array
        self.original_image = image_array.copy()  # Save original image for resetting if needed
        self.input_type = input_type
        self.eraser_size = 10  # Default size of the eraser tool
        self.drawing = False  # Flag to indicate if mouse is being dragged for erasing
        self.maxxed_fov = maxxed_fov

        # Create a mask array to handle erased pixels
        self.mask = np.zeros_like(image_array, dtype=bool)

        # Stack to store previous states for undo functionality
        self.undo_stack = []

        # Set up the plot
        if input_type == 'ST Map':
            self.fig, self.ax = plt.subplots(figsize=(8,8))
            self.im = self.ax.imshow(self.image_array, cmap='plasma')            
        if input_type == 'Active Site':
            self.fig, self.ax = plt.subplots(figsize=(8,8))
            self.im_fov = self.ax.imshow(self.maxxed_fov, cmap='gray', vmax=np.quantile(maxxed_fov, .99))

            cmap = plt.cm.brg  # Start with a grayscale colormap
            # Create a new colormap with alpha values
            new_cmap = cmap.copy()
            new_cmap.set_under('none')

            self.im = self.ax.imshow(self.image_array, cmap=new_cmap, alpha=0.4, vmin=0.9)            

        # Connect mouse events to the handler functions
        self.cid_press = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.cid_motion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_drag)
        self.cid_release = self.fig.canvas.mpl_connect('button_release_event', self.on_release)

        # Add a slider to control the eraser size
        ax_slider = self.fig.add_axes([0.2, 0.01, 0.5, 0.03], facecolor='lightgoldenrodyellow')  # Slider position
        self.slider = Slider(ax_slider, 'Eraser Size', 1, 50, valinit=self.eraser_size, valstep=1)
        self.slider.on_changed(self.update_eraser_size)

        # Add an undo button to undo the last erase action
        ax_button = self.fig.add_axes([0.8, 0.01, 0.1, 0.05])  # Button position
        self.undo_button = Button(ax_button, 'Undo')
        self.undo_button.on_clicked(self.undo)

        ax_save_button = self.fig.add_axes([0.8, 0.06, 0.1, 0.05])  # Position below the undo button
        self.save_button = Button(ax_save_button, 'Save')
        self.save_button.on_clicked(self.save_image)


    def erase(self, x, y):
        # Save current state to undo stack before erasing
        #self.undo_stack.append(self.mask.copy())  # Store the previous mask state

        # Erase the portion of the image by masking it
        y_start = max(0, int(y - self.eraser_size))
        y_end = min(self.image_array.shape[0], int(y + self.eraser_size))
        x_start = max(0, int(x - self.eraser_size))
        x_end = min(self.image_array.shape[1], int(x + self.eraser_size))

        # Mask the erased pixels (set them as "erased")
        self.mask[y_start:y_end, x_start:x_end] = True
        self.update_image()

    def update_image(self):
        # Use the mask to display the image with erased parts as white
        global display_image
        display_image = np.ma.masked_where(self.mask, self.image_array)  # Mask the erased parts
        self.im.set_data(display_image)
        self.fig.canvas.draw()

    def update_eraser_size(self, val):
        # Update the eraser size from the slider
        self.eraser_size = int(val)

    def on_click(self, event):
        if event.inaxes != self.ax:
            return  # Ignore clicks outside the image area
        self.drawing = True
        self.undo_stack.append(self.mask.copy())  # Store the previous mask state

        self.erase(event.xdata, event.ydata)


    def on_drag(self, event):
        if self.drawing and event.inaxes == self.ax:
            self.erase(event.xdata, event.ydata)

    def on_release(self, event):
        self.drawing = False

    def undo(self, event=None):
        # Check if there is a previous state to undo
        if self.undo_stack:
            # Pop the last mask state and revert to it
            self.mask = self.undo_stack.pop()
            self.update_image()

    # New method to save the image
    def save_image(self, event=None):
        # Ask the user for the file path to save the image
        file_path = filedialog.asksaveasfilename(defaultextension=".npy", filetypes=[("NumPy files", "*.npy"), ("PNG files", "*.png")])

        masked_img = np.ma.masked_where(self.mask, self.image_array)
        fill_val = 0 if input_type == 'Active Site' else np.nan
        erased_img = masked_img.filled(fill_value=fill_val)

        if file_path:
            # Check if the user chose to save as PNG
            if file_path.endswith(".png"):
                # Save the image as a PNG file using matplotlib's imsave
                plt.imsave(file_path, erased_img, cmap='plasma', format='png')
            else:
                # Save the numpy array as a .npy file
                np.save(file_path, erased_img)
                #erased_stmap.dump(file_path)


def load_image_from_file(imtype='array'):
    # Create a tkinter window (it will not show)
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Ask the user to select a .npz file
    if imtype=='array':
        file_path = filedialog.askopenfilename(title="Select a NumPy array file", filetypes=[("NumPy files", "*.npy"), ("NumPy files", "*.npz")])
    elif imtype=='tiff':
        file_path = filedialog.askopenfilename(title="Select a Tiff file (.tif or .tiff)", filetypes=[("Tiff files", "*.tif"), ("Tiff files", "*.tiff")])
    elif imtype=='png':
        file_path = filedialog.askopenfilename(title="Select a png file", filetypes=[("PNG files", "*.png")])

    if file_path:
        # Load the numpy array from the selected file
        if imtype == 'array':
            return np.load(file_path)
        elif imtype == 'tiff':
            return tiff.imread(file_path)
        elif imtype =='png':
            return None

    else:
        print("No file selected. Please select a file.")
        return None



# Create a sample numpy array (e.g., a grayscale image of 100x100)

input_type = input("""Are you erasing an Active Site Map or ST Map? Type 'Active Site' or 'ST Map' and hit Enter --> \n""")

num_tries=0
while input_type not in {'Active Site', 'ST Map'}:
    input_type = input(f"""Invalid Input{'!'*(num_tries+1)} {'ヽ༼ ಠ益ಠ ༽ﾉ '*(num_tries if num_tries<4 else 3)} You must type 'Active Site' or 'ST Map' and hit Enter --> \n""")
    num_tries+=1

if input_type == 'Active Site':
    print("""
    
        The purpose of the eraser tool for active site masks is to remove artifacts 
        and/or to separate active sites on branched or curved structures. 

        You will need 2 files for this tool:
            1) Image of FOV, ideally plasma, and should be denoised and motion corrected (saved as a .npy file)
            2) Active site masks (saved as a .npy file)

        You will first select the .npy file for the video, and then you will select the file for the active site masks.

        To continue, hit Enter...

        """)
    input()

    plasma_fov = load_image_from_file(imtype='array')
    while not isinstance(plasma_fov, np.ndarray):
        plasma_fov = load_image_from_file(imtype='array')
        print(plasma_fov.type)

    input("""Tiff file has been selected. Next, you will select the file corresponding to active site masks. Hit Enter to continue.""")

image_array = load_image_from_file(imtype='array')
while not isinstance(plasma_fov, np.ndarray):
    image_array = load_image_from_file(imtype='array')

if type(image_array) == np.lib.npyio.NpzFile:
    first_key = list(image_array.keys())[0]
    image_array = image_array[first_key]
#image_array = np.load('/Users/lij49/oldanesth_part1_STMaps/egta_baseline_8_stmaparray.npz')['arr_0']
# Create the EraserTool instance and start the interactive session

maxxed_fov = plasma_fov if input_type == 'Active Site' else None
eraser_tool = EraserTool(image_array, input_type, maxxed_fov=maxxed_fov)

# Display the plot
plt.show()
