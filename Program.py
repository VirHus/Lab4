import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
from sklearn.cluster import KMeans

class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing Application")
        self.root.geometry("1000x600")

        self.image = None
        self.original_image = None
        self.camera_active = False

        self.create_widgets()

    def create_widgets(self):
        # Create a menu
        menu = tk.Menu(self.root)
        self.root.config(menu=menu)

        # Add a File menu
        file_menu = tk.Menu(menu, tearoff=0)
        menu.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Image", command=self.load_image)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Add a Camera menu
        camera_menu = tk.Menu(menu, tearoff=0)
        menu.add_cascade(label="Camera", menu=camera_menu)
        camera_menu.add_command(label="Start Camera", command=self.start_camera)
        camera_menu.add_command(label="Stop Camera", command=self.stop_camera)

        # Add a Processing menu
        processing_menu = tk.Menu(menu, tearoff=0)
        menu.add_cascade(label="Processing", menu=processing_menu)
        processing_menu.add_command(label="Sobel Edge Detection", command=self.sobel_edge_detection)
        processing_menu.add_command(label="Canny Edge Detection", command=self.canny_edge_detection)
        processing_menu.add_command(label="Thresholding", command=self.thresholding)
        processing_menu.add_command(label="K-Means Clustering", command=self.kmeans_clustering)

        # Create an image panel to display images
        self.image_panel = tk.Label(self.root)
        self.image_panel.pack(side="left", padx=10, pady=10)
        
        self.image_panel_result = tk.Label(self.root)
        self.image_panel_result.pack(side="right", padx=10, pady=10)

        # Create control frame for sliders and buttons
        self.control_frame = tk.Frame(self.root)
        self.control_frame.pack(side="top", fill="both", expand=True)

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image = cv2.imread(file_path)
            self.original_image = self.image.copy()
            self.display_image(self.image, self.image_panel, width=450, height=500)  # Set desired width and height
            self.clear_result_panel()

    def display_image(self, image, panel, width=400, height=500):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (width, height))  # Resize the image
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        panel.configure(image=image)
        panel.image = image
        
    def clear_result_panel(self):
        # Get the background color of the root window
        bg_color = self.root.cget("background")
        # Convert the color to a format suitable for creating the blank image
        bg_color_rgb = self.root.winfo_rgb(bg_color)
        bg_color_rgb = (bg_color_rgb[0]//256, bg_color_rgb[1]//256, bg_color_rgb[2]//256)
        blank_image = np.ones((450, 500, 3), np.uint8) * np.array(bg_color_rgb, dtype=np.uint8)
        self.display_image(blank_image, self.image_panel_result, width=450, height=500)
        
    def start_camera(self):
        if not self.camera_active:
            self.camera_active = True
            self.capture_video()

    def stop_camera(self):
        self.camera_active = False
        self.clear_result_panel()

    def capture_video(self):
        camera = cv2.VideoCapture(0)
        while self.camera_active:
            _, frame = camera.read()
            self.image = frame
            self.original_image = self.image.copy()
            self.display_image(self.image, self.image_panel, width=450, height=500)
            self.root.update()
        camera.release()
        
    def sobel_edge_detection(self):
        if self.image is None:
            messagebox.showerror("Error", "No image loaded!")
            return
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        sobel_combined = cv2.magnitude(sobelx, sobely)
        sobel_combined = cv2.convertScaleAbs(sobel_combined)
        self.display_image(sobel_combined, self.image_panel_result)

    def canny_edge_detection(self):
        if self.image is None:
            messagebox.showerror("Error", "No image loaded!")
            return
        self.control_frame.pack_forget()
        self.control_frame = tk.Frame(self.root)
        self.control_frame.pack(side="right", fill="both", expand=True)

        def apply_canny():
            threshold1 = thresh1_slider.get()
            threshold2 = thresh2_slider.get()
            edges = cv2.Canny(self.image, threshold1, threshold2)
            self.display_image(edges, self.image_panel_result)

        tk.Label(self.control_frame, text="Threshold1").pack()
        thresh1_slider = tk.Scale(self.control_frame, from_=0, to=255, orient=tk.HORIZONTAL)
        thresh1_slider.pack()
        tk.Label(self.control_frame, text="Threshold2").pack()
        thresh2_slider = tk.Scale(self.control_frame, from_=0, to=255, orient=tk.HORIZONTAL)
        thresh2_slider.pack()

        apply_button = tk.Button(self.control_frame, text="Apply", command=apply_canny)
        apply_button.pack()

    def thresholding(self):
        if self.image is None:
            messagebox.showerror("Error", "No image loaded!")
            return
        self.control_frame.pack_forget()
        self.control_frame = tk.Frame(self.root)
        self.control_frame.pack(side="right", fill="both", expand=True)

        def apply_thresholding():
            threshold_value = thresh_slider.get()
            _, thresh_image = cv2.threshold(cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY), threshold_value, 255, cv2.THRESH_BINARY)
            self.display_image(thresh_image, self.image_panel_result)

        tk.Label(self.control_frame, text="Threshold Value").pack()
        thresh_slider = tk.Scale(self.control_frame, from_=0, to=255, orient=tk.HORIZONTAL)
        thresh_slider.pack()

        apply_button = tk.Button(self.control_frame, text="Apply", command=apply_thresholding)
        apply_button.pack()

    def kmeans_clustering(self):
        if self.image is None:
            messagebox.showerror("Error", "No image loaded!")
            return
        self.control_frame.pack_forget()
        self.control_frame = tk.Frame(self.root)
        self.control_frame.pack(side="right", fill="both", expand=True)

        def apply_kmeans():
            k = k_slider.get()
            image = self.image.reshape((-1, 3))
            kmeans = KMeans(n_clusters=k, random_state=0)
            kmeans.fit(image)
            clustered = kmeans.cluster_centers_[kmeans.labels_]
            clustered = clustered.reshape(self.image.shape).astype(np.uint8)
            self.display_image(clustered, self.image_panel_result)

        tk.Label(self.control_frame, text="Number of Clusters (K)").pack()
        k_slider = tk.Scale(self.control_frame, from_=2, to=10, orient=tk.HORIZONTAL)
        k_slider.pack()

        apply_button = tk.Button(self.control_frame, text="Apply", command=apply_kmeans)
        apply_button.pack()



if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()
