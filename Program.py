import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from sklearn.cluster import KMeans

class Camera:
    def __init__(self, app):
        self.app = app
        self.capture = None

    def start_video(self):
        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            messagebox.showerror("Error", "Failed to open the video camera.")
            return

        self.process_video()

    def stop_video(self):
        if self.capture and self.capture.isOpened():
            self.capture.release()
            self.capture = None

    def process_video(self):
        if self.capture and self.capture.isOpened():
            ret, frame = self.capture.read()
            if ret:
                self.app.display_image(frame, self.app.image_panel)
                processed_frame = self.app.apply_processing(frame)
                self.app.display_image(processed_frame, self.app.image_panel_result)
            self.app.root.after(10, self.process_video)

class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("GUI-BASED EDGE DETECTION AND IMAGE SEGMENTATION")
        self.root.geometry("1000x600")
        self.root.configure(bg="lightgray")

        self.image = None
        self.original_image = None
        self.camera_active = False

        self.filter_type = tk.StringVar(value="None")
        
        self.video_processor = Camera(self)

        self.create_widgets()

    def create_widgets(self):

        menu = tk.Menu(self.root)
        self.root.config(menu=menu)


        file_menu = tk.Menu(menu, tearoff=0)
        menu.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Image", command=self.load_image)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        

        camera_menu = tk.Menu(menu, tearoff=0)
        menu.add_cascade(label="Camera", menu=camera_menu)
        camera_menu.add_command(label="Start Camera", command=self.start_camera)
        camera_menu.add_command(label="Stop Camera", command=self.stop_camera)

        processing_menu = tk.Menu(menu, tearoff=0)
        menu.add_cascade(label="Processing", menu=processing_menu)
        processing_menu.add_command(label="Sobel Edge Detection", command=lambda: self.set_filter("Sobel"))
        processing_menu.add_command(label="Canny Edge Detection", command=lambda: self.set_filter("Canny"))
        processing_menu.add_command(label="Thresholding", command=lambda: self.set_filter("Threshold"))
        processing_menu.add_command(label="K-Means Clustering", command=lambda: self.set_filter("KMeans"))

        self.image_panel = tk.Label(self.root)
        self.image_panel.pack(side="left", padx=10, pady=10)
        
        self.image_panel_result = tk.Label(self.root)
        self.image_panel_result.pack(side="right", padx=10, pady=10)

        self.control_frame = tk.Frame(self.root)
        self.control_frame.pack(side="top", fill="both", expand=True)
        
        

    def set_filter(self, filter_type):
        self.filter_type.set(filter_type)
        filter = self.filter_type.get()
        self.clear_control_frame()
        
        if filter == "Sobel":
            self.sobel_edge_detection()
        elif filter == "Canny":
            self.canny_edge_detection()
        elif filter == "Threshold":
            self.thresholding()
        elif filter == "KMeans":
            self.kmeans_clustering()
            
    def clear_control_frame(self):
        for widget in self.control_frame.winfo_children():
            widget.destroy()
        

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if self.camera_active == True:
            self.stop_camera()
        if file_path:
            self.image = cv2.imread(file_path)
            self.original_image = self.image.copy()
            self.display_image(self.image, self.image_panel, width=450, height=500)
            self.clear_result_panel()

    def display_image(self, image, panel, width=400, height=500):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (width, height))
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        panel.configure(image=image)
        panel.image = image

    def clear_result_panel(self):
        bg_color = self.root.cget("background")
        bg_color_rgb = self.root.winfo_rgb(bg_color)
        bg_color_rgb = (bg_color_rgb[0] // 256, bg_color_rgb[1] // 256, bg_color_rgb[2] // 256)
        blank_image = np.ones((450, 500, 3), np.uint8) * np.array(bg_color_rgb, dtype=np.uint8)
        self.display_image(blank_image, self.image_panel_result, width=450, height=500)
        
    def clear_original_panel(self):
        bg_color = self.root.cget("background")
        bg_color_rgb = self.root.winfo_rgb(bg_color)
        bg_color_rgb = (bg_color_rgb[0] // 256, bg_color_rgb[1] // 256, bg_color_rgb[2] // 256)
        blank_image = np.ones((450, 500, 3), np.uint8) * np.array(bg_color_rgb, dtype=np.uint8)
        self.display_image(blank_image, self.image_panel, width=450, height=500)

    def start_camera(self):
        if not self.camera_active:
            self.camera_active = True
            self.video_processor.start_video()

    def stop_camera(self):
        self.camera_active = False
        self.video_processor.stop_video()
        self.clear_original_panel()
        self.clear_result_panel()

    def apply_processing(self, frame):
        filter_type = self.filter_type.get()
        if filter_type == "Sobel":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel = cv2.magnitude(sobelx, sobely)
            sobel = cv2.convertScaleAbs(sobel)
            return sobel
        elif filter_type == "Canny":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            return edges
        elif filter_type == "Threshold":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            return thresh_image
        elif filter_type == "KMeans":
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pixel_values = image.reshape((-1, 3))
            pixel_values = np.float32(pixel_values)

            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
            k = 3
            _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

            centers = np.uint8(centers)
            labels = labels.flatten()
            segmented_image = centers[labels.flatten()]
            segmented_image = segmented_image.reshape(image.shape)
            return cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
        return frame
    
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
