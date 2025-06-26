import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import matplotlib.pyplot as plt


class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Aplikasi Pengolahan Citra Digital")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2c3e50')
        
        # Variables
        self.original_image = None
        self.processed_image = None
        self.image_path = ""
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main title
        title_frame = tk.Frame(self.root, bg='#2c3e50')
        title_frame.pack(pady=10)
        
        title_label = tk.Label(title_frame, text="🖼️ APLIKASI PENGOLAHAN CITRA DIGITAL", 
                              font=('Arial', 20, 'bold'), fg='#ecf0f1', bg='#2c3e50')
        title_label.pack()
        
        # Main container
        main_frame = tk.Frame(self.root, bg='#2c3e50')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Left panel for controls
        # control_frame = tk.Frame(main_frame, bg='#34495e', relief=tk.RAISED, bd=2)
        # control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # ==== SCROLLABLE CONTROL PANEL ====
        control_canvas = tk.Canvas(main_frame, bg='#2c3e50', highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=control_canvas.yview)
        scrollable_frame = tk.Frame(control_canvas, bg='#34495e')

        scrollable_frame.bind(
            "<Configure>",
            lambda e: control_canvas.configure(
                scrollregion=control_canvas.bbox("all")
            )
        )

        control_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        control_canvas.configure(yscrollcommand=scrollbar.set)

        control_canvas.pack(side=tk.LEFT, fill=tk.Y)
        control_canvas.config(width=250)  # atau sesuaikan dengan lebar panel kiri

        scrollbar.pack(side=tk.LEFT, fill=tk.Y)

        def _on_mousewheel(event):
            control_canvas.yview_scroll(int(-1*(event.delta/120)), "units")

        control_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        scrollable_frame.bind("<Enter>", lambda e: control_canvas.bind_all("<MouseWheel>", _on_mousewheel))
        scrollable_frame.bind("<Leave>", lambda e: control_canvas.unbind_all("<MouseWheel>"))


        
        # Input section
        input_section = tk.LabelFrame(scrollable_frame, text="📁 INPUT GAMBAR", 
                                     font=('Arial', 12, 'bold'), fg='#ecf0f1', bg='#34495e')
        input_section.pack(fill=tk.X, padx=10, pady=10)
        
        self.load_btn = tk.Button(input_section, text="Pilih Gambar", command=self.load_image,
                                 bg='#3498db', fg='white', font=('Arial', 10, 'bold'),
                                 relief=tk.FLAT, padx=20, pady=8)
        self.load_btn.pack(pady=10)

        # Save section
        save_section = tk.LabelFrame(scrollable_frame, text="💾 SIMPAN", 
                                    font=('Arial', 12, 'bold'), fg='#ecf0f1', bg='#34495e')
        save_section.pack(fill=tk.X, padx=10, pady=10)
        
        self.save_btn = tk.Button(save_section, text="Simpan Hasil", command=self.save_image,
                                 bg='#27ae60', fg='white', font=('Arial', 10, 'bold'),
                                 relief=tk.FLAT, padx=20, pady=8)
        self.save_btn.pack(pady=10)
        
        # Processing section
        # Histogram 
        process_section = tk.LabelFrame(scrollable_frame, text="⚙️ PROSES GAMBAR", 
                                       font=('Arial', 12, 'bold'), fg='#ecf0f1', bg='#34495e')
        process_section.pack(fill=tk.X, padx=10, pady=10)
        histogram_btn = tk.Button(process_section, text="📊 Histogram", command=self.show_histogram,
                          bg='#1abc9c', fg='white', font=('Arial', 9, 'bold'),
                          relief=tk.FLAT, padx=15, pady=5)
        histogram_btn.pack(fill=tk.X, pady=2)

        # Basic processing buttons
        buttons_config = [
            ("Grayscale", self.convert_to_grayscale, '#95a5a6'),
            ("Binary", self.convert_to_binary, '#9b59b6'),
        ]
        
        for text, command, color in buttons_config:
            btn = tk.Button(process_section, text=text, command=command,
                           bg=color, fg='white', font=('Arial', 9, 'bold'),
                           relief=tk.FLAT, padx=15, pady=5)
            btn.pack(fill=tk.X, pady=2)
        
        # Arithmetic operations
        arith_section = tk.LabelFrame(scrollable_frame, text="➕ OPERASI ARITMATIKA", 
                                     font=('Arial', 12, 'bold'), fg='#ecf0f1', bg='#34495e')
        arith_section.pack(fill=tk.X, padx=10, pady=5)
        
        # Frame untuk input skalar
        scalar_frame = tk.Frame(arith_section, bg='#34495e')
        scalar_frame.pack(fill=tk.X, pady=5)
        
        # Label untuk input skalar
        scalar_label = tk.Label(scalar_frame, text="Nilai Skalar:", 
                               font=('Arial', 9), fg='#ecf0f1', bg='#34495e')
        scalar_label.pack(anchor=tk.W, padx=5)
        
        # Entry untuk input skalar
        self.scalar_var = tk.StringVar(value="1.5")
        self.scalar_entry = tk.Entry(scalar_frame, textvariable=self.scalar_var,
                                    font=('Arial', 9), width=10, justify=tk.CENTER)
        self.scalar_entry.pack(pady=2)
        
        # Operasi perkalian dengan skalar konstanta
        multiply_btn = tk.Button(arith_section, text="Perkalian Skalar", command=self.multiply_scalar_operation,
                                bg='#e67e22', fg='white', font=('Arial', 9, 'bold'),
                                relief=tk.FLAT, padx=15, pady=5)
        multiply_btn.pack(fill=tk.X, pady=2)
        
        # Logic operations
        logic_section = tk.LabelFrame(scrollable_frame, text="🔧 OPERASI LOGIKA", 
                                     font=('Arial', 12, 'bold'), fg='#ecf0f1', bg='#34495e')
        logic_section.pack(fill=tk.X, padx=10, pady=5)
        
        # Hanya operasi NOT
        not_btn = tk.Button(logic_section, text="NOT (Inversi)", command=self.not_operation,
                           bg='#8e44ad', fg='white', font=('Arial', 9, 'bold'),
                           relief=tk.FLAT, padx=15, pady=5)
        not_btn.pack(fill=tk.X, pady=2)
        
        # Convolution section
        conv_section = tk.LabelFrame(scrollable_frame, text="🔍 KONVOLUSI & FILTER", 
                                    font=('Arial', 12, 'bold'), fg='#ecf0f1', bg='#34495e')
        conv_section.pack(fill=tk.X, padx=10, pady=10)
        
        filter_buttons = [
            ("Blur", self.apply_blur, '#16a085'),
            ("Sharpen", self.apply_sharpen, '#f39c12'),
            ("Edge Detection", self.apply_edge_detection, '#8e44ad'),
        ]
        
        
        for text, command, color in filter_buttons:
            btn = tk.Button(conv_section, text=text, command=command,
                           bg=color, fg='white', font=('Arial', 9, 'bold'),
                           relief=tk.FLAT, padx=15, pady=5)
            btn.pack(fill=tk.X, pady=2)
        
        # Dilatasi Section 
        self.kernel_options = {
            "Kotak 3x3 (Full 1)": np.array([[1,1,1], [1,1,1], [1,1,1]], dtype=np.uint8),
            "Salib": np.array([[0,1,0], [1,1,1], [0,1,0]], dtype=np.uint8),
            "Horizontal": np.array([[0,0,0], [1,1,1], [0,0,0]], dtype=np.uint8),
            "Vertikal": np.array([[0,1,0], [0,1,0], [0,1,0]], dtype=np.uint8),
            "Diagonal Kiri-Kanan": np.array([[0,0,1], [0,1,0], [1,0,0]], dtype=np.uint8),
            "Diagonal Kanan-Kiri": np.array([[1,0,0], [0,1,0], [0,0,1]], dtype=np.uint8),
        }
        self.selected_kernel_name = tk.StringVar(value="Kotak 3x3 (Full 1)")

        # MORFOLOGI SECTION
        morph_section = tk.LabelFrame(scrollable_frame, text="🔲 OPERASI MORFOLOGI", 
            font=('Arial', 12, 'bold'), fg='#ecf0f1', bg='#34495e')
        morph_section.pack(fill=tk.X, padx=10, pady=10)

        # Dropdown kernel
        kernel_label = tk.Label(morph_section, text="Pilih Kernel:", font=('Arial', 9),
            fg='#ecf0f1', bg='#34495e')
        kernel_label.pack(anchor=tk.W, padx=5)

        kernel_dropdown = ttk.Combobox(morph_section, textvariable=self.selected_kernel_name,
            values=list(self.kernel_options.keys()), state='readonly')
        kernel_dropdown.pack(fill=tk.X, padx=5, pady=5)

        # Tombol Dilasi
        dilate_btn = tk.Button(morph_section, text="Dilasi", command=self.apply_dilation,
            bg='#2980b9', fg='white', font=('Arial', 9, 'bold'),
            relief=tk.FLAT, padx=15, pady=5)
        dilate_btn.pack(fill=tk.X, pady=5)

        
        
        # Reset button
        self.reset_btn = tk.Button(save_section, text="Reset", command=self.reset_image,
                                  bg='#c0392b', fg='white', font=('Arial', 10, 'bold'),
                                  relief=tk.FLAT, padx=20, pady=8)
        self.reset_btn.pack(pady=5)
        
        # Right panel for images
        image_frame = tk.Frame(main_frame, bg='#2c3e50')
        image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Original image section
        original_section = tk.LabelFrame(image_frame, text="📷 GAMBAR ASLI", 
                                        font=('Arial', 12, 'bold'), fg='#ecf0f1', bg='#2c3e50')
        original_section.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(0, 5))
        
        self.original_canvas = tk.Canvas(original_section, bg='white', width=400, height=300)
        self.original_canvas.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Processed image section
        processed_section = tk.LabelFrame(image_frame, text="✨ HASIL PROSES", 
                                         font=('Arial', 12, 'bold'), fg='#ecf0f1', bg='#2c3e50')
        processed_section.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, pady=(5, 0))
        
        self.processed_canvas = tk.Canvas(processed_section, bg='white', width=400, height=300)
        self.processed_canvas.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Selamat datang! Silakan pilih gambar untuk memulai.")
        status_bar = tk.Label(self.root, textvariable=self.status_var, 
                             bg='#34495e', fg='#ecf0f1', font=('Arial', 10))
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Pilih Gambar",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.gif")]
        )
        
        if file_path:
            self.image_path = file_path
            self.original_image = cv2.imread(file_path)
            self.processed_image = self.original_image.copy()
            
            if self.original_image is not None:
                self.display_image(self.original_image, self.original_canvas)
                self.display_image(self.processed_image, self.processed_canvas)
                self.status_var.set(f"Gambar berhasil dimuat: {os.path.basename(file_path)}")
            else:
                messagebox.showerror("Error", "Gagal memuat gambar!")
    
    def display_image(self, image, canvas):
        if image is None:
            return
            
        # Get canvas dimensions
        canvas.update()
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        # Convert BGR to RGB for display
        if len(image.shape) == 3:
            display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            display_image = image
        
        # Resize image to fit canvas while maintaining aspect ratio
        h, w = display_image.shape[:2]
        aspect_ratio = w / h
        
        if canvas_width / canvas_height > aspect_ratio:
            new_height = canvas_height - 20
            new_width = int(new_height * aspect_ratio)
        else:
            new_width = canvas_width - 20
            new_height = int(new_width / aspect_ratio)
        
        display_image = cv2.resize(display_image, (new_width, new_height))
        
        # Convert to PIL Image and then to PhotoImage
        pil_image = Image.fromarray(display_image)
        photo = ImageTk.PhotoImage(pil_image)
        
        # Clear canvas and display image
        canvas.delete("all")
        canvas.create_image(canvas_width//2, canvas_height//2, image=photo)
        
        canvas.image = photo  # Keep a reference
    def show_histogram(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Silakan pilih gambar terlebih dahulu!")
            return

        try:
            # Grayscale histogram
            image_gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            hist_gray = cv2.calcHist([image_gray], [0], None, [256], [0, 256])

            # RGB histograms
            channels = ('b', 'g', 'r')
            colors = {'b': 'blue', 'g': 'green', 'r': 'red'}
            hist_rgb = {}

            for i, ch in enumerate(channels):
                hist_rgb[ch] = cv2.calcHist([self.original_image], [i], None, [256], [0, 256])

            # Tampilkan hasil dalam dua subplot
            plt.figure("Histogram Gambar", figsize=(12, 5))

            # Subplot 1: Grayscale
            plt.subplot(1, 2, 1)
            plt.plot(hist_gray, color='gray')
            plt.title("Histogram Grayscale")
            plt.xlabel("Nilai Intensitas (0-255)")
            plt.ylabel("Jumlah Piksel")
            plt.grid(True)

            # Subplot 2: RGB
            plt.subplot(1, 2, 2)
            for ch in channels:
                plt.plot(hist_rgb[ch], color=colors[ch], label=ch.upper())
            plt.title("Histogram RGB")
            plt.xlabel("Nilai Intensitas (0-255)")
            plt.ylabel("Jumlah Piksel")
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.show()

        except Exception as e:
            messagebox.showerror("Error", f"Gagal menampilkan histogram:\n{str(e)}")

    def convert_to_grayscale(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Silakan pilih gambar terlebih dahulu!")
            return
        
        self.processed_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        self.display_image(self.processed_image, self.processed_canvas)
        self.status_var.set("Gambar berhasil dikonversi ke grayscale")
    
    def convert_to_binary(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Silakan pilih gambar terlebih dahulu!")
            return
        
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        _, self.processed_image = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        self.display_image(self.processed_image, self.processed_canvas)
        self.status_var.set("Gambar berhasil dikonversi ke binary")
    
    # =========================
    # OPERASI ARITMATIKA
    # =========================
    
    def multiply_scalar_operation(self):
        """Operasi Perkalian dengan Skalar Konstanta"""
        if self.original_image is None:
            messagebox.showwarning("Warning", "Silakan pilih gambar terlebih dahulu!")
            return
        
        # Ambil nilai skalar dari input field
        try:
            scalar_value = float(self.scalar_var.get())
            
            # Validasi range nilai
            if scalar_value < 0.1 or scalar_value > 3.0:
                messagebox.showwarning("Warning", "Nilai skalar harus antara 0.1 dan 3.0!")
                return
                
        except ValueError:
            messagebox.showerror("Error", "Masukkan nilai skalar yang valid (angka)!")
            return
        
        try:
            # Konversi ke float64 untuk mencegah overflow
            img_float = self.original_image.astype(np.float64)
            
            # Operasi perkalian dengan skalar konstanta
            multiplied = img_float * scalar_value
            
            # Clip nilai agar tetap dalam range 0-255 dan konversi kembali ke uint8
            self.processed_image = np.clip(multiplied, 0, 255).astype(np.uint8)
            
            self.display_image(self.processed_image, self.processed_canvas)
            self.status_var.set(f"Operasi Perkalian Skalar (×{scalar_value}) berhasil diterapkan")
            
        except Exception as e:
            messagebox.showerror("Error", f"Terjadi kesalahan saat melakukan operasi perkalian:\n{str(e)}")
    
    # =========================
    # OPERASI LOGIKA
    # =========================
    
    def not_operation(self):
        """Operasi NOT - inversi gambar"""
        if self.original_image is None:
            messagebox.showwarning("Warning", "Silakan pilih gambar terlebih dahulu!")
            return
        
        # Operasi NOT - membalik semua bit
        self.processed_image = cv2.bitwise_not(self.original_image)
        self.display_image(self.processed_image, self.processed_canvas)
        self.status_var.set("Operasi Logika NOT (inversi) berhasil diterapkan")
    
    def apply_blur(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Silakan pilih gambar terlebih dahulu!")
            return
        
        # Gaussian blur kernel
        self.processed_image = cv2.GaussianBlur(self.original_image, (15, 15), 0)
        self.display_image(self.processed_image, self.processed_canvas)
        self.status_var.set("Filter Blur berhasil diterapkan")
    
    def apply_sharpen(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Silakan pilih gambar terlebih dahulu!")
            return
        
        # Sharpening kernel yang lebih halus
        kernel = np.array([[0, -1, 0],
                          [-1, 5, -1],
                          [0, -1, 0]], dtype=np.float32)
        
        # Alternatif: gunakan unsharp masking untuk hasil yang lebih natural
        # Blur gambar terlebih dahulu
        blurred = cv2.GaussianBlur(self.original_image, (0, 0), 1.0)
        
        # Unsharp mask = Original + alpha * (Original - Blurred)
        alpha = 1.5  # faktor penajaman
        self.processed_image = cv2.addWeighted(self.original_image, 1 + alpha, blurred, -alpha, 0)
        
        # Pastikan nilai pixel dalam range 0-255
        self.processed_image = np.clip(self.processed_image, 0, 255).astype(np.uint8)
        
        self.display_image(self.processed_image, self.processed_canvas)
        self.status_var.set("Filter Sharpen (Unsharp Masking) berhasil diterapkan")
    
    def apply_edge_detection(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Silakan pilih gambar terlebih dahulu!")
            return
        
        # Canny edge detection
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        self.processed_image = edges
        self.display_image(self.processed_image, self.processed_canvas)
        self.status_var.set("Filter Edge Detection berhasil diterapkan")
    
    def apply_dilation(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Silakan pilih gambar terlebih dahulu!")
            return

        try:
            # Konversi ke grayscale → threshold → biner
            gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

            # Ambil kernel sesuai pilihan
            kernel_name = self.selected_kernel_name.get()
            kernel = self.kernel_options[kernel_name]

            # Lakukan dilasi
            dilated = cv2.dilate(binary, kernel, iterations=1)

            self.processed_image = dilated
            self.display_image(self.processed_image, self.processed_canvas)
            self.status_var.set(f"Dilasi berhasil dengan kernel: {kernel_name}")

        except Exception as e:
            messagebox.showerror("Error", f"Gagal melakukan dilasi:\n{str(e)}")


    def save_image(self):
        if self.processed_image is None:
            messagebox.showwarning("Warning", "Tidak ada gambar hasil untuk disimpan!")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Simpan Gambar",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                cv2.imwrite(file_path, self.processed_image)
                messagebox.showinfo("Success", f"Gambar berhasil disimpan ke:\n{file_path}")
                self.status_var.set(f"Gambar disimpan: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Gagal menyimpan gambar:\n{str(e)}")
    
    def reset_image(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Tidak ada gambar untuk direset!")
            return
        
        self.processed_image = self.original_image.copy()
        self.display_image(self.processed_image, self.processed_canvas)
        self.status_var.set("Gambar berhasil direset ke kondisi awal")

def main():
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()