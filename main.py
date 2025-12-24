import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import torch
from torchvision import transforms

# --------------------------------------------
# DEVICE CONFIGURATION
# --------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------
# LOAD ANIMEGANv2 MODEL
# --------------------------------------------
def load_animegan_model():
    try:
        model = torch.hub.load(
            'bryandlee/animegan2-pytorch:main',
            'generator',
            pretrained='face_paint_512_v2'
        )
        model.to(device).eval()
        print("AnimeGANv2 model loaded.")
        return model
    except Exception as e:
        print(f"Error loading AnimeGAN model: {e}")
        return None

anime_model = load_animegan_model()

# --------------------------------------------
# PILLOW RESAMPLING
# --------------------------------------------
try:
    RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE = Image.LANCZOS

# --------------------------------------------
# AnimeGAN FUNCTION
# --------------------------------------------
def apply_anime_style(img_bgr):
    if anime_model is None:
        return img_bgr
    try:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        tfm = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        tensor = tfm(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            out_tensor = anime_model(tensor)[0].cpu() 
        out_tensor = out_tensor * 0.5 + 0.5
        out_pil = transforms.ToPILImage()(out_tensor)
        out_np = np.array(out_pil)
        out_bgr = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)
        h, w = img_bgr.shape[:2]
        out_bgr = cv2.resize(out_bgr, (w, h))
        return out_bgr
    except Exception as e:
        print(f"AnimeGAN Error: {e}")
        return img_bgr

# --------------------------------------------------
# REFINED PROFESSIONAL PINK THEME
# --------------------------------------------------
class Theme:
    COLOR_BG         = "#fdf6f7"   # Very soft pink background
    COLOR_SIDEBAR    = "#f8e4e8"   # Slightly deeper for sidebar
    COLOR_CARD       = "#ffffff"   # White cards for previews
    COLOR_BORDER     = "#e8c7cc"   # Soft border
    COLOR_ACCENT     = "#a53e56"   # Deep rose accent for buttons
    COLOR_HOVER      = "#c14d66"   # Hover shade
    COLOR_TEXT_MAIN  = "#5a2d3a"   # Dark elegant text
    COLOR_TEXT_SUB   = "#824b57"   # Subtle secondary text

    @staticmethod
    def get_fonts():
        return {
            "header": ctk.CTkFont("Poppins SemiBold", 32),
            "title": ctk.CTkFont("Poppins Medium", 18),
            "body": ctk.CTkFont("Poppins", 14),
            "button": ctk.CTkFont("Poppins Medium", 14),
            "mono": ctk.CTkFont("Consolas", 11),
        }

# --------------------------------------------------
# MAIN APPLICATION
# --------------------------------------------------
class CartoonifyApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Cartoonify Me")
        self.geometry("1400x900")
        self.minsize(1200, 800)
        self.configure(fg_color=Theme.COLOR_BG)

        self.fonts = Theme.get_fonts()
        self.original_image = None
        self.cartoon_image = None

        self.setup_ui()

    def setup_ui(self):
        # Header
        header = ctk.CTkLabel(self, text="Cartoonify Me", font=self.fonts["header"],
                              text_color=Theme.COLOR_ACCENT)
        header.pack(pady=20)

        # Main container
        main_frame = ctk.CTkFrame(self, fg_color="transparent")
        main_frame.pack(fill="both", expand=True, padx=30, pady=(0, 20))

        # Sidebar (controls)
        sidebar = ctk.CTkFrame(main_frame, width=300, corner_radius=20,
                               fg_color=Theme.COLOR_SIDEBAR,
                               border_width=2, border_color=Theme.COLOR_BORDER)
        sidebar.pack(side="left", fill="y", padx=(0, 20))
        sidebar.pack_propagate(False)

        ctk.CTkLabel(sidebar, text="Controls", font=self.fonts["title"],
                     text_color=Theme.COLOR_ACCENT).pack(pady=(20, 10))

        # File input
        input_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        input_frame.pack(fill="x", padx=20, pady=10)

        self.path_entry = ctk.CTkEntry(input_frame, placeholder_text="No image selected...",
                                       height=40, font=self.fonts["body"])
        self.path_entry.pack(fill="x", side="left", expand=True, padx=(0, 10))

        browse_btn = ctk.CTkButton(input_frame, text="Browse", width=100,
                                   fg_color=Theme.COLOR_ACCENT, hover_color=Theme.COLOR_HOVER,
                                   command=self.browse_image)
        browse_btn.pack(side="right")

        # Effect buttons
        effects_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        effects_frame.pack(fill="both", expand=True, padx=20, pady=20)

        buttons = [
            ("Cartoonify", self.cartoonify_image),
            ("Anime Style", self.apply_anime),
            ("Pencil Sketch", self.apply_pencil_sketch),
            ("Edge Only", self.apply_edge_only),
            ("Save Output", self.save_output),
        ]

        for text, cmd in buttons:
            btn = ctk.CTkButton(effects_frame, text=text, height=45,
                                font=self.fonts["button"],
                                fg_color=Theme.COLOR_ACCENT,
                                hover_color=Theme.COLOR_HOVER,
                                command=cmd)
            btn.pack(fill="x", pady=8)

        # Preview area
        preview_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        preview_frame.pack(side="right", fill="both", expand=True)

        preview_frame.grid_rowconfigure(0, weight=1)
        preview_frame.grid_columnconfigure((0, 1), weight=1)

        # Original preview
        orig_card = ctk.CTkFrame(preview_frame, fg_color=Theme.COLOR_CARD,
                                 corner_radius=20, border_width=2, border_color=Theme.COLOR_BORDER)
        orig_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        ctk.CTkLabel(orig_card, text="Original Image", font=self.fonts["title"],
                     text_color=Theme.COLOR_TEXT_MAIN).pack(pady=15)
        self.original_preview = ctk.CTkLabel(orig_card, text="No Image Loaded",
                                             text_color=Theme.COLOR_TEXT_SUB)
        self.original_preview.pack(expand=True)

        # Output preview
        out_card = ctk.CTkFrame(preview_frame, fg_color=Theme.COLOR_CARD,
                                corner_radius=20, border_width=2, border_color=Theme.COLOR_BORDER)
        out_card.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        ctk.CTkLabel(out_card, text="Processed Output", font=self.fonts["title"],
                     text_color=Theme.COLOR_TEXT_MAIN).pack(pady=15)
        self.cartoon_preview = ctk.CTkLabel(out_card, text="No Output Yet",
                                            text_color=Theme.COLOR_TEXT_SUB)
        self.cartoon_preview.pack(expand=True)

        # Console / Log
        console_frame = ctk.CTkFrame(self, fg_color=Theme.COLOR_SIDEBAR,
                                    corner_radius=20, border_width=2, border_color=Theme.COLOR_BORDER)
        console_frame.pack(fill="x", padx=30, pady=(0, 20))

        ctk.CTkLabel(console_frame, text="Log Console", font=self.fonts["body"],
                     text_color=Theme.COLOR_ACCENT).pack(anchor="w", padx=20, pady=(15, 5))

        self.progress = ctk.CTkProgressBar(console_frame, height=8,
                                           progress_color=Theme.COLOR_ACCENT)
        self.progress.set(0)
        self.progress.pack(fill="x", padx=20, pady=(0, 10))

        self.console = ctk.CTkTextbox(console_frame, height=120,
                                      font=self.fonts["mono"], text_color=Theme.COLOR_TEXT_MAIN)
        self.console.pack(fill="x", padx=20, pady=(0, 15))

    # ---------------- Helpers ----------------
    def log(self, msg):
        self.console.insert("end", f">> {msg}\n")
        self.console.see("end")

    def show_image(self, label, img_cv):
        if img_cv is None:
            return
        pil = Image.fromarray(img_cv)
        pil = pil.resize((500, 450), RESAMPLE)
        tk_img = ImageTk.PhotoImage(pil)
        label.configure(image=tk_img, text="")
        label.image = tk_img  # Keep reference

    def browse_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png")])
        if path:
            self.path_entry.delete(0, "end")
            self.path_entry.insert(0, path)
            img = cv2.imread(path)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.original_image = img_rgb
                self.show_image(self.original_preview, img_rgb)
                self.log(f"Loaded: {path}")
            else:
                self.log("Failed to load image.")

    # ---------------- Effects ----------------
    def cartoonify_image(self):
        if self.original_image is None:
            self.log("❌ Load an image first.")
            return
        self.log("Applying cartoon effect...")
        self.progress.set(0.3)
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)
        blur = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, 9, 9)
        color = cv2.bilateralFilter(self.original_image, 9, 200, 200)
        cartoon = cv2.bitwise_and(color, color, mask=edges)
        self.cartoon_image = cartoon
        self.show_image(self.cartoon_preview, cartoon)
        self.progress.set(1)
        self.log("✨ Cartoonify complete!")

    def apply_anime(self):
        if self.original_image is None:
            self.log("❌ Load an image first.")
            return
        self.log("Applying Anime style...")
        self.progress.set(0.3)
        bgr = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2BGR)
        anime_bgr = apply_anime_style(bgr)
        anime_rgb = cv2.cvtColor(anime_bgr, cv2.COLOR_BGR2RGB)
        self.cartoon_image = anime_rgb
        self.show_image(self.cartoon_preview, anime_rgb)
        self.progress.set(1)
        self.log("🌸 Anime style applied!")

    def apply_pencil_sketch(self):
        if self.original_image is None:
            self.log("❌ Load an image first.")
            return
        self.log("Applying pencil sketch...")
        self.progress.set(0.5)
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)
        inv = 255 - gray
        blur = cv2.GaussianBlur(inv, (21, 21), 0)
        sketch = cv2.divide(gray, 255 - blur, scale=256)
        sketch_rgb = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)
        self.cartoon_image = sketch_rgb
        self.show_image(self.cartoon_preview, sketch_rgb)
        self.progress.set(1)
        self.log("✏️ Pencil sketch complete!")

    def apply_edge_only(self):
        if self.original_image is None:
            self.log("❌ Load an image first.")
            return
        self.log("Generating edges...")
        self.progress.set(0.5)
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 80, 160)
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        self.cartoon_image = edges_rgb
        self.show_image(self.cartoon_preview, edges_rgb)
        self.progress.set(1)
        self.log("🔲 Edge map generated!")

    def save_output(self):
        if self.cartoon_image is None:
            self.log("❌ No output to save.")
            return
        file = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if file:
            out_bgr = cv2.cvtColor(self.cartoon_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(file, out_bgr)
            self.log(f"💾 Saved: {file}")

# --------------------------------------------------
# RUN APP
# --------------------------------------------------
if __name__ == "__main__":
    ctk.set_appearance_mode("light")
    app = CartoonifyApp()
    app.mainloop()