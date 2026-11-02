"""Image Studio - a small OpenCV image editor built with Tkinter."""
import os
import tkinter as tk
from tkinter import ttk, filedialog, simpledialog, messagebox

import cv2 # type: ignore
import numpy as np
from PIL import Image, ImageTk

# ---------- Theme ----------
BG = "#14151a"
PANEL = "#1d1f27"
CANVAS_BG = "#101116"
BTN = "#2b2e3b"
ACCENT = "#4f8cff"
ACCENT_HOVER = "#6b9fff"
ACCENT_PRESSED = "#3a6fd0"
LINE = "#2a2d3a"
TEXT = "#e8eaf0"
MUTED = "#8a90a6"
FONT = "Segoe UI"


class ImageStudio:
    MAX_HISTORY = 20

    def __init__(self, root):
        self.root = root
        root.title("Image Studio - OpenCV Editor")
        root.geometry("1150x720")
        root.minsize(900, 600)
        root.configure(bg=BG)

        self.original = None    # image as first opened (BGR)
        self.img = None         # current edited image (BGR)
        self.path = None
        self.history = []       # for undo
        self.tk_img = None
        self.scale = 1.0        # display scale (canvas px per image px)
        self.offset = (0, 0)    # top-left of the image on the canvas
        self.crop_mode = False
        self.crop_start = None
        self.crop_rect = None

        self._setup_style()
        self._build_header()
        self._build_statusbar()   # packed before body so it keeps its space
        self._build_body()
        self._bind_keys()
        self.render()

    # ======================= UI building =======================
    def _setup_style(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("Side.TFrame", background=PANEL)
        style.configure("Section.TLabel", background=PANEL, foreground=MUTED,
                        font=(FONT, 8, "bold"))
        for name, normal, hover in [
            ("Tool.TButton", BTN, ACCENT),
            ("Accent.TButton", ACCENT, ACCENT_HOVER),
        ]:
            style.configure(name, background=normal, foreground="white" if name.startswith("Accent") else TEXT,
                            bordercolor=normal, lightcolor=normal, darkcolor=normal,
                            focuscolor=normal, focusthickness=0,
                            padding=(8, 8), font=(FONT, 9))
            style.map(name, background=[("active", hover), ("pressed", ACCENT_PRESSED)],
                      foreground=[("active", "white")])

    def _build_header(self):
        header = tk.Frame(self.root, bg=BG, height=64)
        header.pack(fill="x")
        header.pack_propagate(False)

        titles = tk.Frame(header, bg=BG)
        titles.pack(side="left", padx=18)
        tk.Label(titles, text="Image Studio", bg=BG, fg=TEXT,
                 font=(FONT, 16, "bold")).pack(anchor="w")
        tk.Label(titles, text="Crop, resize, rotate and export images with OpenCV",
                 bg=BG, fg=MUTED, font=(FONT, 9)).pack(anchor="w")

        actions = tk.Frame(header, bg=BG)
        actions.pack(side="right", padx=14)
        for text, cmd, style in [
            ("Open", self.open_image, "Accent.TButton"),
            ("Save", self.save, "Tool.TButton"),
            ("Undo", self.undo, "Tool.TButton"),
            ("Reset", self.reset, "Tool.TButton"),
        ]:
            ttk.Button(actions, text=text, command=cmd, style=style).pack(
                side="left", padx=4, pady=14)

        tk.Frame(self.root, bg=LINE, height=1).pack(fill="x")

    def _build_statusbar(self):
        bar = tk.Frame(self.root, bg=PANEL, height=28)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)
        self.msg = tk.Label(bar, text="Ready", bg=PANEL, fg=MUTED,
                            font=(FONT, 9), anchor="w")
        self.msg.pack(side="left", padx=12)
        self.info = tk.Label(bar, text="", bg=PANEL, fg=MUTED, font=(FONT, 9))
        self.info.pack(side="right", padx=12)

    def _build_body(self):
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True)

        sidebar = ttk.Frame(body, style="Side.TFrame", width=280)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        self._section(sidebar, "Transform", [
            ("Crop", self.start_crop),
            ("Resize", self.resize),
            ("Flip Horizontal", lambda: self.flip(1)),
            ("Flip Vertical", lambda: self.flip(0)),
            ("Rotate CW", lambda: self.rotate(cv2.ROTATE_90_CLOCKWISE)),
            ("Rotate CCW", lambda: self.rotate(cv2.ROTATE_90_COUNTERCLOCKWISE)),
        ])
        self._section(sidebar, "Effects", [
            ("Grayscale", self.grayscale),
            ("Blur", self.blur),
        ])
        self._section(sidebar, "Duplicate & Export", [
            ("Duplicate", self.duplicate_window),
            ("Save Copies", self.save_copies),
            ("Tile Grid", self.tile_grid),
            ("Copies to PDF", self.copies_to_pdf),
        ])

        self.canvas = tk.Canvas(body, bg=CANVAS_BG, highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        self.canvas.bind("<Configure>", lambda e: self.render())
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

    def _section(self, parent, title, items):
        ttk.Label(parent, text=title.upper(), style="Section.TLabel").pack(
            anchor="w", padx=16, pady=(16, 4))
        grid = ttk.Frame(parent, style="Side.TFrame")
        grid.pack(fill="x", padx=12)
        grid.columnconfigure((0, 1), weight=1, uniform="col")
        for i, (text, cmd) in enumerate(items):
            ttk.Button(grid, text=text, command=cmd, style="Tool.TButton").grid(
                row=i // 2, column=i % 2, sticky="ew", padx=4, pady=3)

    def _bind_keys(self):
        self.root.bind("<Control-o>", lambda e: self.open_image())
        self.root.bind("<Control-s>", lambda e: self.save())
        self.root.bind("<Control-z>", lambda e: self.undo())
        self.root.bind("<Escape>", lambda e: self.cancel_crop())

    # ======================= helpers =======================
    def set_msg(self, text):
        self.msg.config(text=text)

    def has_image(self):
        if self.img is None:
            messagebox.showinfo("No image", "Please open an image first (Ctrl+O).")
            return False
        return True

    def to_pil(self):
        """Current OpenCV image (BGR) -> PIL image (RGB)."""
        return Image.fromarray(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))

    def apply(self, new_img, message):
        """Replace the current image, keeping the old one for undo."""
        self.history.append(self.img.copy())
        if len(self.history) > self.MAX_HISTORY:
            self.history.pop(0)
        self.img = new_img
        self.render()
        self.set_msg(message)

    def render(self):
        c = self.canvas
        c.delete("all")
        self.crop_rect = None
        cw, ch = c.winfo_width(), c.winfo_height()
        if self.img is None:
            c.create_text(cw // 2, ch // 2, fill=MUTED, justify="center",
                          font=(FONT, 14),
                          text="Open an image to get started\n\nCtrl+O")
            self.info.config(text="")
            return
        h, w = self.img.shape[:2]
        self.scale = min((cw - 20) / w, (ch - 20) / h, 1.0)
        dw, dh = max(1, int(w * self.scale)), max(1, int(h * self.scale))
        rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb, (dw, dh), interpolation=cv2.INTER_AREA)
        self.tk_img = ImageTk.PhotoImage(Image.fromarray(small))
        self.offset = ((cw - dw) // 2, (ch - dh) // 2)
        c.create_image(*self.offset, anchor="nw", image=self.tk_img)
        self.info.config(text=f"{w} x {h} px   |   Zoom {int(self.scale * 100)}%")

    # ======================= file actions =======================
    def open_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp *.tiff *.tif")])
        if not path:
            return
        img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            messagebox.showerror("Error", "Could not read that image.")
            return
        self.path = path
        self.original = img
        self.img = img.copy()
        self.history.clear()
        self.render()
        self.set_msg(f"Opened {os.path.basename(path)}")

    def save(self):
        if not self.has_image():
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("PDF", "*.pdf")])
        if not path:
            return
        ext = os.path.splitext(path)[1].lower() or ".png"
        try:
            if ext == ".pdf":
                self.to_pil().save(path, "PDF")
            else:
                ok, buf = cv2.imencode(ext, self.img)
                if not ok:
                    raise ValueError("encode failed")
                buf.tofile(path)
        except (cv2.error, ValueError, OSError):
            messagebox.showerror("Error", f"Could not save as '{ext}'.")
            return
        self.set_msg(f"Saved {os.path.basename(path)}")

    def undo(self):
        if self.history:
            self.img = self.history.pop()
            self.render()
            self.set_msg("Undo")
        else:
            self.set_msg("Nothing to undo")

    def reset(self):
        if self.original is not None:
            self.apply(self.original.copy(), "Reset to original")

    # ======================= transform =======================
    def start_crop(self):
        if not self.has_image():
            return
        self.crop_mode = True
        self.canvas.config(cursor="crosshair")
        self.set_msg("Crop: drag a rectangle over the image  (Esc to cancel)")

    def cancel_crop(self):
        if self.crop_mode:
            self.crop_mode = False
            self.crop_start = None
            self.canvas.config(cursor="")
            if self.crop_rect:
                self.canvas.delete(self.crop_rect)
                self.crop_rect = None
            self.set_msg("Crop cancelled")

    def on_press(self, e):
        if not self.crop_mode:
            return
        self.crop_start = (e.x, e.y)
        if self.crop_rect:
            self.canvas.delete(self.crop_rect)
        self.crop_rect = self.canvas.create_rectangle(
            e.x, e.y, e.x, e.y, outline=ACCENT, width=2, dash=(4, 2))

    def on_drag(self, e):
        if self.crop_mode and self.crop_start:
            x0, y0 = self.crop_start
            self.canvas.coords(self.crop_rect, x0, y0, e.x, e.y)

    def on_release(self, e):
        if not (self.crop_mode and self.crop_start):
            return
        x0, y0 = self.crop_start
        self.crop_start = None
        ox, oy = self.offset
        h, w = self.img.shape[:2]
        # canvas coordinates -> image coordinates
        ix0 = int((min(x0, e.x) - ox) / self.scale)
        ix1 = int((max(x0, e.x) - ox) / self.scale)
        iy0 = int((min(y0, e.y) - oy) / self.scale)
        iy1 = int((max(y0, e.y) - oy) / self.scale)
        ix0, ix1 = max(0, ix0), min(w, ix1)
        iy0, iy1 = max(0, iy0), min(h, iy1)
        self.crop_mode = False
        self.canvas.config(cursor="")
        if ix1 - ix0 < 5 or iy1 - iy0 < 5:
            self.render()
            self.set_msg("Crop area too small - try again")
            return
        self.apply(self.img[iy0:iy1, ix0:ix1].copy(),
                   f"Cropped to {ix1 - ix0} x {iy1 - iy0} px")

    def resize(self):
        if not self.has_image():
            return
        percent = simpledialog.askinteger(
            "Resize", "Scale in percent (50 = half size):",
            minvalue=1, maxvalue=500, initialvalue=100)
        if percent:
            f = percent / 100
            self.apply(cv2.resize(self.img, None, fx=f, fy=f), f"Resized to {percent}%")

    def flip(self, code):  # 1 = horizontal, 0 = vertical
        if self.has_image():
            self.apply(cv2.flip(self.img, code),
                       "Flipped horizontally" if code == 1 else "Flipped vertically")

    def rotate(self, code):
        if self.has_image():
            self.apply(cv2.rotate(self.img, code), "Rotated 90 degrees")

    # ======================= effects =======================
    def grayscale(self):
        if self.has_image():
            gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            self.apply(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), "Converted to grayscale")

    def blur(self):
        if not self.has_image():
            return
        s = simpledialog.askinteger("Blur", "Strength (1-25):",
                                    minvalue=1, maxvalue=25, initialvalue=5)
        if s:
            k = s * 2 + 1  # kernel size must be odd
            self.apply(cv2.GaussianBlur(self.img, (k, k), 0), f"Blurred (strength {s})")

    # ======================= duplicate & export =======================
    def duplicate_window(self):
        """Open an independent copy of the current image in a new window."""
        if not self.has_image():
            return
        win = tk.Toplevel(self.root)
        editor = ImageStudio(win)
        win.title("Image Studio - Copy")
        win.geometry("1150x720+60+60")
        editor.path = self.path
        editor.original = self.img.copy()
        editor.img = self.img.copy()
        editor.render()
        editor.set_msg("Working on a duplicate")

    def save_copies(self):
        """Save N duplicate files: copy_1.png, copy_2.png, ..."""
        if not self.has_image():
            return
        n = simpledialog.askinteger("Save Copies", "How many copies?",
                                    minvalue=1, maxvalue=100)
        if not n:
            return
        folder = filedialog.askdirectory(title="Choose folder for copies")
        if not folder:
            return
        ok, buf = cv2.imencode(".png", self.img)
        if not ok:
            messagebox.showerror("Error", "Could not encode the image.")
            return
        for i in range(1, n + 1):
            buf.tofile(os.path.join(folder, f"copy_{i}.png"))
        self.set_msg(f"Saved {n} copies to {folder}")

    def tile_grid(self):
        """Repeat the image in a rows x columns grid."""
        if not self.has_image():
            return
        rows = simpledialog.askinteger("Tile Grid", "Rows:", minvalue=1, maxvalue=10)
        if not rows:
            return
        cols = simpledialog.askinteger("Tile Grid", "Columns:", minvalue=1, maxvalue=10)
        if not cols:
            return
        self.apply(np.tile(self.img, (rows, cols, 1)), f"Tiled {rows} x {cols}")

    def copies_to_pdf(self):
        """Save N copies of the image as one multi-page PDF."""
        if not self.has_image():
            return
        n = simpledialog.askinteger("Copies to PDF", "How many pages (copies)?",
                                    minvalue=1, maxvalue=100)
        if not n:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".pdf", filetypes=[("PDF", "*.pdf")])
        if not path:
            return
        pages = [self.to_pil() for _ in range(n)]
        pages[0].save(path, "PDF", save_all=True, append_images=pages[1:])
        self.set_msg(f"Saved {n}-page PDF: {os.path.basename(path)}")


if __name__ == "__main__":
    root = tk.Tk()
    ImageStudio(root)
    root.mainloop()