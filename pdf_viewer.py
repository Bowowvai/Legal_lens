#!/usr/bin/env python3
"""
Simple PDF Viewer with Zoom using Tkinter + PyMuPDF (fitz) + Pillow
- Open a PDF (via CLI arg or File > Open)
- Zoom In / Zoom Out / 100% / Fit Width
- Next / Prev page
- Scroll the page if larger than the window

Dependencies:
  pip install pymupdf pillow

Run:
  python3 pdf_viewer.py /path/to/file.pdf
  # or just: python3 pdf_viewer.py  (then use File > Open)
"""
import io
import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox

try:
    import fitz  # PyMuPDF
except ImportError:
    print("Missing dependency: pymupdf. Install with `pip install pymupdf`.", file=sys.stderr)
    raise

try:
    from PIL import Image, ImageTk
except ImportError:
    print("Missing dependency: pillow. Install with `pip install pillow`.", file=sys.stderr)
    raise


class PDFViewer(tk.Tk):
    def __init__(self, pdf_path: str | None = None):
        super().__init__()
        self.title("PDF Viewer")
        self.geometry("1000x700")

        # State
        self.doc = None
        self.page_index = 0
        self.zoom = 1.0  # 1.0 == 100%
        self.image_tk = None  # keep reference to avoid GC

        # UI
        self._build_menu()
        self._build_toolbar()
        self._build_canvas()
        self._update_buttons()

        # Key bindings
        self.bind("<Control-plus>", lambda e: self.zoom_in())
        self.bind("<Control-minus>", lambda e: self.zoom_out())
        self.bind("<plus>", lambda e: self.zoom_in())
        self.bind("<minus>", lambda e: self.zoom_out())
        self.bind("=", lambda e: self.zoom_in())
        self.bind("-", lambda e: self.zoom_out())
        self.bind("<Left>", lambda e: self.prev_page())
        self.bind("<Right>", lambda e: self.next_page())
        self.bind("<Key-0>", lambda e: self.reset_zoom())

        if pdf_path:
            self.open_pdf(pdf_path)

    def _build_menu(self):
        menubar = tk.Menu(self)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open…", command=self.open_dialog, accelerator="Ctrl+O")
        file_menu.add_separator()
        file_menu.add_command(label="Quit", command=self.destroy, accelerator="Cmd/Ctrl+Q")
        menubar.add_cascade(label="File", menu=file_menu)
        self.config(menu=menubar)
        # shortcut
        self.bind_all("<Control-o>", lambda e: self.open_dialog())

    def _build_toolbar(self):
        bar = tk.Frame(self, bd=1, relief=tk.RAISED)
        bar.pack(side=tk.TOP, fill=tk.X)

        self.btn_prev = tk.Button(bar, text="◀ Prev", command=self.prev_page)
        self.btn_prev.pack(side=tk.LEFT, padx=2, pady=2)
        self.btn_next = tk.Button(bar, text="Next ▶", command=self.next_page)
        self.btn_next.pack(side=tk.LEFT, padx=2, pady=2)

        tk.Label(bar, text="  ").pack(side=tk.LEFT)  # spacer

        self.btn_zoom_out = tk.Button(bar, text="−", width=3, command=self.zoom_out)
        self.btn_zoom_out.pack(side=tk.LEFT, padx=(8,2), pady=2)
        self.btn_zoom_in = tk.Button(bar, text="+", width=3, command=self.zoom_in)
        self.btn_zoom_in.pack(side=tk.LEFT, padx=2, pady=2)
        self.btn_reset = tk.Button(bar, text="100%", command=self.reset_zoom)
        self.btn_reset.pack(side=tk.LEFT, padx=2, pady=2)
        self.btn_fit_width = tk.Button(bar, text="Fit Width", command=self.fit_width)
        self.btn_fit_width.pack(side=tk.LEFT, padx=2, pady=2)

        self.status = tk.Label(bar, text="Page — / —   Zoom 100%", anchor="w")
        self.status.pack(side=tk.RIGHT, padx=8)

    def _build_canvas(self):
        # Scrollable canvas
        self.canvas = tk.Canvas(self, bg="#1f2937", highlightthickness=0)  # Tailwind slate-800 for comfort
        self.hbar = tk.Scrollbar(self, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.vbar = tk.Scrollbar(self, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=self.hbar.set, yscrollcommand=self.vbar.set)

        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.vbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.hbar.pack(side=tk.BOTTOM, fill=tk.X)

        # Container for the image
        self.image_container = self.canvas.create_image(0, 0, anchor="nw")
        self.canvas.bind("<Configure>", lambda e: self._on_resize())

    # ------------------------ Actions ------------------------
    def open_dialog(self):
        path = filedialog.askopenfilename(
            title="Open PDF",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
        )
        if path:
            self.open_pdf(path)

    def open_pdf(self, path: str):
        try:
            doc = fitz.open(path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open PDF:\n{e}")
            return
        self.doc = doc
        self.page_index = 0
        self.zoom = 1.0
        self._render_page()
        self._update_buttons()
        self.title(f"PDF Viewer — {os.path.basename(path)}")

    def _render_page(self):
        if not self.doc:
            return
        # Clamp index
        self.page_index = max(0, min(self.page_index, len(self.doc)-1))
        page = self.doc.load_page(self.page_index)
        mat = fitz.Matrix(self.zoom, self.zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)

        # Convert to PIL Image
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        self.image_tk = ImageTk.PhotoImage(img)

        # Put into canvas
        self.canvas.itemconfigure(self.image_container, image=self.image_tk)
        self.canvas.config(scrollregion=(0, 0, img.width, img.height))
        self.canvas.coords(self.image_container, 0, 0)
        self._update_status()

    def _update_status(self):
        total = self.doc.page_count if self.doc else 0
        pct = int(self.zoom * 100)
        self.status.config(text=f"Page {self.page_index+1} / {total}   Zoom {pct}%")

    def _update_buttons(self):
        has_doc = self.doc is not None
        for b in (self.btn_prev, self.btn_next, self.btn_zoom_in, self.btn_zoom_out, self.btn_reset, self.btn_fit_width):
            b.config(state=tk.NORMAL if has_doc else tk.DISABLED)

    def next_page(self):
        if not self.doc:
            return
        if self.page_index < self.doc.page_count - 1:
            self.page_index += 1
            self._render_page()

    def prev_page(self):
        if not self.doc:
            return
        if self.page_index > 0:
            self.page_index -= 1
            self._render_page()

    def zoom_in(self):
        if not self.doc:
            return
        self.zoom = min(6.0, self.zoom * 1.15)
        self._render_page()

    def zoom_out(self):
        if not self.doc:
            return
        self.zoom = max(0.2, self.zoom / 1.15)
        self._render_page()

    def reset_zoom(self):
        if not self.doc:
            return
        self.zoom = 1.0
        self._render_page()

    def fit_width(self):
        if not self.doc:
            return
        page = self.doc.load_page(self.page_index)
        rect = page.rect
        # available canvas width (minus scrollbar)
        canvas_width = max(100, self.canvas.winfo_width())
        # We set zoom so that rect.width * zoom ~= canvas_width
        self.zoom = max(0.1, canvas_width / max(1, rect.width))
        self._render_page()

    def _on_resize(self):
        # Optional: if the user used Fit Width last, keep it on resize.
        # For simplicity we do nothing here; user can press Fit Width again.
        pass


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    app = PDFViewer(path)
    app.mainloop()


if __name__ == "__main__":
    main()
