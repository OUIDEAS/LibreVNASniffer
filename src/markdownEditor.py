import tkinter as tk
from tkinter import filedialog, messagebox
import os


class MarkdownEditor(tk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.current_file = None

        # Text area with default text "markdown"
        # Create a title label
        title_label = tk.Label(self, text="Notes", font=("Helvetica", 12))
        title_label.pack(pady=5)  # Add some padding around the title

        self.text_area = tk.Text(
            self,
            wrap="word",
            font=("Arial", 12),
            height=5,
            width=45,
        )
        self.text_area.pack(expand=True, fill="both", padx=5, pady=5, side="left")

    def load_file(self, dir):
        if dir:
            self.current_file = dir
            if not os.path.exists(dir + "/notes.md"):
                with open(dir + "/notes.md", "w", encoding="utf-8") as file:
                    file.write("")  # Create an empty file
            try:
                with open(dir + "/notes.md", "r", encoding="utf-8") as file:
                    content = file.read()
                self.text_area.delete("1.0", tk.END)
                self.text_area.insert(tk.END, content)
            except Exception as e:
                # messagebox.showerror("Error", f"Failed to open markdown file: {e}")
                print(f"Failed to find markdown file: {e}")

    def save_file(self, dir):
        if not self.current_file:
            messagebox.showerror(
                "Error", f"Failed to find markdown file for saving t: {e}"
            )
        if self.current_file:
            try:
                with open(
                    self.current_file + "/notes.md", "w", encoding="utf-8"
                ) as file:
                    content = self.text_area.get("1.0", tk.END).strip()
                    file.write(content)
                messagebox.showinfo("Success", "File saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save file: {e}")
