import os
import tkinter as tk
from tkinter import filedialog
from music21 import converter

def convert_krn_to_xml(folder_path):
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.krn'):
            krn_file = os.path.join(folder_path, file_name)
            musicxml_file = krn_file.replace('.krn', '.xml')
            converter.parse(krn_file).write('musicxml', musicxml_file)
    print("Conversion completed successfully.")

def select_folder():
    folder_path = filedialog.askdirectory()
    if folder_path:
        convert_krn_to_xml(folder_path)

# Create the GUI window
window = tk.Tk()
window.title("Convert .krn to .xml")

# Create and position the upload button
upload_button = tk.Button(window, text="Upload Folder", command=select_folder)
upload_button.pack(pady=20)

# Run the GUI event loop
window.mainloop()
