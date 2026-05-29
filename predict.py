import os

GRAPE_DIR = r"agri_data\grapes"

print("\n📂 SEARCHING ALL FOLDERS:\n")

for root, dirs, files in os.walk(GRAPE_DIR):

    image_files = [
        f for f in files
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if len(image_files) > 0:

        print("FOUND:")
        print(root)

        print("IMAGES:", len(image_files))
        print("-" * 50)