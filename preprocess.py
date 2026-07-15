import os
import cv2
import numpy as np

# Load raw images (48x48x3), normalize, save to .npy

dataset_dir = "datasets/Faces-Datasets"
images = []
labels = []

print("=" * 50)
print("Loading Raw Images (48x48x3)")
print("=" * 50)

# Count total people
total_people = len([d for d in os.listdir(dataset_dir) 
                    if os.path.isdir(os.path.join(dataset_dir, d))])
current_person = 0

# Load images
for person_name in sorted(os.listdir(dataset_dir)):
    person_path = os.path.join(dataset_dir, person_name)
    if not os.path.isdir(person_path):
        continue

    current_person += 1
    person_images = 0

    for img_name in sorted(os.listdir(person_path)):
        img_path = os.path.join(person_path, img_name)
        
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            continue

        # BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to 48x48
        img = cv2.resize(img, (48, 48))

        # Normalize: 0-255 to 0-1
        img = img.astype(np.float32) / 255.0

        images.append(img)
        labels.append(person_name)
        person_images += 1

    print(f"[{current_person}/{total_people}] {person_name}: {person_images} images")

# Convert to numpy array
images_array = np.stack(images, axis=0)  # Shape: (N, 48, 48, 3)
print(f"\nTotal images: {images_array.shape}")
print(f"Classes: {len(np.unique(labels))}")

# Save to .npy
print("\n" + "=" * 50)
print("Saving to .NPY")
print("=" * 50)

np.save("images.npy", images_array)
print(f"OK - images.npy saved: {images_array.shape}")

# Save labels to text file
with open("labels.txt", "w") as f:
    f.write("\n".join(labels))
print(f"OK - labels.txt saved: {len(labels)} labels")

print("\n" + "=" * 50)
print("DONE!")
print("=" * 50)



