import os
import cv2
import shutil
import numpy as np
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from collections import Counter
import matplotlib.pyplot as plt
import insightface
from insightface.app import FaceAnalysis

# Parameters
input_folder = "/input folder path/"
output_folder = "/output folder path/"
min_face_size = 30
debug_mode = True
os.makedirs(output_folder, exist_ok=True)

# Step 1: Initialize InsightFace
print("Step 1: Loading InsightFace model (ArcFace-R100)...")
app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])  # Uses ArcFace-R100
app.prepare(ctx_id=0, det_size=(640, 640))

face_data = []  # Store: embedding, image_path, face_area

# Step 2: Detect and embed faces
print("Step 2: Detecting and embedding faces...")
for image_name in tqdm(os.listdir(input_folder)):
    image_path = os.path.join(input_folder, image_name)
    if not image_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    try:
        img = cv2.imread(image_path)
        faces = app.get(img)
        for face in faces:
            box = face.bbox.astype(int)
            w, h = box[2] - box[0], box[3] - box[1]
            if w < min_face_size or h < min_face_size:
                continue
            embedding = face.embedding
            face_area = w * h
            face_data.append({
                "embedding": embedding,
                "image_path": image_path,
                "face_area": face_area
            })
    except Exception as e:
        print(f"?? Error processing {image_name}: {e}")

if not face_data:
    raise ValueError("No faces found. Check the input folder and images.")

print(f"? Total faces extracted: {len(face_data)}")

# Step 3: Clustering
print("Step 3: Clustering...")
embeddings = np.vstack([fd["embedding"] for fd in face_data])

eps_value = 0.45  # You can tune this value based on visualization
clustering_model = DBSCAN(eps=eps_value, min_samples=2, metric="cosine")
labels = clustering_model.fit_predict(embeddings)

# Summary stats
unique_labels = set(labels)
n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
print(f"Clusters found: {n_clusters}")
print(f"Unclustered faces: {list(labels).count(-1)}")

# Optional: Plot cluster count vs eps
if debug_mode:
    def evaluate_eps_range():
        results = []
        for eps in np.arange(0.3, 0.6, 0.05):
            db = DBSCAN(eps=eps, min_samples=2, metric='cosine').fit(embeddings)
            labels_eps = db.labels_
            n_clusters_eps = len(set(labels_eps)) - (1 if -1 in labels_eps else 0)
            noise = list(labels_eps).count(-1)
            results.append((eps, n_clusters_eps, noise))
        return results

    results = evaluate_eps_range()
    plt.plot([r[0] for r in results], [r[1] for r in results], 'b.-', label='Clusters')
    plt.plot([r[0] for r in results], [r[2] for r in results], 'r.-', label='Noise')
    plt.xlabel('Epsilon')
    plt.ylabel('Count')
    plt.title('Clustering Tuning')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, "eps_tuning.png"))
    plt.close()

# Step 4: Organize output
print("Step 4: Saving clustered images...")

unclustered_folder = os.path.join(output_folder, "unclustered")
os.makedirs(unclustered_folder, exist_ok=True)

person_to_images = {}
for idx, (label, data) in enumerate(zip(labels, face_data)):
    filename = os.path.basename(data["image_path"])
    if label == -1:
        dest_path = os.path.join(unclustered_folder, f"{os.path.splitext(filename)[0]}_face{idx}.jpg")
        shutil.copy2(data["image_path"], dest_path)
    else:
        cluster_folder = os.path.join(output_folder, f"person_{label}")
        os.makedirs(cluster_folder, exist_ok=True)
        dest_path = os.path.join(cluster_folder, filename)
        if os.path.exists(dest_path):
            base, ext = os.path.splitext(filename)
            counter = 1
            while os.path.exists(dest_path):
                dest_path = os.path.join(cluster_folder, f"{base}_{counter}{ext}")
                counter += 1
        shutil.copy2(data["image_path"], dest_path)
        person_to_images.setdefault(label, set()).add(data["image_path"])

# Step 5: Summary
with open(os.path.join(output_folder, "summary.txt"), "w") as f:
    f.write(f"Total images: {len(set(fd['image_path'] for fd in face_data))}\n")
    f.write(f"Total faces detected: {len(face_data)}\n")
    f.write(f"Clusters found: {n_clusters}\n")
    f.write(f"Unclustered faces: {list(labels).count(-1)}\n")
    for label, count in Counter(labels).items():
        if label != -1:
            f.write(f"Person {label}: {count} faces\n")

print(f"\n?? Done! Results saved to: {output_folder}")
 
