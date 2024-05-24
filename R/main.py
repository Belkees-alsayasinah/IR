# قراءة المتجهات من ملف pkl
import pickle

vectors_path = "vectors.pkl"
with open(vectors_path, "rb") as f:
    saved_vectors = pickle.load(f)

# طباعة بعض القيم من المتجهات
for i, vector in enumerate(saved_vectors):
    print(f"Vector {i}: {vector}")
