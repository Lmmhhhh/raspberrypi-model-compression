import kagglehub

# Download latest version
path = kagglehub.dataset_download("tusonggao/imagenet-train-subset-100k")

print("Path to dataset files:", path)