import os
import tensorflow as tf

# --- Configuration ---
# Make sure this points to your dataset folder.
data_dir = 'Cutout Files'
# ---

print(f"--- Starting TensorFlow image verification in '{data_dir}' ---")
print("This may take a moment...")
bad_files = []

# Walk through all files in the directory and its subdirectories
for dirpath, _, filenames in os.walk(data_dir):
    for filename in filenames:
        file_path = os.path.join(dirpath, filename)

        # We use a try-except block to catch the error from TensorFlow
        try:
            # 1. Read the file's raw byte content
            image_bytes = tf.io.read_file(file_path)

            # 2. Try to decode it using TensorFlow's function
            # This is the exact step that fails during training.
            tf.image.decode_image(image_bytes)

        except tf.errors.InvalidArgumentError as e:
            # This error is thrown when TensorFlow cannot decode the image
            print(f"\n!!! Found problematic file: {file_path}")
            # The error message from TF is often not very helpful, so we won't print it.
            bad_files.append(file_path)
        except Exception as e:
            # Catch any other potential errors
            print(f"\n!!! Found file with other error: {file_path}")
            print(f"    Reason: {e}")
            bad_files.append(file_path)

if not bad_files:
    print("\nVerification complete. No problematic files found for TensorFlow.")
else:
    print(f"\nVerification complete. Found {len(bad_files)} problematic file(s).")
    print("Please delete these files and run your training script again.")