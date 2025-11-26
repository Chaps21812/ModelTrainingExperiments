def write_count(filename, image_count, annotation_count, counter):
    with open(filename, 'w') as f:
        f.write(f"Images: {image_count}, Annotations: {annotation_count}.\n")
        for item, count in counter.items():
            f.write(f"{item}: {count} \n")
