import os
# 1. Thư mục chứa 100.000 ảnh của bạn
IMAGE_DIR = r"C:\Users\Admin\Desktop\OCR\images"

# 2. File text đáp án mà bạn đang có 
INPUT_LABEL_FILE = r"C:\Users\Admin\Desktop\OCR\labels.txt"

# 3. File danh sách chuẩn sẽ được tạo ra để cho AI đọc
OUTPUT_FILE = r"C:\Users\Admin\Desktop\OCR\train.txt"
count = 0
with open(INPUT_LABEL_FILE, "r", encoding="utf-8") as infile, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
    
    for line in infile:
        line = line.strip()
        if not line:
            continue
            
        # Cắt dòng thành 2 phần: tên file ảnh và chữ đáp án
        parts = line.split(maxsplit=1)
        
        if len(parts) == 2:
            filename = parts[0]
            label = parts[1]
            
            # Ghép thành đường dẫn đầy đủ (Ví dụ: C:\...\images\00001.jpg)
            full_path = os.path.join(IMAGE_DIR, filename)
            
            # Ghi ra file chuẩn
            outfile.write(f"{full_path} {label}\n")
            count += 1

print(f"Xong! Đã chuẩn hóa thành công {count} ảnh vào file {OUTPUT_FILE}.")