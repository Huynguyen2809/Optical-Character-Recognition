import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image
import models.crnn as crnn_model
import argparse
from collections import OrderedDict

# Thiết lập đường dẫn từ dòng lệnh
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_path', required=True, help='Đường dẫn đến file trí nhớ (.pth)')
parser.add_argument('-i', '--image_path', required=True, help='Đường dẫn đến bức ảnh cần đọc')
opt = parser.parse_args()

# BẢNG CHỮ CÁI (Phải y hệt như lúc Train)

alphabet = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ '
nclass = len(alphabet) + 1

# 1. Khởi tạo mô hình AI
print(f'Đang đánh thức AI từ file: {opt.model_path} ...')
model = crnn_model.CRNN(32, 1, nclass, 256)

# Xử lý lỗi "module." nếu lúc train dùng nhiều GPU/DataParallel
state_dict = torch.load(opt.model_path)
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] if k.startswith('module.') else k
    new_state_dict[name] = v

model.load_state_dict(new_state_dict)

# Đưa lên Card đồ họa nếu có
if torch.cuda.is_available():
    model = model.cuda()

# Chuyển mô hình sang chế độ "Làm việc" (Không học nữa)
model.eval()

# 2. Xử lý bức ảnh đầu vào
converter = utils.strLabelConverter(alphabet)
transformer = dataset.resizeNormalize((100, 32))

try:
    image = Image.open(opt.image_path).convert('L') # Chuyển thành ảnh đen trắng
except Exception as e:
    print(f"Lỗi không mở được ảnh: {e}")
    exit()

image = transformer(image)
if torch.cuda.is_available():
    image = image.cuda()
image = image.view(1, *image.size())
image = Variable(image)

# 3. AI bắt đầu dự đoán
preds = model(image)

# 4. Dịch kết quả từ dạng số sang dạng chữ
_, preds = preds.max(2)
preds = preds.transpose(1, 0).contiguous().view(-1)
preds_size = Variable(torch.IntTensor([preds.size(0)]))
sim_pred = converter.decode(preds.data, preds_size.data, raw=False)

print('\n' + '='*50)
print(f'KẾT QUẢ ĐỌC ĐƯỢC:  >> {sim_pred} <<')
print('='*50 + '\n')