from __future__ import print_function
from __future__ import division

import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
from torch.nn import CTCLoss
import os
import utils
import dataset

# Đổi tên import để tránh xung đột với biến mô hình
import models.crnn as crnn_model

parser = argparse.ArgumentParser()
parser.add_argument('--trainRoot', required=True, help='path to dataset')
parser.add_argument('--valRoot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
parser.add_argument('--nepoch', type=int, default=25, help='number of epochs')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--pretrained', default='', help="path to pretrained model")
#parser.add_argument('--alphabet', type=str, default='0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ- ')
parser.add_argument('--alphabet', type=str, default='0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ ')
parser.add_argument('--expr_dir', default='expr', help='Where to store samples and models')
parser.add_argument('--displayInterval', type=int, default=500, help='Interval to be displayed')
parser.add_argument('--n_test_disp', type=int, default=10, help='Number of samples to display')
parser.add_argument('--valInterval', type=int, default=500, help='Interval to be displayed')
parser.add_argument('--saveInterval', type=int, default=500, help='Interval to be displayed')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
parser.add_argument('--adam', action='store_true', help='Use adam')
parser.add_argument('--adadelta', action='store_true', help='Use adadelta')
parser.add_argument('--keep_ratio', action='store_true', help='keep ratio for resize')
parser.add_argument('--manualSeed', type=int, default=1234, help='reproduce experiment')
parser.add_argument('--random_sample', action='store_true', help='random sampler')
opt = parser.parse_args()

if not os.path.exists(opt.expr_dir):
    os.makedirs(opt.expr_dir)

random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

#cudnn.benchmark = True
cudnn.benchmark = False
torch.backends.cudnn.enabled = False
# Data loading
train_dataset = dataset.lmdbDataset(root=opt.trainRoot)
assert train_dataset
if not opt.random_sample:
    sampler = dataset.randomSequentialSampler(train_dataset, opt.batchSize)
else:
    sampler = None
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=opt.batchSize,
    shuffle=False, sampler=sampler,
    num_workers=int(opt.workers),
    collate_fn=dataset.alignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio=opt.keep_ratio))

test_dataset = dataset.lmdbDataset(
    root=opt.valRoot, transform=dataset.resizeNormalize((100, 32)))

nclass = len(opt.alphabet) + 1
nc = 1
converter = utils.strLabelConverter(opt.alphabet)
criterion = CTCLoss()

# Khởi tạo trọng số
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# Tạo mô hình
crnn = crnn_model.CRNN(opt.imgH, nc, nclass, opt.nh)
crnn.apply(weights_init)

if opt.pretrained != '':
    print('loading pretrained model from %s' % opt.pretrained)
    crnn.load_state_dict(torch.load(opt.pretrained))

if opt.cuda:
    crnn.cuda()
    crnn = torch.nn.DataParallel(crnn, device_ids=range(opt.ngpu))
    criterion = criterion.cuda()

# Loss averager
loss_avg = utils.averager()

# Setup optimizer
if opt.adam:
    optimizer = optim.Adam(crnn.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
elif opt.adadelta:
    optimizer = optim.Adadelta(crnn.parameters())
else:
    optimizer = optim.RMSprop(crnn.parameters(), lr=opt.lr)

def val(net, dataset, criterion, max_iter=100):
    print('Start val')
    for p in crnn.parameters():
        p.requires_grad = False
    net.eval()
    data_loader = torch.utils.data.DataLoader(
        dataset, shuffle=True, batch_size=opt.batchSize, num_workers=int(opt.workers))
    val_iter = iter(data_loader)
    i = 0
    n_correct = 0
    loss_avg = utils.averager()
    max_iter = min(max_iter, len(data_loader))
    for i in range(max_iter):
        data = next(val_iter)
        cpu_images, cpu_texts = data
        batch_size = cpu_images.size(0)
        t, l = converter.encode(cpu_texts)
        image = cpu_images.cuda() if opt.cuda else cpu_images
        text = t.cuda() if opt.cuda else t
        length = l.cuda() if opt.cuda else l
        preds = crnn(image)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        cost = criterion(preds, text, preds_size, length) / batch_size
        loss_avg.add(cost)
        _, preds = preds.max(2)
        if preds.dim() == 3: # Chỉ squeeze nếu nó vẫn còn 3 chiều
            preds = preds.squeeze(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        for pred, target in zip(sim_preds, cpu_texts):
            if pred == target.lower():
                n_correct += 1
    accuracy = n_correct / float(max_iter * opt.batchSize)
    print('Test loss: %f, accuracy: %f' % (loss_avg.val(), accuracy))

def trainBatch(net, criterion, optimizer):
    data = next(train_iter)
    cpu_images, cpu_texts = data
    batch_size = cpu_images.size(0)
    t, l = converter.encode(cpu_texts)
    image = cpu_images.cuda() if opt.cuda else cpu_images
    text = t.cuda() if opt.cuda else t
    length = l.cuda() if opt.cuda else l
    preds = crnn(image)
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
    cost = criterion(preds, text, preds_size, length) / batch_size
    crnn.zero_grad()
    cost.backward()
    optimizer.step()
    return cost

# Vòng lặp chính
for epoch in range(opt.nepoch):
    train_iter = iter(train_loader)
    i = 0
    while i < len(train_loader):
        for p in crnn.parameters():
            p.requires_grad = True
        crnn.train()
        cost = trainBatch(crnn, criterion, optimizer)
        loss_avg.add(cost)
        i += 1
        if i % opt.displayInterval == 0:
            print('[%d/%d][%d/%d] Loss: %f' % (epoch, opt.nepoch, i, len(train_loader), loss_avg.val()))
            loss_avg.reset()
        if i % opt.valInterval == 0:
            val(crnn, test_dataset, criterion)
        if i % opt.saveInterval == 0:
            torch.save(crnn.state_dict(), '{0}/netCRNN_{1}_{2}.pth'.format(opt.expr_dir, epoch, i))