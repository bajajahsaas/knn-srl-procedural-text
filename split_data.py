import os
import shutil
import re
dire = 'wlpdata'
try:
    os.mkdir('wlpdata_split')
    os.mkdir('wlpdata_split/train')
    os.mkdir('wlpdata_split/test')
    os.mkdir('wlpdata_split/valid')
except:
    pass
files = set([x.split('.')[0] for x in os.listdir(dire) if \
             re.match(r'protocol_[0-9]*.(ann|txt)', x)])

trainsplit = int(0.8 * len(files))
valsplit = int(0.1* len(files))
testsplit = len(files) - trainsplit - valsplit

def move_to_dir(src, tgt, fnames):
    for fil in fnames:
        shutil.copy2(os.path.join(src, fil + '.txt'), tgt)
        shutil.copy2(os.path.join(src, fil + '.ann'), tgt)

files = list(files)
traindata = files[:trainsplit]
valdata = files[trainsplit:-testsplit]
testdata = files[-testsplit:]

move_to_dir(dire, 'wlpdata_split/train', traindata)
move_to_dir(dire, 'wlpdata_split/test', testdata)
move_to_dir(dire, 'wlpdata_split/valid', valdata)
