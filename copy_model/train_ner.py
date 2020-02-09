import nltk
import pickle
import sklearn
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.metrics import precision_score, recall_score, f1_score
# from sklearn.cross_validation import cross_val_score
# from sklearn.grid_search import RandomizedSearchCV

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics

from biobert import *
from bert_crf import *

with open('entity_types.txt', 'r') as f:
    entities = f.read().splitlines()
all_labels = ['O'] + ['I_' + x for x in entities] + ['B_'+x for x in entities]
label_dic = {l:i for i, l in enumerate(all_labels)}

with open('tagger_dict.pkl', 'wb') as f:
    pickle.dump({'list':all_labels, 'dict':label_dic}, f)

tokenizer, _ = getscibertmodel()

def get_sent_labels(sent, entities):
    tokenized = tokenizer.tokenize(sent)
    enc = tokenizer.encode(sent, add_special_tokens = True)
    labels = ['O' for i in enc]
    for ent, typ, _, span in entities:
        for i in range(span[0]+1, span[1]+1):
            labels[i] = 'I_' + typ
        labels[span[0]] = 'B_' + typ
    return tokenized,enc,labels

def get_batch(dataset):
    for i in np.random.permutation(len(dataset)):
        el = dataset[i]
        toks,enc, labs = get_sent_labels(el['query_sent'], el['entities'])
        lab_ids = [label_dic[i] for i in labs]
        yield torch.tensor([enc]).cuda(), torch.tensor([lab_ids]).cuda()

def eval(dataset):
    pred = []
    target = []
    for enc, lab in get_batch(dataset):
        with torch.no_grad():
            pred_labs = tagger(enc)[0]
        target.append(lab.cpu().detach()[0,:].numpy())
        pred.append(pred_labs)
    target = np.concatenate(target)
    pred = np.concatenate(pred)
    print(float(np.sum(np.equal(target, pred)))/target.shape[0])
        
    labels = list(range(1,len(all_labels)))
    micro_f1 = str(round(f1_score(target, pred, labels=labels, average="micro"), 4)*100)
    macro_f1 = str(round(f1_score(target, pred, labels=labels, average="macro"), 4)*100)
    print('Micro F1 = %s \nMacro F1 = %s'%(micro_f1, macro_f1))





with open('val.pkl', 'rb') as f:
    val_data = pickle.load(f)

with open('train.pkl', 'rb') as f:
    train_data = pickle.load(f)

tagger = BERT_CRF(2*19+1).cuda()
print(val_data[0]['query_sent'])
toks,enc, labs = get_sent_labels(val_data[0]['query_sent'], val_data[0]['entities'])
print(toks)
print(enc)
print(labs)
batch_size = 16
learning_rate = 1e-4

toks,enc, labs = get_sent_labels(val_data[0]['query_sent'], val_data[0]['entities'])
pred = tagger(torch.tensor([enc]).cuda())
print(labs)
print([all_labels[x] for x in pred[0]])



optimizer = torch.optim.Adam(tagger.parameters(), lr=learning_rate, \
                                 weight_decay=0.001)
eval(val_data)
for epoch in range(10):
    i = 1
    optimizer.zero_grad()
    total_loss = 0.0
    for enc, lab in get_batch(train_data):
        if i%batch_size ==0:
            optimizer.step()
            optimizer.zero_grad()
            print('Batch # %d done'%(i/batch_size))
            print('Total loss = %f'%total_loss)
            total_loss = 0
        loss = tagger(enc, lab)
        loss.backward()
        optimizer.step()
        total_loss += loss.cpu().detach().numpy()
        # print(loss.cpu().detach().numpy())
        i+=1
    print('Epoch # %d done'%epoch)
    toks,enc, labs = get_sent_labels(val_data[0]['query_sent'], val_data[0]['entities'])
    pred = tagger(torch.tensor([enc]).cuda())
    print(labs)
    print([all_labels[x] for x in pred[0]])
    torch.save(tagger.state_dict(), 'bert_crf.pt')
    eval(val_data)
    # print(pred)


# torch.save(tagger.state_dict(), 'bert_crf.pt')
with open('tagger_dict.pkl', 'wb') as f:
    pickle.dump({'list':all_labels, 'dict':label_dic}, f)




        

