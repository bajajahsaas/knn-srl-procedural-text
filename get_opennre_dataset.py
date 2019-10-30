from bratreader.repomodel import RepoModel
import collections
import pickle
import json

r = RepoModel("wlpdata")
numcrosssent = 0
numwithinsent = 0
num_frames = 0
num_links = []


def get_words(annotation):
    return ' '.join([x.form for x in annotation.words])
relset = set()
docs = list(r.documents.keys())
trainlen = int(0.8 * len(docs))
traindata = docs[:trainlen]
testdata = docs[trainlen:]

def process(files, outputfile):
    dataset = []
    for fil in files:
        doc = r.documents[fil]
        for annotation in set(doc.annotations):
            for key in annotation.links:
                for tail in annotation.links[key]:
                    dataset.append({
                            'sentence':get_words(doc.sentences[annotation.words[0].sentkey]),
                            'head' : {'word': get_words(annotation), 'id': \
                                      '%s_%s'%(fil,annotation.id)},
                        'tail' : {'word': get_words(tail), 'id' : \
                                  '%s_%s'%(fil, tail.id)},
                        'relation': key
                    })
                relset.add(key)
    
    with open(outputfile, 'w') as f:
        json.dump(dataset, f, indent = 4)
process(traindata, 'train.json')
process(testdata, 'test.json')
dic = {'NA' : 0}
for i, rel in enumerate(relset):
    dic[rel] = i + 1

with open('rel2id.json', 'w') as f:
    json.dump(dic, f, indent = 4)
