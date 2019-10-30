from bratreader.repomodel import RepoModel
import collections
import pickle

r = RepoModel("wlpdata")
numcrosssent = 0
numwithinsent = 0
num_frames = 0
num_links = []

for fil in r.documents:
    doc = r.documents[fil]
    for annotation in set(doc.annotations):
        label = list(annotation.labels.keys())[0]
        if label == 'Action':
            num_frames +=1
            num_links.append( sum([len(annotation.links[key]) for key in annotation.links]))

print(collections.Counter(num_links))
