from bratreader.repomodel import RepoModel
import pickle

r = RepoModel("wlpdata")
numcrosssent = 0
numwithinsent = 0

for fil in r.documents:
    doc = r.documents[fil]
    for annotation in set(doc.annotations):
        for key in annotation.links:
            for linked_annotation in annotation.links[key]:
                if linked_annotation.words[0].sentkey == annotation.words[0].sentkey:
                    numwithinsent += 1
                else:
                    numcrosssent += 1

print(numwithinsent)
print(numcrosssent)


