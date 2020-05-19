import torch
import numpy as np

batch = 1
n = 10
context_size = 10
dim = 500
num_classes = 13

prototypes = torch.rand((num_classes+1, dim))
proto_labels = torch.randint(0,num_classes+1,size=(num_classes+1,))

context = torch.rand((batch, context_size, dim))
context_labels = torch.randint(0,num_classes+1,size= (batch, context_size))  # generates [0,num_classes]

# print(context.size())
# print(context_labels.size())

# method 1
# context = torch.cat((context, prototypes.unsqueeze(0)), dim=1)
# context_labels = torch.cat((context_labels, proto_labels.unsqueeze(0)), dim=1)

# missing label will be different per batch (but we use bz = 1)
all_labels = np.arange(num_classes+1)
present_labels = torch.unique(context_labels.flatten()).cpu().numpy()
missing_labels = np.setdiff1d(all_labels, present_labels)

# print(missing_labels)
select_prototypes = prototypes[missing_labels, :]
select_proto_labels = proto_labels[missing_labels]

# print(select_prototypes.size())
context = torch.cat((context, select_prototypes.unsqueeze(0)), dim=1)
context_labels = torch.cat((context_labels, select_proto_labels.unsqueeze(0)), dim=1)

# print(context.size())
# print(context_labels.size())