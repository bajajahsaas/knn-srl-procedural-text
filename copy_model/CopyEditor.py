import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dims):
        super(MLP, self).__init__()
        all_dims = [input_dims] + hidden_dims + [output_dims]
        layers = []
        for i in range(len(hidden_dims)):
            layers.append(nn.Linear(all_dims[i], all_dims[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(all_dims[-2], all_dims[-1]))
        self.network = nn.Sequential(*layers)

    def forward(self, input):
        return self.network(input)


class AttentionDist(nn.Module):
    def __init__(self, dim, num_classes):
        super(AttentionDist, self).__init__()
        self.dim = dim
        self.num_classes = num_classes

    def forward(self, queries, context, context_labels, mask):
        # queries: Batch x n x dim
        # context: Batch x context_size x dim
        # context_labels: Batch x context_size  (which class)
        # mask: Batch x context_size // because batch may have different
        #               context sizes

        # output: Context vector, Probability distribution over numclasses+1 
        #                   +1 for none class which is always 0

        # Batch x n x context_size
        dotprod = torch.sum(queries.unsqueeze(2) * context.unsqueeze(1), dim=-1) * mask.unsqueeze(1)# \
                        # +(1.0-mask.unsqueeze(1))*(-np.inf)
        l_softmax = F.log_softmax(dotprod, dim = -1)

        # Batch x n x context_size
        context_vector = torch.sum(torch.exp(l_softmax.unsqueeze(-1)) * context.unsqueeze(1), dim = 2)

        # Batch x n x (num_classes+1)
        _, n, __ = l_softmax.size()
        onehot_labels =\
            F.one_hot(context_labels, self.num_classes+1).unsqueeze(1).repeat((1,n,1,1))
        onehot_logprobs = onehot_labels * l_softmax.unsqueeze(-1)
        onehot_logprobs[torch.where(onehot_labels == 0)] = -float('Inf')


        l_softmax_classes = torch.logsumexp(onehot_logprobs, dim = 2)
        return context_vector, l_softmax_classes


class RelationEmbedding(nn.Module):
    def __init__(self, emb_dim,num_entities,type_embedding_dim, output_dim,
                 hidden_dims = [], use_entity=False):
        super(RelationEmbedding, self).__init__()
        self.emb_dim = emb_dim
        self.use_entity = use_entity
        self.output_dim = output_dim
        self.entity_embedding_dim = entity_embedding_dim
        layers = []     
        if use_entity:
            total_input_dim = 2*emb_dim + 2*type_embedding_dim
            self.type_embeddings = nn.Embedding(num_entities,
                                                type_embedding_dim)
        else:
            total_input_dim = 2*emb_dim

        self.network = MLP(total_input_dim, hidden_dims,
                           output_dim) 
            
            
            # experiment with mlp,
                    
                    # non linearity

    def forward(self, headtail):
        # head, tail : * x input_dim
        #
        # output : * x output_dim
        head_emb_type, tail_emb_type = headtail
        head, headtype = head_emb_type
        tail, tailtype = tail_emb_type
        init_shape = head.size()
        final_shape = tuple(list(init_shape)[:-1] + [self.output_dim])
        head = head.view(-1, self.input_dim)
        tail = tail.view(-1, self.input_dim)
        if self.use_entity:
            head_type_embedding = self.type_embeddings(headtype)
            tail_type_embedding = self.type_embeddings(tailtype)
            concat = torch.cat((head, tail, headtype, tailtype), dim = -1)
        else:
            concat = torch.cat((head, tail), dim = -1)

        mapped = self.network(concat)
        return torch.reshape(mapped, final_shape)


class ShouldCopy(nn.Module): # MLP to get probability of copying vs generating
    # input: context vector, query embedding
    def __init__(self, dim, hidden_dims = []):
        super(ShouldCopy, self).__init__()
        self.dim = dim
        self.network = MLP(2*dim, hidden_dims, 1)

    def forward(self, query, context):
        # query, context: Batch x n x dim
        #
        # output: Batch x n
        concat = torch.cat((query, context), -1)
        logits = self.network(concat).squeeze(-1)
        prob = F.logsigmoid(logits)
        neg_prob = F.logsigmoid(-logits)
        return prob, neg_prob



class MultiClass(nn.Module):
    def __init__(self, dim, num_classes, hidden_dims):
        super(MultiClass, self).__init__()
        self.dim = dim
        self.num_classes = num_classes
        layers = []
        self.network = MLP(dim, hidden_dims, num_classes +1)

    def forward(self, rel_emb):
        logits = self.network(rel_emb)
        return F.log_softmax(logits, dim = -1)


class CopyEditor(nn.Module):
    def __init__(self, emb_dim, args):
        super(CopyEditor, self).__init__()
        assert copy or generate, 'Either copy or generate must be true'
        self.copy = args.copy
        self.generate = args.generate
        self.emb_dim = emb_dim
        self.num_classes = args.classes
        self.attention = AttentionDist(args.relation_output_dim, args.num_classes)
        self.rel_embedding = RelationEmbedding(emb_dim, args.relation_output_dim,
                                args.relation_hidden_dims, args.use_entity) # share
                        # embeddings between query and context
        self.should_copy = ShouldCopy(args.relation_output_dim)
        self.multiclass = MultiClass(args.relation_output_dim,
                                     self.num_classes,
                                     args.relation_hidden_dims)

    def forward(self, query_vectors, context_vectors, context_labels,  mask):
        # query_vector: (head, tail) each of Batch x n x dim
        # context_vectors: (head, tail) each of Batch x context_size x dim
        # Context_labels: Batch x Context_size
        # mask : Batch x context_size
        #
        # output: Batch x (Num_classes+1) log distribution


        # Batch x n x dim
        query_embedding = self.rel_embedding(query_vectors)

        context_vec, copy_dist = None, None
        if self.generate:
            gen_dist = self.multiclass(query_embedding)
        if self.copy:
            # Batch x context_size x dim
            if context_labels is not None: # if contex
                context_embedding = self.rel_embedding(context_vectors)
                #Batch x n x dim, Batch x n x num_classes+1
                context_vec, copy_dist_unsmooth = self.attention(query_embedding, context_embedding, \
                                                 context_labels, mask)
                # We need to smooth since copy can assign p(relation) = 0 if it
                # does not exist in context. This is not an issue in
                # Copy&Generate but if Generate is turned off, this makes
                # NLL=-log(P(y)) -> inf.


                # P_smooth = 0.99 * P_copy + 0.01 * p_uniform
                # we stay in log space to avoid nans due to overflow
                # log(P_smooth) = logsumexp(log(0.99)+log(p_copy),log(0.01)+log(1/N))
                # TODO: smoothing with actual distribution of labels rather
                # than uniform
                copy_dist = torch.logsumexp(torch.stack((copy_dist_unsmooth +
                                                               np.log(0.99),
                                                               torch.ones_like(copy_dist_unsmooth)*np.log(0.01/(self.num_classes+1))),
                                                               dim=-1), dim = -1)

        # if generate is enabled and copy is disabled or no cotext provided
        if self.generate and (not self.copy or copy_dist is None):
            return gen_dist
        #if copy is enabled but not generate
        elif self.copy and not self.generate:
            # If no context provided we are stuck since we do not generate
            # In this case always return norel
            if copy_dist is None:
                # If no context always return P(norel) = 0.99 # using 1.0 causes loss to be Inf
                probs = torch.ones(query_embedding.size()[0],query_embedding.size()[1],self.num_classes+1)*(np.log(0.01/self.num_classes))
                probs[:,:,-1] = np.log(0.99)
                if query_embedding.is_cuda:
                    probs = probs.cuda()
                return probs
            # Else return just the copy distribution
            else:
                return copy_dist
        # if both are enabled and context is available
        else:
            copy_prob, gen_prob = self.should_copy(query_embedding, \
                                         context_vec)
            copy_prob = copy_prob.unsqueeze(-1)
            gen_prob = gen_prob.unsqueeze(-1)

            # Batch x n x num_classes+1
            log_probs = torch.stack([copy_prob + copy_dist, gen_prob +
                                     gen_dist], dim = -1)
            final_probs = torch.logsumexp(log_probs, dim = -1)
            return final_probs




