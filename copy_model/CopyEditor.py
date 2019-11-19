import numpy as np
from transformers import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset


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
    def __init__(self, input_dim, output_dim, hidden_dims = []):
        super(RelationEmbedding, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.network = nn.Linear(2*input_dim,output_dim) # experiment with mlp,
                    # non linearity

    def forward(self, head, tail):
        # head, tail : * x input_dim
        #
        # output : * x output_dim
        init_shape = head.size()
        final_shape = tuple(list(init_shape)[:-1] + [self.output_dim])
        head = head.view(-1, self.input_dim)
        tail = tail.view(-1, self.input_dim)
        concat = torch.cat((head, tail), dim = -1)
        mapped = self.network(concat)
        return torch.reshape(mapped, final_shape)


class ShouldCopy(nn.Module): # MLP to get probability of copying vs generating
    # input: context vector, query embedding
    def __init__(self, dim):
        super(ShouldCopy, self).__init__()
        self.dim = dim
        self.network = nn.Linear(2*dim, 1)

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
    def __init__(self, dim, num_classes, num_hidden):
        super(MultiClass, self).__init__()
        self.dim = dim
        self.num_classes = num_classes
        layers = []
        dims = [dim] + num_hidden
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], num_classes +1))
        self.network = nn.Sequential(*layers)

    def forward(self, rel_emb):
        logits = self.network(rel_emb)
        return F.log_softmax(logits, dim = -1)


class CopyEditor(nn.Module):
    def __init__(self, emb_dim, num_classes, copy=True, generate=True):
        super(CopyEditor, self).__init__()
        assert copy or generate, 'Either copy or generate must be true'
        self.copy = copy
        self.generate = generate
        self.emb_dim = emb_dim
        self.num_classes = num_classes
        self.attention = AttentionDist(emb_dim, num_classes)
        self.rel_embedding = RelationEmbedding(emb_dim, emb_dim) # share
                        # embeddings between query and context
        self.should_copy = ShouldCopy(emb_dim)
        self.multiclass = MultiClass(emb_dim, num_classes, [])

    def forward(self, query_vectors, context_vectors, context_labels,  mask):
        # query_vector: (head, tail) each of Batch x n x dim
        # context_vectors: (head, tail) each of Batch x context_size x dim
        # Context_labels: Batch x Context_size
        # mask : Batch x context_size
        #
        # output: Batch x (Num_classes+1) log distribution


        # Batch x n x dim
        query_embedding = self.rel_embedding(query_vectors[0], \
                                             query_vectors[1])
        B, n, dim = query_vectors[0].size()
        copy_prob = torch.zeros(B, n)
        context_vec, copy_dist = None, None
        if self.generate:
            gen_dist = self.multiclass(query_embedding)
        if self.copy:
            # Batch x context_size x dim
            if context_labels is not None: # if contex
                context_embedding = self.rel_embedding(context_vectors[0], \
                                                       context_vectors[1])
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
            return gen_dist, copy_prob
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
                return probs, copy_prob
            # Else return just the copy distribution
            else:
                return copy_dist, copy_prob
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
            return final_probs, copy_prob



class CopyEditorBertWrapper(nn.Module):
    def __init__(self,emb_dim, num_classes, copy=True, generate=True):
        super(CopyEditorBertWrapper, self).__init__()
        self.emb_dim = emb_dim
        self.copy_editor = CopyEditor(emb_dim, num_classes, copy=copy,\
                                      generate=generate)
        self.bert_transformer = BertModel.from_pretrained('../../biobert')  
    
    def get_vectors(self, tokens, spans):
        N = spans.size()[0]
        bert_embeddings = self.bert_transformer(tokens.unsqueeze(0))[-2].squeeze(0)
        head_embeddings = []
        tail_embeddings = []
        for i in range(N):
            hstart, hend = spans[i][0][0], spans[i][0][1]
            tstart, tend = spans[i][1][0], spans[i][1][1]
            head_embeddings.append(torch.mean(bert_embeddings[hstart:hend+1,:],\
                                             dim=0))
            tail_embeddings.append(torch.mean(bert_embeddings[tstart:tend+1,:],\
                                             dim=0))
        head_embeddings = torch.stack(head_embeddings, dim=0)
        print(head_embeddings.size())
        tail_embeddings = torch.stack(tail_embeddings, dim=0)
        return (head_embeddings.unsqueeze(0), tail_embeddings.unsqueeze(0))

    def forward(self,query_tokens, query_spans, context_tokens, context_spans, context_labels):
        query_vectors = self.get_vectors(query_tokens, query_spans)
        context_heads = []
        context_tails = []
        for tokens, spans in zip(context_tokens, context_spans):
            ch,ct = self.get_vectors(tokens, spans)
            context_heads.append(ch)
            context_tails.append(ct)
        if len(context_heads) > 0:
            context_heads = torch.cat(context_heads, dim=1)
            context_tails = torch.cat(context_tails, dim=1)
            context_labels = torch.cat(context_labels).unsqueeze(0)
        else:
            context_heads, context_tails, context_labels = None, None, None
        context_vectors = (context_heads, context_tails)
        return self.copy_editor(query_vectors, context_vectors, context_labels,\
                                torch.ones(context_heads.size()[:-1]))

        

