import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset
from transformers import *
from biobert import *


class MLP(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims):
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
    def __init__(self, dim, num_classes, context_label_dims, attnmethod, hidden_dims=[]):
        super(AttentionDist, self).__init__()
        self.dim = dim
        self.num_classes = num_classes
        self.attention_method = attnmethod
        if self.attention_method == 'mlp':
            self.label_embedding = nn.Embedding(num_classes + 1, context_label_dims)
            total_input_dim = 2 * dim + context_label_dims
        else:
            total_input_dim = 2*dim
        self.network = MLP(total_input_dim, hidden_dims, 1)

    def forward(self, queries, context, context_labels, mask):
        # queries: Batch x n x dim
        # context: Batch x context_size x dim
        # context_labels: Batch x context_size  (which class)
        # mask: Batch x context_size // because batch may have different
        #               context sizes

        # output: Context vector, Probability distribution over numclasses+1 
        #                   +1 for none class

        # Batch x n x context_size
        # Computes matrix [M_{ij}] where M_{ij} = q_i^t c_j

        if self.attention_method == 'dotprod':
            attn = torch.sum(queries.unsqueeze(2) * context.unsqueeze(1), dim=-1) * mask.unsqueeze(1)
            # queries: (Batch x n x dim),  queries.unsqueeze(2): (Batch x n x 1 x dim)
            # context: (Batch x context_size x dim),  context.unsqueeze(1): (Batch x 1 x context_size x dim)
            # dot_prod: (Batch x n x context_size x dim)
            # torch.sum(dot_prod, dim = -1): (Batch x n x context_size)

        elif self.attention_method == 'mlp':
            _,n,__ = queries.size()
            _,cs,__ = context.size()
            label_emb = self.label_embedding(context_labels)  # Batch x context_size  x context_label_dims
            # context_label_dims should be = dim
            concat = torch.cat((queries.unsqueeze(2).repeat(1,1,cs,1),\
                                context.unsqueeze(1).repeat(1,n,1,1),\
                                label_emb.unsqueeze(1).repeat(1,n,1,1)), -1)
            attn = self.network(concat).squeeze(-1)

            # TODO:
            # [M_{ij}] = concat(q_i, c_j, cl_j)
            # log(p(j|i)) = MLP(M_{ij}) + k
            # inputdim = 2*rel + label_emb
            # MLP: inputdim -> score(scalar)
            # B x n x cs x dim -> B x n x cs x 1
        elif self.attention_method == 'no_label':
            _,n,__ = queries.size()
            _,cs,__ = context.size()
            # context_label_dims should be = dim
            concat = torch.cat((queries.unsqueeze(2).repeat(1,1,cs,1),\
                                context.unsqueeze(1).repeat(1,n,1,1)), -1)
            attn = self.network(concat).squeeze(-1)


        # +(1.0-mask.unsqueeze(1))*(-np.inf)
        # Batch x n x context_size
        l_softmax = F.log_softmax(attn, dim=-1)

        # Batch x n x dim
        # Batch x 1x cs x dim
        # Batch x n x cs x 1
        # Batch x n x dim
        context_vector = torch.sum(torch.exp(l_softmax.unsqueeze(-1)) * context.unsqueeze(1), dim=2)

        # Batch x n x (num_classes+1)
        _, n, __ = l_softmax.size()

        # Batch x 1 x cs x num_classes+1
        # Batch x n x cs x 1
        # Batch x n x cs x num_classes+1
        #  c1, c2, c3
        #  p1, p2, p3
        #  l1, l2, l1
        #  P_l1, P_l2

        # l_softmax = [[p1, p2, p3]]
        # cl = [l1,l2,l1]

        # [[1,0],[0,1],[1,0]]  
        onehot_labels =\
            F.one_hot(context_labels, self.num_classes+1).unsqueeze(1).repeat((1,n,1,1))

        #  [[[p1,0],[0,p2],[p3,0]]] //1 x 3 x 2
        onehot_logprobs = onehot_labels * l_softmax.unsqueeze(-1)
        onehot_logprobs[torch.where(onehot_labels == 0)] = -float('Inf')

        #  [[p1+p3, p2]]
        l_softmax_classes = torch.logsumexp(onehot_logprobs, dim = 2)
        return context_vector, l_softmax_classes, attn


class RelationEmbedding(nn.Module):
    def __init__(self,
                 emb_dim,num_entities,num_buckets,type_embedding_dim,pos_embedding_dim,output_dim,
                 hidden_dims = [], use_entity=True, use_pos = True):
        super(RelationEmbedding, self).__init__()
        self.emb_dim = emb_dim
        self.use_entity = use_entity
        self.use_pos = use_pos
        self.output_dim = output_dim
        self.type_embedding_dim = type_embedding_dim
        self.pos_embedding_dim = pos_embedding_dim
        layers = []
        total_input_dim = 2*emb_dim
        if use_entity:
            total_input_dim += 2*type_embedding_dim
            self.type_embeddings = nn.Embedding(num_entities,
                                                type_embedding_dim)

        if use_pos:
            self.pos_embeddings = nn.Embedding(num_buckets,
                                               pos_embedding_dim)
            total_input_dim += pos_embedding_dim
        self.network = MLP(total_input_dim, hidden_dims,
                           output_dim)




    def forward(self, headtail):
        # (head, headtype), (tail,tailtype), posbucket : * x input_dim
        #
        # output : * x output_dim
        head_emb_type, tail_emb_type, pos = headtail
        head, headtype = head_emb_type
        tail, tailtype = tail_emb_type
        init_shape = head.size()
        final_shape = tuple(list(init_shape)[:-1] + [self.output_dim])
        head = head.view(-1, self.emb_dim)
        tail = tail.view(-1, self.emb_dim)
        features = [head, tail]
        if self.use_entity:
            head_type_embedding = self.type_embeddings(headtype).squeeze(0)
            tail_type_embedding = self.type_embeddings(tailtype).squeeze(0)
            features.extend([head_type_embedding, tail_type_embedding])
        if self.use_pos:
            pos_embeddings = self.pos_embeddings(pos).squeeze(0)
            features.append(pos_embeddings)

        concat = torch.cat(features, dim = -1)
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
        self.network = MLP(dim, hidden_dims, num_classes +1)

    def forward(self, rel_emb):
        logits = self.network(rel_emb)
        return F.log_softmax(logits, dim = -1)


class CopyEditor(nn.Module):
    def __init__(self, emb_dim, args):
        super(CopyEditor, self).__init__()
        assert args.copy or args.generate, 'Either copy or generate must be true'
        self.copy = args.copy
        self.generate = args.generate
        self.emb_dim = emb_dim
        self.num_classes = args.classes
        self.attention = AttentionDist(args.relation_output_dim, args.classes, args.context_label_dim, args.attnmethod,
                                       args.attention_hidden_dims)
        self.rel_embedding = RelationEmbedding(emb_dim, args.num_entities, \
                                               args.num_buckets, args.type_dim, args.pos_dim, \
                                               args.relation_output_dim, args.relation_hidden_dims, \
                                               args.use_entity, args.use_pos)  # share
        # embeddings between query and context
        self.should_copy = ShouldCopy(args.relation_output_dim,
                                      args.shouldcopy_hidden_dims)
        self.multiclass = MultiClass(args.relation_output_dim,
                                     self.num_classes,
                                     args.relation_hidden_dims)

    def forward(self, query_vectors, context_vectors, context_labels,  mask):
        # query_vector: ((head, headtype), (tail, tailtype), posbucket) head, tail of Batch x n x dim
        #                                                               headtype,tailtype, posbucket of size Batch x n
        # context_vectors: ((head, headtype), (tail, tailtype), posbucket) each of Batch x context_size x dim
        # Context_labels: Batch x Context_size
        # mask : Batch x context_size,
        #
        # output: Batch x (Num_classes+1) log distribution


        # Batch x n x dim
        query_embedding = self.rel_embedding(query_vectors)

        context_vec, copy_dist = None, None
        copy_prob = torch.zeros_like(query_vectors[2]).float()
        l_softmax = None
        if self.generate:
            gen_dist = self.multiclass(query_embedding)
        if self.copy:
            # Batch x context_size x dim
            if context_labels is not None: # if contex
                context_embedding = self.rel_embedding(context_vectors)
                #Batch x n x dim, Batch x n x num_classes+1
                context_vec, copy_dist_unsmooth, l_softmax = self.attention(query_embedding, context_embedding, \
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
                return probs, copy_prob, l_softmax
            # Else return just the copy distribution
            else:
                return copy_dist, copy_prob, l_softmax
        # if both are enabled and context is available
        else:
            copy_prob, gen_prob = self.should_copy(query_embedding, \
                                         context_vec)
            copy_prob = copy_prob.unsqueeze(-1)
            gen_prob = gen_prob.unsqueeze(-1)

            # Batch x n x num_classes+1
            # p_final = p_copy*copy_dist + p_gen*gen_dist
            log_probs = torch.stack([copy_prob + copy_dist, gen_prob +
                                     gen_dist], dim = -1)
            final_probs = torch.logsumexp(log_probs, dim = -1)
            return final_probs, copy_prob

# We use this somewhat unintuitive complicated architecture because in some
# configurations, we precompute all the inputs to CopyEditor ( when we want to
# keep the BERT parameters frozen ). CopyEditorBertWrapper uses a trainable BERT
# module to compute the same for the case when we want to finetune BERT.

# Compute bert embeddings in test time
class CopyEditorBertWrapper(nn.Module):
    def __init__(self,emb_dim, args):
        super(CopyEditorBertWrapper, self).__init__()
        self.emb_dim = emb_dim
        self.copy_editor = CopyEditor(emb_dim, args)
        self.bert_tokenizer, self.bert_transformer = getscibertmodel()
        if args.gpu:
            # use multiple GPUs for bert
            self.bert_transformer = nn.DataParallel(self.bert_transformer, \
                                                    list(range(torch.cuda.device_count())))
        self.bert_transformer.eval()

    def get_vectors(self, bert_embeddings, spans):
        N = spans.size()[0]
        if N == 0:
            print(N)
            return None, None, None, None, None
        # bert_embeddings = self.bert_transformer(tokens.unsqueeze(0))[-2].squeeze(0)
        head_embeddings = []
        tail_embeddings = []
        for i in range(N):
            hstart, hend = spans[i][0], spans[i][1]
            tstart, tend = spans[i][3], spans[i][4]
            if hend < hstart:
                print("%d\t%d\n"%(hstart, hend))
                raise Exception
            head_embeddings.append(torch.mean(bert_embeddings[hstart:hend+1,:],\
                                             dim=0))
            tail_embeddings.append(torch.mean(bert_embeddings[tstart:tend+1,:],\
                                             dim=0))
        head_embeddings = torch.stack(head_embeddings, dim=0)
        tail_embeddings = torch.stack(tail_embeddings, dim=0)
        head_type = spans[:,2]
        tail_type = spans[:,5]
        pos = spans[:,6]
        return head_embeddings.unsqueeze(0),head_type.unsqueeze(0),\
                tail_embeddings.unsqueeze(0), tail_type.unsqueeze(0),\
                    pos.unsqueeze(0)

    def forward(self,query_tokens, query_spans, context_tokens, context_spans, context_labels):
        cat = [query_tokens] + context_tokens
        mxlen = max([len(x) for x in cat])
        for x in cat:
            x.extend([self.bert_tokenizer.pad_token_id] * (mxlen - len(x)))
        cat_tensor = torch.tensor(cat)
        if query_spans.is_cuda:
            cat_tensor = cat_tensor.cuda()
        bert_embeddings = self.bert_transformer(cat_tensor)[-2]
        qh,qht,qt,qtt,pos = self.get_vectors(bert_embeddings[0,:], query_spans)
        query_vectors = ((qh,qht),(qt,qtt),pos)
        context_heads = []
        context_tails = []
        context_tail_type = []
        context_head_type = []
        context_pos = []
        for i, spans in enumerate(context_spans):
            ch,cht,ct,ctt,pos = self.get_vectors(bert_embeddings[i+1,:], spans)
            context_heads.append(ch)
            context_tails.append(ct)
            context_tail_type.append(ctt)
            context_head_type.append(cht)
            context_pos.append(pos)
        if len(context_heads) > 0:
            context_heads = torch.cat(context_heads, dim=1)
            context_tails = torch.cat(context_tails, dim=1)
            context_labels = torch.cat(context_labels).unsqueeze(0)
            context_head_type = torch.cat(context_head_type, dim=1)
            context_tail_type = torch.cat(context_tail_type, dim=1)
            context_pos = torch.cat(context_pos, dim=1)
        else:
            context_heads, context_tails, context_labels,\
                context_pos, context_head_type, context_tail_type =\
                    None, None,None, None, None, None
        context_vectors = ((context_heads, context_head_type),\
                           (context_tails, context_tail_type), context_pos)
        if context_heads is not None:
            mask = torch.ones(context_heads.size()[:-1])
        else:
            mask = None
        if mask is not None and query_spans.is_cuda:
            mask = mask.cuda()
        return self.copy_editor(query_vectors, context_vectors, context_labels,\
                                mask)
