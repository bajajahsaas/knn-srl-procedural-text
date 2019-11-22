import argparse

parser = argparse.ArgumentParser(description='Settings for training')
parser.add_argument('--entity', dest='use_entity', action='store_true')
parser.add_argument('--no-entity', dest='use_entity', action='store_false')
parser.set_defaults(use_entity=True)
parser.add_argument('--copy', dest='copy', action='store_true', help='Copy')
parser.add_argument('--no-copy', dest='copy', action='store_false',
                    help='Don\'t copy')
parser.set_defaults(copy=True)
parser.add_argument('--generate', dest='generate', action='store_true',
                    help='Generate')
parser.add_argument('--no-generate', dest='generate', action='store_false',
                    help='Don\'t generate')
parser.set_defaults(generate=True)
parser.add_argument('--gpu', dest='gpu', action='store_true', help='Use GPU')
parser.set_defaults(gpu=False)
parser.add_argument('--traindata', action='store', dest='traindata', type=str,
                    help='Pickle file with training data', default='train.pkl')
parser.add_argument('--valdata', action='store', dest='valdata', type=str,
                    help='Pickle file with validation data', default='val.pkl')
parser.add_argument('--testdata', action='store', dest='testfile', type=str,
                    help='Pickle file with test data', default='test.pkl')
parser.add_argument('--model_path', action='store', dest='model_path', type=str,
                    help='File to save pytorch model parameters',
                    default='model.pt')
parser.add_argument('--test_output_path', action='store', dest='test_output_path', type=str,
                    help='File to save test csv',
                    default='test_output.csv')
parser.add_argument('--epochs', action='store', dest='epochs', type=int,
                    help='Number of epochs', default=200)
parser.add_argument('--shouldcopy_hidden_dims', action='store',
                    dest='shouldcopy_hidden_dims', type=list,
                    help='hidden dimensions for mlp for copyprob', default=[256,128,64])
parser.add_argument('--relation_hidden_dims', action='store',
                    dest='relation_hidden_dims', type=list,
                    help='hidden dims for relation embedding', default=[512,256])
parser.add_argument('--relation_output_dim', action='store',
                    dest='relation_output_dim', type=int,
                    help='output dim of relation embedding', default=256)
parser.add_argument('--type_embedding_dim', action='store',
                    dest='type_dim', type=int,
                    help='Embedding size of type embedding', default=128)
parser.add_argument('--classes', action='store', dest='classes', type=int,
                    help='Number of classes', default=13)
parser.add_argument('--num_entities', action='store', dest='num_entities', type=int,
                    help='Number of Entities', default=19)
parser.add_argument('--batch_size', action='store', dest='batch_size', type=int,
                    help='Batch size', default=16)
parser.add_argument('--grad_maxnorm', action='store', dest='grad_maxnorm',
                    type=float, help='Max norm for gradient clipping',
                    default=100)
parser.add_argument('--lr', action='store', dest='lr',
                    type=float, help='Learning rate',
                    default=1e-4)
parser.add_argument('--weight_decay', action='store', dest='weight_decay',
                    type=float, help='Weight decay parameter',
                    default=1e-4)
parser.add_argument('--plot_path', action='store', dest='plot_path',
                    type=str, help='Path for plot',
                    default='logs')
args = parser.parse_args()
