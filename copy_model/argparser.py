import argparse

parser = argparse.ArgumentParser(description='Settings for training')
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
                    help='Number of epochs', default=50)
parser.add_argument('--classes', action='store', dest='classes', type=int,
                    help='Number of classes', default=14)
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
args = parser.parse_args()
