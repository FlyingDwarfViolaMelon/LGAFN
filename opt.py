import argparse

parser = argparse.ArgumentParser(description='LGAFN', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--name', type=str, default='acm')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--n_clusters', type=int, default=3)
parser.add_argument('--n_z', type=int, default=20)
parser.add_argument('--n_input', type=int, default=100)
parser.add_argument('--data_path', type=str, default='.txt')
parser.add_argument('--label_path', type=str, default='.txt')
parser.add_argument('--adj_path', type=str, default='.txt')
parser.add_argument('--save_path', type=str, default='.txt')
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--n_components', type=int, default=100)
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--pca_status', type=bool, default=False) 
parser.add_argument('--shuffle', type=bool, default=True)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--encoder_dim', type=int, default=[500, 500, 2000])
parser.add_argument('--decoder_dim', type=int, default=[2000, 500, 500])
parser.add_argument('--hidden_gsa_dim', type=int, default=[500, 500, 2000])

args = parser.parse_args()