import argparse
import os
import datetime
import torch
import numpy as np
import random
import scanpy as sc
import anndata as ad
from collections import Counter
from torch.nn import MSELoss

from genomedata import GenomeData
from autoencoder import AutoEncoder
from graph_function import create_edge_index, get_dist_matrix


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def leiden_clustering(data, res=0.5):
    adata = ad.AnnData(data)
    sc.pp.neighbors(adata, use_rep='X')
    sc.tl.umap(adata)
    sc.tl.leiden(
        adata,
        key_added="clusters",
        resolution=res,
        flavor="igraph",
        n_iterations=2,
    )
    labels_p = adata.obs['clusters']
    labels_uniq = np.unique(labels_p)
    counter = Counter(labels_p)
    for value, count in counter.items():
        print(f"{value}: {count}")

    return labels_p


def main(args):
    start_t = datetime.datetime.now()

    if not os.path.isdir(args.output):
        os.mkdir(args.output)

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.cuda.set_device(args.gpu)

    gd = GenomeData(args.input)
    gd.load_data()
    gd.preprocess_data()

    train_set = gd.data_rc_all.copy()

    num_rows = train_set.shape[0]

    e, A, indices_tuples = create_edge_index(train_set, 10)
    dist_matrix = get_dist_matrix(num_rows, indices_tuples)
    A = torch.from_numpy(A).to(device)
    dist_matrix = torch.from_numpy(dist_matrix).to(device)

    train_set = torch.from_numpy(train_set).to(device)
    
    model = AutoEncoder(input_dim=train_set.size(1), n_layers=3, z_dim=args.latent_dim).to(device)

    criterion = MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-08, weight_decay=0,
                                 amsgrad=False)
    # Start training the model
    model.train()
    train_loss = []
    metric_all = [[]]
    lr = args.lr
    for epoch in range(args.epochs):
        outputs, embeddings = model(train_set, A, dist_matrix)
        if outputs.size(1) < train_set.size(1):
            target, _ = torch.split(train_set, outputs.size(1), dim=-1)
        else:
            target = train_set
        loss = criterion(outputs, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % 50 == 0:
            lr = lr * 0.9
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        train_loss.append(loss.item())
        print(f'epoch: {epoch+1:.0f}, loss: {train_loss[-1]:.5f}')

    ll_file = args.output + '/loss.txt'
    file_o = open(ll_file, 'w')
    np.savetxt(file_o, np.c_[np.reshape(train_loss, (1, len(train_loss)))], fmt='%f', delimiter=',')
    file_o.close()

    # get cell embeddings
    model.eval()
    with torch.no_grad():
        outputs, embeddings = model(train_set, A, dist_matrix)

    embeddings = embeddings.cpu().detach().numpy()

    labels_p = leiden_clustering(embeddings, args.resolution)

    label_file = args.output + '/labels.txt'
    file_o = open(label_file, 'w')
    np.savetxt(file_o, np.c_[np.reshape(labels_p, (1, len(labels_p)))], fmt='%s', delimiter=',')
    file_o.close()

    latent_file = args.output + '/embeddings.txt'
    file_o = open(latent_file, 'w')
    np.savetxt(file_o, np.c_[embeddings], fmt='%.3f', delimiter=',')
    file_o.close()

    end_t = datetime.datetime.now()
    print('elapsed time: ', (end_t - start_t).seconds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="scMTC")
    parser.add_argument('--gpu', type=int, default=0, help='which GPU to use.')
    parser.add_argument('--epochs', type=int, default=200, help='number of epoches to train the model.')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate.')
    parser.add_argument('--resolution', type=float, default=0.5, help='resolution parameter of the Leiden algorithm.')
    parser.add_argument('--latent_dim', type=int, default=64, help='the latent dimension.')
    parser.add_argument('--seed', type=int, default=1, help='random seed.')
    parser.add_argument('--input', type=str, default='', help='a file containing read counts and GC-content data.')
    parser.add_argument('--output', type=str, default='', help='a directory to save results.')
    args = parser.parse_args()
    main(args)
