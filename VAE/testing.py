import argparse
import os
import random
import warnings
from importlib import import_module

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
from collections import Counter

import hdbscan
import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import tensorflow as tf
# from brokenaxes import brokenaxes
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import models, optimizers

import losses
import utils

plt.rcParams.update({'font.size': 16})
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

'''
0: other
1: class 1
2: class 2
...
'''

def test(MODEL, DRAW_PACK=True,  PLOT_COMP=False, PLOT_GIF=False, EXPORT_PIPE=False):
    network = import_module(f'model.{MODEL}.training.network')

    args_dict = json.load(open(f'model/{MODEL}/training/arguments.json'))
    MODEL_NAME = args_dict['model_name']
    DATASET = args_dict['dataset']
    NON_CENTER = args_dict['non_center']

    REPEAT_SIZE = args_dict['repeat_size']
    BATCH_SIZE = args_dict['batch_size']
    CLOUD_SIZE = args_dict['cloud_size']
    DIMEN_SIZE = args_dict['dimen_size']
    LATENT_DIM = args_dict['latent_dim']

    EPOCHS = args_dict['epochs']
    LEARNING_RATE = args_dict['learning_rate']

    L2_WEIGHT = args_dict['l2_weight']
    CLOSS_WEIGHT = args_dict['closs_weight']
    OLOSS_WEIGHT = args_dict['oloss_weight']
    RLOSS_WEIGHT = args_dict['rloss_weight']
    WLOSS_WEIGHT = args_dict['wloss_weight']

    RADIUS = args_dict['radius']

    # Plot curve
    history = pd.read_csv(f'model/{MODEL}/training/logging.csv')
    plt.plot(history['loss'])
    plt.title("Training History")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.savefig(f'model/{MODEL}/training/curve.png')
    plt.close()

    history.drop(columns=['epoch', 'loss']).plot()
    plt.title("Training Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.savefig(f'model/{MODEL}/training/loss.png')
    plt.close()

    print('[*] Plot curve and loss successfully!')

    # Read data
    pointcloud_dataset = utils.PointCloudDataset(f'data/{DATASET}', CLOUD_SIZE, NON_CENTER)
    dataset = pointcloud_dataset.make_dataset(10000)
    DATASET_SIZE, CLOUD_SIZE, DIMEN_SIZE = dataset.shape
    
    print(f'[*] Read {DATASET_SIZE} samples with batch size {BATCH_SIZE}.')

    # Load model
    model = network.PointNetAutoEncoder(REPEAT_SIZE, BATCH_SIZE, CLOUD_SIZE, DIMEN_SIZE, LATENT_DIM, L2_WEIGHT, CLOSS_WEIGHT, OLOSS_WEIGHT, RADIUS)
    model.compile(loss=losses.CombinedReconstructionLoss(2*RADIUS, RLOSS_WEIGHT, WLOSS_WEIGHT))

    # Best checkpoint
    ckpts, ckpt_epochs, ckpt_losses = [], [],[]
    for root, dirs, files in os.walk(f'model/{MODEL}/training/bestcheckpoint'):
        for name in files:
            prefix, ext = os.path.splitext(name)
            if ext == '.index':
                ckpts.append(os.path.join(root, prefix))
                ckpt_epoch, ckpt_loss = prefix[8:].split('-')
                ckpt_epochs.append(int(ckpt_epoch))
                ckpt_losses.append(float(ckpt_loss))
                
    # sort by epoch
    ckpts = np.array(ckpts)
    ckpt_epochs = np.array(ckpt_epochs)
    ckpt_losses = np.array(ckpt_losses)
    sorted_index = ckpt_epochs.argsort()

    ckpts = ckpts[sorted_index]
    ckpt_epochs = ckpt_epochs[sorted_index]
    ckpt_epochs = ckpt_losses[sorted_index]

    best_ckpt = ckpts[np.argmin(ckpt_losses)]
    # Latest checkpoint
    latest_ckpt = tf.train.latest_checkpoint(f'model/{MODEL}/training/latestcheckpoint')
    
    checkpoint = tf.train.Checkpoint(model)
    checkpoint.restore(latest_ckpt).expect_partial()
    print(f'[*] Load model {MODEL} successfully!')

    # Cluster
    if not os.path.exists(f'model/{MODEL}/cluster'):
        os.mkdir(f'model/{MODEL}/cluster')

    if DATASET[:2] == 'lj':
        n_clusters = 4
    elif DATASET[:2] == 'SP':
        n_clusters = 3
    elif DATASET[:2] == 'CP':
        n_clusters = 2
    n_clusters = 2
    struiden_pipeline = utils.StructureIdentificationPipeline(model, dataset, n_clusters)

    # Plot cluster with center
    viz_tool = utils.VisualizationTool(RADIUS, PLOT_COMP)
    msc = MinMaxScaler(feature_range=(0, 1))
    if LATENT_DIM > 3:
        viz_points = TSNE(init='pca', random_state=2021).fit_transform(struiden_pipeline.encoded)
        viz_points = msc.fit_transform(viz_points)
    else:
        viz_points = msc.fit_transform(struiden_pipeline.encoded)
    plt.figure(figsize=(10, 10))
    viz_tool.plot_cluster(viz_points,
                          struiden_pipeline.labels, 
                          struiden_pipeline.probabilities,
                          medoid_idx=struiden_pipeline.medoid_idx)
    plt.savefig(f'model/{MODEL}/cluster/cluster.png', bbox_inches='tight')
    plt.close()

    print('[*] Save cluster image successfully!')
    
    # LJ baseline
    # https://github.com/rsdefever/GenStrIde
    if DATASET[:2] == 'lj':
        true_labels = np.load('data/lj_labels.npy')[pointcloud_dataset.slice]
        true_labels = np.argmax(true_labels, axis=-1)
        accuracy, confusion = utils.cluster_accuracy(true_labels, struiden_pipeline.labels)
        print(f'[*] LJ baseline accuracy: {accuracy:.6f}')
        label_names = ['liquid', 'fcc', 'hcp', 'bcc']

        new2old = []
        for i in struiden_pipeline.medoid_idx:
            new2old.append(true_labels[i])

        old2new = {j:i for i,j in enumerate(new2old)}
        true_labels = np.array([old2new[i] for i in true_labels])
        label_names = np.array([label_names[old2new[i]] for i in range(4)])

        plt.figure(figsize=(10, 10))
        cmap = LinearSegmentedColormap.from_list('true_labels', viz_tool.palette[:4], N=4)
        scatter = plt.scatter(viz_points[:,0], viz_points[:,1], c=true_labels, cmap=cmap, alpha=0.2)
        plt.legend(scatter.legend_elements()[0], label_names)
        plt.savefig(f'model/{MODEL}/cluster/cluster_true.png', bbox_inches='tight')
        plt.close()

        with open(f'model/{MODEL}/report.txt', 'w') as f:
            f.write(str(accuracy) + '\n')
            np.savetxt(f, confusion)
        
        # plot
        # recall = confusion / np.sum(confusion, axis=-1, keepdims=True)
        
        # x = np.arange(len(label_names))
        # width = 0.2
        # plt.figure(figsize=(10,7))
        # bax = brokenaxes(xlims=((-0.99, 3.99),), ylims=((0, .199), (.8, 1)), hspace=.05, despine=False)
        # for i in range(4):
        #     bax.bar(x + width*(i-1.5), recall[i], width=width, label=label_names[i], tick_label=label_names, color=viz_tool.palette[i])
        #     plt.text(0.2*(i+1), -0.08, s=label_names[i], ha='center')
        # bax.legend(loc='upper center', bbox_to_anchor=(0.5,1.1), ncol=4, framealpha=0)
        # bax.set_xlabel('True labels')
        # bax.set_ylabel('Predicted percentage', labelpad=50)
        # plt.savefig(f'model/{MODEL}/recall.png', bbox_inches='tight')
        # plt.close()

    # Save cells
    if not os.path.exists(f'model/{MODEL}/cells'):
        os.mkdir(f'model/{MODEL}/cells')

    for i in range(n_clusters):
        points_true = dataset[struiden_pipeline.medoid_idx[i]]
        points_pred = struiden_pipeline.reconstructure_predict(np.expand_dims(points_true, axis=0))[0]

        n = len(points_true)
        cell_centers = np.concatenate([points_true, points_pred])
        cell_labels = [0] * n + [i] * n
        cell_xyz = f'model/{MODEL}/cells/cell_{i}.xyz'
        viz_tool.save_xyz(cell_xyz, cell_centers, cell_labels)
        
        cell_true_centers = points_true
        cell_true_labels = [i] * n
        cell_true_xyz = f'model/{MODEL}/cells/cell_true_{i}.xyz'
        viz_tool.save_xyz(cell_true_xyz, cell_true_centers, cell_true_labels)

        cell_pred_centers = points_pred
        cell_pred_labels = [i] * n
        cell_pred_xyz = f'model/{MODEL}/cells/cell_pred_{i}.xyz'
        viz_tool.save_xyz(cell_pred_xyz, cell_pred_centers, cell_pred_labels)
    
    print('[*] Save cells xyz files successfully!')
    
    plt.figure(figsize=(10*n_clusters, 10))
    for i in range(n_clusters):
        cell_xyz = f'model/{MODEL}/cells/cell_{i}.xyz'
        cell_png = f'model/{MODEL}/cells/cell_{i}.png'
        viz_tool.draw_xyz(cell_png, cell_xyz)
        plt.subplot(1, n_clusters, i+1)
        image = plt.imread(cell_png)
        plt.imshow(image)
        plt.title(f'loss: {struiden_pipeline.cell_losses[i]:.6f}', fontdict={'fontsize':16})
        plt.axis('off')
    plt.savefig(f'model/{MODEL}/cells/cells.png')
    plt.close()

    plt.figure(figsize=(10*n_clusters, 10))
    for i in range(n_clusters):
        cell_true_png = f'model/{MODEL}/cells/cell_true_{i}.png'
        cell_true_xyz = f'model/{MODEL}/cells/cell_true_{i}.xyz'
        viz_tool.draw_xyz(cell_true_png, cell_true_xyz)
        plt.subplot(1, n_clusters, i+1)
        image = plt.imread(cell_true_png)
        plt.imshow(image)
        plt.axis('off')
    plt.savefig(f'model/{MODEL}/cells/cells_true.png')
    plt.close()

    plt.figure(figsize=(10*n_clusters, 10))
    for i in range(n_clusters):
        cell_pred_png = f'model/{MODEL}/cells/cell_pred_{i}.png'
        cell_pred_xyz = f'model/{MODEL}/cells/cell_pred_{i}.xyz'
        viz_tool.draw_xyz(cell_pred_png, cell_pred_xyz)
        plt.subplot(1, n_clusters, i+1)
        image = plt.imread(cell_pred_png)
        plt.imshow(image)
        plt.axis('off')
    plt.savefig(f'model/{MODEL}/cells/cells_pred.png')
    plt.close()

    print('[*] Save cells image successfully!')

    # Save packings
    if DRAW_PACK and DATASET[-3:] not in {'npy', 'npz'}:
        if not os.path.exists(f'model/{MODEL}/packing'):
            os.mkdir(f'model/{MODEL}/packing')
        
        report = pd.DataFrame()
        if not NON_CENTER:
            for i in range(len(pointcloud_dataset)):
                lattice_i = pointcloud_dataset.lattice_list[i]
                centers_i = pointcloud_dataset.centers_list[i]
                neighbors_i = pointcloud_dataset.neighbors_list[i]
                filename_i = pointcloud_dataset.filename_list[i]

                loss_i, labels_i, probas_i = struiden_pipeline.cluster_predict(neighbors_i)
                info = {'density': float(filename_i[:-4].split('_')[-1]), 'loss': loss_i, 'clusters': len(Counter(labels_i))-1, 'ratio': sum(labels_i) / len(labels_i)}

                pack_xyz = f'model/{MODEL}/packing/{filename_i}'
                pack_png = f'model/{MODEL}/packing/{os.path.splitext(filename_i)[0]}.png'
                viz_tool.save_xyz(pack_xyz, centers_i, labels_i, probas_i, lattice=lattice_i)
                viz_tool.draw_xyz(pack_png, pack_xyz)

                if not os.path.exists(f'model/{MODEL}/baseline'):
                    os.makedirs(f'model/{MODEL}/baseline')

                if DIMEN_SIZE == 2 and CLOUD_SIZE == 6:
                    comp_gif = f'model/{MODEL}/baseline/{os.path.splitext(filename_i)[0]}.gif'
                    comp_png = f'model/{MODEL}/baseline/{os.path.splitext(filename_i)[0]}.png'
                    cluster_metrics = viz_tool.disk_comparison(pack_xyz, pack_png, comp_gif, comp_png)
                    info.update(cluster_metrics)
                elif DIMEN_SIZE == 3 and CLOUD_SIZE == 12:
                    comp_gif = f'model/{MODEL}/baseline/{os.path.splitext(filename_i)[0]}.gif'
                    comp_png = f'model/{MODEL}/baseline/{os.path.splitext(filename_i)[0]}.png'
                    cluster_metrics = viz_tool.sphere_comparsion(pack_xyz, pack_png, comp_gif, comp_png)
                    info.update(cluster_metrics)
                    
                report = report.append(info, ignore_index=True)

        else:
            for i in range(len(pointcloud_dataset)):
                lattice_i = pointcloud_dataset.lattice_list[i]
                centers_i = pointcloud_dataset.centers_list[i]
                neighbors_i = pointcloud_dataset.neighbors_list[i]
                filename_i = pointcloud_dataset.filename_list[i]

                loss_i, labels_i, probas_i = struiden_pipeline.cluster_predict(neighbors_i)
                info = {'density': float(filename_i[:-4].split('_')[-1]), 'loss': loss_i, 'clusters': len(Counter(labels_i))-1, 'ratio': sum(labels_i) / len(labels_i)}
                centers_i_, labels_i_, probas_i_ = [], [], []
                for j in range(len(centers_i)):
                    modified_neighbour = neighbors_i[j] + centers_i[j]
                    for neigh in modified_neighbour:
                        if (0 <= neigh).all() and (neigh < np.diagonal(lattice_i)).all():
                            centers_i_ += [neigh]
                            labels_i_ += [labels_i[j]]
                            probas_i_ += [probas_i[j]]
                info = {'density': float(filename_i[:-4].split('_')[2]), 'loss': loss_i, 'clusters': len(Counter(labels_i))-1}

                pack_xyz = f'model/{MODEL}/packing/{filename_i}'
                pack_png = f'model/{MODEL}/packing/{os.path.splitext(filename_i)[0]}.png'
                viz_tool.save_xyz(pack_xyz, centers_i_, labels_i_, probas_i_, lattice=lattice_i)
                viz_tool.draw_xyz(pack_png, pack_xyz)

                report = report.append(info, ignore_index=True)

        report.to_csv(f'model/{MODEL}/report.csv')
        print('[*] Save packing image successfully!')
    
    # Plot gif
    if PLOT_GIF:
        if not os.path.exists(f'model/{MODEL}/cluster/frames'):
            os.mkdir(f'model/{MODEL}/cluster/frames')

        for ckpt_epoch, ckpt in zip(ckpt_epochs, ckpts):

            checkpoint = tf.train.Checkpoint(model)
            checkpoint.restore(ckpt).expect_partial()
            ckpt_encoded = model.encoder.predict(dataset, batch_size=BATCH_SIZE)
            ckpt_loss = model.evaluate(dataset, dataset, verbose=2, batch_size=BATCH_SIZE)

            plt.figure(figsize=(10, 10))
            viz_tool.plot_cluster(ckpt_encoded[:,:2], struiden_pipeline.labels, struiden_pipeline.probabilities, ckpt_epoch, ckpt_loss)
            plt.savefig(f'model/{MODEL}/cluster/frames/cluster_{ckpt_epoch:06d}.png')
            plt.close()

        frames = []
        image_list = os.listdir(f'model/{MODEL}/cluster/frames')
        image_list.sort()
        for image_name in image_list:
            frames.append(imageio.imread(f'model/{MODEL}/cluster/frames/{image_name}'))
        imageio.mimsave(f'model/{MODEL}/cluster/cluster.gif', frames, duration=0.2)

        print('[*] Save cluster gif successfully!')

    # For export
    if EXPORT_PIPE:
        if not os.path.exists(f'model/{MODEL}/export'):
            os.mkdir(f'model/{MODEL}/export')

        labels = struiden_pipeline.labels
        probas = struiden_pipeline.probabilities
        # cluster without ticks
        cluster_colors = viz_tool.color_palette(labels, probas)
        plt.figure(figsize=(10,10))
        scatter = plt.scatter(viz_points[:,0], viz_points[:,1], c=cluster_colors, alpha=0.2)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(f'model/{MODEL}/export/cluster.png')
        plt.close()

        # raw cluster
        plt.figure(figsize=(10,10))
        scatter = plt.scatter(viz_points[:,0], viz_points[:,1], color=viz_tool.palette[0], alpha=0.2)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(f'model/{MODEL}/export/cluster_raw.png')
        plt.close()

        labels_sample = [1, 0, 2, 1, 1, 2, 2]
        for i, label in enumerate(labels_sample):
            cell_sample = random.sample(list(dataset[labels==label]), 1)[0]
            # colored
            cell_sample_labels = [label] * len(cell_sample)
            cell_sample_xyz = f'model/{MODEL}/export/cell_sample_{i}.xyz'
            viz_tool.save_xyz(cell_sample_xyz, cell_sample, cell_sample_labels)
            cell_sample_png = f'model/{MODEL}/export/cell_sample_{i}.png'
            viz_tool.draw_xyz(cell_sample_png, cell_sample_xyz)
            # raw
            cell_sample_labels = [0] * len(cell_sample)
            cell_sample_xyz = f'model/{MODEL}/export/cell_sample_{i}_raw.xyz'
            viz_tool.save_xyz(cell_sample_xyz, cell_sample, cell_sample_labels)
            cell_sample_png = f'model/{MODEL}/export/cell_sample_{i}_raw.png'
            viz_tool.draw_xyz(cell_sample_png, cell_sample_xyz)
        
        # colored
        plt.figure(figsize=(10,10))
        loc = [(0,1),(0,3), (1,0),(1,2),(1,4), (2,1),(2,3)]
        for i in range(7):
            plt.subplot2grid((3,6), loc[i], colspan=2)
            image = Image.open(f'model/{MODEL}/export/cell_sample_{i}.png')
            image = image.crop((250,250,750,750))
            plt.imshow(image)
            plt.axis('off')
        plt.savefig(f'model/{MODEL}/export/cells.png')
        plt.close()

        # raw
        plt.figure(figsize=(10,10))
        for i in range(7):
            plt.subplot2grid((3,6), loc[i], colspan=2)
            image = Image.open(f'model/{MODEL}/export/cell_sample_{i}_raw.png')
            image = image.crop((250,250,750,750))
            plt.imshow(image)
            plt.axis('off')
        plt.savefig(f'model/{MODEL}/export/cells_raw.png')
        plt.close()
        
        # draw example
        example_dataset = utils.PointCloudDataset('data/sphere_packing_0.701597.xyz', CLOUD_SIZE)
        dataset = example_dataset.make_dataset()
        
        _, example_labels, example_probas = struiden_pipeline.cluster_predict(dataset)
        example_centers = example_dataset.centers_list[0]
        example_lattice = example_dataset.lattice_list[0]
        # colored
        pack_xyz = f'model/{MODEL}/export/example.xyz'
        pack_png = f'model/{MODEL}/export/example.png'
        viz_tool.save_xyz(pack_xyz, example_centers, example_labels, example_probas, lattice=example_lattice)
        viz_tool.draw_xyz(pack_png, pack_xyz)
        # raw
        pack_raw_xyz = f'model/{MODEL}/export/example_raw.xyz'
        pack_raw_png = f'model/{MODEL}/export/example_raw.png'
        viz_tool.save_xyz(pack_raw_xyz, example_centers, [0] * len(example_centers), lattice=example_lattice)
        viz_tool.draw_xyz(pack_raw_png, pack_raw_xyz)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Setting Testing Parameters')
    parser.add_argument('-m', '--model', type=str,
                        help='model name in application.py')
    parser.add_argument('-dp', '--draw_pack', action='store_true',
                        help='whether to draw packing')
    parser.add_argument('-pg', '--plot_gif', action='store_true',
                        help='whether to plot gif')
    parser.add_argument('-pc', '--plot_comp', action='store_true',
                        help='whether to plot comparison')
    parser.add_argument('-ep', '--export_pipe', action='store_true',
                        help='whether to export pipeline')                        
    args = parser.parse_args()
    
    MODEL = args.model
    DRAW_PACK = args.draw_pack
    PLOT_COMP = args.plot_comp
    PLOT_GIF = args.plot_gif
    EXPORT_PIPE = args.export_pipe

    if MODEL == None:
        dirs = os.listdir('model')
        dirs.sort()
        MODEL = dirs[-1]

    test(MODEL, DRAW_PACK, PLOT_COMP, PLOT_GIF, EXPORT_PIPE)