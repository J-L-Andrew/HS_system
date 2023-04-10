import colorsys
import itertools
import math
import os
from collections import Counter, defaultdict
from icecream import ic

import freud
import hdbscan
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import tensorflow as tf
from ovito.io import export_file, import_file
from ovito.modifiers import (AcklandJonesModifier,
                             CommonNeighborAnalysisModifier,
                             PolyhedralTemplateMatchingModifier)
from ovito.vis import ParticlesVis, TachyonRenderer, Viewport
from ovito.data import NearestNeighborFinder
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture

# plt.rcParams.update({'font.size': 18})

def cluster_accuracy(y_true, y_pred):
    y_true_ = y_true

    n = max(max(y_true)+1, max(y_pred)+1)

    accuracy_list = []
    confusion_list = []
    for perm in itertools.permutations(range(0,n)):
        y_pred_ = [perm[i] for i in y_pred]
        accuracy_list.append(accuracy_score(y_true_, y_pred_))
        confusion_list.append(confusion_matrix(y_true_, y_pred_))
    
    max_idx = np.argmax(accuracy_list)
    return accuracy_list[max_idx], confusion_list[max_idx]

class PointCloudDataset():

    def __init__(self, data_path, cloud_size, non_center):
        self.data_path = data_path
        self.cloud_size = cloud_size
        self.non_center = non_center
        dataset_name = data_path.split('/')[-1] + '_' + str(cloud_size) + '_' + str(non_center)
        if os.path.exists(f'data/{dataset_name}.npz'):
            dataset = np.load(f'data/{dataset_name}.npz', allow_pickle=True)
            self.lattice_list = np.array(dataset['lattice_list'])
            self.centers_list = np.array(dataset['centers_list'])
            self.neighbors_list = np.array(dataset['neighbors_list'])
            self.filename_list = np.array(dataset['filename_list'])

        else:
            self.read_packing()
            np.savez_compressed(f'data/{dataset_name}.npz',
                lattice_list=self.lattice_list,
                centers_list=self.centers_list,
                neighbors_list=self.neighbors_list,
                filename_list=self.filename_list)

    def __len__(self):
        return len(self.neighbors_list)

    # def read_packing(self):
    #     if os.path.isfile(self.data_path):
    #         if self.data_path[-3:] == 'xyz':
    #             box, points = self.read_xyz(self.data_path)
    #             centers, neighbors = self.get_neighbors(box, points)
    #             filename = os.path.split(self.data_path)[-1]
    #         elif self.data_path[-3:] == 'npy':
    #             neighbors = np.load(self.data_path)[:, :self.cloud_size, :]
    #             centers = None
    #             box = None
    #             filename = None

    #         self.box_list = np.array([box])
    #         self.centers_list = np.array([centers])
    #         self.neighbors_list = np.array([neighbors])
    #         self.filename_list = np.array([filename])
    #     else:
    #         box_list = []
    #         centers_list = []
    #         neighbors_list = []
    #         filename_list = []
    #         pack_list = os.listdir(self.data_path)
    #         pack_list.sort()
    #         for pack_name in pack_list:
    #             box, points = self.read_xyz(os.path.join(self.data_path, pack_name))
    #             centers, neighbors = self.get_neighbors(box, points)
    #             box_list.append(box)
    #             centers_list.append(centers)
    #             neighbors_list.append(neighbors)
    #             filename_list.append(pack_name)

    #         self.box_list = np.array(box_list)
    #         self.centers_list = np.array(centers_list)
    #         self.neighbors_list = np.array(neighbors_list)
    #         self.filename_list = np.array(filename_list)

    # def read_xyz(self, filename):
    #     with open(filename, 'r') as f:
    #         N = int(next(f))
    #         L = list(map(float, next(f).split()))
    #         points = np.loadtxt(f)
    #         assert len(points) == N

    #         if points.shape[1] == 3:
    #             box = freud.Box(*L)
    #         elif points.shape[1] == 2:
    #             box = freud.Box(*L, is2D=True)
    #         return box, points

    # def get_neighbors(self, box, points):
    #     # 2D -> 3D
    #     if box.is2D:
    #         points = np.insert(points, 2, 0, -1)

    #     linkcell = freud.locality.LinkCell(box, points)
    #     query_args = dict(mode='nearest', num_neighbors=self.cloud_size, r_min=0.1)
    #     query = linkcell.query(points, query_args)
    #     nlist = query.toNeighborList()

    #     centers = [[] for i in range(nlist.num_query_points)]
    #     neighbors = [[] for i in range(nlist.num_query_points)]
    #     for i, j in nlist[:]:
    #         point = points[j] - points[i]
    #         for k in range(3):
    #             if point[k] > box.L[k] / 2:
    #                 point[k] -= box.L[k]
    #             elif point[k] < -box.L[k] / 2:
    #                 point[k] += box.L[k]
    #         centers[i] = points[i]
    #         neighbors[i].append(point)

    #     centers, neighbors = np.array(centers), np.array(neighbors)

    #     # 3D -> 2D
    #     if box.is2D:
    #         centers = np.delete(centers, 2, -1)
    #         neighbors = np.delete(neighbors, 2, -1)
            
    #     return centers, neighbors

    def make_dataset(self, max_length=np.inf):
        dataset = np.concatenate(self.neighbors_list, axis=0)
        ic(dataset.shape)
        dataset_size = len(dataset)
        if dataset_size > max_length:
            # np.random.seed(2021)
            self.slice = np.random.choice(dataset_size, size=max_length, replace=False)
            dataset = dataset[self.slice]
            
        return dataset

    def read_packing(self):
        lattice_list = []
        centers_list = []
        neighbors_list = []
        filename_list = []
        pack_list = os.listdir(self.data_path)
        pack_list.sort()
        for pack_name in pack_list:
            filename = os.path.join(self.data_path, pack_name)
            lattice, centers, neighbors = self.get_neighbors(filename)

            lattice_list.append(lattice)
            centers_list.append(centers)
            neighbors_list.append(neighbors)
            filename_list.append(pack_name)

        self.lattice_list = np.array(lattice_list)
        self.centers_list = np.array(centers_list)
        self.neighbors_list = np.array(neighbors_list)
        self.filename_list = np.array(filename_list)

    def get_neighbors(self, filename):
        pipeline = import_file(filename)
        data = pipeline.source.data
        shape_name = filename.split('/')[-1].split('(')[0]
        if shape_name == 'Sphere':
            data.cell_.pbc = [True, True, True]
            dim = 3
        elif shape_name == 'Circle':
            data.cell_.pbc = [True, True, False]
            dim = 2
        
        if self.non_center:
            centers, neighbors = self.noncenter_particle(data)
        else:
            centers, neighbors = self.center_particle(data)
            
        return data.cell[:3,:3], np.array(centers)[...,:dim], np.array(neighbors)[...,:dim]

    def center_particle(self, data):
        centers, neighbors = [], []
        finder = NearestNeighborFinder(self.cloud_size, data)

        for index in range(data.particles.count):
            centers.append(data.particles["Position"][index])
            neighbor = []
            for neigh in finder.find(index):
                neighbor.append(neigh.delta)
            neighbors.append(neighbor)

        return centers, neighbors

    def noncenter_particle(self, data):
        grids = defaultdict(list)
        box_len = data.cell.matrix[0][0]
        positions = data.particles["Position"]

        resolution = 1
        grid_num = int(box_len // resolution)
        grid_len = box_len / grid_num
        for index in range(data.particles.count):
            grid_pos = tuple(map(int, positions[index] // grid_len))
            grids[grid_pos].append(index)

        centers = []
        neighbors = []
        neighbor_indices = set()
        for i in range(grid_num):
            for j in range(grid_num):
                for k in range(grid_num):
                    corrected = {}
                    distance = {}
                    pixel = np.array([i+0.5, j+0.5, k+0.5]) * grid_len
                    for p in range(-2,3):
                        for q in range(-2,3):
                            for r in range(-2,3):
                                grid_pos = [i+p, j+q, k+r]
                                offset = np.zeros(3)
                                for t in range(3):
                                    if grid_pos[t] < 0:
                                        grid_pos[t] += grid_num
                                        offset[t] = -box_len
                                    elif grid_pos[t] >= grid_num:
                                        grid_pos[t] -= grid_num
                                        offset[t] = box_len

                                for index in grids[tuple(grid_pos)]:
                                    particle_pos = positions[index] + offset
                                    distance[index] = np.linalg.norm(pixel - particle_pos)
                                    corrected[index] = particle_pos

                    nearest = sorted(distance, key=distance.get)[:self.cloud_size]
                    if tuple(nearest) not in neighbor_indices:
                        neighbor_indices.add(tuple(nearest))
                        point_cloud = np.array([corrected[index] for index in nearest])
                        neighbors.append(point_cloud - point_cloud.mean(axis=0))
                        centers.append(point_cloud.mean(axis=0))
        
        return centers, neighbors

class StructureIdentificationPipeline():

    def __init__(self, network_model, dataset, n_clusters, method='gmm'):
        self.network_model = network_model
        self.dataset = dataset
        self.n_clusters = n_clusters
        self.batch_size = self.network_model.batch_size
        self.method = method
        self.build_pipeline()

    def build_pipeline(self):
        
        # autoencoder
        encoded = self.network_model.encoder.predict(self.dataset, batch_size=self.batch_size)
        loss = self.network_model.evaluate(self.dataset, self.dataset, verbose=2, batch_size=self.batch_size)
        
        # gausian mixture model
        if self.method == 'gmm':
            self.cluster = GaussianMixture(n_components=self.n_clusters, warm_start=True, n_init=10, max_iter=1000)
            original_labels = self.cluster.fit_predict(encoded)
            probabilities = np.max(self.cluster.predict_proba(encoded), axis=-1)
            means = self.cluster.means_
        elif self.method == 'hdbscan':
            self.cluster = hdbscan.HDBSCAN(min_cluster_size = len(self.dataset) // 10,
                                        min_samples = 50,
                                        alpha=2.,
                                        cluster_selection_epsilon=2,
                                        gen_min_span_tree=True,
                                        prediction_data=True)
            self.cluster.fit(encoded)
            original_labels = self.cluster.labels_
            probabilities = self.cluster.probabilities_
            self.n_clusters = max(original_labels) + 1
            means = [self.cluster.weighted_cluster_medoid(i) for i in range(self.n_clusters)]

        # medoid loss
        self.network_model.batch_size = 1

        medoid_idx = []
        cell_losses = []
        for i in range(self.n_clusters):
            medoid = means[i]
            idx = np.argmin(np.linalg.norm(encoded - medoid, axis=-1))
            medoid_idx.append(idx)
            X_true = np.expand_dims(self.dataset[idx], axis=0)
            cell_loss = self.network_model.evaluate(X_true, X_true, verbose=2, batch_size=1)
            cell_losses.append(cell_loss)

        self.network_model.batch_size = self.batch_size

        # label map
        #           0    1    2
        # losses = [1.1, 3.3, 2.2]
        # new2old = [1, 2, 0]
        # old2new = [2, 0, 1]
        new2old = list(reversed(np.argsort(cell_losses)))
        old2new = {j:i for i,j in enumerate(new2old)}
        old2new[-1] = -1
        labels = np.array([old2new[i] for i in original_labels])
        self.label_map = old2new

        # convert medoid & cell_losses
        self.medoid_idx = [medoid_idx[new2old[i]] for i in range(self.n_clusters)]
        self.cell_losses = [cell_losses[new2old[i]] for i in range(self.n_clusters)]

        self.encoded = encoded
        self.labels = labels
        self.probabilities = probabilities

    def cluster_predict(self, point_cloud):
        self.network_model.batch_size = len(point_cloud)

        encoded = self.network_model.encoder.predict(point_cloud, batch_size=len(point_cloud))
        loss = self.network_model.evaluate(point_cloud, point_cloud, verbose=2, batch_size=len(point_cloud))
        
        if self.method == 'gmm':
            original_labels = self.cluster.predict(encoded)
            probas = np.max(self.cluster.predict_proba(encoded), axis=-1)
        elif self.method == 'hdbscan':
            original_labels, probas = hdbscan.approximate_predict(self.cluster, encoded)
        
        labels = np.array([self.label_map[i] for i in original_labels])

        self.network_model.batch_size = self.batch_size

        return loss, labels, probas

    def reconstructure_predict(self, point_cloud):
        self.network_model.batch_size = len(point_cloud)

        reconstructed = self.network_model.predict(point_cloud, batch_size=len(point_cloud))

        self.network_model.batch_size = self.batch_size

        return reconstructed

class VisualizationTool():

    def __init__(self, radius, plot_comp):
        self.palette = []
        for c in sns.color_palette('Set1', n_colors=6):
            nc = colorsys.rgb_to_hls(*c)
            self.palette.append(colorsys.hls_to_rgb(nc[0],0.75,1))
        self.palette.insert(0, colorsys.hls_to_rgb(0,0.75,0))

        self.baseline_palette = []
        for c in sns.color_palette('Set2', n_colors=6):
            nc = colorsys.rgb_to_hls(*c)
            self.baseline_palette.append(colorsys.hls_to_rgb(nc[0],0.75,1))
        self.baseline_palette.insert(0, colorsys.hls_to_rgb(0,0.75,0))

        self.radius = radius
        self.plot_comp = plot_comp

    def color_palette(self, labels, probabilities):
        colors = []
        for col, pro in zip(labels, probabilities):
            hls = colorsys.rgb_to_hls(*self.palette[col])
            # rgb = colorsys.hls_to_rgb(hls[0], -0.3*pro+0.8, hls[2])
            rgb = colorsys.hls_to_rgb(hls[0], -0.25*pro+1, hls[2])
            colors.append(rgb)
        return colors

    def plot_cluster(self, points, labels, probabilities, epoch=None, loss=None, medoid_idx=[]):
        latent_dim = points.shape[1]
        cluster_colors = self.color_palette(labels, probabilities)
        # 2D
        if latent_dim == 2:
            scatter = plt.scatter(points[:,0], points[:,1], c=cluster_colors, alpha=0.2)
            plt.xlabel('Feature1')
            plt.ylabel('Feature2')
            if len(medoid_idx):
                plt.scatter(points[medoid_idx,0], points[medoid_idx,1], c=self.palette[:len(medoid_idx)], linewidths=1, edgecolors='k')
            if epoch and loss:
                plt.title(f'epoch: {epoch} | loss: {loss:.6f}')

            # insect
            if -1 not in labels:
                plt.axes([0.7, 0.68, 0.2, 0.2])
                count = np.bincount(labels)
                plt.pie(count, colors=self.palette[:len(count)], autopct='%1.1f%%')
                plt.axis('off')
            else:
                plt.axes([0.7, 0.68, 0.2, 0.2])
                count = np.bincount(labels+1)
                plt.pie(count, colors=[[1,1,1]] + self.palette[:len(count)-1], autopct='%1.1f%%')
                plt.axis('off')

        elif latent_dim == 3:
            elevs = [30, 30, 30, 30]
            azims = [45, 135, 225, 315]

            for i in range(4):
                ax = plt.subplot(2, 2, i+1, projection='3d')
                ax.set(xlabel='X', ylabel='Y', zlabel='Z')
                ax.view_init(elev=elevs[i], azim=azims[i])

                scatter = ax.scatter(points[:,0], points[:,1], points[:,2], c=cluster_colors, alpha=0.2)
                if len(medoid_idx):
                    ax.scatter(points[medoid_idx,0], points[medoid_idx,1], points[medoid_idx,2], c=self.palette[:len(medoid_idx)], linewidths=1, edgecolors='k')
                if loss and epoch:
                    ax.title(f'epoch: {epoch} | loss: {loss:.6f}')

            # insect
            plt.axes([0.8, 0.78, 0.2, 0.2])
            count = np.bincount(labels)
            plt.pie(count, colors=self.palette[:len(count)], autopct='%1.1f%%')
            plt.axis('off')

    def ovito_preprocess(self, shape):
        if shape == 'disk':
            def circle_preprocess(frame, data):
                if 'Comment' in data.attributes:
                    comment = data.attributes['Comment']
                    lattice = np.fromstring(comment, dtype=float, sep=' ').reshape(3, 3)
                    cell_mat = np.hstack([lattice, [[0.],[0.],[0.]]])
                    data.cell_.matrix = cell_mat
                data.cell_.pbc = (True, True, False)
                data.particles.vis.radius = self.radius
                data.particles.vis.shape = ParticlesVis.Shape.Cylinder

            return circle_preprocess
            
        elif shape == 'sphere':
            def sphere_preprocess(frame, data):
                if 'Comment' in data.attributes:
                    comment = data.attributes['Comment']
                    lattice = np.fromstring(comment, dtype=float, sep=' ').reshape(3, 3)
                    cell_mat = np.hstack([lattice, [[0.],[0.],[0.]]])
                    data.cell_.matrix = cell_mat
                data.cell_.pbc = (True, True, True)
                data.particles.vis.radius = self.radius

            return sphere_preprocess

    def disk_comparison(self, pack_xyz, pack_png, comp_gif, comp_png):
        def hexatic_order(frame, data):
            box = freud.Box.from_matrix(data.cell.matrix[:3,:3])
            points = data.particles.position[:]
            hex_order = freud.order.Hexatic(k=6)
            hex_order.compute(system=(box, points))
            label = [1 if prob > THRESHOLD else 0 for prob in np.abs(hex_order.particle_order)]
            data.particles_.create_property('Structure Type', data=label)

        def disk_baseline(metrics):
            pipeline = import_file(pack_xyz, columns=['Position.X', 'Position.Y', 'Particle Type'])
            pipeline.modifiers.append(self.ovito_preprocess('disk'))
            pipeline.modifiers.append(hexatic_order)
            data = pipeline.compute()
            pred_type = data.particles.particle_types[:]
            stru_type = data.particle_properties.structure_types[:]
            export_file(pipeline, f'{pack_xyz[:-4]}_true.xyz', 'xyz', columns=['Position.X', 'Position.Y', 'Particle Type'])
            
            if self.plot_comp:
                pipeline.add_to_scene()
                vp = Viewport(type = Viewport.Type.Ortho, camera_dir = (0, 0, -1))
                vp.zoom_all(size=(1000,1000))
                qimage = vp.render_image(filename=comp_png, size=(1000,1000), renderer=TachyonRenderer())
                pipeline.remove_from_scene()
                image = Image.fromqimage(qimage)
            else:
                image = None
            
            rand_score = sklearn.metrics.rand_score(stru_type, pred_type)

            return rand_score, image

        cluster_metrics = {}
        baseline_images = []
        metrics_dict = {'Hex.90': 0.9,
                        'Hex.95': 0.95,
                        'Hex.99': 0.99}
        for metrics in metrics_dict.keys():
            THRESHOLD = metrics_dict[metrics]
            rand_score, image = disk_baseline(metrics)
            cluster_metrics[metrics] = rand_score
            baseline_images.append(image)

        # Plot gif
        if self.plot_comp:
            frame1 = np.concatenate(baseline_images, axis=-2)
            images = [imageio.imread(pack_png)] * 3 
            frame2 = np.concatenate(images, axis=-2)
            imageio.mimsave(comp_gif, [frame1, frame2], duration=1)

        return cluster_metrics

    def sphere_comparsion(self, pack_xyz, pack_png, comp_gif, comp_png):
        def sphere_baseline(metrics):
            modifier = metrics_dict[metrics]
            pipeline = import_file(pack_xyz, columns=['Position.X', 'Position.Y', 'Position.Z', 'Particle Type'])
            pipeline.modifiers.append(self.ovito_preprocess('sphere'))
            pipeline.modifiers.append(modifier)
            data = pipeline.compute()
            pred_type = data.particles.particle_types[:]
            stru_type = data.particle_properties.structure_types[:]

            for i in range(5):
                modifier.structures[i].color = self.baseline_palette[i]

            if self.plot_comp:
                pipeline.add_to_scene()
                vp = Viewport(type = Viewport.Type.Ortho, camera_dir = (-1, -1, -1))
                vp.zoom_all(size=(1000,1000))
                qimage = vp.render_image(filename=comp_png, size=(1000,1000), renderer=TachyonRenderer())
                pipeline.remove_from_scene()
                image = Image.fromqimage(qimage)
            else:
                image = None
            
            rand_score = sklearn.metrics.rand_score(stru_type, pred_type)

            return rand_score, image

        cluster_metrics = {}
        baseline_images = []

        metrics_dict = {'AJA': AcklandJonesModifier(),
                        'CNA': CommonNeighborAnalysisModifier(),
                        'PTM': PolyhedralTemplateMatchingModifier()}
        for metrics in metrics_dict.keys():
            rand_score, image = sphere_baseline(metrics)
            cluster_metrics[metrics] = rand_score
            baseline_images.append(image)

        # Plot gif
        if self.plot_comp:
            frame1 = np.concatenate(baseline_images, axis=-2)
            images = [imageio.imread(pack_png)] * 3 
            frame2 = np.concatenate(images, axis=-2)
            imageio.mimsave(comp_gif, [frame1, frame2], duration=1)

        return cluster_metrics

    def save_xyz(self, filename, centers, labels, probabilities=[], lattice=[]):
        if len(probabilities) == 0:
            probabilities = [1] * len(centers)

        n = len(centers)
        colors = self.color_palette(labels, probabilities)
        with open(filename, 'w') as f:
            f.write(str(n) + '\n')
            f.write(' '.join([str(L) for L in np.array(lattice).flat]) + '\n')
            np.savetxt(f, np.column_stack([centers, labels, colors]))

    def draw_xyz(self, filename, xyz_file, camera_dir=(0, 0, -1)):
        with open(xyz_file) as f:
            next(f)
            L = next(f)
            vis_cell = L != '\n'
            pos = next(f)
            dimen_size = len(pos.split()) - 4

        if dimen_size == 2:
            pipeline = import_file(xyz_file, columns=['Position.X', 'Position.Y', 'Particle Type', 'Color.R', 'Color.G', 'Color.B'])
            pipeline.source.data.cell.vis.enabled = vis_cell
            pipeline.modifiers.append(self.ovito_preprocess('disk'))

            pipeline.add_to_scene()
            # vp = Viewport(type = Viewport.Type.Ortho, camera_dir=(0, 0, -1))
            vp = Viewport(type = Viewport.Type.Ortho, camera_dir=camera_dir)
            vp.zoom_all(size=(1000,1000))
            vp.render_image(filename=filename, size=(1000,1000), renderer=TachyonRenderer())
            pipeline.remove_from_scene()
        else:
            pipeline = import_file(xyz_file, columns=['Position.X', 'Position.Y', 'Position.Z', 'Particle Type', 'Color.R', 'Color.G', 'Color.B'])
            pipeline.source.data.cell.vis.enabled = vis_cell
            pipeline.modifiers.append(self.ovito_preprocess('sphere'))

            pipeline.add_to_scene()
            # vp = Viewport(type = Viewport.Type.Ortho, camera_dir=(-1, -1, -1))
            vp = Viewport(type = Viewport.Type.Ortho, camera_dir=camera_dir)
            vp.zoom_all(size=(1000,1000))
            vp.render_image(filename=filename, size=(1000,1000), renderer=TachyonRenderer())
            pipeline.remove_from_scene()

# def gaussian_mixture_cluster(encoded, prefix):
#     min_bic = np.inf
#     bic_list = []
#     aic_list = []
#     n_range = list(range(1,10))
#     for n_components in n_range:
#         gmm = GaussianMixture(n_components=n_components, warm_start=True)
#         gmm.fit(encoded)
#         aic = gmm.aic(encoded)
#         aic_list.append(aic)
#         bic = gmm.bic(encoded)
#         bic_list.append(bic)
#         if bic < min_bic:
#             min_bic = bic
#             best_gmm = gmm
#             best_n = n_components

#     plt.figure(figsize=(8,6))
#     plt.plot(n_range, aic_list, 'o:')
#     plt.xlabel('N')
#     plt.ylabel('AIC')
#     plt.savefig(os.path.join(prefix, 'aic.png'))
#     plt.close()

#     plt.figure(figsize=(8,6))
#     plt.plot(n_range, bic_list, 'o:')
#     plt.xlabel('N')
#     plt.ylabel('BIC')
#     plt.savefig(os.path.join(prefix, 'bic.png'))
#     plt.close()

#     k_range = list(range(best_n-1, 0, -1))
#     proba = best_gmm.predict_proba(encoded)

#     def combine_two(proba):
#         N, K = proba.shape
#         comb = itertools.combinations(range(K), 2)
#         Sk_list = []
#         merged_list = []
#         for i, j in comb:
#             merged = np.delete(proba, j, -1)
#             merged[:, i] = proba[:, i] + proba[:, j]
#             Sk = -np.sum(merged * np.log(merged + 1e-6))
#             Sk_list.append(Sk)
#             merged_list.append(merged)
#         idx = np.argmin(Sk_list)
#         return Sk_list[idx], merged_list[idx]

#     Sk_list = []
#     proba_list = []
#     for k in k_range:
#         Sk, proba = combine_two(proba)
#         Sk_list.append(Sk)
#         proba_list.append(proba)

#     plt.figure(figsize=(8,6))
#     plt.plot(k_range, Sk_list, 'o:')
#     plt.xlabel('K')
#     plt.ylabel('$S_k$')
#     plt.savefig(os.path.join(prefix, 'entropy.png'))
#     plt.close()

#     # n = int(input('Best n components: '))
#     n = 5

#     proba = proba_list[n]
#     labels = np.argmax(proba, axis=-1)
#     probabilities = np.max(proba, axis=-1)
#     best_gmm = gmm

#     # gmm = GaussianMixture(n_components=n, warm_start=True, n_init=10, max_iter=1000)
#     # labels = gmm.fit_predict(encoded)
#     # probabilities = np.max(gmm.predict_proba(encoded), axis=-1)
#     return labels, probabilities, gmm


# def draw_xyz(box, centers, labels, probabilities):
#     scene = fresnel.Scene()
#     geometry = fresnel.geometry.Sphere(scene, N=len(centers))
#     geometry.material = fresnel.material.Material(roughness=0.8)
#     geometry.outline_width = 0.01
#     geometry.position[:] = centers - np.mean(centers, axis=-2)
#     geometry.radius[:] = [1]
#     if box:
#         bbox = fresnel.geometry.Box(scene, box, box_radius=0.1)
#     # colors
#     cluster_colors = [sns.desaturate(palette[col], sat) for col, sat in zip(labels, probabilities)]
#     geometry.material.primitive_color_mix = 1
#     geometry.color[:] = fresnel.color.linear(cluster_colors)
#     # camera
#     scene.camera = fresnel.camera.Orthographic.fit(scene)
#     img = fresnel.preview(scene, w=1000, h=1000)
#     image = Image.fromarray(img[:], mode='RGBA')
#     return image

# def cluster_accuracy(y_true, y_pred):
#     y_true_ = -y_true

#     n = max(max(y_true)+1, max(y_pred)+1)

#     acc_list = []
#     for perm in itertools.permutations(range(1,n)):
#         y_pred_ = [-perm[i-1] if i != 0 else 0 for i in y_pred]
#         acc_list.append(accuracy_score(y_true_, y_pred_))
    
#     return max(acc_list)