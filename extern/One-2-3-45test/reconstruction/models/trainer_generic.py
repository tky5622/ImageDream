"""
decouple the trainer with the renderer
"""
import os
import cv2 as cv
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import trimesh

from utils.misc_utils import visualize_depth_numpy

from utils.training_utils import numpy2tensor

from loss.depth_loss import DepthLoss, DepthSmoothLoss

from models.sparse_neus_renderer import SparseNeuSRenderer


class GenericTrainer(nn.Module):
    def __init__(self,
                 rendering_network_outside,
                 pyramid_feature_network_lod0,
                 pyramid_feature_network_lod1,
                 sdf_network_lod0,
                 sdf_network_lod1,
                 variance_network_lod0,
                 variance_network_lod1,
                 rendering_network_lod0,
                 rendering_network_lod1,
                 n_samples_lod0,
                 n_importance_lod0,
                 n_samples_lod1,
                 n_importance_lod1,
                 n_outside,
                 perturb,
                 alpha_type='div',
                 conf=None,
                 timestamp="",
                 mode='train',
                 base_exp_dir=None,
                 ):
        super(GenericTrainer, self).__init__()

        self.conf = conf
        self.timestamp = timestamp

        
        self.base_exp_dir = base_exp_dir 
        

        self.anneal_start = self.conf.get_float('train.anneal_start', default=0.0)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)
        self.anneal_start_lod1 = self.conf.get_float('train.anneal_start_lod1', default=0.0)
        self.anneal_end_lod1 = self.conf.get_float('train.anneal_end_lod1', default=0.0)

        # network setups
        self.rendering_network_outside = rendering_network_outside
        self.pyramid_feature_network_geometry_lod0 = pyramid_feature_network_lod0  # 2D pyramid feature network for geometry
        self.pyramid_feature_network_geometry_lod1 = pyramid_feature_network_lod1  # use differnet networks for the two lods

        # when num_lods==2, may consume too much memeory
        self.sdf_network_lod0 = sdf_network_lod0
        self.sdf_network_lod1 = sdf_network_lod1

        # - warpped by ModuleList to support DataParallel
        self.variance_network_lod0 = variance_network_lod0
        self.variance_network_lod1 = variance_network_lod1

        self.rendering_network_lod0 = rendering_network_lod0
        self.rendering_network_lod1 = rendering_network_lod1

        self.n_samples_lod0 = n_samples_lod0
        self.n_importance_lod0 = n_importance_lod0
        self.n_samples_lod1 = n_samples_lod1
        self.n_importance_lod1 = n_importance_lod1
        self.n_outside = n_outside
        self.num_lods = conf.get_int('model.num_lods')  # the number of octree lods
        self.perturb = perturb
        self.alpha_type = alpha_type

        # - the two renderers
        self.sdf_renderer_lod0 = SparseNeuSRenderer(
            self.rendering_network_outside,
            self.sdf_network_lod0,
            self.variance_network_lod0,
            self.rendering_network_lod0,
            self.n_samples_lod0,
            self.n_importance_lod0,
            self.n_outside,
            self.perturb,
            alpha_type='div',
            conf=self.conf)

        self.sdf_renderer_lod1 = SparseNeuSRenderer(
            self.rendering_network_outside,
            self.sdf_network_lod1,
            self.variance_network_lod1,
            self.rendering_network_lod1,
            self.n_samples_lod1,
            self.n_importance_lod1,
            self.n_outside,
            self.perturb,
            alpha_type='div',
            conf=self.conf)

        self.if_fix_lod0_networks = self.conf.get_bool('train.if_fix_lod0_networks')

        # sdf network weights
        self.sdf_igr_weight = self.conf.get_float('train.sdf_igr_weight')
        self.sdf_sparse_weight = self.conf.get_float('train.sdf_sparse_weight', default=0)
        self.sdf_decay_param = self.conf.get_float('train.sdf_decay_param', default=100)
        self.fg_bg_weight = self.conf.get_float('train.fg_bg_weight', default=0.00)
        self.bg_ratio = self.conf.get_float('train.bg_ratio', default=0.0)

        self.depth_loss_weight = self.conf.get_float('train.depth_loss_weight', default=1.00)

        print("depth_loss_weight: ", self.depth_loss_weight)
        self.depth_criterion = DepthLoss()

        # - DataParallel mode, cannot modify attributes in forward()
        # self.iter_step = 0
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')

        # - True for finetuning; False for general training
        self.if_fitted_rendering = self.conf.get_bool('train.if_fitted_rendering', default=False)

        self.prune_depth_filter = self.conf.get_bool('model.prune_depth_filter', default=False)

    def get_trainable_params(self):
        # set trainable params

        self.params_to_train = []

        if not self.if_fix_lod0_networks:
            #  load pretrained featurenet
            self.params_to_train += list(self.pyramid_feature_network_geometry_lod0.parameters())
            self.params_to_train += list(self.sdf_network_lod0.parameters())
            self.params_to_train += list(self.variance_network_lod0.parameters())

            if self.rendering_network_lod0 is not None:
                self.params_to_train += list(self.rendering_network_lod0.parameters())

        if self.sdf_network_lod1 is not None:
            #  load pretrained featurenet
            self.params_to_train += list(self.pyramid_feature_network_geometry_lod1.parameters())

            self.params_to_train += list(self.sdf_network_lod1.parameters())
            self.params_to_train += list(self.variance_network_lod1.parameters())
            if self.rendering_network_lod1 is not None:
                self.params_to_train += list(self.rendering_network_lod1.parameters())

        return self.params_to_train

    def export_mesh_step(self, sample,
                        iter_step=0,
                        chunk_size=512,
                        resolution=360,
                        save_vis=False,
                        ):
        # * only support batch_size==1
        # ! attention: the list of string cannot be splited in DataParallel
        batch_idx = sample['batch_idx'][0]
        meta = sample['meta'][batch_idx]  # the scan lighting ref_view info

        sizeW = sample['img_wh'][0][0]
        sizeH = sample['img_wh'][0][1]
        H, W = sizeH, sizeW

        partial_vol_origin = sample['partial_vol_origin']  # [B, 3]
        near, far = sample['query_near_far'][0, :1], sample['query_near_far'][0, 1:]

        # the ray variables
        sample_rays = sample['rays']
        rays_o = sample_rays['rays_o'][0]
        rays_d = sample_rays['rays_v'][0]

        imgs = sample['images'][0]
        intrinsics = sample['intrinsics'][0]
        intrinsics_l_4x = intrinsics.clone()
        intrinsics_l_4x[:, :2] *= 0.25
        w2cs = sample['w2cs'][0]
        # target_candidate_w2cs = sample['target_candidate_w2cs'][0]
        proj_matrices = sample['affine_mats']


        # - the image to render
        scale_mat = sample['scale_mat']  # [1,4,4]  used to convert mesh into true scale
        trans_mat = sample['trans_mat']
        query_c2w = sample['query_c2w']  # [1,4,4]
        true_img = sample['query_image'][0]
        true_img = np.uint8(true_img.permute(1, 2, 0).cpu().numpy() * 255)

        # depth_min, depth_max = near.cpu().numpy(), far.cpu().numpy()

        # scale_factor = sample['scale_factor'][0].cpu().numpy()
        # true_depth = sample['query_depth'] if 'query_depth' in sample.keys() else None
        # # if true_depth is not None:
        # #     true_depth = true_depth[0].cpu().numpy()
        # #     true_depth_colored = visualize_depth_numpy(true_depth, [depth_min, depth_max])[0]
        # # else:
        # #     true_depth_colored = None

        rays_o = rays_o.reshape(-1, 3).split(chunk_size)
        rays_d = rays_d.reshape(-1, 3).split(chunk_size)

        # - obtain conditional features
        with torch.no_grad():
            # - obtain conditional features
            geometry_feature_maps = self.obtain_pyramid_feature_maps(imgs, lod=0)
            # - lod 0
            conditional_features_lod0 = self.sdf_network_lod0.get_conditional_volume(
                feature_maps=geometry_feature_maps[None, :, :, :, :],
                partial_vol_origin=partial_vol_origin,
                proj_mats=proj_matrices,
                sizeH=sizeH,
                sizeW=sizeW,
                lod=0,
            )

        con_volume_lod0 = conditional_features_lod0['dense_volume_scale0']
        con_valid_mask_volume_lod0 = conditional_features_lod0['valid_mask_volume_scale0']
        coords_lod0 = conditional_features_lod0['coords_scale0']  # [1,3,wX,wY,wZ]

        if self.num_lods > 1:
            sdf_volume_lod0 = self.sdf_network_lod0.get_sdf_volume(
                con_volume_lod0, con_valid_mask_volume_lod0,
                coords_lod0, partial_vol_origin)  # [1, 1, dX, dY, dZ]

        depth_maps_lod0, depth_masks_lod0 = None, None


        if self.num_lods > 1:
            geometry_feature_maps_lod1 = self.obtain_pyramid_feature_maps(imgs, lod=1)

            if self.prune_depth_filter:
                pre_coords, pre_feats = self.sdf_renderer_lod0.get_valid_sparse_coords_by_sdf_depthfilter(
                    sdf_volume_lod0[0], coords_lod0[0], con_valid_mask_volume_lod0[0], con_volume_lod0[0],
                    depth_maps_lod0, proj_matrices[0],
                    partial_vol_origin, self.sdf_network_lod0.voxel_size,
                    near, far, self.sdf_network_lod0.voxel_size, 12)
            else:
                pre_coords, pre_feats = self.sdf_renderer_lod0.get_valid_sparse_coords_by_sdf(
                    sdf_volume_lod0[0], coords_lod0[0], con_valid_mask_volume_lod0[0], con_volume_lod0[0])

            pre_coords[:, 1:] = pre_coords[:, 1:] * 2

            with torch.no_grad():
                conditional_features_lod1 = self.sdf_network_lod1.get_conditional_volume(
                    feature_maps=geometry_feature_maps_lod1[None, :, :, :, :],
                    partial_vol_origin=partial_vol_origin,
                    proj_mats=proj_matrices,
                    sizeH=sizeH,
                    sizeW=sizeW,
                    pre_coords=pre_coords,
                    pre_feats=pre_feats,
                )

            con_volume_lod1 = conditional_features_lod1['dense_volume_scale1']
            con_valid_mask_volume_lod1 = conditional_features_lod1['valid_mask_volume_scale1']


        # - extract mesh
        if (iter_step % self.val_mesh_freq == 0):
            torch.cuda.empty_cache()
            self.validate_colored_mesh(
                                        density_or_sdf_network=self.sdf_network_lod0,
                                        func_extract_geometry=self.sdf_renderer_lod0.extract_geometry,
                                        resolution=resolution,
                                        conditional_volume=con_volume_lod0,
                                        conditional_valid_mask_volume = con_valid_mask_volume_lod0,
                                        feature_maps=geometry_feature_maps,
                                        color_maps=imgs,
                                        w2cs=w2cs,
                                        target_candidate_w2cs=None,
                                        intrinsics=intrinsics,
                                        rendering_network=self.rendering_network_lod0,
                                        rendering_projector=self.sdf_renderer_lod0.rendering_projector,
                                        lod=0,
                                        threshold=0,
                                        query_c2w=query_c2w,
                                        mode='val_bg', meta=meta,
                                        iter_step=iter_step, scale_mat=scale_mat, trans_mat=trans_mat
                                    )
            torch.cuda.empty_cache()

            if self.num_lods > 1:
                self.validate_colored_mesh(
                            density_or_sdf_network=self.sdf_network_lod1,
                            func_extract_geometry=self.sdf_renderer_lod1.extract_geometry,
                            resolution=resolution,
                            conditional_volume=con_volume_lod1,
                            conditional_valid_mask_volume = con_valid_mask_volume_lod1,
                            feature_maps=geometry_feature_maps,
                            color_maps=imgs,
                            w2cs=w2cs,
                            target_candidate_w2cs=None,
                            intrinsics=intrinsics,
                            rendering_network=self.rendering_network_lod1,
                            rendering_projector=self.sdf_renderer_lod1.rendering_projector,
                            lod=1,
                            threshold=0,
                            query_c2w=query_c2w,
                            mode='val_bg', meta=meta,
                            iter_step=iter_step, scale_mat=scale_mat, trans_mat=trans_mat
                        )
            torch.cuda.empty_cache()
            save_path = self.base_exp_dir+'/con_volume_lod_150.pth'
            coords_lod0_150 = F.interpolate(con_volume_lod0, size=(150, 150, 150), mode='trilinear', align_corners=False)
            torch.save(coords_lod0_150, save_path)
        
            # torch.save(con_valid_mask_volume_lod0, '/home/bitterdhg/Code/nerf/Learn/One-2-3-45-master/con_valid_mask_volume_lod0.pth')
            print("save_path: " + save_path)

    def forward(self, sample,
                perturb_overwrite=-1,
                background_rgb=None,
                alpha_inter_ratio_lod0=0.0,
                alpha_inter_ratio_lod1=0.0,
                iter_step=0,
                mode='train',
                save_vis=False,
                resolution=360,
                ):
        import time
        begin = time.time()
        result =  self.export_mesh_step(sample,
                                iter_step=iter_step,
                                save_vis=save_vis,
                                resolution=resolution,
                                )
        end = time.time()

    def obtain_pyramid_feature_maps(self, imgs, lod=0):
        """
        get feature maps of all conditional images
        :param imgs:
        :return:
        """

        if lod == 0:
            extractor = self.pyramid_feature_network_geometry_lod0
        elif lod >= 1:
            extractor = self.pyramid_feature_network_geometry_lod1

        pyramid_feature_maps = extractor(imgs)

        # * the pyramid features are very important, if only use the coarst features, hard to optimize
        fused_feature_maps = torch.cat([
            F.interpolate(pyramid_feature_maps[0], scale_factor=4, mode='bilinear', align_corners=True),
            F.interpolate(pyramid_feature_maps[1], scale_factor=2, mode='bilinear', align_corners=True),
            pyramid_feature_maps[2]
        ], dim=1)

        return fused_feature_maps
    def validate_colored_mesh(self, density_or_sdf_network, func_extract_geometry, world_space=True, resolution=360,
                                threshold=0.0, mode='val',
                                # * 3d feature volume
                                conditional_volume=None,
                                conditional_valid_mask_volume=None,
                                feature_maps=None,
                                color_maps = None,
                                w2cs=None,
                                target_candidate_w2cs=None,
                                intrinsics=None,
                                rendering_network=None,
                                rendering_projector=None,
                                query_c2w=None,
                                lod=None, occupancy_mask=None,
                                bound_min=[-1, -1, -1], bound_max=[1, 1, 1], meta='', iter_step=0, scale_mat=None,
                                trans_mat=None
                                ):

        bound_min = torch.tensor(bound_min, dtype=torch.float32)
        bound_max = torch.tensor(bound_max, dtype=torch.float32)

        vertices, triangles, fields = func_extract_geometry(
            density_or_sdf_network,
            bound_min, bound_max, resolution=resolution,
            threshold=threshold, device=conditional_volume.device,
            # * 3d feature volume
            conditional_volume=conditional_volume, lod=lod,
            occupancy_mask=occupancy_mask
        )
        

        with torch.no_grad():
            ren_geo_feats, ren_rgb_feats, ren_ray_diff, ren_mask, _, _ = rendering_projector.compute_view_independent(
                torch.tensor(vertices).to(conditional_volume),
                lod=lod,
                # * 3d geometry feature volumes
                geometryVolume=conditional_volume[0],
                geometryVolumeMask=conditional_valid_mask_volume[0],
                sdf_network=density_or_sdf_network,
                # * 2d rendering feature maps
                rendering_feature_maps=feature_maps, # [n_view, 56, 256, 256]
                color_maps=color_maps,
                w2cs=w2cs,
                target_candidate_w2cs=target_candidate_w2cs,
                intrinsics=intrinsics,
                img_wh=[256,256],
                query_img_idx=0,  # the index of the N_views dim for rendering
                query_c2w=query_c2w,
            )


            vertices_color, rendering_valid_mask = rendering_network(
                ren_geo_feats, ren_rgb_feats, ren_ray_diff, ren_mask)
        


        if scale_mat is not None:
            scale_mat_np = scale_mat.cpu().numpy()
            vertices = vertices * scale_mat_np[0][0, 0] + scale_mat_np[0][:3, 3][None]

        if trans_mat is not None: # w2c_ref_inv
            trans_mat_np = trans_mat.cpu().numpy()
            vertices_homo = np.concatenate([vertices, np.ones_like(vertices[:, :1])], axis=1)
            vertices = np.matmul(trans_mat_np, vertices_homo[:, :, None])[:, :3, 0]

        vertices_color = np.array(vertices_color.squeeze(0).cpu() * 255, dtype=np.uint8)
        mesh = trimesh.Trimesh(vertices, triangles, vertex_colors=vertices_color)
        # os.makedirs(os.path.join(self.base_exp_dir, 'meshes_' + mode, 'lod{:0>1d}'.format(lod)), exist_ok=True)
        # mesh.export(os.path.join(self.base_exp_dir, 'meshes_' + mode, 'lod{:0>1d}'.format(lod),
        #                          'mesh_{:0>8d}_{}_lod{:0>1d}.ply'.format(iter_step, meta, lod)))  
        
        mesh.export(os.path.join(self.base_exp_dir, 'mesh.ply'))