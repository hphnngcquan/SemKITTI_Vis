import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time
from tools.utils import read_poses, transform_point_cloud
class ScanVis:
    def __init__(self, cfg):
        self.cfg = cfg
        self.data_cfg = self.parse_cfg(self.cfg['data_cfg'])
        self.offset = cfg['offset']
        self.point_size = cfg['point_size']
        self.type = cfg['type']
        self.frgrnd_mask = cfg['frgrnd_mask']
        self.user_pcl = cfg.get('user_pcl_path', False)
        self.save_num = 0
        self.thing_color = self.get_thing_color()
        self.reset()
        self.load_frame()
    
    def reset(self):
        if self.cfg['save_multiple'] != 0:
            self.plotter = pv.Plotter(off_screen=True)
        else:
            self.plotter = pv.Plotter()
        self.plotter.window_size = [1920, 1080]
        self.points_actor = None
        self.plotter.set_background(self.cfg['background'])
        self.plotter.add_key_event("n", self.front)
        self.plotter.add_key_event("b", self.back)
        self.plotter.add_key_event("s", self.save_graphics)
        self.plotter.add_key_event("c", lambda: print(self.plotter.camera_position))
        self.text = self.plotter.add_text(f"Sequence: {self.cfg['seq']}, Frame: {self.offset}", font_size=12, color='red', position='upper_left')
        self.plotter.add_key_event("q", lambda: sys.exit(0))

    def load_frame(self):
        if self.user_pcl:
            pcd = np.fromfile(self.cfg['user_pcl_path'], dtype=np.int32).reshape(-1, 3)
            self.pcd = pcd
            self.label = np.zeros((pcd.shape[0],), dtype=np.uint32)
            return
        glob_pcd = []
        glob_label = []
        if self.type == "sem_errors":
            glob_gt_label = []
        for i in range(self.cfg['sweep']):
            offset = self.offset + i
            if offset >= self.cfg['max_offset']:
                offset = self.offset
            pcd = np.fromfile(self.cfg['pcl_path'] + '/sequences/{:02d}/velodyne/{}.bin'.format(self.cfg['seq'], str(offset).zfill(6)), dtype=np.float32).reshape(-1, 4)

            if self.cfg['pred']:
                label = np.fromfile(self.cfg['pred_path'] + '/sequences/{:02d}/predictions/{}.label'.format(self.cfg['seq'], str(offset).zfill(6)), dtype=np.uint32)
                if self.type == "sem_errors":
                    gt_label = np.fromfile(self.cfg['pcl_path'] + '/sequences/{:02d}/labels/{}.label'.format(self.cfg['seq'], str(offset).zfill(6)), dtype=np.uint32)
            else:
                label = np.fromfile(self.cfg['pcl_path'] + '/sequences/{:02d}/labels/{}.label'.format(self.cfg['seq'], str(offset).zfill(6)), dtype=np.uint32)
            if self.frgrnd_mask:
                mask = (label >> 16) != 0
                pcd = pcd[mask]
                label = label[mask]
                if self.type == "sem_errors":
                    gt_label = gt_label[mask]
            pcd[:,:3] = pcd[:,:3] - np.mean(pcd[:,:3], axis=0)

            pose = read_poses(os.path.join(self.cfg['pcl_path'], f"sequences/{self.cfg['seq']:02d}"))
            pcd[:,:3] = transform_point_cloud(pcd[:,:3], pose[offset], pose[0])
            glob_pcd.append(pcd)
            glob_label.append(label)
            if self.type == "sem_errors":
                glob_gt_label.append(gt_label)
        self.pcd = np.concatenate(glob_pcd, axis=0)
        self.label = np.concatenate(glob_label, axis=0)
        if self.type == "sem_errors":
            self.gt_label = np.concatenate(glob_gt_label, axis=0)

        if self.cfg['bbox']:
            self.bbox = []
            for lab in np.unique(self.label):
                if lab >> 16 == 0:
                    continue
                mask = self.label == lab
                center = self.pcd[mask][:,:3].mean(axis=0)
                size = np.max(self.pcd[mask][:,:3], axis=0) - np.min(self.pcd[mask][:,:3], axis=0)
                self.bbox.append((lab, center, size))

        

    def front(self):
        self.offset += 1
        if self.type != "4d_ins_traj":
            self.plotter.remove_actor(self.points_actor)
        self.load_frame()
        self.plotter.remove_actor(self.text)
        if self.cfg['bbox'] and self.box_actors is not None:
            for actor in self.box_actors:
                self.plotter.remove_actor(actor)
        self.text = self.plotter.add_text(f"Sequence: {self.cfg['seq']}, Frame: {self.offset}", font_size=12, color='red', position='upper_left')
        if self.cfg['save_multiple'] == 0:
            self.show()
    
    def back(self):
        self.offset -= 1
        if self.offset < 0:
            self.offset = self.cfg['max_offset'] - 1
        self.plotter.remove_actor(self.points_actor)
        self.load_frame()
        self.plotter.remove_actor(self.text)
        if self.cfg['bbox'] and self.box_actors is not None:
            for actor in self.box_actors:
                self.plotter.remove_actor(actor)
        self.text = self.plotter.add_text(f"Sequence: {self.cfg['seq']}, Frame: {self.offset}", font_size=12, color='red', position='upper_left')
        if self.cfg['save_multiple'] == 0:
            self.show()

    def show(self):
        self.colors = np.zeros((self.pcd.shape[0], 3), dtype=np.uint8)
        if self.type == "sem":
            self.apply_semantic_colors()
        elif self.type == "3d_ins":
            self.apply_panoptic_3d_colors()
        elif self.type == "4d_ins" or self.type == "4d_ins_traj":
            self.apply_panoptic_4d_colors()
        elif self.type == "range_color":
            self.apply_range_colors()
        elif self.type == "sem_errors":
            self.apply_sem_error_colors()
        elif self.type == "user":
            self.apply_user_colors()
        elif self.type == "pcl":
            self.colors[:] = [200, 200, 200]
        else:
            raise ValueError("No visualization type selected.")
        
        if self.cfg['sphere']:
             self.points_actor = self.plotter.add_points(self.pcd[:, :3], scalars=self.colors, rgb=True, point_size=self.point_size, render_points_as_spheres=True)
        else:
            self.points_actor = self.plotter.add_points(self.pcd[:, :3], scalars=self.colors, rgb=True, point_size=self.point_size)
        
        if self.cfg['bbox']:
            self.box_actors = []
            self.box_color = np.zeros((len(self.bbox), 3), dtype=np.uint8)
            for i, (lab, center, size) in enumerate(self.bbox):
                if self.cfg["bbox_type"] == "track":
                    color = [int(x * 255) for x in self.thing_color[str(lab >> 16)]]
                elif self.cfg["bbox_type"] == "det":
                    color = [255, 255, 255]
                else:
                    raise ValueError("bbox_type must be one of ['det', 'track']")
                self.box_color[i] = color
                mins = center - size / 2
                maxs = center + size / 2
                box = pv.Box(bounds=(mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2]))
                actor = self.plotter.add_mesh(box, color=color, line_width=4, style='wireframe')
                self.box_actors.append(actor)
        
        if self.cfg['cam_pose'] is not None:
            self.plotter.camera_position = self.cfg['cam_pose']
        self.plotter.show()

    def save_multiple(self, num):
        for _ in range(num):
            # Remove previous points
            if self.points_actor is not None:
                self.plotter.remove_actor(self.points_actor)

            # Apply colors
            self.colors = np.zeros((self.pcd.shape[0], 3), dtype=np.uint8)
            if self.type == "sem":
                self.apply_semantic_colors()
            elif self.type == "3d_ins":
                self.apply_panoptic_3d_colors()
            elif self.type in ["4d_ins", "4d_ins_traj"]:
                self.apply_panoptic_4d_colors()
            elif self.type == "range_color":
                self.apply_range_colors()
            elif self.type == "pcl":
                self.colors[:] = [200, 200, 100]
            else:
                raise ValueError("No visualization type selected.")

            # Add new actor
            if self.cfg['sphere']:
                self.points_actor = self.plotter.add_points(
                    self.pcd[:, :3], scalars=self.colors, rgb=True,
                    point_size=self.point_size, render_points_as_spheres=True
                )
            else:
                self.points_actor = self.plotter.add_points(
                    self.pcd[:, :3], scalars=self.colors, rgb=True,
                    point_size=self.point_size
                )

            # Set camera
            if self.cfg['cam_pose'] is not None:
                self.plotter.camera_position = self.cfg['cam_pose']

            # Save frame
            self.save_graphics()
            self.save_num += 1

            # Move to next frame
            self.front()

        print(f"Saved {num} frames. Exiting.")
        sys.exit(0)

    def apply_semantic_colors(self):
        sem_labels = self.label & 0xFFFF
        for sem in np.unique(sem_labels):
            color = self.data_cfg['color_map'][sem]
            self.colors[sem_labels == sem] = color

    def apply_panoptic_3d_colors(self):
        for lab in np.unique(self.label):
            if lab >> 16 == 0:
                lab2 = lab & 0xFFFF
                if not self.cfg['frgrnd_color_mask']:
                    self.colors[self.label == lab] = self.data_cfg['color_map'][lab2]
                else:
                    self.colors[self.label == lab] = [200, 200, 200]
                continue
            mask = self.label == lab
            color = np.random.randint(0, 255, size=3)
            self.colors[mask] = color

    def apply_panoptic_4d_colors(self):
        for lab in np.unique(self.label):
            if lab >> 16 == 0:
                lab = lab & 0xFFFF
                if not self.cfg['frgrnd_color_mask']:
                    self.colors[self.label == lab] = self.data_cfg['color_map'][lab]
                else:
                    self.colors[self.label == lab] = [200, 200, 200]
                continue
            mask = self.label == lab
            if str(lab >> 16) in self.thing_color:
                color = [int(x * 255) for x in self.thing_color[str(lab >> 16)]]
            else:
                raise ValueError(f"Thing class {lab >> 16} not found in thing_color mapping.")
            self.colors[mask] = color
    
    def apply_range_colors(self):
        ranges = self.pcd[:,3]
        norm_ranges = (ranges - np.min(ranges)) / (np.max(ranges) - np.min(ranges))
        self.colors = (plt.get_cmap('viridis')(norm_ranges)[:,:3] * 255).astype(np.uint8)
    
    def apply_sem_error_colors(self):
        self.colors[:] = [200, 200, 200]  # Default gray
        sem_labels = self.label & 0xFFFF
        gt_sem_labels = self.gt_label & 0xFFFF
        class_remap = self.data_cfg["learning_map"]
        maxkey = max(class_remap.keys())
        class_lut = np.zeros((maxkey + 100), dtype=np.int32)
        class_lut[list(class_remap.keys())] = list(class_remap.values())

        sem_labels = class_lut[sem_labels]
        gt_sem_labels = class_lut[gt_sem_labels]
        for sem in np.unique(gt_sem_labels):
            if sem == 0:  # ignore class
                continue
            mask = gt_sem_labels == sem
            correct_mask = (sem_labels == gt_sem_labels) & mask
            error_mask = (sem_labels != gt_sem_labels) & mask
            self.colors[correct_mask] = [200, 200, 200]  # Green for correct
            self.colors[error_mask] = [255, 0, 0]    # Red for errors

    def apply_user_colors(self):
        raise NotImplementedError("User-defined colors not implemented yet.")
    
    def save_graphics(self):
        if self.cfg['save_graphics'] not in ['png', 'pdf', 'svg']:
            raise ValueError("save_graphics must be one of ['png', 'pdf', 'svg']")
        
        if self.cfg['name_graphics']:
            save_path = os.path.join(self.cfg['save_dir'], f"{self.cfg['name_graphics']}_{self.save_num}.{self.cfg['save_graphics']}")
            if os.path.exists(save_path):
                print(f"File {save_path} already exists. Save with time.")
                save_path = os.path.join(self.cfg['save_dir'], f"{self.cfg['name_graphics']}_{self.save_num}_{int(time.time())}.{self.cfg['save_graphics']}")
        
        else:
            save_path = os.path.join(self.cfg['save_dir'], f"seq_{self.cfg['seq']}_frame_{str(self.offset).zfill(6)}.{self.cfg['save_graphics']}")
            if os.path.exists(save_path):
                print(f"File {save_path} already exists. Save with time.")
                save_path = os.path.join(self.cfg['save_dir'], f"seq_{self.cfg['seq']}_frame_{str(self.offset).zfill(6)}_{int(time.time())}.{self.cfg['save_graphics']}")


        self.plotter.remove_actor(self.text)
        if self.cfg['save_graphics'] == 'png':
            self.plotter.screenshot(save_path)
        else:
            self.plotter.save_graphic(save_path, raster=True, painter=True)
        print(f"Saved visualization to {save_path}")

    def get_thing_color(self):
        import json
        with open(self.cfg['thing_color_file'], 'r') as f:
            thing_color = json.load(f)
        return thing_color
    
    def parse_cfg(self, cfg):
        import yaml
        with open(cfg, 'r') as f:
            cfg_dict = yaml.safe_load(f)
        return cfg_dict