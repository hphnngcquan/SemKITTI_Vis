import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from tools.utils import read_poses, transform_point_cloud
class ScanVis:
    def __init__(self, cfg):
        self.cfg = cfg
        self.data_cfg = self.parse_cfg(self.cfg['data_cfg'])
        self.offset = cfg['offset']
        self.point_size = cfg['point_size']
        self.type = cfg['type']
        self.frgrnd_mask = cfg['frgrnd_mask']
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
        self.plotter.set_background("white")
        self.plotter.add_key_event("n", self.front)
        self.plotter.add_key_event("b", self.back)
        self.plotter.add_key_event("s", self.save_graphics)
        self.plotter.add_key_event("c", lambda: print(self.plotter.camera_position))
        self.text = self.plotter.add_text(f"Sequence: {self.cfg['seq']}, Frame: {self.offset}", font_size=12, color='white', position='upper_left')
        self.plotter.add_key_event("q", lambda: sys.exit(0))

    def load_frame(self):
        glob_pcd = []
        glob_label = []
        for i in range(self.cfg['sweep']):
            offset = self.offset + i
            if offset >= self.cfg['max_offset']:
                offset = self.offset
            pcd = np.fromfile(self.cfg['pcl_path'] + '/sequences/{:02d}/velodyne/{}.bin'.format(self.cfg['seq'], str(offset).zfill(6)), dtype=np.float32).reshape(-1, 4)
            label = np.fromfile(self.cfg['pcl_path'] + '/sequences/{:02d}/labels/{}.label'.format(self.cfg['seq'], str(offset).zfill(6)), dtype=np.uint32)
            if self.frgrnd_mask:
                mask = (self.label >> 16) != 0
                pcd = pcd[mask]
                label = label[mask]
            pcd[:,:3] = pcd[:,:3] - np.mean(pcd[:,:3], axis=0)

            pose = read_poses(os.path.join(self.cfg['pcl_path'], f"sequences/{self.cfg['seq']:02d}"))
            pcd[:,:3] = transform_point_cloud(pcd[:,:3], pose[offset], pose[0])
            glob_pcd.append(pcd)
            glob_label.append(label)
        self.pcd = np.concatenate(glob_pcd, axis=0)
        self.label = np.concatenate(glob_label, axis=0)

        

    def front(self):
        self.offset += 1
        if self.type != "4d_ins_traj":
            self.plotter.remove_actor(self.points_actor)
        self.load_frame()
        self.plotter.remove_actor(self.text)
        self.text = self.plotter.add_text(f"Sequence: {self.cfg['seq']}, Frame: {self.offset}", font_size=12, color='white', position='upper_left')
        if self.cfg['save_multiple'] == 0:
            self.show()
    
    def back(self):
        self.offset -= 1
        if self.offset < 0:
            self.offset = self.cfg['max_offset'] - 1
        self.plotter.remove_actor(self.points_actor)
        self.load_frame()
        self.plotter.remove_actor(self.text)
        self.text = self.plotter.add_text(f"Sequence: {self.cfg['seq']}, Frame: {self.offset}", font_size=12, color='white', position='upper_left')
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
        elif self.type == "pcl":
            self.colors[:] = [200, 200, 100]
        else:
            raise ValueError("No visualization type selected.")
        
        if self.cfg['sphere']:
             self.points_actor = self.plotter.add_points(self.pcd[:, :3], scalars=self.colors, rgb=True, point_size=self.point_size, render_points_as_spheres=True)
        else:
            self.points_actor = self.plotter.add_points(self.pcd[:, :3], scalars=self.colors, rgb=True, point_size=self.point_size)
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
                lab = lab & 0xFFFF
                self.colors[self.label == lab] = self.data_cfg['color_map'][lab]
                continue
            mask = self.label == lab
            color = np.random.randint(0, 255, size=3)
            self.colors[mask] = color

    def apply_panoptic_4d_colors(self):
        for lab in np.unique(self.label):
            if lab >> 16 == 0:
                lab = lab & 0xFFFF
                self.colors[self.label == lab] = self.data_cfg['color_map'][lab]
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

    def save_graphics(self):
        if self.cfg['save_graphics'] not in ['png', 'pdf', 'svg']:
            raise ValueError("save_graphics must be one of ['png', 'pdf', 'svg']")
        save_path = os.path.join(self.cfg['save_dir'], f"seq_{self.cfg['seq']}_frame_{str(self.offset).zfill(6)}.{self.cfg['save_graphics']}")
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