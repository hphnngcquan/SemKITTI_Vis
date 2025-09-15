import pyvista as pv
import numpy as np
import sys

class ScanVis:
    def __init__(self, cfg):
        self.cfg = cfg
        self.data_cfg = self.parse_cfg(self.cfg['data_cfg'])
        self.offset = cfg['offset']
        self.point_size = cfg['point_size']
        self.type = type
        self.frgrnd_mask = cfg['frgrnd_mask']
        self.thing_color = self.get_thing_color()
        self.reset()
        self.load_frame()
    
    def reset(self):
        self.plotter = pv.Plotter()
        self.points_actor = None
        self.plotter.set_background("black")
        self.plotter.add_key_event("n", self.front)
        self.plotter.add_key_event("b", self.back)
        self.plotter.add_key_event("q", lambda: sys.exit(0))

    def load_frame(self):
        self.pcd = np.fromfile(self.cfg['pcl_path'] + '/sequences/{:02d}/velodyne/{}.bin'.format(self.cfg['seq'], str(self.offset).zfill(6)), dtype=np.float32).reshape(-1, 4)
        self.label = np.fromfile(self.cfg['pcl_path'] + '/sequences/{:02d}/labels/{}.label'.format(self.cfg['seq'], str(self.offset).zfill(6)), dtype=np.uint32)
        if self.frgrnd_mask:
            mask = (self.label >> 16) != 0
            self.pcd = self.pcd[mask]
            self.label = self.label[mask]
    def front(self):
        self.offset += 1
        if self.type != "4d_ins_traj" and self.offset >= 4541:
            self.plotter.remove_actor(self.points_actor)
        self.load_frame()
        self.show()
    
    def back(self):
        self.offset -= 1
        if self.offset < 0:
            self.offset = 0
            print("Already at the first frame.")
        self.plotter.remove_actor(self.points_actor)
        self.load_frame()
        self.show()

    def show(self):
        self.colors = np.zeros((self.pcd.shape[0], 3), dtype=np.uint8)
        if self.type == "sem":
            self.apply_semantic_colors()
        elif self.type == "3d_ins":
            self.apply_panoptic_3d_colors()
        elif self.type == "4d_ins" or self.type == "4d_ins_traj":
            self.apply_panoptic_4d_colors()
        else:
            raise ValueError("No visualization type selected.")
        
        if self.cfg['sphere']:
             self.points_actor = self.plotter.add_points(self.pcd[:, :3], scalars=self.colors, rgb=True, point_size=self.point_size, render_points_as_spheres=True)
        else:
            self.points_actor = self.plotter.add_points(self.pcd[:, :3], scalars=self.colors, rgb=True, point_size=self.point_size)
        self.plotter.show()
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