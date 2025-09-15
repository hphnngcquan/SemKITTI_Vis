import argparse
import yaml
from tools.vis_module import ScanVis

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize results from a directory.")
    parser.add_argument("--cfg_file", type=str, default="./cfg/cfg.yaml", help="Config file path.")
    parser.add_argument("--type", type=str, default="4d_ins_traj", choices=["sem", "3d_ins", "4d_ins", "4d_ins_traj"], help="Type of visualization.")
    parser.add_argument("--pred", type=bool, default=True, help="Whether to visualize predictions or ground truth.")
    parser.add_argument("--seq", type=int, default=8, help="Sequence number to visualize.")
    parser.add_argument("--sphere", action="store_true", default=True, help="Flag to use sphere glyphs for point rendering.")
    parser.add_argument("--point_size", type=int, default=15, help="Point size for visualization.")
    parser.add_argument("--offset", type=int, default=0, help="Offset for sequence numbering.")
    parser.add_argument("--frgrnd_mask", default=True, action="store_true", help="Flag to visualize only foreground points.")

    # savings
    parser.add_argument("--save_graphics", type=str, default='pdf', help="Flag to save visualizations as graphics files.")
    parser.add_argument("--save_dir", type=str, default=".", help="Directory to save visualizations. Defaults to input directory.")
    
    args = parser.parse_args()

    with open(args.cfg_file, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg['seq'] = args.seq
    cfg['sphere'] = args.sphere
    cfg['save_dir'] = args.save_dir
    cfg['save_graphics'] = args.save_graphics
    cfg['type'] = args.type
    cfg['pred'] = args.pred
    cfg['frgrnd_mask'] = args.frgrnd_mask
    cfg['offset'] = args.offset
    cfg['point_size'] = args.point_size
    scan_vis = ScanVis(cfg=cfg)

    scan_vis.show()
