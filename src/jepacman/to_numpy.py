import os, pickle, numpy as np, glob, argparse
from tqdm import tqdm

def convert(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(src_dir, "**", "*.pkl"), recursive=True))
    
    for fpath in tqdm(files):
        with open(fpath, "rb") as f:
            session = pickle.load(f)

        n_frames = len(session)
        n_agents = len(session[0])

        cases     = np.zeros((n_frames, n_agents, 81, 8),  dtype=np.float32)
        positions = np.zeros((n_frames, n_agents, 81, 2),  dtype=np.float32)
        pad_masks = np.zeros((n_frames, n_agents, 81),     dtype=bool)
        actions   = np.zeros((n_frames, n_agents),         dtype=np.int8)

        for t, frame in enumerate(session):
            for a, obs in enumerate(frame):
                for i, c in enumerate(obs["visible_cells"]):
                    cases[t, a, i] = [
                        c["wall_north"], c["wall_south"],
                        c["wall_east"],  c["wall_west"],
                        c["has_pellet"], c["has_super_pellet"],
                        c["has_ghost"],  c["has_pacman"],
                    ]
                    positions[t, a, i] = [c["dx"], c["dy"]]
                    pad_masks[t, a, i] = c["is_pad"]
                actions[t, a] = obs["action"]

        rel = os.path.relpath(fpath, src_dir)
        out = os.path.join(dst_dir, rel).replace(".pkl", ".npz")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.savez_compressed(out, cases=cases, positions=positions,
                            pad_masks=pad_masks, actions=actions)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("dst")
    args = p.parse_args()
    convert(args.src, args.dst)