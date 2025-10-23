from dataset import AntibodyGraphDatasetFolder
from common.dataset_utils import extract_graph, convert_dgl_to_pyg

class AntibodyGraphDatasetPPA(AntibodyGraphDatasetFolder):
    def __getitem__(self, idx):
        g = self.load_graph(idx)
        ab = convert_dgl_to_pyg(extract_graph(g, 0))
        ig = convert_dgl_to_pyg(extract_graph(g, 1))
        ag = convert_dgl_to_pyg(extract_graph(g, 2))

        return (ab, ig, ag), self.labels[idx], self.orig_labels[idx]

