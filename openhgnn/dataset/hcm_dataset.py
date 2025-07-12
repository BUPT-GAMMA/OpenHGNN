import os
import numpy as np
import pandas as pd
import torch as th # Use torch as th for consistency
import dgl # Import dgl for graph construction
from dgl.data.utils import download, extract_archive, load_graphs, save_graphs # Import save_graphs
from dgl.data import DGLDataset
from sklearn.model_selection import KFold



def neg_data_generate(adj_data_all, train_data_fix, val_data_fix, neg_num_test, seed):
    """A function used to generate negative samples."""
    neg_num_train = 1
    np.random.seed(seed)
    train_neg_1_ls = []
    train_neg_2_ls = []
    train_neg_3_ls = []
    train_neg_4_ls = []
    val_neg_1_ls = []
    val_neg_2_ls = []
    val_neg_3_ls = []
    val_neg_4_ls = []

    # Determine dimensions for arr_true based on the max indices in adj_data_all
    # Assuming adj_data_all is (gene_id, microbe_id, disease_id, label)
    # The original code uses fixed sizes (301, 176, 153), let's make it dynamic
    # Ensure adj_data_all has at least 3 columns for indices
    if adj_data_all.shape[1] < 3:
        raise ValueError("adj_data_all must have at least 3 columns for gene, microbe, disease IDs.")

    max_gene_id = int(adj_data_all[:, 0].max()) + 1
    max_mic_id = int(adj_data_all[:, 1].max()) + 1
    max_dis_id = int(adj_data_all[:, 2].max()) + 1
    arr_true = np.zeros((max_gene_id, max_mic_id, max_dis_id))

    for line in adj_data_all:
        arr_true[int(line[0]), int(line[1]), int(line[2])] = 1

    # Initialize arr_false_train with dynamic sizes
    arr_false_train = np.zeros((max_gene_id, max_mic_id, max_dis_id))

    for i in train_data_fix:
        ctn_1 = 0
        ctn_2 = 0
        ctn_3 = 0
        ctn_4 = 0
        tr_gene_ls = list(range(arr_true.shape[0]))
        tr_mic_ls = list(range(arr_true.shape[1]))
        tr_dis_ls = list(range(arr_true.shape[2]))

        # Type 1: Fully random negative samples
        while ctn_1 < neg_num_train:
            a = np.random.randint(0, arr_true.shape[0])
            b = np.random.randint(0, arr_true.shape[1])
            c = np.random.randint(0, arr_true.shape[2])
            if arr_true[a, b, c] != 1 and arr_false_train[a, b, c] != 1:
                arr_false_train[a, b, c] = 1
                ctn_1 += 1
                train_neg_1_ls.append((a, b, c, 0))

        # Type 2: Fix microbe and disease, random gene
        while ctn_2 < neg_num_train:
            b = int(i[1])
            c = int(i[2])
            # Ensure a valid gene index is chosen
            if not tr_gene_ls: # If list is empty, refill or break
                tr_gene_ls = list(range(arr_true.shape[0])) # Refill to allow more choices
            a = np.random.choice(tr_gene_ls)
            if arr_true[a, b, c] != 1 and arr_false_train[a, b, c] != 1:
                arr_false_train[a, b, c] = 1
                ctn_2 += 1
                train_neg_2_ls.append((a, b, c, 0))
            # Remove chosen 'a' to ensure distinct negative samples for this positive sample
            if a in tr_gene_ls:
                tr_gene_ls.remove(a)


        # Type 3: Fix gene and disease, random microbe
        while ctn_3 < neg_num_train:
            a = int(i[0])
            c = int(i[2])
            if not tr_mic_ls:
                tr_mic_ls = list(range(arr_true.shape[1]))
            b = np.random.choice(tr_mic_ls)
            if arr_true[a, b, c] != 1 and arr_false_train[a, b, c] != 1:
                arr_false_train[a, b, c] = 1
                ctn_3 += 1
                train_neg_3_ls.append((a, b, c, 0))
            if b in tr_mic_ls:
                tr_mic_ls.remove(b)

        # Type 4: Fix gene and microbe, random disease
        while ctn_4 < neg_num_train:
            a = int(i[0])
            b = int(i[1])
            if tr_dis_ls:
                c = np.random.choice(tr_dis_ls)
                tr_dis_ls.remove(c)
                if arr_true[a, b, c] != 1 and arr_false_train[a, b, c] != 1:
                    arr_false_train[a, b, c] = 1
                    ctn_4 += 1
                    train_neg_4_ls.append((a, b, c, 0))
            else:
                # If no more unique diseases can be found, repeat existing ones
                distance_t4 = neg_num_train - ctn_4
                if train_neg_4_ls: # Ensure there's at least one element to repeat
                    last_ind = len(train_neg_4_ls) - 1
                    for k in range(distance_t4):
                        train_neg_4_ls.append(train_neg_4_ls[last_ind])
                else: # If no samples generated yet, just break
                    break
                break

    train_neg_1_arr = np.array(train_neg_1_ls) if train_neg_1_ls else np.array([])
    train_neg_2_arr = np.array(train_neg_2_ls) if train_neg_2_ls else np.array([])
    train_neg_3_arr = np.array(train_neg_3_ls) if train_neg_3_ls else np.array([])
    train_neg_4_arr = np.array(train_neg_4_ls) if train_neg_4_ls else np.array([])

    # Ensure all arrays are not empty before vstack
    all_train_neg_arrays = [arr for arr in [train_neg_1_arr, train_neg_2_arr, train_neg_3_arr, train_neg_4_arr] if arr.size > 0]
    if all_train_neg_arrays:
        train_neg_all = np.vstack(all_train_neg_arrays)
        train_neg_all = np.vstack((train_neg_all, train_data_fix))
    else: # If no negative samples were generated, just use positive data
        train_neg_all = train_data_fix

    np.random.shuffle(train_neg_all)

    # Initialize arr_false_val with dynamic sizes
    arr_false_val_1 = np.zeros((max_gene_id, max_mic_id, max_dis_id))
    arr_false_val_2 = np.zeros((max_gene_id, max_mic_id, max_dis_id))
    arr_false_val_3 = np.zeros((max_gene_id, max_mic_id, max_dis_id))
    arr_false_val_4 = np.zeros((max_gene_id, max_mic_id, max_dis_id))

    for i in val_data_fix:
        # Convert i to tuple if it's a numpy array row to ensure hashability for set/list operations
        neg_1_i = [tuple(i)]; neg_2_i = [tuple(i)]; neg_3_i = [tuple(i)]; neg_4_i = [tuple(i)] # Start with positive sample
        cva_1 = 0; cva_2 = 0; cva_3 = 0; cva_4 = 0
        gene_ls = list(range(arr_true.shape[0]))
        mic_ls = list(range(arr_true.shape[1]))
        dis_ls = list(range(arr_true.shape[2]))

        # Type 1: Fully random negative samples for validation
        while cva_1 < neg_num_test:
            a_1 = np.random.randint(0, arr_true.shape[0])
            b_1 = np.random.randint(0, arr_true.shape[1])
            c_1 = np.random.randint(0, arr_true.shape[2])
            if arr_true[a_1, b_1, c_1] != 1 and arr_false_train[a_1, b_1, c_1] != 1 and arr_false_val_1[a_1, b_1, c_1] != 1:
                arr_false_val_1[a_1, b_1, c_1] = 1
                cva_1 += 1
                neg_1_i.append((a_1, b_1, c_1, 0))
        np.random.shuffle(neg_1_i)
        val_neg_1_ls.extend(neg_1_i)

        # Type 2: Fix microbe and disease, random gene for validation
        while cva_2 < neg_num_test:
            b_2 = int(i[1])
            c_2 = int(i[2])
            if gene_ls:
                a_2 = np.random.choice(gene_ls)
                gene_ls.remove(a_2)
                if arr_true[a_2, b_2, c_2] != 1 and arr_false_train[a_2, b_2, c_2] != 1 and arr_false_val_2[a_2, b_2, c_2] != 1:
                    arr_false_val_2[a_2, b_2, c_2] = 1
                    cva_2 += 1
                    neg_2_i.append((a_2, b_2, c_2, 0))
            else:
                distance_2 = neg_num_test - cva_2
                if neg_2_i:
                    last_ind = len(neg_2_i) - 1
                    for k in range(distance_2):
                        neg_2_i.append(neg_2_i[last_ind])
                break
        np.random.shuffle(neg_2_i)
        val_neg_2_ls.extend(neg_2_i)

        # Type 3: Fix gene and disease, random microbe for validation
        while cva_3 < neg_num_test:
            a_3 = int(i[0])
            c_3 = int(i[2])
            if mic_ls:
                b_3 = np.random.choice(mic_ls)
                mic_ls.remove(b_3)
                if arr_true[a_3, b_3, c_3] != 1 and arr_false_train[a_3, b_3, c_3] != 1 and arr_false_val_3[a_3, b_3, c_3] != 1:
                    arr_false_val_3[a_3, b_3, c_3] = 1
                    cva_3 += 1
                    neg_3_i.append((a_3, b_3, c_3, 0))
            else:
                distance_3 = neg_num_test - cva_3
                if neg_3_i:
                    last_ind = len(neg_3_i) - 1
                    for k in range(distance_3):
                        neg_3_i.append(neg_3_i[last_ind])
                break
        np.random.shuffle(neg_3_i)
        val_neg_3_ls.extend(neg_3_i)

        # Type 4: Fix gene and microbe, random disease for validation
        while cva_4 < neg_num_test:
            a_4 = int(i[0])
            b_4 = int(i[1])
            if dis_ls:
                c_4 = np.random.choice(dis_ls)
                dis_ls.remove(c_4)
                if arr_true[a_4, b_4, c_4] != 1 and arr_false_train[a_4, b_4, c_4] != 1 and arr_false_val_4[a_4, b_4, c_4] != 1:
                    arr_false_val_4[a_4, b_4, c_4] = 1
                    cva_4 += 1
                    neg_4_i.append((a_4, b_4, c_4, 0))
            else:
                distance_4 = neg_num_test - cva_4
                if neg_4_i:
                    last_ind = len(neg_4_i) - 1
                    for k in range(distance_4):
                        neg_4_i.append(neg_4_i[last_ind])
                break
        np.random.shuffle(neg_4_i)
        val_neg_4_ls.extend(neg_4_i)

    return train_neg_all, train_neg_1_ls, train_neg_2_ls, train_neg_3_ls, train_neg_4_ls, \
           val_neg_1_ls, val_neg_2_ls, val_neg_3_ls, val_neg_4_ls


def get_train_val_data(all_data, train_ind, val_ind, adj, seed):
    """To generate negative samples for the training set for 5-fold CV."""
    neg_num_test = 30
    train_data_pos, val_data_pos = all_data[train_ind], all_data[val_ind]
    train_data_all, tr_neg_1_ls, tr_neg_2_ls, tr_neg_3_ls, tr_neg_4_ls, \
    te_neg_1_ls, te_neg_2_ls, te_neg_3_ls, te_neg_4_ls = neg_data_generate(
        adj, train_data_pos, val_data_pos, neg_num_test, seed)
    np.random.shuffle(train_data_all)
    val_neg_data = []
    for i in te_neg_1_ls:
        if list(i)[-1] == 0:
            val_neg_data.append(i)
    train_data_pos = train_data_pos.copy().astype(int)
    val_data_pos = val_data_pos.copy().astype(int)
    return train_data_pos, np.array(tr_neg_1_ls), val_data_pos, np.array(val_neg_data)


def get_indep_data(adj, train_data_pos, val_data_pos, seed):
    """To generate negative samples for the training set for independent test."""
    neg_num_test = 30
    train_data_all, tr_neg_1_ls, tr_neg_2_ls, tr_neg_3_ls, tr_neg_4_ls, \
    te_neg_1_ls, te_neg_2_ls, te_neg_3_ls, te_neg_4_ls = neg_data_generate(
        adj, train_data_pos, val_data_pos, neg_num_test, seed)
    np.random.shuffle(train_data_all)
    val_neg_data = []
    for i in te_neg_1_ls:
        if list(i)[-1] == 0:
            val_neg_data.append(i)
    train_data_pos = train_data_pos.copy().astype(int)
    val_data_pos = val_data_pos.copy().astype(int)
    return train_data_pos, np.array(tr_neg_1_ls), val_data_pos, np.array(val_neg_data)


# --- End of helper functions ---


class HCMDataset(DGLDataset):
    _prefix = 'https://s3.cn-north-1.amazonaws.com.cn/dgl-data/'
    _urls = {
        # Define URL for dataHCMGNN if it's different from the default construction
        # 'dataHCMGNN': 'https://example.com/path/to/dataHCMGNN.zip'
    }

    def __init__(self, name, raw_dir=None, force_reload=False, verbose=True, args=None):
        assert name in ['dataHCMGNN']
        # self.name = name
        self.args = args
        # Corrected data_path and g_path to be within the raw_dir/name subdirectory
        self.data_path = os.path.join('./openhgnn/dataset', f'{name}.zip') # Path to downloaded zip
        self.g_path = os.path.join('./openhgnn/dataset', name, 'graph.bin') # Path where processed DGL graph will be saved
        raw_dir = './openhgnn/dataset'
        url = self._urls.get(name, f'{self._prefix}dataset/{name}.zip')

        super(HCMDataset, self).__init__(name=name,
                                         url=url,
                                         raw_dir=raw_dir,
                                         force_reload=force_reload,
                                         verbose=verbose)

    def _download(self):
        """
        Download raw data to local disk and extract it.
        """
        if os.path.exists(self.data_path):
           pass
        else:
            file_path = os.path.join(self.raw_dir)
            # download file
            download(self.url, path=file_path)
        extract_archive(self.data_path, os.path.join(self.raw_path))


    def process(self):
        """
        Process raw data to graphs, features, and generate positive/negative samples (CSVs).
        This method now creates the DGL graph from raw data.
        """


        # Base path for raw data files within the extracted directory
        extracted_data_dir = os.path.join(self.raw_path, self.name)

        # 1. Load initial features (gene, microbe, disease similarities)
        try:
            mic_feat_path = os.path.join(extracted_data_dir, 'mic_sim176.txt')
            gene_feat_path = os.path.join(extracted_data_dir, 'gene_sim_BP301.csv')
            disease_feat_path = os.path.join(extracted_data_dir, 'dis_sim153.txt')

            self.microbe_feat = pd.read_csv(mic_feat_path, header=None, sep='\t').values
            self.gene_feat = pd.read_csv(gene_feat_path, header=None).values
            self.disease_feat = pd.read_csv(disease_feat_path, header=None, sep='\t').values

            self.features = {
                'g': th.FloatTensor(self.gene_feat),
                'm': th.FloatTensor(self.microbe_feat),
                'd': th.FloatTensor(self.disease_feat)
            }
            self.in_size = {
                'g': self.gene_feat.shape[1],
                'm': self.microbe_feat.shape[1],
                'd': self.disease_feat.shape[1]
            }
            print("Feature files loaded.")
        except FileNotFoundError as e:
            print(f"Error loading feature files: {e}. Ensure they are in the extracted dataset directory.")
            self.features = None
            self.in_size = None
            raise # Re-raise to stop if features are critical

        # 2. Load positive pairs (g_m_d_pos_pairs.txt)
        adj_data_path = os.path.join(extracted_data_dir, 'g_m_d_pos_pairs.txt')
        try:
            adj_data = np.loadtxt(adj_data_path)
            print(f"Positive pairs loaded from {adj_data_path}.")
        except FileNotFoundError as e:
            print(f"Error loading positive pairs: {e}. Ensure 'g_m_d_pos_pairs.txt' is in the extracted dataset directory.")
            raise # Re-raise to stop if positive pairs are critical


        # 3. Generate negative samples and save to CSVs (from data_process.py's generate_dataset logic)
        np.random.shuffle(adj_data) # Shuffle positive data again for splitting

        # self.args.output_dir = os.path.join('./openhgnn/output', self.args.model)
        args = self.args # Get arguments for seed, etc.
        np.random.seed(args.seed) # Set seed for reproducibility
        # Create output directories if they don't exist
        cv_data_dir = os.path.join(args.output_dir, 'CV_data')
        indep_data_dir = os.path.join(args.output_dir, 'indepent_data')
        os.makedirs(cv_data_dir, exist_ok=True)
        os.makedirs(indep_data_dir, exist_ok=True)

        # Split for Independent Test (10% of data)
        cv_data = adj_data[int(0.1 * len(adj_data)):, :]
        indep_data = adj_data[:int(0.1 * len(adj_data)), :]

        # 5-Fold Cross Validation Data Generation
        kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
        for fold_num, (train_index, val_index) in enumerate(kf.split(cv_data)):
            current_fold_dir = os.path.join(cv_data_dir, f'CV_{fold_num + 1}')
            os.makedirs(current_fold_dir, exist_ok=True)

            train_data_pos, train_data_neg, val_data_pos, val_data_neg = \
                get_train_val_data(cv_data, train_index, val_index, adj_data, args.seed)

            np.savetxt(os.path.join(current_fold_dir, 'train_data_pos.csv'), train_data_pos, delimiter=",", fmt='%d')
            np.savetxt(os.path.join(current_fold_dir, 'train_data_neg.csv'), train_data_neg, delimiter=',', fmt='%d')
            np.savetxt(os.path.join(current_fold_dir, 'val_data_pos.csv'), val_data_pos, delimiter=',', fmt='%d')
            np.savetxt(os.path.join(current_fold_dir, 'val_data_neg.csv'), val_data_neg, delimiter=',', fmt='%d')
            print(f"Saved CV fold {fold_num + 1} data to {current_fold_dir}")

        # Independent Test Data Generation
        train_data_pos_indep, train_data_neg_indep, test_data_pos_indep, test_data_neg_indep = \
            get_indep_data(adj_data, cv_data, indep_data, args.seed)

        np.savetxt(os.path.join(indep_data_dir, 'train_data_pos.csv'), train_data_pos_indep, delimiter=",", fmt='%d')
        np.savetxt(os.path.join(indep_data_dir, 'train_data_neg.csv'), train_data_neg_indep, delimiter=",", fmt='%d')
        np.savetxt(os.path.join(indep_data_dir, 'test_data_pos.csv'), test_data_pos_indep, delimiter=",", fmt='%d')
        np.savetxt(os.path.join(indep_data_dir, 'test_data_neg.csv'), test_data_neg_indep, delimiter=",", fmt='%d')
        print(f"Saved independent test data to {indep_data_dir}")

        # Store paths to the generated CSVs for external access
        self.train_data_path = os.path.join(indep_data_dir, 'train_data_pos.csv')
        self.train_neg_data_path = os.path.join(indep_data_dir, 'train_data_neg.csv')
        self.test_data_pos_path = os.path.join(indep_data_dir, 'test_data_pos.csv')
        self.test_data_neg_path = os.path.join(indep_data_dir, 'test_data_neg.csv')

        # Store the independent test data for direct access if needed, or rely on paths
        self.train_data_indep = np.vstack((train_data_pos_indep, train_data_neg_indep))
        self.test_data_indep = np.vstack((test_data_pos_indep, test_data_neg_indep))


        # 4. Create DGL Heterogeneous Graph from independent training positive data
        # Determine node counts from feature file shapes (number of rows)
        # These are already class attributes from step 1
        num_genes = self.gene_feat.shape[0]
        num_microbes = self.microbe_feat.shape[0]
        num_diseases = self.disease_feat.shape[0]

        # Use train_data_pos_indep to build the graph
        pos_data_for_graph = train_data_pos_indep  # This is the key change

        g_m_edges, m_d_edges, g_d_edges = [list() for x in range(3)]
        for i in range(len(pos_data_for_graph)):
            one_g_m_edge = []
            one_g_m_edge.extend(pos_data_for_graph[i][0:2].tolist())
            one_m_d_edge = []
            one_m_d_edge.extend(pos_data_for_graph[i][1:3].tolist())
            one_g_d_edge = []
            one_g_d_edge.extend([pos_data_for_graph[i][0], pos_data_for_graph[i][2]])
            if one_g_m_edge not in g_m_edges:  # Use 'not in' for list membership check
                g_m_edges.append(one_g_m_edge)
            if one_m_d_edge not in m_d_edges:
                m_d_edges.append(one_m_d_edge)
            if one_g_d_edge not in g_d_edges:
                g_d_edges.append(one_g_d_edge)

        # Convert to numpy arrays and sort
        g_m_edges = np.array(sorted(g_m_edges, key=(lambda x: x[0])), dtype=int)
        m_d_edges = np.array(sorted(m_d_edges, key=(lambda x: x[0])), dtype=int)
        g_d_edges = np.array(sorted(g_d_edges, key=(lambda x: x[0])), dtype=int)

        # Define the data dictionary for the heterogeneous graph
        data_dict = {
            ('g', 'g_m', 'm'): (th.LongTensor(g_m_edges[:, 0]), th.LongTensor(g_m_edges[:, 1])),
            ('m', 'm_d', 'd'): (th.LongTensor(m_d_edges[:, 0]), th.LongTensor(m_d_edges[:, 1])),
            ('g', 'g_d', 'd'): (th.LongTensor(g_d_edges[:, 0]), th.LongTensor(g_d_edges[:, 1])),
            ('m', 'm_g', 'g'): (th.LongTensor(g_m_edges[:, 1]), th.LongTensor(g_m_edges[:, 0])),  # Reverse edge
            ('d', 'd_m', 'm'): (th.LongTensor(m_d_edges[:, 1]), th.LongTensor(m_d_edges[:, 0])),  # Reverse edge
            ('d', 'd_g', 'g'): (th.LongTensor(g_d_edges[:, 1]), th.LongTensor(g_d_edges[:, 0]))  # Reverse edge
        }

        # Define the number of nodes for each type
        num_nodes_dict = {
            'g': num_genes,
            'm': num_microbes,
            'd': num_diseases
        }

        # Create the DGL heterogeneous graph
        self._g = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)
        print("DGL heterogeneous graph created.")

        # Attach features to nodes
        if self.features is not None:
            self._g.nodes['g'].data['feat'] = self.features['g']
            self._g.nodes['m'].data['feat'] = self.features['m']
            self._g.nodes['d'].data['feat'] = self.features['d']
            print("Node features attached to graph.")

        # Modify the g_path to save in extracted_data_dir as requested
        self.g_path = os.path.join(extracted_data_dir, 'graph.bin')
        os.makedirs(os.path.dirname(self.g_path), exist_ok=True)
        save_graphs(self.g_path, [self._g])
        print(f"Generated DGL graph saved to {self.g_path}.")

        print("Dataset processing complete.")


    def __getitem__(self, idx):
        """
        Get one example by index.
        Returns the DGL graph at index 0.
        """
        if idx == 0 and hasattr(self, '_g') and self._g is not None:
            return self._g
        raise IndexError("HCMDataset only supports index 0 for graph access.")


    def __len__(self):
        """
        Number of data examples.
        Returns 1 as it provides a single primary graph object.
        """
        return 1

    def save(self):
        """
        Save processed data to directory `self.save_path`.
        The DGL graph is saved in process() method.
        """
        # The graph is already saved in process()
        pass

    def load(self):
        """
        Load processed data from directory `self.save_path`.
        This method is called by DGLDataset's superclass if has_cache() is true.
        It should load the DGL graph and potentially set other attributes if not done in process().
        """
        if os.path.exists(self.g_path):
            g, _ = load_graphs(self.g_path)
            self._g = g[0] # DGL load_graphs returns a list of graphs, take the first one
            print(f"DGL graph loaded from cache: {self.g_path}.")
            # Load features and set paths for CSVs
            self._load_features_and_set_paths()
        else:
            raise FileNotFoundError(f"Graph cache not found at {self.g_path}. Please ensure process() has been run.")


    def has_cache(self):
        self.args.output_dir = os.path.join('./openhgnn/output', self.args.model)
        self.g_path = os.path.join(self.raw_path, self.name, 'graph.bin')
        graph_exists = os.path.exists(self.g_path)

        extracted_raw_data_dir = os.path.join(self.raw_path, self.name)
        raw_data_folder_exists = os.path.exists(extracted_raw_data_dir)

        train_csv_exists = os.path.exists(os.path.join(self.args.output_dir, 'indepent_data', 'train_data_pos.csv'))
        test_csv_exists = os.path.exists(os.path.join(self.args.output_dir, 'indepent_data', 'test_data_pos.csv'))

        return raw_data_folder_exists and graph_exists and train_csv_exists and test_csv_exists

    def _load_features_and_set_paths(self):
        """Helper to load features and set paths when loading from cache or after processing."""
        extracted_data_dir = os.path.join(self.raw_path, self.name)
        mic_feat_path = os.path.join(extracted_data_dir, 'mic_sim176.txt')
        gene_feat_path = os.path.join(extracted_data_dir, 'gene_sim_BP301.csv')
        disease_feat_path = os.path.join(extracted_data_dir, 'dis_sim153.txt')

        self.microbe_feat = pd.read_csv(mic_feat_path, header=None, sep='\t').values
        self.gene_feat = pd.read_csv(gene_feat_path, header=None).values
        self.disease_feat = pd.read_csv(disease_feat_path, header=None, sep='\t').values

        self.features = {
            'g': th.FloatTensor(self.gene_feat),
            'm': th.FloatTensor(self.microbe_feat),
            'd': th.FloatTensor(self.disease_feat)
        }
        self.in_size = {
            'g': self.gene_feat.shape[1],
            'm': self.microbe_feat.shape[1],
            'd': self.disease_feat.shape[1]
        }
        args = self.args

        indep_data_dir = os.path.join(args.output_dir, 'indepent_data')
        self.train_data_path = os.path.join(indep_data_dir, 'train_data_pos.csv')
        self.train_neg_data_path = os.path.join(indep_data_dir, 'train_data_neg.csv')
        self.test_data_pos_path = os.path.join(indep_data_dir, 'test_data_pos.csv')
        self.test_data_neg_path = os.path.join(indep_data_dir, 'test_data_neg.csv')

        # Also load the independent train/test data arrays if needed
        train_data_pos_indep = np.loadtxt(self.train_data_path, delimiter=",")
        train_data_neg_indep = np.loadtxt(self.train_neg_data_path, delimiter=",")
        test_data_pos_indep = np.loadtxt(self.test_data_pos_path, delimiter=",")
        test_data_neg_indep = np.loadtxt(self.test_data_neg_path, delimiter=",")

        self.train_data_indep = np.vstack((train_data_pos_indep, train_data_neg_indep))
        self.test_data_indep = np.vstack((test_data_pos_indep, test_data_neg_indep))

