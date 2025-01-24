import os
import torch,requests,zipfile,io
import os.path as osp
from typing import Any, Callable, List, Optional
from collections.abc import Sequence
import copy
import ssl
import sys
import urllib
import errno



class NBF_Dataset():

    def __init__(self, root, name, version, transform=None, pre_transform=None):#   root/name/version == ~/WN18RR/v1
        if isinstance(root, str):
            root = osp.expanduser(osp.normpath(root))
        self.root = root
        self.name = name
        self.version = version
        self.transform = transform
        self.pre_transform = pre_transform

        assert name in ["FB15k-237", "WN18RR"]
        assert version in ["v1", "v2", "v3", "v4"]

        self.urls = {
        "FB15k-237": [
            "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s_ind/train.txt",
            "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s_ind/test.txt",
            "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s/train.txt",
            "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s/valid.txt"
        ],
        "WN18RR": [
            "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s_ind/train.txt",
            "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s_ind/test.txt",
            "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s/train.txt",
            "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s/valid.txt"
        ]
    }
        
        

        self._download()
        self._process()

        self.train_data = torch.load(self.processed_paths[0]) 
        self.valid_data = torch.load(self.processed_paths[1]) 
        self.test_data = torch.load(self.processed_paths[2])  
        print(self.processed_paths[0])


    def _download(self):
        if files_exist(self.raw_paths):  
            return

        makedirs(self.raw_dir)
        self.download()
    
    def _process(self):

        if files_exist(self.processed_paths): 
            return
        
        makedirs(self.processed_dir)
        self.process()


    @property 
    def num_relations(self):
        #return int(self.data.edge_type.max()) + 1
        return max(int(self.train_data.edge_type.max()),
                   int(self.valid_data.edge_type.max()),
                   int(self.test_data.edge_type.max()),
                   ) + 1

    @property
    def raw_dir(self): 
        return os.path.join(self.root, self.name, self.version, "raw") 

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name, self.version, "processed")   

    @property
    def processed_file_names(self): 
        return ["train_data.pt","valid_data.pt","test_data.pt"]

    @property
    def raw_file_names(self):
        return [
            "train_ind.txt", "test_ind.txt", "train.txt", "valid.txt"
        ]


    def download(self):

        url = "https://s3.cn-north-1.amazonaws.com.cn/dgl-data/dataset/openhgnn/{}.zip".format(self.name)
        response = requests.get(url)
        with zipfile.ZipFile(io.BytesIO(response.content)) as myzip:
            myzip.extractall(self.raw_dir)
        print("---  download data finished---")

        

    @property
    def raw_paths(self) :
        r"""The absolute filepaths that must be present in order to skip
        downloading."""
        files = self.raw_file_names
        # Prevent a common source of error in which `file_names` are not
        # defined as a property.
        if isinstance(files, Callable):
            files = files()
        return [osp.join(self.raw_dir, f) for f in to_list(files)]
    

    @property
    def processed_paths(self) -> List[str]:
        r"""The absolute filepaths that must be present in order to skip
        processing."""
        files = self.processed_file_names
        # Prevent a common source of error in which `file_names` are not
        # defined as a property.
        if isinstance(files, Callable):
            files = files()
        return [osp.join(self.processed_dir, f) for f in to_list(files)]
    


    def process(self):
        test_files = self.raw_paths[:2]
        train_files = self.raw_paths[2:]
        inv_train_entity_vocab = {}
        inv_test_entity_vocab = {}
        inv_relation_vocab = {}
        triplets = []
        num_samples = []

        for txt_file in train_files:
            with open(txt_file, "r") as fin:
                num_sample = 0
                for line in fin:
                    h_token, r_token, t_token = line.strip().split("\t")
                    if h_token not in inv_train_entity_vocab:
                        inv_train_entity_vocab[h_token] = len(inv_train_entity_vocab)
                    h = inv_train_entity_vocab[h_token]
                    if r_token not in inv_relation_vocab:
                        inv_relation_vocab[r_token] = len(inv_relation_vocab)
                    r = inv_relation_vocab[r_token]
                    if t_token not in inv_train_entity_vocab:
                        inv_train_entity_vocab[t_token] = len(inv_train_entity_vocab)
                    t = inv_train_entity_vocab[t_token]
                    triplets.append((h, t, r))
                    num_sample += 1
            num_samples.append(num_sample)
        count = 0
        for txt_file in test_files:
            with open(txt_file, "r") as fin:
                num_sample = 0
                for line in fin:
                    h_token, r_token, t_token = line.strip().split("\t")
                    if h_token not in inv_test_entity_vocab:
                        inv_test_entity_vocab[h_token] = len(inv_test_entity_vocab)
                    h = inv_test_entity_vocab[h_token]
                    count += 1
                    if r_token in inv_relation_vocab: # assert r_token in inv_relation_vocab
                        r = inv_relation_vocab[r_token]
                        if t_token not in inv_test_entity_vocab:
                            inv_test_entity_vocab[t_token] = len(inv_test_entity_vocab)
                        t = inv_test_entity_vocab[t_token]
                        triplets.append((h, t, r))
                        num_sample += 1
            num_samples.append(num_sample)

        triplets = torch.tensor(triplets)

        edge_index = triplets[:, :2].t()
        edge_type = triplets[:, 2]

        num_relations = int(edge_type.max()) + 1


        train_fact_slice = slice(None, sum(num_samples[:1]))
        test_fact_slice = slice(sum(num_samples[:2]), sum(num_samples[:3]))
        train_fact_index = edge_index[:, train_fact_slice]
        train_fact_type = edge_type[train_fact_slice]
        test_fact_index = edge_index[:, test_fact_slice]
        test_fact_type = edge_type[test_fact_slice]
        # add flipped triplets for the fact graphs
        train_fact_index = torch.cat([train_fact_index, train_fact_index.flip(0)], dim=-1)
        train_fact_type = torch.cat([train_fact_type, train_fact_type + num_relations])
        test_fact_index = torch.cat([test_fact_index, test_fact_index.flip(0)], dim=-1)
        test_fact_type = torch.cat([test_fact_type, test_fact_type + num_relations])

        train_slice = slice(None, sum(num_samples[:1]))
        valid_slice = slice(sum(num_samples[:1]), sum(num_samples[:2]))
        test_slice = slice(sum(num_samples[:3]), sum(num_samples))

        train_data = NBF_Data(edge_index=train_fact_index, edge_type=train_fact_type,
                             num_nodes=len(inv_train_entity_vocab),num_edges = train_fact_index.shape[1],
                          target_edge_index=edge_index[:, train_slice], target_edge_type=edge_type[train_slice])
        
        valid_data = NBF_Data(edge_index=train_fact_index, edge_type=train_fact_type, 
                            num_nodes=len(inv_train_entity_vocab),num_edges = train_fact_index.shape[1],
                          target_edge_index=edge_index[:, valid_slice], target_edge_type=edge_type[valid_slice])
        
        test_data = NBF_Data(edge_index=test_fact_index, edge_type=test_fact_type,
                            num_nodes=len(inv_test_entity_vocab),num_edges = test_fact_index.shape[1],
                         target_edge_index=edge_index[:, test_slice], target_edge_type=edge_type[test_slice])

        if self.pre_transform is not None:
            train_data = self.pre_transform(train_data)
            valid_data = self.pre_transform(valid_data)
            test_data = self.pre_transform(test_data)

        #torch.save((train_data, valid_data, test_data), self.processed_paths[0])
        torch.save(train_data, self.processed_paths[0]) 
        torch.save(valid_data, self.processed_paths[1])     
        torch.save(test_data, self.processed_paths[2]) 


        

    def __repr__(self):
        return "%s()" % self.name





class NBF_Data():
    def __init__(self,
                 num_nodes,num_edges,
                 edge_index,edge_type,
                 target_edge_index,target_edge_type
                 ):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.edge_index = copy.copy(edge_index)
        self.edge_type = copy.copy(edge_type)
        self.target_edge_index = copy.copy(target_edge_index)
        self.target_edge_type  = copy.copy(target_edge_type)



def files_exist(files: List[str]) -> bool: 
# NOTE: We return `False` in case `files` is empty, leading to a
# re-processing of files on every instantiation.
    return len(files) != 0 and all([osp.exists(f) for f in files])



def makedirs(path: str): 
    r"""Recursively creates a directory.

    Args:
        path (str): The path to create.
    """
    try:
        os.makedirs(osp.expanduser(osp.normpath(path)))
    except OSError as e:
        if e.errno != errno.EEXIST and osp.isdir(path):
            raise e



def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]



def download_url(url: str, folder: str, log: bool = True,
                 filename: Optional[str] = None):
    r"""Downloads the content of an URL to a specific folder.

    Args:
        url (str): The URL.
        folder (str): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """

    if filename is None:
        filename = url.rpartition('/')[2]
        filename = filename if filename[0] == '?' else filename.split('?')[0]

    path = osp.join(folder, filename)

    if osp.exists(path):  # pragma: no cover
        if log and 'pytest' not in sys.modules:
            print(f'Using existing file {filename}', file=sys.stderr)
        return path

    if log and 'pytest' not in sys.modules:
        print(f'Downloading {url}', file=sys.stderr)

    makedirs(folder)

    context = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=context)

    with open(path, 'wb') as f:
        # workaround for https://bugs.python.org/issue42853
        while True:
            chunk = data.read(10 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    return path




    