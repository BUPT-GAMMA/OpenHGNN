# KTN[NIPS 2022]

-   paper: [Zero-shot Transfer Learning within a Heterogeneous Graph via Knowledge Transfer Networks](https://arxiv.org/abs/2203.02018)
-   Code from author: [KTN](https://github.com/minjiyoon/KTN)

## Description

KTN is a zero-shot cross-type transfer learning method for heterogeneous graphs. It aims to transfer knowledge from a label-abundant source node type to a label-scarce target node type in the same heterogeneous graph.

KTN is designed to tackle the issue of HGNNs learning different feature extractors for different node types, which hinders direct transfer across types.

The core of KTN is a learnable graph convolution network $t_{KTN}$ that transforms target node embeddings into the source embedding space:               
$$
\textbf{t}_{\text{KTN}}(H^{(L)}_{t}) = A_{ts} H^{(L)}_{t} T_{ts} 
$$
And loss is now calculated by:
$$
\mathcal{L}_{\text{KTN}} = \left\|H^{(L)}_{s} - \textbf{t}_{\text{KTN}}(H^{(L)}_{t})\right\|_{2}
$$

Where $H^{(L)}_t$ is the L-th layer embeddings of target nodes, $A_{ts}$ is the adjacency matrix from target to source types, and $T_{ts}$ is a learnable transformation matrix. By minimizing the L2 distance between source embeddings $H^{(L)}_s$ and mapped target embeddings $t_{KTN}(H^{(L)}_t)$, $T_{ts}$ learns to project target nodes into source space.

At test time, the learned $t_{KTN}$ is simply applied to transform $H^{(L)}_t$, which is then fed into the source classifier for prediction.

## How to run

- Clone the Openhgnn-DGL

```bash
python main.py -m HMPNN -d OAG_CS -t KTN -g 0 --use_best_config
```

  If you do not have gpu, set -gpu -1.

## Performance: Recommendation

-   Device: GPU, **NVIDIA L4**

| Source | Target | Task | NDCG Base | NDCG KTN | Gain   |
| :----: | :----: | :--: | --------- | -------- | ------ |
| paper  | author |  L1  | 0.2781    | 0.6058   | 117.8% |
| paper  | author |  L2  | 0.2164    | 0.2782   | 28.56% |
| venue  | author |  L1  | 0.3527    | 0.5815   | 64.87% |
| author | paper  |  L1  | 0.2592    | 0.6116   | 136.0% |
| author | venue  |  L2  | 0.2570    | 0.4623   | 79.89% |

## Dataset

Candidate dataset: OAG_CS

### OAG_CS

The OAG dataset, comprises five types of nodes: papers (P), authors (A), institutions (I), venues (V), and fields (F), along with their respective relationships. Paper and author nodes have attributes based on text, while institution, venue, and field nodes have attributes based on both text and the graph structure. Additionally, paper, author, and venue nodes are assigned labels indicating research fields categorized into two hierarchical levels, L1 and L2. OAG_CS is a subgraph constructed from OAG specifically for the field of computer science.

#### Nodes

| Type        | Count  |
| ----------- | ------ |
| affiliation | 9079   |
| author      | 510189 |
| field       | 45717  |
| paper       | 544244 |
| venue       | 6934   |

#### Edges

| Type               | Count    |
| ------------------ | -------- |
| affiliation-author | 612872   |
| author-affiliation | 612872   |
| author-paper       | 1091559  |
| field-field        | 525052   |
| field-paper        | 3709710  |
| paper-author       | 1091559  |
| paper-field        | 3709710  |
| paper-paper        | 11577794 |
| paper-venue        | 544244   |
| venue-paper        | 544244   |

## Batching

Full batch train on OAG_CS dataset with source domain of paper and target domain of author on task L1 and L1 would cost about 36 GiB and 174 GiB of RAM respectively. So it is recommended to do mini-batch train instead of full batch train.

You can choose the batch_size to fit your VRAM. The following is reference for batch size and the VRAM required:

| Source | Target | Task | Batch Size | VRAM Required |
| ------ | ------ | ---- | ---------- | ------------- |
| paper  | author | L1   | 4096       | 19.3 GiB      |
| paper  | author | L1   | 3072       | 15.9 GiB      |
| paper  | author | L1   | 2048       | 14.8 GiB      |
| paper  | author | L1   | 1024       | 8.63 GiB      |
| author | paper  | L1   | 8192       | 20.1 GiB      |
| author | venue  | L2   | 1024       | 19.7 GiB      |

To speed up training, user can choose to only iterates through only part of the dataset each epoch while training and testing by setting `source_train_batch`, `source_test_batch`, and `target_test_batch`. If the set batch count is larger than the number of batches, the trainer would just iterates through all batches. For reference, a batch size of 128 would batch the graph into 1001 batches with paper as source domain.

## Task

Despite its name of `KTN4MultiLabelNodeClassification`, the task can be used for any multi-label node classification.

The task re-implemented NDCG and MRR metrics in PyTorch for evaluation.

The `Classifier` module is a simple linear classifier that computes the loss and evaluation metrics. It serves as the common classifier for both source domain and target domain in the KTN task.

## Trainerflow

The `KTN_NodeClassification` Trainerflow implements the Knowledge Transfer Network (KTN) for transfer learning in node classification tasks. It trains the HGNN, the node classifier and the matching_w (corresponding to $T_{ts}$ in the paper) jointly during the training process. Specifically:

- The HGNN generates node embeddings for both source and target
- The classifier is trained on source node embeddings and labels
- The matching_w matrices are learned to project target embeddings into the source space
- Matching loss aligns the source and projected target representations

After training, the flow applies the learned components to enable knowledge transfer:

- Apply learned matching_w to map target nodes into source embedding space
- Feed transferred target node embeddings into the trained source classifier
- Predict labels for target nodes using the classifier

This enables label knowledge transfer from label-rich source nodes to scarce target nodes within the heterogeneous graph.

## More

#### Contributor

Yang Hongchen

#### If you have any questions,

Submit an issue or email to  [yanghongchen@bupt.edu.cn](mailto:yanghongchen@bupt.edu.cn).



