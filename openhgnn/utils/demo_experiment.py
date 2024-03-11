from openhgnn.utils import hgbi
from openhgnn.utils.tsne_g import draw_tsne
from openhgnn.utils.visualize import plot_degree_dist
from openhgnn.utils.meta_path_analyse import meta_path_heterophily

#simple demo to load graph
def demo_load_g():
    ds_node = hgbi.build_dataset(name = 'acm4NSHE',task = 'node_classification')
    print(ds_node.g)

#simple demo to load and rebuild, it takes a long time
def demo_load_rebuild_g():
    ds_link = hgbi.build_dataset(
        name = 'ohgbl-yelp2',task = 'link_prediction')
    print(ds_link.g)

    #Build Risk Product Detection Dataset
    ds_node = hgbi.build_dataset(
        name = 'RPDD',task = 'node_classification')
    '''
    ds_node contains dataset details:
    e.g. in_dim, meta_paths, category, num_classes
    ds_node.g is Graph on DGL format
    '''

    #Build Takeout Recommendation Dataset
    ds_link =  hgbi.build_dataset(
        name = 'TRD',task = 'link_prediction')
    '''
    ds_link contains dataset details:
    e.g. target_link, target_link_r, node_type,
    ds_link.g is Graph on DGL format
    '''

    #construct own dataset
    #rebuild
    ds = hgbi.MyDataset(
        name="my_graph",path="./graph.bin")
    ds_link = hgbi.AsLinkPredictionDataset(
        ds, target_link=['user-buy-poi'],
        target_link_r=['rev_user-buy-poi'],
        split_ratio=[0.5, 0.3, 0.3],
        neg_ratio=3,
        neg_sampler='global'
    )

#analyse of degree
def demo_degree():
    dataset = hgbi.build_dataset(
    name = 'dblp4GTN',task = 'node_classification')
    plot_degree_dist(dataset.g,'./degree.png')

#visualization of tsne visualization on data split
def demo_tsne():
    dataset = hgbi.build_dataset(
        name = 'dblp4GTN',task = 'node_classification')
    draw_tsne(dataset,'./tsne.png')

#analyse of heterophily
def demo_meta_path_heterophily():
    dataset = hgbi.build_dataset(
    name = 'dblp4GTN',task = 'node_classification')
    g = dataset[0]
    meta_path_nums, heterophily, edge_radio  = meta_path_heterophily(g, meta_paths_dict=dataset.meta_paths_dict, strength=2)

if __name__ == "__main__":
    #demo_load_g() 
    #demo_load_rebuild_g()
    #demo_degree()
    #demo_tsne()
    demo_meta_path_heterophily()
