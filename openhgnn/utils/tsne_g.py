import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from dgl.data import CoraGraphDataset
import matplotlib.font_manager as fm

def draw_tsne(ds,save_path):
    dataset = ds
    graph = dataset.g
    target = dataset.target_ntype
    train_mask = graph.ndata['train_mask'][target]
    val_mask = graph.ndata['val_mask'][target]
    test_mask = graph.ndata['test_mask'][target]
    mask_map = {'train':0, 'val':1, 'test':2}
    num_nodes = graph.num_nodes(target)
    mask = [ 0 for i in range(num_nodes)]
    for i in range(num_nodes):
        if train_mask[i]==True:
            mask[i]= (0.5451, 0.1019,0.1019) 
        elif val_mask[i]==True:
            mask[i]= (0.0941, 0.4549, 0.8039)
        else:
            mask[i]= (0.9333, 0.6784, 0.0549)


    tsne = TSNE(n_components=2, random_state=42, perplexity=10)
    if len(graph.ndata['feat'])>0 :
        feat = graph.ndata['feat'][target]  
    elif len(graph.ndata['h'])>0:
        feat = graph.ndata['h'][target] 
    else:
        print("error: no feat of nodes")
        return None
    
    aft_tsne = tsne.fit_transform(feat)

    plt.scatter(aft_tsne[:, 0], aft_tsne[:, 1] , c=mask, s=10)
    plt.style.use('seaborn')
    max_x = max(aft_tsne[:,0])
    max_y = max(aft_tsne[:,1])
    min_x = min(aft_tsne[:,0])
    min_y = min(aft_tsne[:,1])
    plt.xlim( min_x+30, max_x-20)
    plt.ylim( min_y+30, max_y-50)
    #plt.title('T-SNE Visualization of Cora Dataset',pad=20)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    color1 = (0.9333, 0.6784, 0.0549)
    color2 = (0.0941, 0.4549, 0.8039)
    color3 = (0.5451, 0.1019,0.1019)
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markersize=12,markerfacecolor=color1, label='Train'),
                    plt.Line2D([0], [0], marker='o', color='w', markersize=12, markerfacecolor=color2, label='Val'),
                    plt.Line2D([0], [0], marker='o', color='w',  markersize=12, markerfacecolor=color3, label='Test')]

    # 添加图例
    font = fm.FontProperties(weight='bold', size=12)
    plt.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.98), ncol=3,frameon=False, prop = font)
    plt.gca().set_aspect(1)
    plt.xlabel('t-SNE',fontsize=13)
    plt.ylabel('')
    #plt.axis('off')
    axes = plt.gca()
    [axes.spines[loc_axis].set_visible(False) for loc_axis in ['top','right','bottom','left']]
    #plt.gca().set_title("t_SNE")
    axes.set_xticks([])
    axes.set_yticks([])
    #plt.title('t_SNE')
    plt.savefig(save_path,dpi=300)