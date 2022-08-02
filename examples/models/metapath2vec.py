from openhgnn import Experiment
import numpy

if __name__ == '__main__':
    experiment = Experiment(model='Metapath2vec', dataset='dblp4MAGNN', task='node_classification', gpu=-1,
                            meta_path_key='APVPA', learning_rate=0.01, dim=128,
                            max_epoch=1, batch_size=512, window_size=5, num_workers=4, rw_length=20, rw_walks=1,
                            neg_size=5)
    experiment.run()
    emb = numpy.load('./openhgnn/output/Metapath2vec/dblp4MAGNN_mp2vec_embeddings.npy')
    print(emb)