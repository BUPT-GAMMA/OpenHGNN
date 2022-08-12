from openhgnn.dataset import build_dataset


def datasets_info(datasets: list, task='link_prediction'):
    for dataset_name in datasets:
        dataset = build_dataset(dataset=dataset_name, task=task, logger=None)
        g = dataset.g
        print('Information of {} dataset:'.format(dataset_name))
        print(g)
