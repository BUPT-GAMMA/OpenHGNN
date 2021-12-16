# TrainerFlow

A trainerflow is an abstraction of a predesigned workflow that trains and evaluate a model on a given dataset for a specific use case. It must contain a unique training mechanism involving loss calculation and a specific sampler(sample something used in loss calculation) .

Once we select the model and the task, the func *[get_trainerflow](../start.py)* will help us select the trainerflow. So the customized trainerflow needed be added in this func.

#### Included Object:

- task : [Task](https://quip-amazon.com/jr0EANkmK5mj/OpenHGNN#ULa9CAeWfba)
- model : [Model](../models/#Model) (built through given args.model)
- optimizer : torch.optim.Optimizer
- dataloader(if mini_batch_flag is True) : 
  - torch.utils.data.DataLoader 
  - [dgl.dataloading](https://docs.dgl.ai/en/latest/api/python/dgl.dataloading.html#)

#### Method:

- train()
  - decorated with @abstractmethod, so it must be overridden.
- _full_train_setp()
  - train with a full_batch graph
- _mini_train_step()
  - train with a mini_batch seed nodes graph
- _test_step()
  - evaluate in training/validation/testing

#### Supported trainerflow

- ##### Node classification flow
  
  - Supported Model: HAN/MAGNN/GTN
  - The task: node classification
    - The task.dataset must include the splited[train/valid/test.] mask.
    - The task.dataset will give the value of input dimension to the args.in_dim.
  - The sampler in this flow is supported by [dgl.dataloading](https://docs.dgl.ai/en/latest/api/python/dgl.dataloading.html).
  - The flow is the most common in the GNNs cause most GNNs model are involved in the task semi-supervised node classification. Here the task is to classify the nodes of HIN(Heterogeneous Information Network).
  - Note: we will set the args.out_dim with num_classes if they are not equivalent.
  
- ##### Link prediction
  
  - The same with entity classification except that it is used for link prediction.
  - Supported Model: RGCN/CompGCN/RSHN
  - Supported Task: link prediction
- ##### HetGNN trainerflow
- ##### NSHE trainerflow

### How to build a new trainerflow

1. Create a class *your_trainerflow* that inherits the [*BaseFlow*](./base_flow.py) and register the trainerflow with *@register_flow(str)*.
2. We decorate the func *train()* with @abstractmethod. So the *train()* must be overridden, or the your_trainerflow cannot be instantiated. Besides *train()*, the __*init*__ and *_test_step()* should both be implement. One of the  *_full_train_step()* and *_mini_train_step()* must be implemented at least.
3. Add your_trainerflow into the func *[get_trainerflow](../start.py)*.
4. Fill the dict *SUPPORTED_FLOWS* in [trainerflow/__init__.py](./__init__.py)

