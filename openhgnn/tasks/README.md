# Task

The task means downstream task. So it is related to dataset closely.

#### Included Object:

- [Dataset](../dataset/#Dataset), containing the heterogeneous graph as DGLGraph, as well as node/edge features and additional dataset-specific information.
- Evaluation Metric, a callable object which takes in the prediction and the ground truth label, and returns an output.
  - Currently we will implement evaluation metrics for node classification and link prediction.
- Training Loss, a callable object which takes in the prediction and the ground truth as well as any other necessary information and returns an output. It must be differentiable.
  - Ranking loss for self-supervision, or contrastive learning losses, all fall into this category.
- Training-validation-test Split.

#### Method:

- get_graph()
  - return the DGLGraph
- get_loss_fn()
  - train with a full_batch graph
- evaluate()
  - Evaluate Metric
- get_labels()
  - return labels

#### Supported task

- ##### Node classification

- ##### Link prediction

  - 

- **Recommendation**

### How to build a new trainerflow

1. Create a class *your_task* that inherits the [*BaseTask*](./base_task.py) and register the task with *@register_task(str)*.
2. We decorate the func *evaluate()* with @abstractmethod. So the *evaluate()* must be overridden, or the your_task cannot be instantiated. 
4. Fill the dict *SUPPORTED_TASKS* in [task/__init__.py](./__init__.py)

