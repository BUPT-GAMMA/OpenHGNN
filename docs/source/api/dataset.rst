.. _api-dataset:

Dataset
=======

.. currentmodule:: openhgnn.dataset

.. autosummary::
    :nosignatures:
    {% for cls in openhgnn.dataset.classes %}
      {{ cls }}
    {% endfor %}

.. automodule:: openhgnn.dataset
    :members:
    :exclude-members: BaseDataset, NodeClassificationDataset

    .. autoclass:: BaseDataset
        :special-members:
        :show-inheritance:

    .. _api-base-node-dataset:

    .. autoclass:: NodeClassificationDataset
        :special-members:
        :show-inheritance:
