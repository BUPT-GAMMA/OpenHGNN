HCMGNN
======

* Paper: Heterogeneous Causal Metapath Graph Neural Network for
  Gene-Microbe-Disease Association Prediction, IJCAI 2024
* Registered model name: ``HCMGNN``
* Task: ``recommendation``

Reproduction
------------

.. list-table::
   :header-rows: 1

   * - Field
     - Value
   * - Data source
     - ``GMD4HCMGNN`` processed from the upstream HCMGNN dataset and stored in
       the OpenHGNN S3 dataset mirror.
   * - Preprocessing
     - Runs cross-validation and independent testing over gene-microbe-disease
       association data.
   * - Command
     - ``python main.py -m HCMGNN -d GMD4HCMGNN -t recommendation -g 0 --use_best_config``
   * - Metrics
     - Hits@K, NDCG@K, and MRR.
   * - Expected result
     - HCMGNN-CV: Hits@5 0.9647, Hits@3 0.9381, Hits@1 0.7791, NDCG@5
       0.8861, NDCG@3 0.8752, NDCG@1 0.7791, MRR 0.8629. HCMGNN-Indep:
       Hits@5 0.9650, Hits@3 0.9534, Hits@1 0.8222, NDCG@5 0.9064,
       NDCG@3 0.9015, NDCG@1 0.8222, MRR 0.8899.
   * - Hardware/runtime
     - Reported on GeForce RTX 2080. The run performs approximately six
       train/evaluation passes due to cross-validation plus independent test.
   * - Seed
     - ``seed = 0`` in ``config.ini``.

