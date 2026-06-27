AI Assistant Skills
===================

OpenHGNN ships optional Codex skills under the repository ``skills/``
directory. These skills are not required to use OpenHGNN, but they help users
and contributors follow the same model, trainerflow, dataset, task, docs, and
test conventions.

Available Skills
----------------

.. list-table::
   :header-rows: 1

   * - Skill
     - Purpose
   * - ``openhgnn-algorithm-developer``
     - Helps users run existing models, reproduce results, debug
       ``Experiment`` failures, and integrate new algorithms into OpenHGNN.

Install Locally
---------------

After cloning OpenHGNN, copy the skill into your Codex skills directory:

.. code:: bash

   mkdir -p ~/.codex/skills
   cp -r skills/openhgnn-algorithm-developer ~/.codex/skills/

Then start a new Codex session and ask with the skill name, for example:

.. code:: text

   Use $openhgnn-algorithm-developer to help me run SEHTGNN on node_regression.
   Use $openhgnn-algorithm-developer to integrate a new heterogeneous graph model.

What It Checks
--------------

The skill guides the assistant to inspect registry state, choose the right
task, follow DGL-first implementation rules, write reproduction documentation,
and run validation commands such as:

.. code:: bash

   openhgnn validate-registry --format json
   python -m sphinx -W -b html docs/source /tmp/openhgnn-docs-build
   git diff --check
