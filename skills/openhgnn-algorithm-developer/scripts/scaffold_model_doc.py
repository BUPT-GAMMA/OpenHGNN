#!/usr/bin/env python3
"""Create a standard OpenHGNN model reproduction page."""

from __future__ import annotations

import argparse
from pathlib import Path


TEMPLATE = """{title}
{underline}

* Paper: <paper title>, {venue}
* Registered model name: ``{model}``
* Task: ``{task}``

Reproduction
------------

.. list-table::
   :header-rows: 1

   * - Field
     - Value
   * - Data source
     - ``{dataset}``; replace this with the upstream source or OpenHGNN mirror.
   * - Preprocessing
     - TODO: describe how raw data becomes DGL/OpenHGNN inputs.
   * - Command
     - ``python main.py -m {model} -d {dataset} -t {task} -g 0 --use_best_config``
   * - Metric
     - TODO: metric name.
   * - Expected result
     - TODO: validated result, mean/std if available.
   * - Hardware/runtime
     - TODO: GPU/CPU, memory, runtime.
   * - Seed
     - ``seed = 0``

Implementation Notes
--------------------

TODO: describe DGL operators, optional dependencies, and known limitations.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--venue", default="<venue year>")
    parser.add_argument("--task", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    title = args.model
    text = TEMPLATE.format(
        title=title,
        underline="=" * len(title),
        model=args.model,
        venue=args.venue,
        task=args.task,
        dataset=args.dataset,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
