Gastric Cancer Risk Calculator
==============================

Educational framework for gastric cancer risk stratification and survival modeling.

.. warning::

   This is an educational tool only. Not intended for clinical decision-making.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api/index
   contributing

Installation
------------

.. code-block:: bash

   pip install gastric-cancer-risk-calculator

For development:

.. code-block:: bash

   pip install -e ".[dev]"

Quick Start
-----------

.. code-block:: python

   from risk_calculator import calculate_risk

   # Calculate risk for a patient
   result = calculate_risk(
       age=65,
       tumor_stage="T3",
       lymph_node_status="N1",
       # ... other parameters
   )
   print(f"Risk score: {result['risk_score']}")

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
