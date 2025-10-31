.. _api_reference:

=============
API Reference
=============

This section provides the API documentation for the ``TiRank`` package.

Core Model & Training
---------------------
This is the core logic for the TiRank multi-task learning model, loss functions, and training/prediction pipeline.

.. autosummary::
   :toctree: generated/
   
   tirank.Model
   tirank.TrainPre
   tirank.Loss

Data Handling & Preprocessing
-----------------------------
Functions and classes for loading, filtering, and preprocessing data before it enters the model.

.. autosummary::
   :toctree: generated/

   tirank.LoadData
   tirank.SCSTpreprocess
   tirank.Dataloader

Feature Engineering
-------------------
The core modules for Relative Expression Ordering (REO) and H&E image feature extraction.

.. autosummary::
   :toctree: generated/

   tirank.GPextractor
   tirank.Imageprocessing

Plotting & Visualization
------------------------
Functions for visualizing TiRank results.

.. autosummary::
   :toctree: generated/
   
   tirank.Visualization