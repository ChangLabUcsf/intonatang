Invariance analysis
===================

To determine whether the pattern of neural activity to different intonation contours is the same across 
different data sets (e.g. speech data vs. nonspeech control data), we used linear discriminant analysis (LDA)
to fit models to predict intonation contour (i.e. the intonation/sentence_type condition between 1-4) from 
the neural activity time series from each electrode. These models were fit on one data set and tested on another, 
and the performance accuracies of the model on different test sets were compared.

.. automodule:: intonatang.intonation_invariance
   :members:
