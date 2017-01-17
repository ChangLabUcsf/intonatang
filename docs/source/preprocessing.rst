.. _preprocessing:

Preprocessing
=============

Preprocessing for the intonation project consists of extracting the high-gamma analytic amplitude over the entire recording block and manually marking bad channels and bad time segments.

1. Inspect raw ECoG for artifacts and large spikes. Mark bad time segments. 
2. Find bad channels (channels with no activity or continuous spiking) and save to a MATLAB variable ``bcs`` (use 0 indexing, because analysis then goes to Python)
3. Output Hilbert (Chang lab recipe with 8 bands)
4. Get ``ECxx_Bxxx_hg_100Hz`` by taking the average of 8 bands and downsampling to 100Hz
5. Get ``ECxx_Bxxx_log_hg_100Hz`` by first taking the log of the each of the 8 bands, averaging, and then downsampling.
6. Save ``ECxx_Bxxx.mat`` file with:
    * ``ECxx_Bxxx_hg_100Hz``
    * ``ECxx_Bxxx_log_hg_100Hz``
    * ``bcs``
    * ``badTimeSegments`` (loaded from GUI-generated Artifacts/badTimeSegments.mat)
