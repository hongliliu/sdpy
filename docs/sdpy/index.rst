************
sdpy package
************

sdpy stands for Single Dish OTF python mapping and contains tools for OTF
mapping and gridding for single-dish data.

Python codes for Arecibo, GBT, APEX, and TripleSpec data cubes.  Forked from
casaradio: http://code.google.com/p/casaradio

Renamed from "gbtpy" to "sdpy" to indicate that it isn't just for GBT but
for
all single dish heterodyne or other cubemaking machines.


What's inside?
--------------

Tools for:

 * Generation of artificial off positions by averaging all positions in OTF
   maps
 * calibration of position-switched data using those off positions
 * Gridding OTF data using a user-specified kernel

Requirements
------------

 * astropy
 * matplotlib

Reference/API
=============

.. automodapi:: sdpy
