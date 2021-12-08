# tesseraeFinder
Finds if pre-computed paths intersect with pre-identified tesserae regions on Venus. The algorithm is also applicable to any path-crosses-box problem.

tesseraeFinder exists to primarially visualize how a balloon path intersects with tesserae regions and has 3 pre-digitized balloon paths from Lebonnois 2017.

tesseraeStatz exists to primarially calculate the statistics of multiple balloon path variations crossing over tesserae regions. It uses data from Lebonnois 2017 or newer versions and can be procured through contacting Lebonnois.

Both codes have parameters that can be adjusted in the codes themselves such as days per circumnavigation, sample rate, and landing coordinates to name a few.

To run either codes you need a Python instance. Anaconda is usually the easiest way to get Python and all of the packages needed by this code and the included Spyder IDE can be used to run this code.

The optional function subfun_figFitter can be downloaded from https://github.com/dinsmoro/GRITI/blob/main/subfun_figFitter.py and will plot the figure better. It isn't required to run the code and isn't included here since it is GPL-3.0 licensed.

The optional map of Venus (named tesseraeMap.png) can be made from the map of Venus and its tesserae regions in Gilmore et al., 2020 https://science.nasa.gov/science-red/s3fs-public/atoms/files/Venus%20Flagship%20Mission.pdf.

More tesserae region quadrangles can be added, but they must be convex trapezium (Brit. Eng. calls this an irregular quadrilateral) with no interior angle greater than 180 degrees and their 4 corner points must be added along with a major tesserae region designation number (0 represents no major tesseae region).

Coded to support a Venus mission design for the 2021 NASA-JPL Planetary Science Summer School (PSSS).
