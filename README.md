# Peak Detection and Gaussian Fitting Program to pinpoint the location of a molecule

## Overview
This Java program detects peaks from an input image, fits Gaussian distributions to these peaks, and reconstructs the image based on the fitted Gaussian parameters. It utilizes the Apache Commons Math library for least squares optimization and JFreeChart for visualization.

## Features
- **Peak Detection**: Utilizes algorithms to identify peaks or local maxima within the input image.
- **Gaussian Fitting**: Fits Gaussian distributions to detected peaks to model their spatial intensity profiles.
- **Image Reconstruction**: Reconstructs the image by combining the fitted Gaussian distributions.
- **Visualization**: Generates scatter plots using JFreeChart to display the fitted peak coordinates.
- **Customization**: Parameters such as peak detection thresholds and Gaussian fitting parameters can be adjusted to optimize performance.

## Usage
1. **Input Image**: Provide the input image path to the program.
2. **Run Program**: Compile and execute the `Gaussian2D.java` file.
3. **View Results**: The program generates an output file (`output.txt`) containing the coordinates of the fitted peaks. Use the `ImagePlot.java` file to visualize the fitted peaks as a scatter plot.

## Dependencies
- Apache Commons Math 3.x
- JFreeChart

## Installation
1. Download and install the Apache Commons Math library (https://commons.apache.org/proper/commons-math/).
2. Download and install the JFreeChart library (https://www.jfree.org/jfreechart/).

## Acknowledgements
**Akshat Kumar** developed this program under **Bosanta Ranjan Boruah** (HOD Physics Department, IIT Guwahati). Thanks to the Apache Commons Math and JFreeChart development teams for their contributions.

For questions or inquiries, please contact akshatkumar1712@gmail.com.

