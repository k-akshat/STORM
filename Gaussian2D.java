import org.apache.commons.math3.fitting.leastsquares.LeastSquaresBuilder;
import org.apache.commons.math3.fitting.leastsquares.LeastSquaresOptimizer;
import org.apache.commons.math3.fitting.leastsquares.LeastSquaresProblem;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.fitting.leastsquares.LevenbergMarquardtOptimizer;
import org.apache.commons.math3.analysis.MultivariateVectorFunction;
import org.apache.commons.math3.analysis.MultivariateMatrixFunction;
import java.awt.image.*;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import javax.imageio.ImageIO;

public class Gaussian2D {

    public static void main(String[] args) throws IOException {

        BufferedImage img = ImageIO.read(new File("./img/1.tif")); //Read the image file
        double[][] gaussianMat = getMatrixOfImage(img); // Create pixel matrix from image
        double[][] normalizedData = normalizeData(gaussianMat); // Normalize the pixel Matrix
        double estimatedMean = estimateMean(normalizedData); // Calculate the mean of the data
        int [][] peakLocations = findPeaks(normalizedData, estimatedMean); // Find all the peaks in the normalized pixel matrix
        double[][] fittedPeaks = new double[peakLocations.length][2]; // To store the fitted coordinates of all the peaks

        // Iterating through each peak to find its fitted location
        for (int i = 0; i < peakLocations.length; i++) {
            // Seperating a 3x3 matrix around each peak to do the fitting
            int size = 3;
            double[][] data = new double[size][size];
            for(int k = 0; k < size; k++){
                for(int j = 0; j < size; j++){
                    data[k][j] = normalizedData[peakLocations[i][0] - size/2 + k][peakLocations[i][1] - size/2 + j];
                }
            }
            // Create a matrix that contains the {x, y, observed value at (x, y))
            double[][] valueMat = makeGaussian(data);

            // Intial Guess for the fitting
            double[] initialGuess = {1.0, 1.0, 1.0, 1.0, 1.0};

            // Define the least square problem
            LeastSquaresProblem problem = new LeastSquaresBuilder()
                    .start(initialGuess)
                    .model(modelFunction(valueMat), modelJacobian(valueMat))
                    .target(target(valueMat))
                    .lazyEvaluation(false)
                    .maxEvaluations(1000)
                    .maxIterations(1000)
                    .build();

            // Solve the least squares problem using Levenberg-Marquardt optimizer
            LeastSquaresOptimizer optimizer = new LevenbergMarquardtOptimizer();
            LeastSquaresOptimizer.Optimum optimum = optimizer.optimize(problem);
            RealVector fittedParameters = optimum.getPoint();
            double x_cord = fittedParameters.getEntry(1) - 2 + peakLocations[i][0];
            double y_cord = fittedParameters.getEntry(2) - 2 + peakLocations[i][1];

            //Round off the coordinates of peaks till three decimal digits
            fittedPeaks[i][0] = (double)Math.round(x_cord*1000)/1000;
            fittedPeaks[i][1] = (double)Math.round(y_cord*1000)/1000;
            System.out.println("Peak Detected at: " + peakLocations[i][0] + "," + peakLocations[i][1]);
            System.out.println("Fitted Parameters: "+ fittedParameters);
        }
        // Write the fitted peaks coordinate to a txt file
        writeCSB("output.txt", fittedPeaks);
        System.out.println("Run the Image Plot file to get the Plot of Fitted Coordinates");
    }

    // Function to construct a target vector
    private static RealVector target(double [][] data) {
        int matLen = data.length;
        double[] target = new double[matLen];
        for(int k = 0; k < matLen; k++){
            target[k] = data[k][2];
        }
        return new ArrayRealVector(target, false);
    }

    // Function to define the model function
    private static MultivariateVectorFunction modelFunction(double [][] data) {
        return new MultivariateVectorFunction() {
            @Override
            public double[] value(double[] parameters) {
                int matLen = data.length;
                double[] values = new double[matLen];
                for (int i = 0; i < matLen; i++) {
                    double x = data[i][0];
                    double y = data[i][1];
                    values[i] = gaussian(x, y, parameters);
                }
                return values;
            }
        };
    }

    // Function to define the Jacobian Matrix
    private static MultivariateMatrixFunction modelJacobian(double [][] data) {
        return new MultivariateMatrixFunction() {
            @Override
            public double[][] value(double[] parameters) {
                int matLen = data.length;
                double[][] jacobian = new double[matLen][5];
                for (int i = 0; i < matLen; i++) {
                    double x = data[i][0];
                    double y = data[i][1];
                    double[][] partialDerivatives = jacobian(x, y, parameters);
                    jacobian[i] = partialDerivatives[0];
                }
                return jacobian;
            }
        };
    }

    // Function to determine the gaussian function value
    private static double gaussian(double x, double y, double[] p) {
        double A = p[0];
        double x0 = p[1];
        double y0 = p[2];
        double sigmaX = p[3];
        double sigmaY = p[4];
        return A*Math.exp(-((x - x0) * (x - x0) / (2 * sigmaX * sigmaX) + (y - y0) * (y - y0) / (2 * sigmaY * sigmaY)));
    }

    // Function to determine the value of jacobian
    private static double[][] jacobian(double x, double y, double[] p) {
        double A = p[0];
        double x0 = p[1];
        double y0 = p[2];
        double sigmaX = p[3];
        double sigmaY = p[4];
        double fValue = gaussian(x, y, p);

        double dFdA = fValue / A;
        double dFdx0 = A * (x - x0) / (sigmaX * sigmaX) * fValue;
        double dFdy0 = A * (y - y0) / (sigmaY * sigmaY) * fValue;
        double dFdsigmaX = A * (x - x0) * (x - x0) / (sigmaX * sigmaX * sigmaX) * fValue;
        double dFdsigmaY = A * (y - y0) * (y - y0) / (sigmaY * sigmaY * sigmaY) * fValue;

        return new double[][]{{dFdA, dFdx0, dFdy0, dFdsigmaX, dFdsigmaY}};
    }

    // Function to create a matrix that contains the {x, y, observed value at (x, y))
    private static double[][] makeGaussian(double[][] Mat) {
        int matLen = Mat.length;
        double[][] newGaussianMat = new double[matLen*matLen][3];
        int k = 0;
        for(int i = 0; i < matLen; i++){
            for(int j = 0; j < matLen ; j++, k++){
                newGaussianMat[k][0] = i;
                newGaussianMat[k][1] = j;
                newGaussianMat[k][2] = Mat[i][j];
            }
        }
        return newGaussianMat;
    }

    // Function to normalize the matrix
    public static double[][] normalizeData(double[][] matrix) {
        double maxValue = Arrays.stream(matrix)
                .flatMapToDouble(Arrays::stream) // Flatten the 2D array to a stream of doubles
                .max() // Find the maximum value
                .orElse(Double.NaN); // Handle the case where the matrix is empty

        int rows = matrix.length;
        int cols = matrix[0].length;

        double[][] normalizedData = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                normalizedData[i][j] = (matrix[i][j]) / maxValue;
            }
        }

        return normalizedData;
    }

    // Function to find the mean value of given data
    public static double estimateMean(double[][] matrix) {
        double sum = 0.0;
        int count = 0;

        for (double[] row : matrix) {
            for (double value : row) {
                sum += value;
                count++;
            }
        }

        return sum / count;
    }

    // Function to create pixel matrix from a image
    public static double[][] getMatrixOfImage(BufferedImage bufferedImage) {
        int width = bufferedImage.getWidth(null);
        int height = bufferedImage.getHeight(null);
        double[][] pixels = new double[width][height];
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                int pixelValue = bufferedImage.getRGB(i, j);
                pixels[i][j] = (pixelValue >> 16) & 0xff;
            }
        }

        return pixels;
    }

    // Function to determine the peaks in a matrix
    public static int[][] findPeaks(double[][] matrix, double estMean) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        int[][] peaks = new int[rows * cols][2]; // Assuming maximum all elements are peaks (worst case)
        int peakCount = 0;

        for (int i = 2; i < rows - 2; i++) {
            for (int j = 2; j < cols - 2; j++) {
                if (isPeak(matrix, i, j) && matrix[i][j] > 0.40) {
                    peaks[peakCount][0] = i;
                    peaks[peakCount][1] = j;
                    peakCount++;
                }
            }
        }

        int[][] actualPeaks = new int[peakCount][2];
        System.arraycopy(peaks, 0, actualPeaks, 0, peakCount);

        return actualPeaks;
    }

    // Helper function for determining the peaks
    private static boolean isPeak(double[][] matrix, int row, int col) {
        double value = matrix[row][col];
        return value > matrix[row - 1][col] && value > matrix[row + 1][col] &&
                value > matrix[row][col - 1] && value > matrix[row][col + 1] && value > matrix[row - 1][col + 1] && value > matrix[row - 1][col - 1] && value > matrix[row + 1][col + 1] && value > matrix[row + 1][col - 1];
    }

    // Function to write the coordinates of fitted peaks in a txt file
    public static void writeCSB(String filename, double[][] peaks) throws IOException {
        FileWriter writer = new FileWriter(filename);

        // Write Peaks section
        for (double[] peak : peaks) {
            writer.write(peak[0] + "," + peak[1] + "\n");
        }
        writer.close();
    }
}

