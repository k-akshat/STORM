import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import javax.swing.*;
import java.awt.*;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class ImagePlot extends JFrame {

    public ImagePlot(String title) {
        super(title);

        // Create the dataset
        XYDataset dataset = createDataset();

        // Create the chart
        JFreeChart chart = ChartFactory.createScatterPlot(
                "Scatter Plot",          // chart title
                "X",                     // x-axis label
                "Y",                     // y-axis label
                dataset);                // data

        // Create a panel to display the chart
        ChartPanel panel = new ChartPanel(chart);
        panel.setPreferredSize(new Dimension(800, 600));
        setContentPane(panel);
    }

    private XYDataset createDataset() {
        XYSeries series = new XYSeries("Peaks");

        // Read data from the file and add it to the series
        try (BufferedReader br = new BufferedReader(new FileReader("output.txt"))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] parts = line.split(",");
                double x = Double.parseDouble(parts[0]);
                double y = Double.parseDouble(parts[1]);
                series.add(x, y);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        return new XYSeriesCollection(series);
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            ImagePlot example = new ImagePlot("JFreeChart Example");
            example.setSize(800, 600);
            example.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
            example.setLocationRelativeTo(null);
            example.setVisible(true);
        });
    }
}
