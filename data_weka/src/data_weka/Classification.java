package data_weka;

import javax.swing.JFrame;
import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.gui.treevisualizer.PlaceNode2;
import weka.gui.treevisualizer.TreeVisualizer;
import weka.classifiers.Evaluation;

public class Classification {

    public static void main(String[] args) {
        try {
            // Load datasets
            Instances trainDataset = loadDataset("src/data_weka/train.arff");
            Instances testDataset = loadDataset("src/data_weka/test.arff");
            Instances evalDataset = loadDataset("src/data_weka/evaluate.arff");

            // Set class index
            setClassIndex(trainDataset);
            setClassIndex(testDataset);
            setClassIndex(evalDataset);

            // Evaluate J48
            evaluateWithConfig(
                new J48(),
                trainDataset,
                testDataset,
                evalDataset,
                "J48",
                new String[][]{
                    {"minNumObj=5"}, // Train set config
                    {"minNumObj=5"},  // Test set config
                    {"minNumObj=1"}   // Eval set config
                }
            );

            // Evaluate RandomForest
            evaluateWithConfig(
                new RandomForest(),
                trainDataset,
                testDataset,
                evalDataset,
                "RandomForest",
                new String[][]{
                    {}, // Train set config
                    {},  // Test set config
                    {}   // Eval set config
                }
            );

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // Load a dataset from file
    private static Instances loadDataset(String path) throws Exception {
        DataSource source = new DataSource(path);
        return source.getDataSet();
    }

    // Set class index for a dataset
    private static void setClassIndex(Instances dataset) {
        dataset.setClassIndex(dataset.numAttributes() - 1);
    }

    private static <T extends Classifier> void evaluateWithConfig(
        T classifier,
        Instances trainDataset,
        Instances testDataset,
        Instances evalDataset,
        String classifierName,
        String[][] config
    ) throws Exception {
        System.out.println("\n=== Evaluating " + classifierName + " ===");

        // Train Dataset
        setOptionsFromConfig(classifier, config[0]);
        classifier.buildClassifier(trainDataset);
        
        // Save the trained classifier
        saveClassifier(classifier, classifierName + "_model.bin");

        long startTime = System.currentTimeMillis();
        evaluateAndPrintResults(classifier, trainDataset, trainDataset, config[0], "Train Set");
        long endTime = System.currentTimeMillis();
        System.out.println("Training time: " + (endTime - startTime) + " ms");
        System.out.println();

        // Test Dataset
        startTime = System.currentTimeMillis();
        evaluateAndPrintResults(classifier, trainDataset, testDataset, config[1], "Test Set");
        endTime = System.currentTimeMillis();
        System.out.println("Testing time: " + (endTime - startTime) + " ms");
        System.out.println();

        // Evaluation Dataset
        setOptionsFromConfig(classifier, config[2]);
        startTime = System.currentTimeMillis();
        Evaluation eval = new Evaluation(evalDataset);
        eval.crossValidateModel(classifier, evalDataset, 10, new java.util.Random(1));
        System.out.println(eval.toSummaryString("Evaluation results:\n", false));
        System.out.println("Confusion Matrix: " + eval.toMatrixString());
        System.out.println("Detailed Accuracy By Class: " + eval.toClassDetailsString());
        visualizeTreeIfPossible(classifier, evalDataset, "Visualize Evaluation Set");
        endTime = System.currentTimeMillis();
        System.out.println("Evaluation time: " + (endTime - startTime) + " ms");
    }

    private static void setOptionsFromConfig(Classifier classifier, String[] config) throws Exception {
        String minNumObj = "2"; // Default value

        // Parse configuration
        for (String param : config) {
            String[] keyValue = param.split("=");
            if (keyValue[0].equals("minNumObj")) {
                minNumObj = keyValue[1];
            }
        }

        // Set options for the classifier
        if (classifier instanceof J48) {
            ((J48) classifier).setOptions(new String[]{"-M", minNumObj});
        }
    }

    private static <T extends Classifier> void evaluateAndPrintResults(
        T classifier,
        Instances trainDataset,
        Instances evalDataset,
        String[] config,
        String datasetType
    ) throws Exception {
        Evaluation evaluation = new Evaluation(trainDataset);
        evaluation.evaluateModel(classifier, evalDataset);
        System.out.println(datasetType + " Evaluation (Config=" + String.join(", ", config) + "):");
        
        System.out.println(evaluation.toSummaryString());
        System.out.println("Confusion Matrix: " + evaluation.toMatrixString());
        System.out.println("Detailed Accuracy By Class: " + evaluation.toClassDetailsString());
        
        if (classifier instanceof J48) {
            visualizeTreeIfPossible(classifier, evalDataset, "Visualize " + datasetType);
        }
    }

    private static void visualizeTreeIfPossible(Classifier classifier, Instances dataset, String title) {
        if (classifier instanceof J48) {
            try {
                TreeVisualizer visualizer = new TreeVisualizer(
                    null,
                    ((J48) classifier).graph(),
                    new PlaceNode2()
                );
                JFrame frame = new JFrame(title);
                frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
                frame.setSize(800, 600);
                frame.getContentPane().add(visualizer);
                frame.setVisible(true);
                visualizer.fitToScreen();
            } catch (Exception e) {
                System.err.println("Tree visualization is not supported for this classifier.");
            }
        }
    }

    private static void saveClassifier(Classifier classifier, String fileName) throws Exception {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(fileName))) {
            oos.writeObject(classifier);
            System.out.println("Saved classifier to " + fileName);
        }
    }
}