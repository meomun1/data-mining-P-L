package data_weka;

import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomTree;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Evaluation;

import javax.swing.*;
import weka.gui.treevisualizer.PlaceNode2;
import weka.gui.treevisualizer.TreeVisualizer;

public class Test {

    public static void main(String[] args) {
        try {
            // Load train | test | evaluation datasets
            Instances trainDataset = loadDataset("src/data_weka/train.arff");
            Instances testDataset = loadDataset("src/data_weka/test.arff");
            Instances evalDataset = loadDataset("src/data_weka/evaluate.arff");
            
            
            // Set Class Index 
            trainDataset.setClassIndex(trainDataset.numAttributes() - 1);
            testDataset.setClassIndex(testDataset.numAttributes() -1 );
            evalDataset.setClassIndex(evalDataset.numAttributes() -1 );
            
            // Supplied test set 
            J48 j48 = new J48();
            j48.setOptions(new String[]{"-M", "5"}); // Set minNumObj = 10
            j48.buildClassifier(trainDataset);
            Evaluation evalTest = new Evaluation(trainDataset);
            evalTest.evaluateModel(j48, testDataset);
            System.out.println(evalTest.toSummaryString("Evaluation test results:\n",false));
            visualizeTree(j48, testDataset, "Visualize test set for J48");
            
            // Use training set
            j48 = new J48();
            j48.setOptions(new String[]{"-M", "5"}); // Set minNumObj = 10
            j48.buildClassifier(trainDataset);
            Evaluation evalTrain = new Evaluation(trainDataset);
            evalTrain.evaluateModel(j48, trainDataset);
            System.out.println(evalTrain.toSummaryString("Evaluation train results:\n",false));
            visualizeTree(j48, trainDataset, "Visualize train set for J48");
            
            
            j48 = new J48();
            j48.setOptions(new String[]{"-M", "1"}); // Set minNumObj = 10
            Evaluation eval = new Evaluation(evalDataset);
            eval.crossValidateModel(j48, evalDataset, 10, new java.util.Random(1));
            System.out.println(eval.toSummaryString("Evaluation train results:\n",false));
            visualizeTree(j48, evalDataset, "Visualize evaluation set for J48");


        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static Instances loadDataset(String path) throws Exception {
        DataSource source = new DataSource(path);
        return source.getDataSet();
    }

    private static void evaluateClassifier(Classifier classifier, Instances dataset) throws Exception {
        Evaluation evaluation = new Evaluation(dataset);
        evaluation.evaluateModel(classifier, dataset);

        System.out.println("Classifier: " + classifier.getClass().getSimpleName());
        System.out.println("Summary: " + evaluation.toSummaryString());
        System.out.println("Confusion Matrix: " + evaluation.toMatrixString());
        System.out.println("Detailed Accuracy By Class: " + evaluation.toClassDetailsString());
    }

    private static void visualizeTree(Classifier tree, Instances dataset, String title) throws Exception {
        TreeVisualizer visualizer = new TreeVisualizer(null, ((weka.classifiers.trees.J48) tree).graph(), new PlaceNode2());
        JFrame frame = new JFrame(title);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(800, 600);
        frame.getContentPane().add(visualizer);
        frame.setVisible(true);
        visualizer.fitToScreen();
    }
}