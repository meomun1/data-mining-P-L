
## Prerequisites

- Java Development Kit (JDK) 8 or higher
- Weka library (included in the project)

## Setup

1. Clone the repository:
    ```sh
    git clone https://github.com/meomun1/data-mining-P-L.git
    cd data_weka
    ```

2. Open the project in your preferred IDE (e.g., Eclipse, IntelliJ IDEA).

3. Ensure that the Weka library is included in the project's build path. The `.classpath` file already includes the path to the Weka library.

## Running the Code

### Classification

The `Classification` class is used to train and evaluate classifiers.

1. Open `src/data_weka/Classification.java`.

2. Run the `main` method. This will:
    - Load the training, testing, and evaluation datasets.
    - Train a J48 classifier and a RandomForest classifier.
    - Evaluate the classifiers on the training, testing, and evaluation datasets.
    - Print the evaluation results to the console.
    - Save the trained classifiers to binary files.

## Datasets

The datasets are located in the `src/data_weka/` directory:
- `train.arff`: Training dataset
- `test.arff`: Testing dataset
- `evaluate.arff`: Evaluation dataset

## Configuration

You can configure the classifiers by modifying the options in the `Classification` class. For example, you can change the `minNumObj` parameter for the J48 classifier.

## License

This project is not licensed 