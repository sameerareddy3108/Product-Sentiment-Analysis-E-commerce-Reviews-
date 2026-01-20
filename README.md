# Sentiment Analysis on Amazon Reviews 2023 Dataset

## Project Overview
This project aims to develop a machine learning-based sentiment analysis system tailored for the Amazon Reviews 2023 dataset. The dataset, collected by McAuley Lab in 2023, includes customer reviews across various products, capturing diverse opinions and sentiments. Specifically, this project focuses on the All_Beauty category, which contains 633 thousand user reviews of beauty products in JSON format.

## Dataset
The dataset used for this project is the Amazon Reviews 2023 dataset (Hou, et al., 2024). It was collected to pretrain BLAIR (Bridging Language and Items for Retrieval and Recommendation) for sentence embedding models specifically for recommendation scenarios. The dataset is essential for understanding customer behavior and enabling businesses to refine their strategies and improve digital products. You can access the dataset from [here](https://amazon-reviews-2023.github.io/).

## Proposed Solution
The proposed solution involves developing a machine learning-based sentiment analysis system. The following algorithms are considered to classify sentiments while addressing the challenges of processing large-scale textual data effectively:
- Naive Bayes
- Support Vector Machines (SVM)
- Logistic Regression
- K-Nearest Neighbors (KNN)

## Installation and Setup
1. Clone the repository:
   ```sh
   git clone https://github.com/3m6d/SentimentAnalysisAmazonDataset.git
   cd SentimentAnalysisAmazonDataset
   ```
2. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
1. Prepare the dataset:
   - Place the JSON dataset file in the `data/` directory.
2. Run the Jupyter Notebook:
   ```sh
   jupyter notebook AmazonDatasetSentimentAnalysis.ipynb
   ```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## License
This project is licensed under the MIT License.

## Acknowledgements
- The dataset was collected by McAuley Lab.
- Special thanks to the researchers who listed the dataset in various categories on the Amazon Review 2023 website.
