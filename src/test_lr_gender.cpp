//#include "hmc.hpp"
#include "utils/c_utils.hpp"

#include <iostream>
#include <stdlib.h>
#include <cfloat>
#include <cmath>
#include <vector>
#include <Eigen/Sparse>
#include <chrono>
#include <ctime>
#include <random>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include "likelihood/CPU_logistic_regression.hpp"

using namespace Eigen;
using namespace std;

int main(int argc, char const *argv[])
{

	C_utils utils;
		
	string data_csv_path, labels_csv_path;
	MatrixXd data;
	VectorXd labels;

	int train_partition = 14000;
	data_csv_path = "../data/ADIENCE/lbp/dataset.csv";
	labels_csv_path = "../data/ADIENCE/lbp/gender_label.csv";

	cout << "Read Data" << endl;

	utils.read_Data(data_csv_path,data);
	utils.read_Labels(labels_csv_path,labels);

	cout << "Data Partition" << endl;
	MatrixXd X_train, X_test;
	VectorXd Y_train, Y_test;
	utils.dataPartition(data, labels, X_train, X_test, Y_train, Y_test, train_partition);

	VectorXd predicted_labels;
	
	double lambda = 0.001;
	int epochs = 10;
	double alpha = 0.90;
	double step_size = 0.1;
	int mini_batch=100;

	CPU_LogisticRegression lr;
	lr.init(X_train, Y_train, lambda, false, false,true);
	int num_batches=X_train.rows()/mini_batch;
	cout << "Lambda: " << lambda <<  ", Iterations: " << num_batches*epochs << ", Alpha: " << alpha << ", Step Size:" << step_size << endl;

	auto start = chrono::high_resolution_clock::now();
	lr.train(num_batches*epochs,mini_batch, alpha, step_size);
	auto finish = chrono::high_resolution_clock::now();
	chrono::duration<double> elapsed = finish - start;
	cout << "Elapsed Train time: " << elapsed.count() << " s\n";
	
	predicted_labels = lr.predict(X_test, false);

	cout << "Init report" << endl;
	utils.report(Y_test, predicted_labels, true);
	utils.confusion_matrix(Y_test, predicted_labels, true);
	utils.calculateAccuracyPercent(Y_test, predicted_labels);
	return 0;
}
