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
	MatrixXd cancer;

  	data_csv_path = "../data/CANCER/breast_cancer.csv";

	utils.read_Data(data_csv_path,cancer);
	MatrixXd data = cancer.block(0,0,cancer.rows(),cancer.cols()-1);
	VectorXd labels = cancer.block(0,cancer.cols()-1,cancer.rows(),1);
	MatrixXd X_train, X_test;
	VectorXd Y_train, Y_test;
	utils.dataPermutation(data, labels);
	utils.dataPartition(data, labels, X_train, X_test, Y_train, Y_test, 400);
	
	cout << "Init" << endl;
	VectorXd predicted_labels;
	double lambda = 0.001;
	int epochs = 10;
	double alpha = 0.90;
	double step_size = 0.1;
	int mini_batch=100;

	CPU_LogisticRegression lr;
	lr.init(X_train, Y_train, lambda, true, true,true);
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
