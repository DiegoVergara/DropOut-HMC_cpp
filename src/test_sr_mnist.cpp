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
#include "likelihood/CPU_softmax_regression.hpp"

using namespace Eigen;
using namespace std;

int main(int argc, char const *argv[])
{
	C_utils utils;
		
	MatrixXd X_train;
	MatrixXd X_test;
	VectorXd Y_train;
	VectorXd Y_test;
	utils.read_Data("../data/MNIST/X_train.csv",X_train);
	utils.read_Data("../data/MNIST/X_test.csv",X_test);
	utils.read_Labels("../data/MNIST/Y_train.csv",Y_train);
	utils.read_Labels("../data/MNIST/Y_test.csv",Y_test);

	//RowVectorXd mean = X_train.colwise().mean();
	//X_train = X_train.rowwise() - mean;
	//X_test =X_test.rowwise() - mean;
	
	cout << "Init" << endl;
	VectorXd predicted_labels;
	
	double reg = 0.01;
	int epochs = 100;
	double momentum = 0.99;
	double learning_rate = 0.0001;
	int mini_batch=100;
	int num_batches=X_train.rows()/mini_batch;
	int iterations = epochs*num_batches;

	cout << "Regularization: " << reg <<  ", Iterations: " << iterations << ", Epochs: " << epochs << ", Mini Batch: " << mini_batch <<", Momentum: " << momentum << ", Learning Rate:" << learning_rate << endl;
	
	CPU_SoftmaxRegression lr;
	lr.init(X_train, Y_train, reg, false, false, true);

	cout << "Init train" << endl;

	auto start = chrono::high_resolution_clock::now();
	lr.train(iterations, mini_batch, momentum, learning_rate);
	auto finish = chrono::high_resolution_clock::now();
	chrono::duration<double> elapsed = finish - start;
	cout << "Elapsed Train time: " << elapsed.count() << " s\n";

	cout << "Init predict" << endl;
	predicted_labels = lr.predict(X_test, false, true);

	cout << "Init report" << endl;
	utils.report(Y_test, predicted_labels, true);
	utils.confusion_matrix(Y_test, predicted_labels, true);
	utils.calculateAccuracyPercent(Y_test, predicted_labels);
	return 0;
}
