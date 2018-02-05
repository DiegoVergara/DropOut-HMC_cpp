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
		
	string train_csv_path, test_csv_path;
	MatrixXd bird;

  	train_csv_path = "../data/BIRDS/CUB200features.txt";

	utils.read_Data(train_csv_path,bird);
	
	MatrixXd X = bird.block(0,1,bird.rows(),bird.cols()-1);
	VectorXd Y = bird.block(0,0,bird.rows(),1);
	
	MatrixXd X_train, X_test;
	VectorXd Y_train, Y_test;
	utils.dataPermutation(X, Y);
	utils.dataPartition(X, Y, X_train, X_test, Y_train, Y_test, 5000);

/*	utils.writeToCSVfile("X_train.csv", X_train);
	utils.writeToCSVfile("X_test.csv", X_test);
	utils.writeToCSVfile("Y_train.csv", Y_train);
	utils.writeToCSVfile("Y_test.csv", Y_test);
*/

	cout << "Init" << endl;
	VectorXd predicted_labels;
	
	double reg = 1e-1;
	int epochs = 100;
	double momentum = 0.99;
	double learning_rate = 1e-1;
	int mini_batch=1000;
	int num_batches=X_train.rows()/mini_batch;

	cout << "Regularization: " << reg <<  ", Iterations: " << num_batches*epochs << ", momentum: " << momentum << ", Learning Rate:" << learning_rate << endl;
	
	CPU_SoftmaxRegression lr;
	lr.init(X_train, Y_train, reg, true, false, false);

	cout << "Init train" << endl;
	auto start = chrono::high_resolution_clock::now();
	lr.train(num_batches*epochs,mini_batch,, momentum, learning_rate);
	auto finish = chrono::high_resolution_clock::now();
	chrono::duration<double> elapsed = finish - start;
	cout << "Elapsed Train time: " << elapsed.count() << " s\n";

	cout << "Init predict" << endl;
	predicted_labels = lr.predict(X_test, true, false);

	cout << "Init report" << endl;
	utils.report(Y_test, predicted_labels, true);
	utils.confusion_matrix(Y_test, predicted_labels, true);
	utils.calculateAccuracyPercent(Y_test, predicted_labels);
	
	return 0;
}
