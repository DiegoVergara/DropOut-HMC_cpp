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
		
		string train_csv_path, test_csv_path;
	MatrixXd train, test;

  	train_csv_path = "../data/MADELON/lbp/madelon.csv";
  	test_csv_path = "../data/MADELON/lbp/madelon_t.csv";

	utils.read_Data(train_csv_path,train);
	utils.read_Data(test_csv_path,test);
	
	MatrixXd X_train = train.block(0,1,train.rows(),train.cols()-1);
	MatrixXd X_test = test.block(0,1,test.rows(),test.cols()-1);
	VectorXd Y_train = train.block(0,0,train.rows(),1);
	VectorXd Y_test = test.block(0,0,test.rows(),1);

	for (int i = 0; i < Y_train.rows(); ++i){
		if(Y_train(i) == -1){
			Y_train(i) = 0;
		}
	}

	for (int i = 0; i < Y_test.rows(); ++i){
		if(Y_test(i) == -1){
			Y_test(i) = 0;
		}
	}
	cout << "Init" << endl;
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
