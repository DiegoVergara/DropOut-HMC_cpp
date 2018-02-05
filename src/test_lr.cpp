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
	MatrixXd iris;

  	data_csv_path = "../data/IRIS/iris.csv";

	utils.read_Data(data_csv_path,iris);
	// Sepal Length, Sepal Width, Petal Length and Petal Width
    // Setosa, Versicolour, and Virginica
	MatrixXd data_train(100, 2);
	data_train << iris.block(0,0,100,1) , iris.block(0,1,100,1);
	VectorXd labels_train=iris.block(0,4,100,1);

	/*for (int i = 0; i < labels_train.rows(); ++i){
		if(labels_train(i) == 2){
			labels_train(i) = 0;
		}
	}*/
	MatrixXd X_train = data_train;
	VectorXd Y_train = labels_train;

	MatrixXd X_test = data_train;
	VectorXd Y_test = labels_train;

	cout << "Init" << endl;
	VectorXd predicted_labels;
	
	double lambda = 0.001;
	int epochs = 10;
	double alpha = 0.90;
	double step_size = 0.1;
	int mini_batch=10;

	CPU_LogisticRegression lr;
	cout << "Init train" << endl;
	lr.init(X_train, Y_train, lambda, false, false,true);
	int num_batches=X_train.rows()/mini_batch;

	cout << "Lambda: " << lambda <<  ", Iterations: " << num_batches*epochs << ", Alpha: " << alpha << ", Step Size:" << step_size << endl;

	auto start = chrono::high_resolution_clock::now();
	lr.train(num_batches*epochs,mini_batch, alpha, step_size);
	auto finish = chrono::high_resolution_clock::now();
	chrono::duration<double> elapsed = finish - start;
	cout << "Elapsed Train time: " << elapsed.count() << " s\n";

	cout << "Init predict" << endl;
	predicted_labels = lr.predict(X_test, false);

	cout << "Init report" << endl;
	utils.report(Y_test, predicted_labels, true);
	utils.confusion_matrix(Y_test, predicted_labels, true);
	utils.calculateAccuracyPercent(Y_test, predicted_labels);
	return 0;
}
