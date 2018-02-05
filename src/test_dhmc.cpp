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
#include "likelihood/CPU_dhmc.hpp"

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
	data_train << iris.block(0,0,100,1) , iris.block(0,1,100,1);//, iris.block(50,2,100,1), iris.block(50,3,100,1);
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
	
	double lambda = 10.0;
	int epochs = 10;
	int mini_batch=100;
	int num_batches=X_train.rows()/mini_batch;
	int iterations = epochs*num_batches;
	int warmup = 100*num_batches;
	double step_size = 1e-2;
	int num_steps = 1e2;
	double mask_rate = 0.1;
	int samples = 100;

	cout << "Lambda: " << lambda << ", Warmup: " << warmup << ", Iterations: " << iterations << ", Mini Batch: "<< mini_batch <<", Samples: " << samples <<", Step Size: " << step_size << ", Num Steps:" << num_steps << ", Mask Rate: "<< mask_rate << endl;

	Mask_CPU_Hamiltonian_MC hmc;
	hmc.init(X_train, Y_train, lambda, warmup, iterations, mini_batch, step_size, num_steps, mask_rate, true, true, samples);

	cout << "Init run" << endl;
	auto start = chrono::high_resolution_clock::now();
	hmc.run();
	auto finish = chrono::high_resolution_clock::now();
	chrono::duration<double> elapsed = finish - start;
	cout << "Elapsed Simulation time: " << elapsed.count() << " s\n";
	//hmc.saveModel("DHMC_gender");

	cout << "Init predict" << endl;
	predicted_labels = hmc.predict(X_test, 10, false);

	cout << "Init report" << endl;
	utils.report(Y_test, predicted_labels, true);
	utils.confusion_matrix(Y_test, predicted_labels, true);
	utils.calculateAccuracyPercent(Y_test, predicted_labels);
	return 0;
}
