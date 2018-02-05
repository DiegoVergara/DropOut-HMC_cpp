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
	
	string data_csv_path;
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
	
	double lambda = 100.0;
	int epochs = 10;
	int mini_batch=100;
	int num_batches=X_train.rows()/mini_batch;
	int iterations = epochs*num_batches;
	int warmup = 100*num_batches;
	double step_size = 1e-1;
	int num_steps = 1e2;
	double mask_rate = 0.1;
	int samples = 1000;

	cout << "Lambda: " << lambda << ", Warmup: " << warmup << ", Iterations: " << iterations << ", Mini Batch: "<< mini_batch <<", Samples: " << samples <<", Step Size: " << step_size << ", Num Steps:" << num_steps << ", Mask Rate: "<< mask_rate << endl;

	Mask_CPU_Hamiltonian_MC hmc;
	hmc.init(X_train, Y_train, lambda, warmup, iterations, mini_batch, step_size, num_steps, mask_rate, true, true, samples);

	cout << "Init run" << endl;
	auto start = chrono::high_resolution_clock::now();
	hmc.run();
	auto finish = chrono::high_resolution_clock::now();
	chrono::duration<double> elapsed = finish - start;
	cout << "Elapsed Simulation time: " << elapsed.count() << " s\n";
	//MatrixXd weights = hmc.get_weights();
	//utils.writeToCSVfile("hmc_weights.csv", weights);
	cout << "Init predict" << endl;
	predicted_labels = hmc.predict(X_test, 30, false);

	cout << "Init report" << endl;
	utils.report(Y_test, predicted_labels, true);
	utils.confusion_matrix(Y_test, predicted_labels, true);
	utils.calculateAccuracyPercent(Y_test, predicted_labels);
	return 0;
}
