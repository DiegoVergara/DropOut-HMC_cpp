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
	string train_csv_path, test_csv_path;
	MatrixXd train, test;

  	train_csv_path = "../data/GISETTE/gisette_scale.csv";
  	test_csv_path = "../data/GISETTE/gisette_scale_t.csv";

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

	double lambda = 10.0;
	int epochs = 10;
	int mini_batch=100;
	int num_batches=X_train.rows()/mini_batch;
	int iterations = epochs*num_batches;
	int warmup = 100*num_batches;
	double step_size = 1e-2;
	int num_steps = 1e2;
	bool mask = true;
	double mask_rate = 0.5;
	int samples = 1000;

	cout << "Lambda: " << lambda << ", Warmup: " << warmup << ", Iterations: " << iterations << ", Mini Batch: "<< mini_batch <<", Samples: " << samples <<", Step Size: " << step_size << ", Num Steps:" << num_steps << ", Mask Rate: "<< mask_rate << endl;

	Mask_CPU_Hamiltonian_MC hmc;
	hmc.init(X_train, Y_train, lambda, warmup, iterations, mini_batch, step_size, num_steps, mask_rate, false, false, samples);

	cout << "Init run" << endl;
	auto start = chrono::high_resolution_clock::now();
	hmc.run();
	auto finish = chrono::high_resolution_clock::now();
	chrono::duration<double> elapsed = finish - start;
	cout << "Elapsed Simulation time: " << elapsed.count() << " s\n";
	//MatrixXd weights = hmc.get_weights();
	//utils.writeToCSVfile("hmc_weights.csv", weights);
	cout << "Init predict" << endl;
	predicted_labels = hmc.predict(X_test, 50);

	cout << "Init report" << endl;
	utils.report(Y_test, predicted_labels, true);
	utils.confusion_matrix(Y_test, predicted_labels, true);
	utils.calculateAccuracyPercent(Y_test, predicted_labels);
	return 0;
}
