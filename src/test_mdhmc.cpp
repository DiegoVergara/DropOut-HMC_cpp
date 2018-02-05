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
#include "likelihood/CPU_mdhmc.hpp"

using namespace Eigen;
using namespace std;

int main(int argc, char const *argv[])
{	
	
	C_utils utils;

	string path;

	if(argc != 3) {
        cerr <<"Incorrect input list" << endl;
        cerr <<"exiting..." << endl;
        return EXIT_FAILURE;
    }
    else{
        string train_csv_path, test_csv_path;
        if(strcmp(argv[1], "-path") == 0) {
            path=argv[2];
        }
    }
 
	const int dir_err = system(("mkdir -p "+path).c_str());
	if (-1 == dir_err)
	{
	    printf("Error creating directory!n");
	    exit(1);
	}
	
	path = path+"/";
	
	MatrixXd X_train;
	MatrixXd X_test;
	VectorXd Y_train;
	VectorXd Y_test;
	utils.read_Data("../data/IRIS/X_train.csv",X_train);
	utils.read_Data("../data/IRIS/X_test.csv",X_test);
	utils.read_Labels("../data/IRIS/Y_train.csv",Y_train);
	utils.read_Labels("../data/IRIS/Y_test.csv",Y_test);
	cout << "Init" << endl;
	VectorXd predicted_labels;
	
	int epochs = 100;
	int mini_batch=100;
	double lambda = 1.0;
	int num_batches=X_train.rows()/mini_batch;
	int iterations = epochs*num_batches;
	int warmup = 100*num_batches;
	double step_size = 1e-3;
	int num_steps = 1e2;
	int samples = 1e3;
	double mask_rate = 0.5;

	cout << "Lambda: " << lambda << ", Warmup: " << warmup << ", Iterations: " << iterations << ", Samples: "<< samples << ", Step Size: " << step_size << ", Num Steps:" << num_steps << ", Mask Rate: "<< mask_rate << endl;
	
	Mask_CPU_Hamiltonian_MC hmc;
	hmc.init(X_train, Y_train, lambda, warmup, iterations, mini_batch, step_size, num_steps, mask_rate, false, false, samples, path);

	cout << "Init run" << endl;
	auto start = chrono::high_resolution_clock::now();
	hmc.run();
	auto finish = chrono::high_resolution_clock::now();
	chrono::duration<double> elapsed = finish - start;
	cout << "Elapsed Simulation time: " << elapsed.count() << " s\n";

	cout << "Init predict" << endl;
	predicted_labels = hmc.predict(X_test, 30, false);

	cout << "Init report" << endl;
	utils.report(Y_test, predicted_labels, true);
	utils.confusion_matrix(Y_test, predicted_labels, true);
	utils.calculateAccuracyPercent(Y_test, predicted_labels);
	return 0;
}
