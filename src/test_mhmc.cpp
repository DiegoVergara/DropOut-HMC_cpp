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
#include "likelihood/CPU_mhmc.hpp"

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
	int mini_batch=20;
	double lambda = 10.0;
	int num_batches=X_train.rows()/mini_batch;
	int iterations = epochs*num_batches;
	int warmup = 100*num_batches;
	double step_size = 1e-3;
	int num_steps = 1e2;
	int samples = 1e3;
	int psamples = 30;

		cout << "Lambda: " << lambda << ", Warmup: " << warmup << ", Iterations: " << iterations << ", Samples: "<< samples << ", Step Size: " << step_size << ", Num Steps:" << num_steps << endl;
	
	CPU_Hamiltonian_MC hmc;
	hmc.init(X_train, Y_train, lambda, warmup, iterations, mini_batch, step_size, num_steps, false, false, samples, path);

	cout << "Init run" << endl;
	auto start = chrono::high_resolution_clock::now();
	hmc.run();
	auto finish = chrono::high_resolution_clock::now();
	chrono::duration<double> elapsed = finish - start;
	cout << "Elapsed Simulation time: " << elapsed.count() << " s\n";

	cout << "Init predict" << endl;
	predicted_labels = hmc.predict(X_test, psamples, false);

	cout << "Prob" << endl;
	MatrixXd predict_proba = hmc.get_predict_proba();
	utils.writeToCSVfile(path+"predict_proba_mean.csv", predict_proba);
	utils.writeToCSVfile(path+"predict_proba_max.csv", (predict_proba.rowwise().maxCoeff()));
	MatrixXd predict_proba_std = hmc.get_predict_proba_std();
	utils.writeToCSVfile(path+"predict_proba_std.csv", predict_proba_std);
	utils.writeToCSVfile(path+"Y_test.csv", Y_test);
	//cout << predict_proba << endl;
	cout << "Mean Prob" << endl;
	VectorXd mean_prob = predict_proba.colwise().mean();
	cout << mean_prob.transpose() << endl;
	cout << "Std Prob" << endl;
	VectorXd std_prob = ((predict_proba.rowwise() - mean_prob.transpose()).array().square().colwise().sum() / (predict_proba.rows())).sqrt();
	cout << std_prob.transpose() << endl;

	MatrixXd predict_history = hmc.get_predict_history();
	VectorXd histogram(psamples);
	for (int i = 0; i < predict_history.cols(); ++i){
		histogram(i) = utils.calculateAccuracyPercent(Y_test, predict_history.col(i));
	}
	utils.writeToCSVfile(path+"histogram.csv", histogram);

	cout << "Init report" << endl;
	utils.report(Y_test, predicted_labels, true);
	utils.confusion_matrix(Y_test, predicted_labels, true);
	utils.calculateAccuracyPercent(Y_test, predicted_labels);
	return 0;
}
