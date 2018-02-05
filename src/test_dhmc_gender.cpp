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
	MatrixXd data;
	VectorXd labels;

	int train_partition = 14000;
	data_csv_path = "../data/ADIENCE/lbp/dataset.csv";
	labels_csv_path = "../data/ADIENCE/lbp/gender_label.csv";

	cout << "Read Data" << endl;

	utils.read_Data(data_csv_path,data);
	utils.read_Labels(labels_csv_path,labels);

	/*int cols = utils.get_Cols(data_csv_path, ',');
	utils.read_Data(data_csv_path,data, 2000, cols);
	utils.read_Labels(labels_csv_path,labels, 2000);	
	int train_partition = 1600;*/

	//cout << "Data Permutation" << endl;
	//utils.dataPermutation(data, labels);

	cout << "Data Partition" << endl;
	MatrixXd data_train, data_test;
	VectorXd labels_train, labels_test;
	utils.dataPartition(data, labels, data_train, data_test, labels_train, labels_test, train_partition);

	
	cout << "Init" << endl;
	VectorXd predicted_labels;

	MatrixXd X_train = data_train;
	MatrixXd X_test = data_test;
	VectorXd Y_train = labels_train;
	VectorXd Y_test = labels_test;

	double lambda = 0.1;
	int epochs = 2;
	int mini_batch=100;
	int num_batches=X_train.rows()/mini_batch;
	int iterations = epochs*num_batches;
	int warmup = 100*num_batches;
	double step_size = 0.01;
	int num_steps = 100;
	bool mask = true;
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

	//hmc.saveModel("DHMC_gender");
	cout << "Init predict" << endl;
	predicted_labels = hmc.predict(X_test, 30, false);
	MatrixXd predict_proba = hmc.get_predict_proba();
	cout << predict_proba << endl;
	cout << "Mean Prob" << endl;
	VectorXd mean_prob = predict_proba.colwise().mean();
	cout << mean_prob.transpose() << endl;
	cout << "Std Prob" << endl;
	VectorXd std_prob = ((predict_proba.rowwise() - mean_prob.transpose()).array().square().colwise().sum() / (predict_proba.rows())).sqrt();
	cout << std_prob.transpose() << endl;

	cout << "Init report" << endl;
	utils.report(Y_test, predicted_labels, true);
	utils.confusion_matrix(Y_test, predicted_labels, true);
	utils.calculateAccuracyPercent(Y_test, predicted_labels);
	return 0;
}
