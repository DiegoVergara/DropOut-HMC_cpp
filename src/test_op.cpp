//#include "hmc.hpp"
#include "utils/c_utils.hpp"
#include "likelihood/multivariate_gaussian.hpp"

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

using namespace Eigen;
using namespace std;


int main(int argc, char const *argv[])
{
	C_utils tools;	
	VectorXd mean(2);
	VectorXd var(2);
	MatrixXd cov(2,2);
	mean << 0,0;
	var << 2,2;
	cov << 2 ,0,
			0, 2;
	
	MVNGaussian mvn = MVNGaussian(mean, var);
	mvn.generate();
	int n = 5000;
	MatrixXd mtx(n,2);
	//for (int i = 0; i < n; ++i) cout << mvn.sample().transpose() << endl;
	for (int i = 0; i < n; ++i) mtx.row(i) = mvn.sample();
	tools.writeToCSVfile("mvn.csv", mtx);
	
	//cout << a << endl;
	return 0;
}
