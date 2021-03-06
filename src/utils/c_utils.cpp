// Author: Diego Vergara
#include <iostream>
#include "c_utils.hpp"

typedef std::numeric_limits< double > dbl;
const static IOFormat CSVFormat(StreamPrecision, DontAlignCols, ", ", "\n");

C_utils::C_utils(){
    initialized=true;
}

MatrixXd C_utils::VtoM(VectorXd &W, int _n_classes, int _dim){
  Map<MatrixXd> M(W.data(), _n_classes, _dim);
  return M;
}

VectorXd C_utils::MtoV(MatrixXd &W){
  VectorXd V(Map<VectorXd>(W.data(), W.size()));
  return V;
}


int C_utils::ReverseInt (int i){
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return((int) ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void C_utils::read_Mnist(string filename, MatrixXd& mnist){
    ifstream file (filename, ios::binary);
    if (file.is_open()){
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = this->ReverseInt(magic_number);
        file.read((char*) &number_of_images,sizeof(number_of_images));
        number_of_images = this->ReverseInt(number_of_images);
        file.read((char*) &n_rows, sizeof(n_rows));
        n_rows = this->ReverseInt(n_rows);
        file.read((char*) &n_cols, sizeof(n_cols));
        n_cols = this->ReverseInt(n_cols);
        mnist.resize(number_of_images, n_rows*n_cols);
        for(int i = 0; i < number_of_images; ++i){
            VectorXd v_temp(n_rows*n_cols);
            for(int r = 0; r < n_rows; ++r){
                for(int c = 0; c < n_cols; ++c){
                    unsigned char temp = 0;
                    file.read((char*) &temp, sizeof(temp));
                    cout << temp << endl;
                    v_temp(r+c) =(double)temp;
                }
            }
            mnist.row(i) = v_temp;
        }
    }
}

void C_utils::read_Mnist_Label(string filename, VectorXd& mnist){
    ifstream file (filename, ios::binary);
    if (file.is_open()){
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = this->ReverseInt(magic_number);
        file.read((char*) &number_of_images,sizeof(number_of_images));
        number_of_images = this->ReverseInt(number_of_images);
        mnist.resize(number_of_images);
        for(int i = 0; i < number_of_images; ++i){
            unsigned char temp = 0;
            file.read((char*) &temp, sizeof(temp));
            mnist(i)= (double)temp;
        }
    }
}

double C_utils::unif(double min, double max){
  random_device rd;
  mt19937 gen(rd());
  uniform_real_distribution<> dis(min, max);
  return dis(gen);
}

Mat C_utils::RGBtoLAB(Mat &frame){
  int channels = frame.channels();
  vector<Mat> frameChannels(channels);
  split(frame, frameChannels); 
  MatrixXd R, G, B;
  cv2eigen(frameChannels[0],R);
  cv2eigen(frameChannels[1],G);
  cv2eigen(frameChannels[2],B);

  if ((R.maxCoeff() >1.0) or (G.maxCoeff() >1.0) or (B.maxCoeff() >1.0))
  { 
    R = R.array()/255.;
    G = G.array()/255.;
    B = B.array()/255.;
  }

  double T = 0.008856;
  int cols = R.cols();
  int rows = R.rows();
  int s = cols * rows;

  MatrixXd RGB(channels,s);
  RGB.row(0) = Map<VectorXd>(R.data(), s);
  RGB.row(1) = Map<VectorXd>(G.data(), s);
  RGB.row(2) = Map<VectorXd>(B.data(), s);

  MatrixXd MAT(3,3);
  MAT << 0.412453, 0.357580, 0.180423,
       0.212671, 0.715160, 0.072169,
       0.019334, 0.119193, 0.950227;

  MatrixXd XYZ = MAT * RGB;
  VectorXd X = XYZ.row(0).array()/0.950456;
  VectorXd Y = XYZ.row(1);
  VectorXd Z = XYZ.row(2).array()/1.088754;

  int Xdim = X.rows();
  VectorXd XT= VectorXd::Zero(Xdim);
  VectorXd YT= VectorXd::Zero(Xdim);
  VectorXd ZT= VectorXd::Zero(Xdim);
  VectorXd nXT= VectorXd::Zero(Xdim);
  VectorXd nYT= VectorXd::Zero(Xdim);
  VectorXd nZT= VectorXd::Zero(Xdim);

  for (int i = 0; i < Xdim; ++i){
    if(X(i)>T){
      XT(i) = 1.0;
    }
    else{
      nXT(i) = 1.0;
    }
    if(Y(i)>T){
      YT(i) = 1.0;
    }
    else{
      nYT(i) = 1.0;
    }
    if(Z(i)>T){
      ZT(i) = 1.0;
    }
    else{
      nZT(i) = 1.0;
    }
  }

  VectorXd Y3 = Y.array().pow(1./3);

  VectorXd fX = XT.array() * X.array().pow(1./3) + nXT.array() * (7.787 * X.array() + 16./116);
  VectorXd fY = YT.array() * Y3.array() + nYT.array() * (7.787 * Y.array() + 16./116);
  VectorXd fZ = ZT.array() * Z.array().pow(1./3) + nZT.array() * (7.787 * Z.array() + 16./116);

  MatrixXd L(cols, rows);
  VectorXd tempL = (YT.array() * (116. * Y3.array() - 16.0) + nYT.array() * (903.3 * Y.array()))/100.;
  L = Map<MatrixXd>((tempL).data(), cols, rows);
  MatrixXd a(cols, rows);
  VectorXd tempa = ((500. * (fX.array() - fY.array())) *3. +110.)/220.;
  a = Map<MatrixXd>((tempa).data(), cols, rows);
  MatrixXd b(cols, rows);
  VectorXd tempb = ((200. * (fY.array() - fZ.array()))*3. +110.)/220.;
  b = Map<MatrixXd>((tempb).data(), cols, rows);

  Mat Lab;
  Mat chL, cha, chb;
  eigen2cv(L, chL);
  eigen2cv(a, cha);
  eigen2cv(b, chb);
  vector<Mat> LabChannels;
  LabChannels.push_back(chL);
  LabChannels.push_back(cha);
  LabChannels.push_back(chb);
  merge(LabChannels, Lab);

  return Lab; 
}

VectorXd C_utils::random_generator(int dimension){
  unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
  mt19937 generator;
  generator.seed(seed1);
  normal_distribution<double> dnormal(0.0,1.0);
  VectorXd random_vector(dimension);

  for (int i = 0; i < dimension; ++i){
    random_vector(i) = dnormal(generator);
  }
  return random_vector;
}

double C_utils::random_uniform(){
  random_device rd;
  mt19937 gen(rd());
  uniform_real_distribution<> dis(0, 1);
  return dis(gen);
}

VectorXd C_utils::random_uniform_generator(int dimension){
  VectorXd random_vector(dimension);

  for (int i = 0; i < dimension; ++i){
    random_vector(i) = this->random_uniform();
  }
  return random_vector;
}

VectorXd C_utils::random_binomial(int n, VectorXd prob, int dim){
  unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
  mt19937 generator;
  generator.seed(seed1);
  VectorXd random_vector(dim);

  for (int i = 0; i < dim; ++i){
    binomial_distribution<int> dbinomial(n,prob(i));
    random_vector(i) = dbinomial(generator);
  }
  return random_vector;
}

VectorXd C_utils::random_binomial(int n, double prob, int dim){
  unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
  mt19937 generator;
  generator.seed(seed1);
  binomial_distribution<int> dbinomial(n,prob);
  VectorXd random_vector(dim);

  for (int i = 0; i < dim; ++i) random_vector(i) = dbinomial(generator);
  return random_vector;
}

void C_utils::writeToCSVfile(string name, MatrixXd matrix, bool append){
    if (append)
    {
        ofstream file(name.c_str(), fstream::app);
        if (file.is_open()){
          file << matrix.format(CSVFormat) << '\n';
          //file << "m" << '\n' <<  colm(matrix) << '\n';
        }
    }
    else{
        ofstream file(name.c_str()); 
        if (file.is_open()){
          file << matrix.format(CSVFormat) << '\n';
          //file << "m" << '\n' <<  colm(matrix) << '\n';
        }
    }
}

double C_utils::calculateAccuracyPercent(VectorXd labels,VectorXd predicted){
    cout << "Accuracy Score:" << endl;
    double ac = 100 * (float) labels.cwiseEqual(predicted).cast<double>().sum() / labels.size();
    cout << ac << endl;
    return ac;
}

void C_utils::printProgBar( int value, int max ) {
  int percent = (value / (float) max) * (float) 100;
  string bar;

  for(int i = 0; i < 50; i++){
    if( i < (percent/2)){
      bar.replace(i,1,"=");
    }else if( i == (percent/2)){
      bar.replace(i,1,">");
    }else{
      bar.replace(i,1," ");
    }
  }

  cout<< "\r" "[" << bar << "] ";
  cout.width( 3 );
  cout<< percent << "%     " << std::flush;
}

void C_utils::dataPermutation(MatrixXd& X_train,VectorXd& Y_train){
  VectorXi indices = VectorXi::LinSpaced(X_train.rows(), 0, X_train.rows());
  std::random_shuffle(indices.data(), indices.data() + X_train.rows());
  X_train.noalias() = indices.asPermutation() * X_train;  
  Y_train.noalias() = indices.asPermutation() * Y_train;
}

/*void C_utils::dataNormalization(MatrixXd& data){
  RowVectorXd mean = data.colwise().mean();
  RowVectorXd var = (data.rowwise() - mean).array().square().colwise().mean();
  data = (data.rowwise() - mean).array().rowwise() / var.array();
}*/

void C_utils::dataNormalization(MatrixXd& data , RowVectorXd& max, RowVectorXd& min){
  max = data.colwise().maxCoeff();
  min =  data.colwise().minCoeff();
  data = (data.rowwise() - min).array().rowwise() / (max-min).array();
}

void C_utils::dataStandardization(MatrixXd& data,RowVectorXd& mean, RowVectorXd& std){
  mean = data.colwise().mean();
  std = ((data.rowwise() - mean).array().square().colwise().sum() / (data.rows())).sqrt();
  if( (std.array() > 0 ).all())
    data = (data.rowwise() - mean).array().rowwise() / std.array();
}

void C_utils::testNormalization(MatrixXd& data , RowVectorXd max, RowVectorXd min){
  data = (data.rowwise() - min).array().rowwise() / (max-min).array();
}

void C_utils::testStandardization(MatrixXd& data,RowVectorXd mean, RowVectorXd std){
  if( (std.array() > 0 ).all())
    data = (data.rowwise() - mean).array().rowwise() / std.array();
}

void C_utils::dataPartition(MatrixXd& data,VectorXd& labels, MatrixXd& X_train, MatrixXd& X_test, VectorXd& Y_train, VectorXd& Y_test, int partition){
  X_train =  data.block(0,0 ,partition, data.cols());
  X_test =  data.block(partition,0 ,data.rows()-partition, data.cols());
  Y_train =  labels.head(partition);
  Y_test =  labels.tail(labels.rows()-partition);
}

VectorXd C_utils::argMin(MatrixXd data, bool row){
    VectorXd result;
    if (row){
        int tam = data.rows();
        result = VectorXd::Zero(tam);
        MatrixXf::Index   minIndex[tam];
        for (int j =0; j< data.rows(); ++j){
            data.row(j).minCoeff(&minIndex[j]);
            result(j) = minIndex[j];
        }
    }
    else{
        int tam = data.cols();
        result = VectorXd::Zero(tam);
        MatrixXf::Index   minIndex[tam];
        for (int j =0; j< data.cols(); ++j){
            data.col(j).minCoeff(&minIndex[j]);
            result(j) = minIndex[j];
        }
    }
    return result;
}

VectorXd C_utils::argMax(MatrixXd data, bool row){
    VectorXd result; 
    if (row){
        int tam = data.rows();
        result = VectorXd::Zero(tam);
        MatrixXf::Index   maxIndex[tam];
        for (int j =0; j< data.rows(); ++j){
            data.row(j).maxCoeff(&maxIndex[j]);
            result(j) = maxIndex[j];
        }
    }
    else{
        int tam = data.cols();
        result = VectorXd::Zero(tam);
        MatrixXf::Index   maxIndex[tam];
        for (int j =0; j< data.cols(); ++j){
            data.col(j).maxCoeff(&maxIndex[j]);
            result(j) = maxIndex[j];
        }
    }
    return result;
}


VectorXd C_utils::matrixDot(MatrixXd &A, VectorXd &x){
    VectorXd aux(A.rows());
    for (int i = 0; i < A.rows(); ++i)
    {
        aux(i) = A.row(i).dot(x);
    }
    return aux;
}

VectorXd C_utils::sign(VectorXd &x){
    VectorXd s(x.rows());
    for (int i = 0; i < x.rows(); ++i){
        if (x(i) > 0.0){
            s(i) = 1.0;
        }
        else if(x(i) < 0.0){
            s(i) = -1.0;
        }
        else{
            s(i) = 0.0;
        }
    }
    return s;
}

VectorXd C_utils::getClasses(VectorXd &x){
    int n_classes = 1;
    VectorXd classes(n_classes);
    classes(0) = x(0);
    for (int i = 1; i < x.size(); ++i){
        int count=0;
        for (int j = 0; j < n_classes; ++j){
          if (x(i) != classes(j)){
            count++;
          }
        }
        if (count == n_classes){
          n_classes++;
          classes.conservativeResize(n_classes);
          classes(n_classes-1)=x(i);
        }
    }
    return classes;
}

int C_utils::getIndex(VectorXd &x, double value){
    int index = 0;
    for (int i = 0; i < x.size(); ++i){
      if(x(i) == value){
        index = i;
        break;
      }
    }
    return index;
  }

MatrixXd C_utils::binarizeLabels(VectorXd &x, VectorXd& classes){
    classes = this->getClasses(x);
    int x_size = x.size();
    int c_size = classes.size();
    MatrixXd binarize = MatrixXd::Zero(x_size, c_size);
    for (int i = 0; i < x_size; ++i){
      binarize(i, this->getIndex(classes, x(i))) = 1.0;
    }
    return binarize;
}

VectorXd C_utils::vecMax(double value, VectorXd &vec){
    VectorXd vector_max(vec.rows());
    for (int i = 0; i < vec.rows(); ++i){
        vector_max(i) = max(value, vec(i));
    }
    return vector_max;
}

void C_utils::read_Labels(const string& filename, VectorXi& labels) {
    int rows = this->get_Rows(filename);
    //cout << "[" << rows << "]"<< endl; 
    ifstream file(filename.c_str());
    if(!file)
        throw exception();
    string line;
    int row = 0;
    labels.resize(rows);
    while (getline(file, line)) {
      if (row < rows){
        //this->printProgBar(row, rows);
        int item=atoi(line.c_str());
        labels(row)=item;
        row++;
      }
    }
    file.close();
    //cout << endl;
}

void C_utils::read_Labels(const string& filename, VectorXd& labels) {
    int rows = this->get_Rows(filename);
    //cout << "[" << rows << "]"<< endl; 
    ifstream file(filename.c_str());
    if(!file)
        throw exception();
    string line;
    int row = 0;
    labels.resize(rows);
    while (getline(file, line)) {
      if (row < rows){
      //  this->printProgBar(row, rows);
        double item=atof(line.c_str());
        labels(row)=item;
        row++;
      }
    }
    file.close();
    //cout << endl;
}

void C_utils::read_Labels(const string& filename, VectorXi& labels, int rows) {
    ifstream file(filename.c_str());
    //cout << "[" << rows << "]"<< endl; 
    if(!file)
        throw exception();
    string line;
    int row = 0;
    labels.resize(rows);
    while (getline(file, line)) {
      if (row < rows){
        //this->printProgBar(row, rows);
        int item=atoi(line.c_str());
        labels(row)=item;
        row++;
      }
    }
    file.close();
    //cout << endl;
}

void C_utils::read_Labels(const string& filename, VectorXd& labels, int rows) {
    ifstream file(filename.c_str());
    //cout << "[" << rows << "]"<< endl; 
    if(!file)
        throw exception();
    string line;
    int row = 0;
    labels.resize(rows);
    while (getline(file, line)) {
      if (row < rows){
        //this->printProgBar(row, rows);
        double item=atof(line.c_str());
        labels(row)=item;
        row++;
      }
    }
    file.close();
    //cout << endl;
}

void C_utils::read_Data(const string& filename, MatrixXd& data) {
    int rows = this->get_Rows(filename);
    int cols = this->get_Cols(filename, ',');
    ifstream file(filename.c_str());
    if(!file)
        throw exception();
    //cout << "[" << rows << ", "<< cols << "]"<< endl; 
    string line, cell;
    int row = 0,col;
    data.resize(rows, cols);
    while (getline(file, line)) {
      col=0;
      stringstream csv_line(line);
      if (row < rows){
        //this->printProgBar(row, rows);
        while (getline(csv_line, cell, ',')){
          if (col<cols){
            double item=atof(cell.c_str());
            data(row,col)=item;
            col++;
          }
        }
      row++;
      }
    }
    file.close();
    //cout << endl;
}

void C_utils::read_Data(const string& filename, MatrixXd& data, int rows, int cols) {
    ifstream file(filename.c_str());
    if(!file)
        throw exception();
    //cout << "[" << rows << ", "<< cols << "]"<< endl; 
    string line, cell;
    int row = 0,col;
    data.resize(rows, cols);
    while (getline(file, line)) {
      col=0;
      stringstream csv_line(line);
      if (row < rows){
        //this->printProgBar(row, rows);
        while (getline(csv_line, cell, ',')){
          if (col<cols){
            double item=atof(cell.c_str());
            data(row,col)=item;
            col++;
          }
        }
      row++;
      }
    }
    file.close();
    //cout << endl;
}

void C_utils::classification_Report(VectorXi &test, VectorXi &predicted)
{
  int count=0;
  for (int i=0; i < test.size(); i++) if (int(predicted(i)) == test(i)) count++;
  double percent = double(count) / double(test.size());
  cout << setprecision(10) << fixed;
  cout << percent*100 << endl;
  
}

void C_utils::classification_Report(VectorXi &test, VectorXd &predicted)
{
  int count=0;
  for (int i=0; i < test.size(); i++) if (int(predicted(i)) == test(i)) count++;
  double percent = double(count) / double(test.size());
  cout << setprecision(10) << fixed;
  cout << percent*100 << endl;
  
}

void C_utils::classification_Report(VectorXd &test, VectorXi &predicted)
{
  int count=0;
  for (int i=0; i < test.size(); i++) if (int(predicted(i)) == test(i)) count++;
  double percent = double(count) / double(test.size());
  cout << setprecision(10) << fixed;
  cout << percent*100 << endl;
  
}

void C_utils::print(VectorXi &test, VectorXi &predicted)
{
  for (int i=0; i < test.size(); i++) cout << test(i) << " " << predicted(i) << endl;
}

int C_utils::get_Rows(const string& filename) {
  int number_of_lines = 0;
  string line;
  ifstream file(filename.c_str());
  while (getline(file, line)) ++number_of_lines;
  return number_of_lines;
}

int C_utils::get_Cols(const string& filename, char separator) {
  int number_of_lines = 0;
  string line, token;
  ifstream file(filename.c_str());
  getline(file, line);
  istringstream aux(line);
  while (getline(aux, token, separator)) ++number_of_lines;
  return number_of_lines;
}

vector<int> C_utils::get_Classes(VectorXi labels){
  vector<int> classes;
  classes.push_back(labels(0));
  for (int i = 1; i < labels.rows(); ++i) {
    int count = 0;
    for (unsigned int k = 0; k < classes.size(); ++k) {
      if (labels(i)!= int(classes.at(k))) count++;
    }
    if (count == int(classes.size())) classes.push_back(labels(i));
  }
  sort(classes.begin(), classes.end());
  return classes;
}

vector<int> C_utils::get_Classes_d(VectorXd labels){
  vector<int> classes;
  classes.push_back(round(labels(0)));
  for (int i = 1; i < labels.rows(); ++i) {
    int count = 0;
    for (unsigned int k = 0; k < classes.size(); ++k) {
      if (round(labels(i))!= int(classes.at(k))) count++;
    }
    if (count == int(classes.size())) classes.push_back(round(labels(i)));
  }
  sort(classes.begin(), classes.end());
  return classes;
}

map<pair<int,int>, int> C_utils::confusion_matrix(VectorXi &test, VectorXi &predicted, bool print){
  vector<int> testedClasses = get_Classes(test);
  vector<int> predictedClasses = get_Classes(predicted);
  set<int> classes;

  classes.insert(testedClasses.begin(), testedClasses.end());
  classes.insert(predictedClasses.begin(), predictedClasses.end());

  map<pair<int,int>, int> confusionMatrix;
  
  for (set<int>::iterator i = classes.begin(); i != classes.end(); ++i)
  {
    for (set<int>::iterator j = classes.begin(); j != classes.end(); ++j)
    {
      confusionMatrix[pair<int,int>(*i,*j)] = 0;
    }
  }

  for(int i = 0; i < test.size(); i++){
    confusionMatrix[pair<int,int>(test(i),predicted(i))] +=1;
  }

  if(print){
    cout << "Confusion Matrix:" << endl;
    for (std::map<pair<int,int>,int>::iterator it = confusionMatrix.begin(); it != confusionMatrix.end(); ++it)
    {
      cout << "class " << it->first.first << " was predicted " << it->second << " times as class " << it->first.second << endl;
    }
  }

  return confusionMatrix;
}

map<pair<int,int>, int> C_utils::confusion_matrix(VectorXi &test, VectorXd &predicted, bool print){
  vector<int> testedClasses = get_Classes(test);
  vector<int> predictedClasses = get_Classes_d(predicted);
  set<int> classes;

  classes.insert(testedClasses.begin(), testedClasses.end());
  classes.insert(predictedClasses.begin(), predictedClasses.end());

  map<pair<int,int>, int> confusionMatrix;
  
  for (set<int>::iterator i = classes.begin(); i != classes.end(); ++i)
  {
    for (set<int>::iterator j = classes.begin(); j != classes.end(); ++j)
    {
      confusionMatrix[pair<int,int>(*i,*j)] = 0;
    }
  }
  
  for(int i = 0; i < test.size(); i++){
    confusionMatrix[pair<int,int>(test(i),predicted(i))] +=1;
  }
  
  if(print){
    cout << "Confusion Matrix:" << endl;
    for (std::map<pair<int,int>,int>::iterator it = confusionMatrix.begin(); it != confusionMatrix.end(); ++it)
    {
      cout << "class " << it->first.first << " was predicted " << it->second << " times as class " << it->first.second << endl;
    }
  }

  return confusionMatrix;
}

map<pair<int,int>, int> C_utils::confusion_matrix(VectorXd &test, VectorXd &predicted, bool print){
  vector<int> testedClasses = get_Classes_d(test);
  vector<int> predictedClasses = get_Classes_d(predicted);
  set<int> classes;

  classes.insert(testedClasses.begin(), testedClasses.end());
  classes.insert(predictedClasses.begin(), predictedClasses.end());

  map<pair<int,int>, int> confusionMatrix;
  
  for (set<int>::iterator i = classes.begin(); i != classes.end(); ++i)
  {
    for (set<int>::iterator j = classes.begin(); j != classes.end(); ++j)
    {
      confusionMatrix[pair<int,int>(*i,*j)] = 0;
    }
  }
  
  for(int i = 0; i < test.size(); i++){
    confusionMatrix[pair<int,int>(test(i),predicted(i))] +=1;
  }
  
  if(print){
    cout << "Confusion Matrix:" << endl;
    for (std::map<pair<int,int>,int>::iterator it = confusionMatrix.begin(); it != confusionMatrix.end(); ++it)
    {
      cout << "class " << it->first.first << " was predicted " << it->second << " times as class " << it->first.second << endl;
    }
  }

  return confusionMatrix;
}

map<int,double> C_utils::precision_score(VectorXi &test, VectorXi &predicted, bool print){
  map<pair<int,int>, int> confusionMatrix = confusion_matrix(test, predicted, false);
  return precision_score(confusionMatrix, print);
}

map<int, double> C_utils::precision_score(VectorXi &test, VectorXd &predicted, bool print){
  map<pair<int,int>, int> confusionMatrix = confusion_matrix(test, predicted, false);
  return precision_score(confusionMatrix, print);
}

map<int, double> C_utils::precision_score(VectorXd &test, VectorXd &predicted, bool print){
  map<pair<int,int>, int> confusionMatrix = confusion_matrix(test, predicted, false);
  return precision_score(confusionMatrix, print);
}

map<int,double> C_utils::precision_score(map<pair<int,int>, int> confusionMatrix, bool print){
  map<int, double> true_positive;
  map<int, double> false_positive;
  map<int, double> precision;

  for (std::map<pair<int,int>,int>::iterator it = confusionMatrix.begin(); it != confusionMatrix.end(); ++it)
  {
    if(it->first.first == it->first.second){
      true_positive[it->first.first] = it->second;
    }
    else{
      if(false_positive.find(it->first.second) == false_positive.end()){
        false_positive[it->first.second] = it->second;
      }
      else{
       false_positive[it->first.second] += it->second; 
      }
    }
  }

  for (std::map<int,double>::iterator it = true_positive.begin(); it != true_positive.end(); ++it)
  {
    precision[it->first] = true_positive[it->first] / (true_positive[it->first] + false_positive[it->first]);
    if(print){
      cout << "class " << it->first << ": " << precision[it->first] << endl;
    }
  }
  
  return precision;
}

map<int, double> C_utils::accuracy_score(VectorXi &test, VectorXi &predicted, bool print){
  map<pair<int,int>, int> confusionMatrix = confusion_matrix(test, predicted, false);
  return accuracy_score(confusionMatrix, print);
}

map<int, double> C_utils::accuracy_score(VectorXi &test, VectorXd &predicted, bool print){
  map<pair<int,int>, int> confusionMatrix = confusion_matrix(test, predicted, false);
  return accuracy_score(confusionMatrix, print);
}

map<int, double> C_utils::accuracy_score(VectorXd &test, VectorXd &predicted, bool print){
  map<pair<int,int>, int> confusionMatrix = confusion_matrix(test, predicted, false);
  return accuracy_score(confusionMatrix, print);
}

map<int, double> C_utils::accuracy_score(map<pair<int,int>, int> confusionMatrix, bool print){
  map<int, double> true_positive;
  map<int, double> false_positive;
  map<int, double> false_negative;
  map<int, double> accuracy;

  for (std::map<pair<int,int>, int>::iterator it = confusionMatrix.begin(); it != confusionMatrix.end(); ++it)
  {
    if(it->first.first == it->first.second){
      true_positive[it->first.first] = it->second;
    }
    else{
      if(false_positive.find(it->first.second) == false_positive.end()){
        false_positive[it->first.second] = it->second;
      }
      else{
       false_positive[it->first.second] += it->second; 
      }

      if(false_negative.find(it->first.first) == false_negative.end()){
        false_negative[it->first.first] = it->second;
      }
      else{
        false_negative[it->first.second] += it->second;
      }
    }
  }

  for (std::map<int, double>::iterator it = true_positive.begin(); it != true_positive.end(); ++it)
  {
    accuracy[it->first] = true_positive[it->first] / (true_positive[it->first] + false_positive[it->first] + false_negative[it->first]);
    if(print){
      cout << "class " << it->first << ": " << accuracy[it->first] << endl;
    }
  }
  return accuracy;
}

map<int, double> C_utils::recall_score(VectorXi &test, VectorXi &predicted, bool print){
  map<pair<int,int>, int> confusionMatrix = confusion_matrix(test, predicted, false);
  return recall_score(confusionMatrix, print);
}

map<int, double> C_utils::recall_score(VectorXi &test, VectorXd &predicted, bool print){
  map<pair<int,int>, int> confusionMatrix = confusion_matrix(test, predicted, false);
  return recall_score(confusionMatrix, print);
}

map<int, double> C_utils::recall_score(VectorXd &test, VectorXd &predicted, bool print){
  map<pair<int,int>, int> confusionMatrix = confusion_matrix(test, predicted, false);
  return recall_score(confusionMatrix, print);
}

map<int, double> C_utils::recall_score(map<pair<int,int>, int> confusionMatrix, bool print){
  map<int, double> true_positive;
  map<int, double> false_negative;
  map<int, double> recall;

  for (std::map<pair<int,int>, int>::iterator it = confusionMatrix.begin(); it != confusionMatrix.end(); ++it)
  {
    if(it->first.first == it->first.second){
      true_positive[it->first.first] = it->second;
    }
    else{
      if(false_negative.find(it->first.first) == false_negative.end()){
        false_negative[it->first.first] = it->second;
      }
      else{
        false_negative[it->first.second] += it->second;
      }
    }
  }

  for (std::map<int, double>::iterator it = true_positive.begin(); it != true_positive.end(); ++it)
  {
    recall[it->first] = true_positive[it->first] / (true_positive[it->first] + false_negative[it->first]);
    if(print){
      cout << "class " << it->first << ": " << recall[it->first] << endl;
    }
  }
  return recall;
}

map<int, double> C_utils::f1_score(map<pair<int, int>, int> confusionMatrix, bool print){
  map<int, double> precision = precision_score(confusionMatrix);
  map<int, double> recall = recall_score(confusionMatrix);
  map<int, double> f1score;
  for (std::map<int, double>::iterator it = precision.begin(); it != precision.end(); ++it)
  {
    f1score[it->first] = 2 * (precision[it->first] * recall[it->first]) / (precision[it->first] + recall[it->first]);
    if(print){
      cout << "class " << it->first << ": " << f1score[it->first] << endl;
    }
  }
  return f1score;
}

map<int, double> C_utils::f1_score(VectorXi &test, VectorXi &predicted, bool print){
  map<pair<int, int>, int> confusionMatrix = confusion_matrix(test, predicted, false);
  return f1_score(confusionMatrix);
}

map<int, double> C_utils::f1_score(VectorXi &test, VectorXd &predicted, bool print){
  map<pair<int, int>, int> confusionMatrix = confusion_matrix(test, predicted, false);
  return f1_score(confusionMatrix);
}

map<int, double> C_utils::f1_score(VectorXd &test, VectorXd &predicted, bool print){
  map<pair<int, int>, int> confusionMatrix = confusion_matrix(test, predicted, false);
  return f1_score(confusionMatrix);
}

map<int, double> C_utils::support_score(VectorXi &test){
  map<int, double> support;
  for (int i = 0; i < test.size(); ++i)
  {
    if(support.find(test(i)) == support.end()){
      support[test(i)] = 1;
    }
    else{
      support[test(i)] +=1;
    }
  }
  return support;
}

map<int, double> C_utils::support_score(VectorXd &test){
  map<int, double> support;
  for (int i = 0; i < test.size(); ++i)
  {
    if(support.find(round(test(i))) == support.end()){
      support[round(test(i))] = 1;
    }
    else{
      support[round(test(i))] +=1;
    }
  }
  return support;
}

map<int, double> C_utils::support_score(map<pair<int, int>, int> confusionMatrix){
  map<int, double> support;
  for (std::map<pair<int, int>, int>::iterator it = confusionMatrix.begin(); it != confusionMatrix.end(); ++it)
  {
    if(support.find(it->first.first) == support.end()){
      support[it->first.first] = it->second;
    }
    else{
      support[it->first.first] += it->second;
    }
  }
  return support;
}

map<int, Metrics> C_utils::report(VectorXi &test, VectorXi &predicted, bool print){
  map<pair<int, int>, int> confusionMatrix = confusion_matrix(test, predicted, false);
  return report(confusionMatrix, print);
}

map<int, Metrics> C_utils::report(VectorXi &test, VectorXd &predicted, bool print){
  map<pair<int, int>, int> confusionMatrix = confusion_matrix(test, predicted, false);
  return report(confusionMatrix, print);
}

map<int, Metrics> C_utils::report(VectorXd &test, VectorXd &predicted, bool print){
  map<pair<int, int>, int> confusionMatrix = confusion_matrix(test, predicted, false);
  return report(confusionMatrix, print);
}

map<int, Metrics> C_utils::report(map<pair<int, int>, int> confusionMatrix, bool print){
  map<int, double> precision = precision_score(confusionMatrix);
  map<int, double> accuracy = accuracy_score(confusionMatrix);
  map<int, double> recall = recall_score(confusionMatrix);
  map<int, double> f1score = f1_score(confusionMatrix);
  map<int, double> support = support_score(confusionMatrix);
  map<int, Metrics> report;
  double avg_precision = 0, avg_accuracy = 0, avg_recall = 0, avg_f1score = 0, total_support = 0;

  if(print){
    cout << setw(15) << left  << "class" << setw(15) << left << "precision" << setw(15) << left 
    << "accuracy" << setw(15) << left << "recall" << setw(15) << left << "f1score" << setw(15) << left << "support" << endl;
  }

  for (std::map<int, double>::iterator it = precision.begin(); it != precision.end(); ++it)
  {
    Metrics m;
    m.precision = precision[it->first]; m.accuracy = accuracy[it->first];
    m.recall = recall[it->first]; m.f1score = f1score[it->first];
    m.support = support[it->first];
    report[it->first] = m;
    if(print){
      cout << setw(15) << left << it->first << setw(15) << left << m.precision << setw(15) << left 
      << m.accuracy << setw(15) << left << m.recall << setw(15) << left << m.f1score << setw(15) << left << m.support << endl;
      avg_precision += m.precision*m.support;
      avg_accuracy += m.accuracy*m.support;
      avg_recall += m.recall*m.support;
      avg_f1score += m.f1score*m.support;
      total_support += m.support;
    }
  }
  avg_precision /= total_support;
  avg_accuracy /= total_support;
  avg_recall /= total_support;
  avg_f1score /= total_support;
  
  if(print){
    cout << setw(15) << left << "avg/total" << setw(15) << left << avg_precision << setw(15) << left
    << avg_accuracy << setw(15) << left << avg_recall << setw(15) << left << avg_f1score << setw(15) 
    << left << total_support << endl;  
  }
  
  return report;
}

void C_utils::read_Mnist(string filename, vector<vector<double> > &vec){
    ifstream file (filename, ios::binary);
    if (file.is_open())
    {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);
        file.read((char*) &number_of_images,sizeof(number_of_images));
        number_of_images = ReverseInt(number_of_images);
        file.read((char*) &n_rows, sizeof(n_rows));
        n_rows = ReverseInt(n_rows);
        file.read((char*) &n_cols, sizeof(n_cols));
        n_cols = ReverseInt(n_cols);
        for(int i = 0; i < number_of_images; ++i)
        {
            vector<double> tp;
            for(int r = 0; r < n_rows; ++r)
            {
                for(int c = 0; c < n_cols; ++c)
                {
                    unsigned char temp = 0;
                    file.read((char*) &temp, sizeof(temp));
                    tp.push_back((double)temp);
                }
            }
            vec.push_back(tp);
        }
    }
}

void C_utils::read_Mnist(string filename, vector<cv::Mat> &vec){
    ifstream file (filename, ios::binary);
    if (file.is_open())
    {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);
        file.read((char*) &number_of_images,sizeof(number_of_images));
        number_of_images = ReverseInt(number_of_images);
        file.read((char*) &n_rows, sizeof(n_rows));
        n_rows = ReverseInt(n_rows);
        file.read((char*) &n_cols, sizeof(n_cols));
        n_cols = ReverseInt(n_cols);
        for(int i = 0; i < number_of_images; ++i)
        {
            cv::Mat tp = Mat::zeros(n_rows, n_cols, CV_8UC1);
            for(int r = 0; r < n_rows; ++r)
            {
                for(int c = 0; c < n_cols; ++c)
                {
                    unsigned char temp = 0;
                    file.read((char*) &temp, sizeof(temp));
                    tp.at<uchar>(r, c) = (int) temp;
                }
            }
            vec.push_back(tp);
        }
    }
}

void C_utils::read_Mnist_Label(string filename, vector<double> &vec){
    ifstream file (filename, ios::binary);
    if (file.is_open())
    {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);
        file.read((char*) &number_of_images,sizeof(number_of_images));
        number_of_images = ReverseInt(number_of_images);
        for(int i = 0; i < number_of_images; ++i)
        {
            unsigned char temp = 0;
            file.read((char*) &temp, sizeof(temp));
            vec[i]= (double)temp;
        }
    }
}
