#include <iostream>
#include <math.h>
#include <random>
#include <cstdio>
#include <cstdlib>
#include <chrono>

#define DATA_LENGTH 20000
#define DATA_DIM 784
#define TRAIN_ROUND 2000
#define REGULARIZATION 0
#define W_SCALE_FACTOR 1
#define LR 0.01
#define BATCH_SIZE 1
using namespace std;
using namespace std::chrono;

mt19937 rng;
uniform_int_distribution<int> gen;

int read_dataset(float *dataset[], char *file_path, int data_len)
{
    FILE *fp = fopen(file_path, "r");

    int i = 0;
    while(!feof(fp)) {
        float* t = (float *)malloc(data_len * sizeof(float));
        for(int j = 0; j < data_len; j++) {
            fscanf(fp, "%f", &t[j]);
        }
        dataset[i++] = t;
    }

    return i;
}

float *X[DATA_LENGTH];
float *Y[DATA_LENGTH];
float *X_t[DATA_LENGTH];
float *Y_t[DATA_LENGTH];

struct dataPoints{
    float** X;
    float* Y;
    //number of samples
    int n;
    //dim of features
    int d;
};

dataPoints get_data(int batchSize, int n){
    const int d = DATA_DIM;
    dataPoints DP;
    DP.X = new float*[batchSize];
    DP.Y = new float[batchSize];
    DP.n = batchSize;
    DP.d = d;
    for(int i=0; i<batchSize; i++){
        int index = gen(rng);
        DP.X[i] = X[index];
        DP.Y[i] = Y[index][0];
    }
    return DP;
}

dataPoints get_data_test(int n){
    //cout<<"Geting test data..."<<endl;
    const int d = DATA_DIM;
    dataPoints DP;
    DP.X = new float*[n];
    DP.Y = new float[n];
    for(int i=0; i<n; i++){
        DP.X[i] = X_t[i];
        DP.Y[i] = Y_t[i][0];
    }
    DP.n = n;
    DP.d = d;
    return DP;
}

template <typename T>
float sgd_step(T* w, int batchSize, float lr, int n){
    float loss = 0;
    for(int i=0; i<batchSize; i++){
        int index = gen(rng);
	    //x*w and w*w
	    float x_times_w = 0;

	    for(int j=0; j<DATA_DIM; j++){
	       x_times_w += X[index][j]*w[j];
	    }

	    float y_predict = 1/(1+exp(-x_times_w));

        //loss += (-Y[index][0]*log(y_predict)-(1-Y[index][0])*log(1-y_predict))/batchSize;
        //loss += 0.5*REGULARIZATION*w_times_w;
        //cout<<"Loss "<<loss<<endl;
        
        float scala = lr*(Y[index][0]-y_predict)/batchSize;
        
	    for(int j=0; j<DATA_DIM; j++){
	        w[j] += X[index][j]*scala;
        }
        //cout<<endl;
    }
    //cout<<loss<<endl;
    return loss;
}

template <typename T>
void train(int maxround, int batchSize, int d, int n, T* w){
    //initialize w
    float loss_all = 0;
    for(int i=0; i<maxround; i++){
        loss_all += sgd_step(w,batchSize,LR,n);
        //cout<<loss_all/(i+1)<<endl;
    }
    //cout<<"Train End"<<endl;
}

template <typename T>
void test(T* w, int n){
    dataPoints DP = get_data_test(n);
    //cout<<"Test Data Got"<<endl;
    float y_predict[DP.n];
    int correct = 0;
    //cout<<DP.d<<" "<<DP.n<<endl;
    for(int i=0; i<DP.n; i++){
        float x_times_w = 0;
        for(int j=0; j<DP.d; j++){
            x_times_w += DP.X[i][j]*w[j];
            //cout<<DP.X[i][j]<<" ";
        }
        //cout<<endl;
        y_predict[i] = 1/(1+exp(-x_times_w));

        //cout<<"Predict and True: "<<y_predict[i]<<" "<<DP.Y[i]<<endl;
        if((y_predict[i]>0.5)==(DP.Y[i]>0.5))
            correct += 1;
    }
    //cout<<"What is the accuracy?"<<endl;
    float acc = float(correct)/DP.n;
    //cout<<"Acc: "<<acc<<endl;
    //cout<<acc<<endl;
}

int main(int argc, char** argv){

    int max_round = stoi(argv[1]);

    int n = read_dataset(Y, "train.label", 1);
    read_dataset(X, "train.out", DATA_DIM);
    float* w;
    w = new float[DATA_DIM];
    
    for(int i=0; i<DATA_DIM; i++){
        w[i] = 0;
    }

    rng = mt19937(100);
    gen = uniform_int_distribution<int>(0,n-1);
    
    auto start = high_resolution_clock::now();
    train(max_round, BATCH_SIZE, DATA_DIM, n, w);
    auto stop = high_resolution_clock::now();

    auto duration = duration_cast<microseconds>(stop - start); 
  
    //cout << "Time taken by training: " << duration.count() << " microseconds" << endl;
    cout<<duration.count()<<endl; 

/*
    cout<<"Weights: ";
    for(int i=0; i<DATA_DIM; i++)
        cout<<w[i]<<" ";
    cout <<endl;
*/
    int n_t = read_dataset(Y_t, "test.label", 1);
    read_dataset(X_t, "test.out", DATA_DIM);
    test(w, n_t);
    return 0;
}
