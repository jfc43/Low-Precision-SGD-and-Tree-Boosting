#include <iostream>
#include <math.h>
#include <random>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <string>

#define DATA_LENGTH 20000
#define DATA_DIM 784
#define TRAIN_ROUND 2000
#define REGULARIZATION 0
#define W_SCALE_FACTOR 10000
#define X_SCALE_FACTOR 10000
#define LEARNING_RATE 0.01
#define BATCH_SIZE 1
using namespace std;
using namespace std::chrono;

mt19937 rng;
uniform_int_distribution<int> gen;

int read_dataset(int16_t *dataset[], char *file_path, int data_len, int scale)
{
    FILE *fp = fopen(file_path, "r");

    int i = 0;
    while(!feof(fp)) {
        int16_t* t = (int16_t *)malloc(data_len * sizeof(int16_t));
        for(int j = 0; j < data_len; j++) {
            float temp;
            fscanf(fp, "%f", &temp);
            t[j] = int16_t(temp*scale);
        }
        dataset[i++] = t;
    }

    return i;
}

int16_t *X[DATA_LENGTH];
int16_t *Y[DATA_LENGTH];
int16_t *X_t[DATA_LENGTH];
int16_t *Y_t[DATA_LENGTH];

struct dataPoints{
    int16_t** X;
    int16_t* Y;
    //number of samples
    int n;
    //dim of features
    int d;
};


dataPoints get_data_test(int n){
    //cout<<"Geting test data..."<<endl;
    const int d = DATA_DIM;
    dataPoints DP;
    DP.X = new int16_t*[n];
    DP.Y = new int16_t[n];
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
        int32_t x_times_w_low = 0;

	    for(int j=0; j<DATA_DIM; j++){
	       x_times_w_low += X[index][j]*w[j];
	    }

        //cout<<"x_times_w_low: "<<x_times_w_low<<endl;

        x_times_w = float(x_times_w_low)/(W_SCALE_FACTOR*X_SCALE_FACTOR);

	    float y_predict = 1/(1+exp(-x_times_w));
        //float y_predict = 0.5;

        //cout<<"y_predict: "<<y_predict<<" y: "<< Y[index][0]<<endl;

        //loss += (-(float)Y[index][0]*log(y_predict)-(1-(float)Y[index][0])*log(1-y_predict))/batchSize;
        //loss += 0.5*REGULARIZATION*w_times_w;
        
        float scala = (lr*(Y[index][0]-y_predict))/batchSize*W_SCALE_FACTOR/X_SCALE_FACTOR;

	    for(int j=0; j<DATA_DIM; j++){
            //cout<<"Scala is "<<scala<<endl;
            //cout<<"Delta w is "<<W_SCALE_FACTOR*scala*DP.X[i][j]/X_SCALE_FACTOR<<endl;
	        w[j] += scala*X[index][j];
        }
    }
    //cout<<loss<<endl;
    return loss;
}

template <typename T>
void train(int maxround, int batchSize, int d, int n, T* w){
    //initialize
    float loss_all = 0;
    for(int i=0; i<maxround; i++){
        loss_all += sgd_step(w,batchSize,LEARNING_RATE,n);
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
        x_times_w = x_times_w/W_SCALE_FACTOR/X_SCALE_FACTOR;
        //cout<<endl;
        y_predict[i] = 1/(1+exp(-x_times_w));

        //cout<<"Predict and True: "<<y_predict[i]<<" "<<DP.Y[i]<<endl;
        if((y_predict[i]>=0.5)==(DP.Y[i]>=0.5))
            correct += 1;
    }
    //cout<<"What is the accuracy?"<<endl;
    float acc = float(correct)/DP.n;
    //cout<<"Acc: "<<acc<<endl;
    //cout<<acc<<endl;
}

int main(int argc, char** argv){

    int max_round = stoi(argv[1]);

    int n = read_dataset(Y, "train.label", 1, 1);
    read_dataset(X, "train.out", DATA_DIM, X_SCALE_FACTOR);
    int16_t* w;
    w = new int16_t[DATA_DIM];

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
        cout<<static_cast<int16_t>(w[i])<<" ";
    cout <<endl;
*/

    int n_t = read_dataset(Y_t, "test.label", 1, 1);
    read_dataset(X_t, "test.out", DATA_DIM, X_SCALE_FACTOR);
    test(w, n_t);
    return 0;
}
