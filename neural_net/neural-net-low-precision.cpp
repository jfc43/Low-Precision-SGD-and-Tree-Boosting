#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>
#include <chrono>

using namespace std;
using namespace std::chrono;

#define INPUT_SCALE 10
#define WEIGHT_SCALE 1000

class DataReader
{
public:
	DataReader(const string filename);
	bool isEof(void)
	{
		return m_trainingDataFile.eof();
	}
    
	// Returns the number of input values read from the file:
	unsigned getNextInputs(vector<int16_t> &inputVals);
	unsigned getTargetOutputs(vector<double> &targetOutputVals);

private:
	ifstream m_trainingDataFile;
};

DataReader::DataReader(const string filename)
{
	m_trainingDataFile.open(filename.c_str());
}


unsigned DataReader::getNextInputs(vector<int16_t> &inputVals)
{
    inputVals.clear();

    string line;
    getline(m_trainingDataFile, line);
    stringstream ss(line);

    double oneValue;
    while (ss >> oneValue) {
	int16_t convertedValue = oneValue * INPUT_SCALE;
        inputVals.push_back(convertedValue);
    }

    return inputVals.size();
}

unsigned DataReader::getTargetOutputs(vector<double> &targetOutputVals)
{
    targetOutputVals.clear();

    string line;
    getline(m_trainingDataFile, line);
    stringstream ss(line);

    double oneValue;
    while (ss >> oneValue) {
        targetOutputVals.push_back(oneValue);
    }

    return targetOutputVals.size();
}

struct Connection
{
	double weight;
	double deltaWeight;
};

class Neuron;

typedef vector<Neuron> Layer;

// ****************** class Neuron ******************

class Neuron
{
public:
	Neuron(unsigned numOutputs, unsigned myIndex);
	void setOutputVal(double val) { m_outputVal = val; }
    void setRawOutputVal(double val) {m_rawOutputVal = val;}
	double getOutputVal(void) const { return m_outputVal; }
    double getRawOutputVal(void) const { return m_rawOutputVal; }
    double getGradient(void) const { return m_gradient; }	
    void feedForward(const Layer &prevLayer, bool isLastLayer);
	void calcOutputGradients(double outputVals, double targetVals);
	void calcHiddenGradients(const Layer &nextLayer);
	void updateInputWeights(Layer &prevLayer);
private:
	static double eta; // [0.0...1.0] overall net training rate
	static double alpha; // [0.0...n] multiplier of last weight change [momentum]
	static double transferFunction(double x);
	static double transferFunctionDerivative(double x);
	// randomWeight: 0 - 1
	static double randomWeight(void) { return rand() / double(RAND_MAX); }
	double sumDOW(const Layer &nextLayer) const;
    double m_rawOutputVal;
	double m_outputVal;
	vector<Connection> m_outputWeights;
	unsigned m_myIndex;
	double m_gradient;
};

double eta = 0.01;
double Neuron::eta = 0.001; // overall net learning rate
double Neuron::alpha = 0.01; // momentum, multiplier of last deltaWeight, [0.0..n]


void Neuron::updateInputWeights(Layer &prevLayer)
{
	// The weights to be updated are in the Connection container
	// in the nuerons in the preceding layer

	for(unsigned n = 0; n < prevLayer.size(); ++n)
	{
		Neuron &neuron = prevLayer[n];
		//double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;

		double newDeltaWeight =
				// Individual input, magnified by the gradient and train rate:
				eta
				* neuron.getOutputVal()
				* m_gradient;
				// Also add momentum = a fraction of the previous delta weight
				//+ alpha
				//* oldDeltaWeight;
		// neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
		neuron.m_outputWeights[m_myIndex].weight -= newDeltaWeight;
	}
}

double Neuron::sumDOW(const Layer &nextLayer) const
{
	double sum = 0.0;

	// Sum our contributions of the errors at the nodes we feed

	for (unsigned n = 0; n < nextLayer.size() - 1; ++n)
	{
		sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
	}

	return sum;
}

void Neuron::calcHiddenGradients(const Layer &nextLayer)
{
	double dow = sumDOW(nextLayer);
	m_gradient = dow * Neuron::transferFunctionDerivative(m_rawOutputVal);
}

void Neuron::calcOutputGradients(double outputVals, double targetVals)
{
    m_gradient = outputVals - targetVals;
}

double Neuron::transferFunction(double x)
{
	// tanh - output range [-1.0..1.0]
    if (x < 0) 
        return 0.0;
    else
	    return x;
}

double Neuron::transferFunctionDerivative(double x)
{
	// tanh derivative
    if (x < 0)
        return 0.0;
    else
	    return 1.0;
}

void Neuron::feedForward(const Layer &prevLayer, bool isLastLayer)
{
	double sum = 0;

	// Sum the previous layer's outputs (which are our inputs)
    // Include the bias node from the previous layer.

	for(unsigned n = 0 ; n < prevLayer.size(); ++n)
	{
		auto o = prevLayer[n].getOutputVal();
		auto w = prevLayer[n].m_outputWeights[m_myIndex].weight;
		sum += o * w;
	}
    
    m_rawOutputVal = sum;
    
    if (isLastLayer == false)
        m_outputVal= Neuron::transferFunction(m_rawOutputVal);
}

Neuron::Neuron(unsigned numOutputs, unsigned myIndex)
{
	for(unsigned c = 0; c < numOutputs; ++c){
		m_outputWeights.push_back(Connection());
		m_outputWeights.back().weight = randomWeight();
	}

	m_myIndex = myIndex;
}


// ****************** class Net ******************
class Net
{
public:
	Net(const vector<unsigned> &topology);
	void feedForward(const vector<int16_t> &inputVals);
	void backProp(const vector<double> &targetVals);
	void getResults(vector<double> &resultVals) const;
	double getRecentAverageError(void) const { return m_recentAverageError; }

private:
	vector<Layer> m_layers; //m_layers[layerNum][neuronNum]
    vector< vector<int16_t> > inputWeights;
    vector<int16_t> inputVals;
    vector<double> firstLayerOutputs;
    unsigned inputDim;
    unsigned m_numClass;
	double m_error;
	double m_recentAverageError;
	static double m_recentAverageSmoothingFactor;
};

double Net::m_recentAverageSmoothingFactor = 100.0; // Number of training samples to average over

void Net::getResults(vector<double> &resultVals) const
{
	resultVals.clear();

	for(unsigned n = 0; n < m_layers.back().size() - 1; ++n)
	{
		resultVals.push_back(m_layers.back()[n].getOutputVal());
	}
}

void Net::backProp(const std::vector<double> &targetVals)
{
	// Calculate overal net error

	Layer &outputLayer = m_layers.back();

    // compute cross entropy loss
    m_error = 0.0;
	for(unsigned n = 0; n < outputLayer.size() - 1; ++n)
	{
        double val = outputLayer[n].getOutputVal();
        val = max(val, 1e-12);
        m_error += -log(val) * targetVals[n];
	}

	// Implement a recent average measurement:

	m_recentAverageError = 
			(m_recentAverageError * m_recentAverageSmoothingFactor + m_error)
			/ (m_recentAverageSmoothingFactor + 1.0);
	// Calculate output layer gradients

	for(unsigned n = 0; n < outputLayer.size() - 1; ++n)
	{
		outputLayer[n].calcOutputGradients(outputLayer[n].getOutputVal(), targetVals[n]);
	}
	// Calculate gradients on hidden layers

	for(int layerNum = m_layers.size() - 2; layerNum >= 0; --layerNum)
	{
		Layer &hiddenLayer = m_layers[layerNum];
		Layer &nextLayer = m_layers[layerNum + 1];

		for(unsigned n = 0; n < hiddenLayer.size(); ++n)
		{
			hiddenLayer[n].calcHiddenGradients(nextLayer);
		}
	}

	// For all layers from outputs to first hidden layer,
	// update connection weights

	for(unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum)
	{
		Layer &layer = m_layers[layerNum];
		Layer &prevLayer = m_layers[layerNum - 1];

		for(unsigned n = 0; n < layer.size() - 1; ++n)
		{
			layer[n].updateInputWeights(prevLayer);
		}
	}
	
	// update input weights
    for (unsigned j = 0; j < m_layers[0].size() - 1; ++j) {
        for (unsigned i = 0; i < inputDim; ++i){
           double deltaWeight = eta * inputVals[i] / INPUT_SCALE * m_layers[0][j].getGradient();
            inputWeights[j][i] -= deltaWeight * WEIGHT_SCALE;
        }
    }
    
}

void Net::feedForward(const vector<int16_t> &input)
{
	// Check the num of inputVals euqal to neuronnum expect bias
	// assert(inputVals.size() == m_layers[0].size() - 1);

	inputVals = input;
        
	// Assign {latch} the input values into the input neurons
	for (unsigned j = 0; j < m_layers[0].size() - 1; ++j) {
        int16_t sum = 0;
        for(unsigned i = 0; i < inputDim; ++i){
            sum += inputVals[i] * inputWeights[j][i];
	    }
	    double converted_sum = double(sum) / INPUT_SCALE / WEIGHT_SCALE + 1;
        m_layers[0][j].setRawOutputVal(converted_sum);
        double after_activation = converted_sum;
        if (converted_sum < 0)
            after_activation = 0;
        m_layers[0][j].setOutputVal(after_activation);
    }

	// Forward propagate
	for(unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum){
	    Layer &prevLayer = m_layers[layerNum - 1];
            for(unsigned n = 0; n < m_layers[layerNum].size() - 1; ++n){
                if (layerNum == m_layers.size() - 1)
                    m_layers[layerNum][n].feedForward(prevLayer, true);
                else
	            m_layers[layerNum][n].feedForward(prevLayer, false);
	    }
	}
    
    // add stable softmax layer
    Layer &lastLayer = m_layers.back();
    double maxValue = lastLayer[0].getRawOutputVal();
    for (unsigned i = 1; i < m_numClass; ++i) {
        if (lastLayer[i].getRawOutputVal() > maxValue) {
            maxValue = lastLayer[i].getRawOutputVal();
        }
    }
   
    vector<double> exps(m_numClass);
    double s = 0.0;

    //cout << "final layer output: ";
    for (unsigned i = 0; i < m_numClass; ++i) {
    //    cout << lastLayer[i].getRawOutputVal() << " ";
        exps[i] = exp(lastLayer[i].getRawOutputVal() - maxValue);
        s += exps[i];
    }

    //cout << endl;

    //cout << "s: " << s << endl;
  
    for (unsigned i = 0; i < m_numClass; ++i) {
        lastLayer[i].setOutputVal(exps[i] / s);
    }
}

Net::Net(const vector<unsigned> &topology)
{
    inputDim = 28 * 28;
    int n = topology[0];
    for (int i = 0; i < n; ++i) {
        vector<int16_t> weights;
        for (int j = 0; j < inputDim; ++j) {
            weights.push_back(rand() / double(RAND_MAX) * WEIGHT_SCALE);
        }
        inputWeights.push_back(weights);
    }
        
	unsigned numLayers = topology.size();
        m_numClass = topology[numLayers - 1];
	for(unsigned layerNum = 0; layerNum < numLayers; ++layerNum){
		m_layers.push_back(Layer());
		// numOutputs of layer[i] is the numInputs of layer[i+1]
		// numOutputs of last layer is 0
		unsigned numOutputs = (layerNum == numLayers - 1) ? 0 :topology[layerNum + 1];

		// We have made a new Layer, now fill it ith neurons, and
		// add a bias neuron to the layer:
		for(unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum){
			m_layers.back().push_back(Neuron(numOutputs, neuronNum));
			// cout << "Mad a Neuron!" << endl;
		}

		// Force the bias node's output value to 1.0. It's the last neuron created above
		m_layers.back().back().setOutputVal(1.0);
	}
}

void showVectorVals(string label, vector<double> &v)
{
	cout << label << " ";
	for(unsigned i = 0; i < v.size(); ++i)
	{
		cout << v[i] << " ";
	}
	cout << endl;
}

int getPrediction(vector<double>& output) {
	int max_index = 0;
	double max_value = output[0];
	int n = output.size();
	for (int i = 1; i < n; ++i) {
		if (output[i] > max_value) {
			max_index = i;
			max_value = output[i];
		}
	}
	return max_index;
}

void train(Net& myNet, string data_path, int maxEpoch) {
    vector< vector<int16_t> > X;
    vector< vector<double> > Y;
    vector<int16_t> inputVals;
    vector<double> targetVals, resultVals;
    
    cout << "Reading training data..." << endl; 

    DataReader trainData(data_path);
    int num_examples = 0;
    while (!trainData.isEof()) {
        trainData.getNextInputs(inputVals);
        if (inputVals.size() == 0)
            break;
        trainData.getTargetOutputs(targetVals);
        X.push_back(inputVals);
        Y.push_back(targetVals);
        num_examples += 1;
    }

    cout << "Finish reading data!" << endl;

    auto start = high_resolution_clock::now();
	for (int i = 0; i < maxEpoch; ++i)
	{
        int step = 0;

        for (int j = 0; j < num_examples; ++j) {
			// Get new input data and feed it forward:
            inputVals = X[j];
            targetVals = Y[j];

			// showVectorVals(": Inputs :", inputVals);
			myNet.feedForward(inputVals);

			// Collect the net's actual results:
			myNet.getResults(resultVals);
			// showVectorVals("Outputs:", resultVals);

			// Train the net what the outputs should have been:
			// showVectorVals("Targets:", targetVals);
			// assert(targetVals.size() == topology.back());

			myNet.backProp(targetVals);
            
            step += 1;
            //if (step % 1000 == 0) {
            //    cout << "step: " << step << ", average error: " << myNet.getRecentAverageError() << endl;
            //}

		}
		// Report how well the training is working, average over recnet
		cout << "epoch: " << i + 1 << ", average error: "
		     << myNet.getRecentAverageError() << endl;
	}
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Time taken by training: " << duration.count() << " microseconds" << endl;
}

void test(Net& myNet, string data_path) {
    vector< vector<int16_t> > X;
    vector< vector<double> > Y;
    vector<int16_t> inputVals;
    vector<double> targetVals, resultVals;

    cout << "Reading test data..." << endl;
    DataReader testData(data_path);
    int num_examples = 0;
    while (!testData.isEof()) {
        testData.getNextInputs(inputVals);
        if (inputVals.size() == 0)
            break;
        testData.getTargetOutputs(targetVals);
        X.push_back(inputVals);
        Y.push_back(targetVals);
        num_examples += 1;
    }
    cout << "Finish reading data!" << endl;
    
    double num_succeed = 0.0;
    for (int i = 0; i < num_examples; ++i) {
		// Get new input data and feed it forward:
        inputVals = X[i];
        targetVals = Y[i];
        
		// showVectorVals(": Inputs :", inputVals);
		myNet.feedForward(inputVals);

		// Collect the net's actual results:
		myNet.getResults(resultVals);
		// showVectorVals("Outputs:", resultVals);

		// Train the net what the outputs should have been:
		// showVectorVals("Targets:", targetVals);

		int pred = getPrediction(resultVals);
		int label = getPrediction(targetVals);
		// cout << "pred: " << pred << "label: " << label << endl;
        if (pred == label) {
			num_succeed += 1;
		}

	}
	cout << "accuracy: " << num_succeed / num_examples << endl;
}

int main()
{
	vector<unsigned> topology;
    // topology.push_back(28 * 28);
    topology.push_back(128);
    topology.push_back(64);
	topology.push_back(10);
    
	Net myNet(topology);

	int maxEpoch = 100;
	
	train(myNet, "train.txt", maxEpoch);
    test(myNet, "test.txt");

    return 0;
}
