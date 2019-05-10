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

typedef double ourType;

class DataReader
{
public:
	DataReader(const string filename);
	bool isEof(void)
	{
		return m_trainingDataFile.eof();
	}
    
	// Returns the number of input values read from the file:
	unsigned getNextInputs(vector<ourType> &inputVals);
	unsigned getTargetOutputs(vector<ourType> &targetOutputVals);

private:
	ifstream m_trainingDataFile;
};

DataReader::DataReader(const string filename)
{
	m_trainingDataFile.open(filename.c_str());
}


unsigned DataReader::getNextInputs(vector<ourType> &inputVals)
{
    inputVals.clear();

    string line;
    getline(m_trainingDataFile, line);
    stringstream ss(line);

    ourType oneValue;
    while (ss >> oneValue) {
        inputVals.push_back(oneValue);
    }

    return inputVals.size();
}

unsigned DataReader::getTargetOutputs(vector<ourType> &targetOutputVals)
{
    targetOutputVals.clear();

    string line;
    getline(m_trainingDataFile, line);
    stringstream ss(line);

    ourType oneValue;
    while (ss >> oneValue) {
        targetOutputVals.push_back(oneValue);
    }

    return targetOutputVals.size();
}

struct Connection
{
	ourType weight;
	ourType deltaWeight;
};

class Neuron;

typedef vector<Neuron> Layer;

// ****************** class Neuron ******************

class Neuron
{
public:
	Neuron(unsigned numOutputs, unsigned myIndex);
	void setOutputVal(ourType val) { m_outputVal = val; }
    void setRawOutputVal(ourType val) {m_rawOutputVal = val;}
	ourType getOutputVal(void) const { return m_outputVal; }
    ourType getRawOutputVal(void) const { return m_rawOutputVal; }
	void feedForward(const Layer &prevLayer, bool isLastLayer);
	void calcOutputGradients(ourType outputVals, ourType targetVals);
	void calcHiddenGradients(const Layer &nextLayer);
	void updateInputWeights(Layer &prevLayer);
private:
	static ourType eta; // [0.0...1.0] overall net training rate
	static ourType alpha; // [0.0...n] multiplier of last weight change [momentum]
	static ourType transferFunction(ourType x);
	static ourType transferFunctionDerivative(ourType x);
	// randomWeight: 0 - 1
	static ourType randomWeight(void) { return rand() / ourType(RAND_MAX); }
	ourType sumDOW(const Layer &nextLayer) const;
    ourType m_rawOutputVal;
	ourType m_outputVal;
	vector<Connection> m_outputWeights;
	unsigned m_myIndex;
	ourType m_gradient;
};

ourType Neuron::eta = 0.0001; // overall net learning rate
ourType Neuron::alpha = 0.0001; // momentum, multiplier of last deltaWeight, [0.0..n]


void Neuron::updateInputWeights(Layer &prevLayer)
{
	// The weights to be updated are in the Connection container
	// in the nuerons in the preceding layer

	for(unsigned n = 0; n < prevLayer.size(); ++n)
	{
		Neuron &neuron = prevLayer[n];
		//ourType oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;

		ourType newDeltaWeight =
				// Individual input, magnified by the gradient and train rate:
				eta
				* neuron.getOutputVal()
				* m_gradient;
				// Also add momentum = a fraction of the previous delta weight
				//+ alpha
				//* oldDeltaWeight;
		//neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
		neuron.m_outputWeights[m_myIndex].weight -= newDeltaWeight;
	}
}
ourType Neuron::sumDOW(const Layer &nextLayer) const
{
	ourType sum = 0.0;

	// Sum our contributions of the errors at the nodes we feed

	for (unsigned n = 0; n < nextLayer.size() - 1; ++n)
	{
		sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
	}

	return sum;
}

void Neuron::calcHiddenGradients(const Layer &nextLayer)
{
	ourType dow = sumDOW(nextLayer);
	m_gradient = dow * Neuron::transferFunctionDerivative(m_rawOutputVal);
}

void Neuron::calcOutputGradients(ourType outputVals, ourType targetVals)
{
    m_gradient = outputVals - targetVals;
}

ourType Neuron::transferFunction(ourType x)
{
	// tanh - output range [-1.0..1.0]
    if (x < 0) 
        return 0.0;
    else
	    return x;
}

ourType Neuron::transferFunctionDerivative(ourType x)
{
	// tanh derivative
    if (x < 0)
        return 0.0;
    else
	    return 1.0;
}

void Neuron::feedForward(const Layer &prevLayer, bool isLastLayer)
{
	ourType sum = 0.0;

	// Sum the previous layer's outputs (which are our inputs)
    // Include the bias node from the previous layer.

	for(unsigned n = 0 ; n < prevLayer.size(); ++n)
	{
		sum += prevLayer[n].getOutputVal() * 
				 prevLayer[n].m_outputWeights[m_myIndex].weight;
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
	void feedForward(const vector<ourType> &inputVals);
	void backProp(const vector<ourType> &targetVals);
	void getResults(vector<ourType> &resultVals) const;
	ourType getRecentAverageError(void) const { return m_recentAverageError; }

private:
	vector<Layer> m_layers; //m_layers[layerNum][neuronNum]
    unsigned m_numClass;
	ourType m_error;
	ourType m_recentAverageError;
	static ourType m_recentAverageSmoothingFactor;
};

ourType Net::m_recentAverageSmoothingFactor = 100.0; // Number of training samples to average over

void Net::getResults(vector<ourType> &resultVals) const
{
	resultVals.clear();

	for(unsigned n = 0; n < m_layers.back().size() - 1; ++n)
	{
		resultVals.push_back(m_layers.back()[n].getOutputVal());
	}
}

void Net::backProp(const std::vector<ourType> &targetVals)
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

	for(unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum)
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
}

void Net::feedForward(const vector<ourType> &inputVals)
{
	// Check the num of inputVals euqal to neuronnum expect bias
	assert(inputVals.size() == m_layers[0].size() - 1);

	// Assign {latch} the input values into the input neurons
	for(unsigned i = 0; i < inputVals.size(); ++i){
		m_layers[0][i].setOutputVal(inputVals[i]); 
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
    ourType maxValue = lastLayer[0].getRawOutputVal();
    for (unsigned i = 1; i < m_numClass; ++i) {
        if (lastLayer[i].getRawOutputVal() > maxValue) {
            maxValue = lastLayer[i].getRawOutputVal();
        }
    }
   
    vector<ourType> exps(m_numClass);
    ourType s = 0.0;

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

void showVectorVals(string label, vector<ourType> &v)
{
	cout << label << " ";
	for(unsigned i = 0; i < v.size(); ++i)
	{
		cout << v[i] << " ";
	}
	cout << endl;
}

int getPrediction(vector<ourType>& output) {
	int max_index = 0;
	ourType max_value = output[0];
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
    vector< vector<ourType> > X, Y;
	vector<ourType> inputVals, targetVals, resultVals;
    
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
    vector< vector<ourType> > X, Y;
	vector<ourType> inputVals, targetVals, resultVals;

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
    
	ourType num_succeed = 0.0;
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
    topology.push_back(28 * 28);
    topology.push_back(128);
    topology.push_back(64);
	topology.push_back(10);
    
	Net myNet(topology);

	int maxEpoch = 100;
	
	train(myNet, "train.txt", maxEpoch);
    test(myNet, "test.txt");

	return 0;
}
