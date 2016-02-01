#ifndef NEURAL_NET
#define NEURAL_NET

#include <vector>
#include <iostream>
#include <stdlib.h>
#include <time.h>

using namespace std;

class NeuralNet
{
public:
	NeuralNet( int nLayers, vector<int> nPerLayer );
	~NeuralNet();

	void printNetwork();
	void trainNetwork(vector<float> errors, vector< vector<float> > results);
	vector< vector<float> > evaluateNet( vector< float > inputs, vector< float > outputs );
	
private:
	int numLayers; // Number of layers in the net
	vector<int> nodesPerLayer; // Number of nodes in each layer of the network

	vector< float > inputLayer;

	// Vector to hold weights for network indexed as:
	// 	networkWeights[ layer index ][ node index i ][ weight from previous node i this, j ] 
	vector<vector<vector< float > > > networkWeights; // Lists of nodes in each layer

	float initRange; // Initial weights range from -initRange/2 to +initRange/2
	float steepness; // Steepness of the sigmoid transfer function
	float learningRate;
	float momentum;

	float initWeight();
	float activationFunction( float x );
	float activationFunctionPrime( float x );
	void updateWeights();
};

#endif