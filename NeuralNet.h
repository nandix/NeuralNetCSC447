#ifndef NEURAL_NET
#define NEURAL_NET

#include <vector>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <string>
#include <fstream>

using namespace std;

class NeuralNet
{
public:
	//Constructors
	NeuralNet( int nLayers, vector<int> nPerLayer );
	NeuralNet( const char* fileName );
	//Destructor
	~NeuralNet();

	void printNetwork(); //Outputs the network and some statistics
	void trainNetwork(vector<float> errors, vector< vector<float> > results);
	void readParameters( string filename );
	vector< vector<float> > readDataFile(string dataFilename);
	vector< vector<float> > evaluateNet( vector< float > inputs, vector< float > outputs );


	vector<int> nodesPerLayer; // Number of nodes in each layer of the network
	int numLayers; // Number of layers in the net
	string trainingFilename;	//file containing training and testing data

	vector< float > inputLayer;

	// Vector to hold weights for network indexed as:
	// 	networkWeights[ layer index ][ node index i ][ weight from previous node i this, j ]
	vector<vector<vector< float > > > networkWeights; // Lists of nodes in each layer

	float initRange; // Initial weights range from -initRange/2 to +initRange/2
	float steepness; // Steepness of the sigmoid transfer function
	float learningRate;
	float momentum;

	//Variables for reading from the parameter file
	vector< vector<float> > data;  //2d vector holding data file. oldest dates at element 0
	vector<string> lines;		//vector to old all the lines to easily remove the useless ones
	string weightFilename;		//filename for storing weights
	int epochs;					//number of training epochs
	float threshold;			//error threshhold
	int yearsBurned;			//number of previous years to use for burned acreage
	int monthsData;				//how many previous months data to use
	int endMonth;				//the last usable month of the current year
	int numOutputClasses;		//number of output classes
	float mediumCutoff;			//minimum burned acreage to be considered medium
	float highCutoff;			//minimum burned acreage to be considered high
	vector<float> years;		//a vector containing all the years from the data file.
	vector<float> burnedAcreage; //vector of all burned acreages

	float burnMin;
	float burnMax;


private:
	int main(int argc, char** argv);
	float initWeight();
	float activationFunction( float x );
	float activationFunctionPrime( float x );
	vector<std::string> split(const string &text, char sep);
	void updateWeights();
};

#endif
