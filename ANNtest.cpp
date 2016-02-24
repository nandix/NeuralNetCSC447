#include "NeuralNet.cpp"
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <math.h>

using namespace std;

/*******************************************************************************
* Function: main()
*
* Description: This file tests the neural net class using the parameters
* 	specified in the parameter file. The parameter file is passed as the first
* 	argument on the command line. This function starts by initializing the
* 	neural net class with the name of the parameter file. This file is then read
* 	and it populates the class variables of the neural net. After that, it
* 	builds a 2 dimensional vector of normalized data points. The size of the
* 	rows are defined in the parameter file and the number of rows is based on
* 	the size of the input file. After that, the input rows are fed into the
* 	neural net one at a time. Then the net is tested for accuracy. This process
* 	is repeated until the net is more accurate than the threshhold defined in
* 	the parameter file, or until it has reached the number of epochs defined in
* 	the parameter file.
*
* Parameters:
* 	argc - The number of arguments passed in on the command line
* 	argv - An array of the command line arguments
* 		argv[0] - The name of the program
* 		argv[1] - The name of the parameter file
*
*
*******************************************************************************/
int main(int argc, char const *argv[])
{
	// Will be read in from a file in the final program
	if( argc != 2)
	{
		cout << "Usage: ./ANNtest params.prm\n";
		return -2;
	}

	NeuralNet net( argv[1] );

	int nLayers = net.numLayers;
	int row, column;
	int lastYearIndex;
	int nSamples;

	//Size of the input and output layers
	float inSize = net.monthsData + net.yearsBurned;
	float outSize = net.numOutputClasses;

	ifstream fin;

	float weight=0;
	fin.open(net.weightFilename.c_str());
	if(fin.good())
	{
		// For each layer beyond the input layer...
		for( int layer = 1; layer < net.numLayers; layer++ )
		{
			// Resize the layer for the correct number of nodes.
			net.networkWeights[ layer-1 ].resize( net.nodesPerLayer[layer] + 1 );

			// Initialize for each node in that layer
			int node;
			for( node = 0; node < net.nodesPerLayer[layer]; node++ )
			{
				// Initialize the correct number of input weights, plus 1 for the bias input
				net.networkWeights[ layer-1 ][ node ].resize( net.nodesPerLayer[layer-1] + 1 );

				// For each node in the layer, initialize the input weights
				for( int wNum = 0; wNum <= net.nodesPerLayer[layer-1]; wNum++ )
				{
					// Initalize to small random values
					fin >> weight;
					net.networkWeights[ layer-1 ][ node ][ wNum ] = weight;
				}
			}
		}
	}
	else
	{
		cout << "Could not find weights file.  Please make sure the weights file exists." << endl;
		fin.close();
		return -1;
	}
	fin.close();

	vector<int> nPerLayer(nLayers);
	vector<vector< float > > data = net.readDataFile(net.trainingFilename);

	//Get the number of possible samples
	int yearsPerSample = ceil(net.monthsData / 12.0);
	if (net.endMonth != 12)
		yearsPerSample++;

	if (yearsPerSample >= net.yearsBurned)
		nSamples = data.size() - yearsPerSample + 1;
	else
		nSamples = data.size() - net.yearsBurned + 1;


	vector<vector< float > > inputs(nSamples, vector<float>(inSize));
	vector<vector< float > > outputs(nSamples, vector<float>(outSize));

	lastYearIndex = data.size() - 1;
	row = lastYearIndex;

	//Populate the input vectors
	for (int i = 0; i < nSamples; i++)
	{

		column = net.endMonth;
		row = lastYearIndex - i;

		for (int j = 0; j < net.yearsBurned; j++)
		{
			inputs[i][j] = data[lastYearIndex - i - j - 1][0];

			float unNormalizeConst = (net.burnMax - net.burnMin);
			if( inputs[i][j]*unNormalizeConst + net.burnMin < net.mediumCutoff )
			{
				outputs[i][0] = 1;
			}
			else if( inputs[i][j]*unNormalizeConst + net.burnMin < net.highCutoff )
			{
				outputs[i][1] = 1;
			}
			else
			{
				outputs[i][2] = 1;
			}
		}

		for (int j = net.yearsBurned ; j < inSize; j++)
		{
			inputs[i][j] = data[row][column];

			if (--column < 1)
			{
				column = 12;
				row--;
			}
		}
	}

	vector<int> sampleIndicies(inputs.size());
	for(int i=0; i < inputs.size(); i++)
	{
		sampleIndicies[i] = i;
	}

	float numWrong = 0;
	for( int i=0; i < inputs.size(); i++ )
	{
		cout << i+1 << ",";
		vector< vector<float> > results = net.evaluateNet( inputs[i], outputs[i] );

		int predictedIndex = -1;
		float maxPrediction = 0;
		vector<int> firePrediction;
		for( int j=0; j < results[nLayers-1].size(); j++ )
		{
			if( results[nLayers-1][j] > maxPrediction )
			{
				predictedIndex = j;
				maxPrediction = results[nLayers-1][j];
			}

		}

		for( int j=0; j < results[nLayers-1].size(); j++ )
		{
			if( j != predictedIndex )
			{
				firePrediction.push_back(0);
			}
			else
			{
				firePrediction.push_back(1);
			}
		}
		
		for( int j=0; j < outputs[i].size(); j++ )
		{
			cout  <<  outputs[i][j];
		}

		cout << ",";
		for( int j=0; j < firePrediction.size(); j++ )
		{
			cout << firePrediction[j];
		}

		for( int j=0; j < firePrediction.size(); j++ )
		{
			if( firePrediction[j] != outputs[i][j])
			{
				cout << ",*";
				numWrong ++;
				break;
			}
		}

		cout << endl;
	}

	cout << "\naccuracy: " << float(nSamples - numWrong)/nSamples *100
			<< " %" << endl;

	return 0;
}
