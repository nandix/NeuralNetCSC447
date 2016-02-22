#include "NeuralNet.cpp"
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <math.h>

using namespace std;

int main(int argc, char const *argv[])
{
	// Will be read in from a file in the final program

	NeuralNet net( argv[1] );

	int nLayers = net.numLayers;
	int row, column;
	int lastYearIndex;
	float errorThreshold = net.threshold;
	float epochThreshold = net.epochs;
	int nSamples;

	float inSize = net.monthsData + net.yearsBurned;
	float outSize = net.numOutputClasses;

	ifstream fin;
	ofstream fout;

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
	fin.close();

	vector<int> nPerLayer(nLayers);
	vector<vector< float > > data = net.readDataFile(net.trainingFilename);

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


	for (int i = 0; i < nSamples; i++)
	{
		column = net.endMonth;
		row = lastYearIndex;

		for (int j = 0; j < net.yearsBurned; j++)
			inputs[i][j] = data[lastYearIndex - i][0];

		for (int j = 1; j < inSize; j++)
		{
			inputs[i][j] = data[row][column];
			if (--column < 1)
			{
				column = 12;
				row--;
			}
		}
	}

	//net.printNetwork();

	float errorProp = 1.0;
	int epochNum = 0;

	vector<int> sampleIndicies(inputs.size());
	for(int i=0; i < inputs.size(); i++)
	{
		sampleIndicies[i] = i;
	}

	// While our network is not well trained and we haven't reached
	//	the maximum number of epochs...
	while( /*fabs(errorProp) > errorThreshold &&*/ epochNum < epochThreshold )
	{
		// cout << "Current error: " << errorProp << endl;
		// Begin another epoch!
		float totalError = 0.0;
		// Use the training data in random order
		vector<int> shuffledIndicies = sampleIndicies;
		random_shuffle( shuffledIndicies.begin(), shuffledIndicies.end() );

		// Output shuffled order of sample indicies
		// cout << "Shuffled indicies: " << endl;
		// for( int i=0; i < shuffledIndicies.size(); i++ )
		// 	cout << shuffledIndicies[i] << " " << sampleIndicies[i] << endl;
		// cout << endl;

		// Loop over the input data
		vector< vector<float> > results;

		vector< float > errors;

		for (int i = 0; i < inputs.size(); ++i)
		{
			// Easier to type than shuffledIndicies[i]...
			int curIndex =  shuffledIndicies[i];

			// cout << "Testing index " << curIndex << endl;
			// Run the training data
			results = net.evaluateNet( inputs[curIndex], outputs[curIndex] );


			// cout << "Resultss: " << endl;
			// for( int Q = 0; Q < results.size(); Q++ )
			// {
			// 	for( int P=0; P < results[Q].size(); P++)
			// 	{
			// 		cout << results[Q][P] << " ";
			// 	}
			// 	cout << endl;
			// }

			// cout << "Evaluated net" << endl;
			errors.resize( results[nLayers-1].size(), 0.0 );

			for(int outNode=0; outNode < results[nLayers-1].size(); outNode++)
			{
				float tempError = outputs[curIndex][outNode] - results[nLayers-1][outNode];
				errors[outNode] = tempError;
				totalError += tempError * tempError;

			}

			// cout << "Desired  Computed Error" << endl;
			// for( int outNode=0; outNode < errors.size(); outNode++)
			// {
				// cout << outputs[curIndex][outNode] << "  " << results[nLayers-1][outNode] << "  " << errors[outNode] << endl;
			// }

			// cout << "Computed errors" << endl;


			net.trainNetwork(errors, results);

			// cout << "Trained network" << endl;

		}

		errorProp = totalError / (results.size() * inputs.size());
		errorProp = sqrt( errorProp );
		//cout << epochNum << "  Error Proportion: " << errorProp << endl;
		if(epochNum%10 == 0)
		{
			cout << "Epoch" << setw(6) << epochNum << ": RMS Error = " << setprecision(3) << errorProp << endl;
		}

		// break;
		epochNum++;
	}

	//write out the weights
	fout.open(net.weightFilename.c_str());
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
				fout << net.networkWeights[ layer-1 ][ node ][ wNum ] << " ";
			}
		}
	}
	fout.close();

	/*for( int i=0; i < inputs.size(); i++ )
	{
		vector< vector<float> > results = net.evaluateNet( inputs[i], outputs[i] );
		cout << "In | Out | Net: " << inputs[i][0] << " " << inputs[i][1] << " | "
				<< outputs[i][0] << " | " << results[nLayers-1][0] << endl;

	}*/
	// for( int i=0; i < inputs.size(); i++ )
	// {
	// 	vector< vector<float> > results = net.evaluateNet( inputs[i], outputs[i] );
	// 	cout << "In | Out | Net: " << inputs[i][0] << " | "
	// 			<< outputs[i][0] << " | " << results[nLayers-1][0] << endl;

	// }

	//net.printNetwork();


	//cout << "The program actually finished" << endl;
	return 0;
}
