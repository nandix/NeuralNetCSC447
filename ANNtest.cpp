#include "NeuralNet.cpp"
#include <algorithm>
#include <iostream>

using namespace std;

int main(int argc, char const *argv[])
{
	// Will be read in from a file in the final program

	NeuralNet net( argv[1] );

	int nLayers = net.numLayers;
	float errorThreshold = net.threshold;
	float epochThreshold = net.epochs;

	float nSamples = net.nodesPerLayer[0];
	float inSize = net.monthsData;
	float outSize = net.numOutputClasses;

	vector<int> nPerLayer(nLayers);
	vector<vector< float > > inputs(nSamples, vector<float>(inSize));
	vector<vector< float > > outputs(nSamples, vector<float>(outSize));

	vector<vector< float > > data = net.readDataFile(net.trainingFilename);

	if (nSamples > data.size())
	{
		cout << "Number of samples invalid in parameter file. Using max samples"
			<< endl;
		nSamples = data.size() - 1;
	}
	for (int i = 0; i < nSamples; i++)
	{
		for (int j = 0; j < inSize; j++)
		{
			int row = (int)(j / data[i].size()) + i;
			int column = j % data[row].size();
			inputs[i][j] = data[row][column];
		}
	}

	net.printNetwork();

	float errorProp = 1.0;
	int epochNum = 0;

	vector<int> sampleIndicies(inputs.size());
	for(int i=0; i < inputs.size(); i++)
	{
		sampleIndicies[i] = i;
	}

	// While our network is not well trained and we haven't reached
	//	the maximum number of epochs...
	while( fabs(errorProp) > errorThreshold && epochNum < epochThreshold )
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
		cout << epochNum << "  Error Proportion: " << errorProp << endl;

		// break;
		epochNum++;
	}

	for( int i=0; i < inputs.size(); i++ )
	{
		vector< vector<float> > results = net.evaluateNet( inputs[i], outputs[i] );
		cout << "In | Out | Net: " << inputs[i][0] << " " << inputs[i][1] << " | "
				<< outputs[i][0] << " | " << results[nLayers-1][0] << endl;

	}
	// for( int i=0; i < inputs.size(); i++ )
	// {
	// 	vector< vector<float> > results = net.evaluateNet( inputs[i], outputs[i] );
	// 	cout << "In | Out | Net: " << inputs[i][0] << " | "
	// 			<< outputs[i][0] << " | " << results[nLayers-1][0] << endl;

	// }

	net.printNetwork();


	cout << "The program actually finished" << endl;
	return 0;
}
