#include "NeuralNet.cpp"
#include <algorithm>
#include <iostream>
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

	NeuralNet net( argv[1] );

	int nLayers = net.numLayers;
	int row, column;
	int lastYearIndex;
	float errorThreshold = net.threshold;
	float epochThreshold = net.epochs;
	int nSamples;

	//Size of the input and output layers
	float inSize = net.monthsData + net.yearsBurned;
	float outSize = net.numOutputClasses;

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

	//Populate the input vectors
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
