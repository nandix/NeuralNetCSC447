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
	if( argc != 2 )
	{
		cout << "Usage: ./CrossValidate params.prm\n";
		return -2;
	}

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


	// cout << "DATA: " << endl;
	// for( int i=0; i < data.size(); i++ )
	// {
	// 	for( int j=0; j < data[i].size(); j++ )
	// 	{
	// 		cout << " " << data[i][j];
	// 	}
	// 	cout << endl;
	// }
	// cout << endl;


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

	float errorProp = 1.0;
	int epochNum = 0;

	vector<int> sampleIndicies(inputs.size());
	for(int i=0; i < inputs.size(); i++)
	{
		sampleIndicies[i] = i;
	}

	float numWrong = 0;
	cout << "Year,Burned,Severity,Predicted (training error)" << endl;

	for( int elimNum = 0; elimNum < inputs.size(); elimNum++ )
	{
		vector<int> shuffledIndicies = sampleIndicies;
		shuffledIndicies.erase(shuffledIndicies.begin()+elimNum);

		// While our network is not well trained and we haven't reached
		//	the maximum number of epochs...
		vector< vector<float> > results;

		// Reconstruct the net for the next cross validation test
		net = NeuralNet(argv[1]);

		epochNum = 0;
		errorProp = 1.0;

		while( fabs(errorProp) > errorThreshold && epochNum < epochThreshold )
		{
			// Begin another epoch!
			float totalError = 0.0;
			// Use the training data in random order
			random_shuffle( shuffledIndicies.begin(), shuffledIndicies.end() );

			// Output shuffled order of sample indicies

			// Loop over the input data

			vector< float > errors;

			for (int i = 0; i < shuffledIndicies.size(); ++i)
			{
				// Easier to type than shuffledIndicies[i]...
				int curIndex =  shuffledIndicies[i];

				// Run the training data
				results = net.evaluateNet( inputs[curIndex], outputs[curIndex] );


				errors.resize( results[nLayers-1].size(), 0.0 );

				for(int outNode=0; outNode < results[nLayers-1].size(); outNode++)
				{
					float tempError = outputs[curIndex][outNode] - results[nLayers-1][outNode];
					errors[outNode] = tempError;
					totalError += tempError * tempError;

				}

				net.trainNetwork(errors, results);
			}

			errorProp = totalError / (results.size() * inputs.size());
			errorProp = sqrt( errorProp );

			epochNum++;
		}


		vector< vector<float> > testResults = net.evaluateNet( inputs[elimNum], outputs[elimNum] );

		int predictedIndex = -1;
		float maxPrediction = 0;
		vector<int> firePrediction;
		for( int j=0; j < testResults[nLayers-1].size(); j++ )
		{
			if( testResults[nLayers-1][j] > maxPrediction )
			{
				predictedIndex = j;
				maxPrediction = testResults[nLayers-1][j];
			}

		}

		for( int j=0; j < testResults[nLayers-1].size(); j++ )
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
		
		cout << net.years[elimNum+2] << "," << setw(6) << ceil(net.burnedAcreage[elimNum+2]) << ", ";
		for( int j=0; j < outputs[elimNum].size(); j++ )
		{
			cout  <<  outputs[elimNum][j];

		}

		cout << ", ";
		for( int j=0; j < firePrediction.size(); j++ )
		{
			cout << firePrediction[j];
		}

		bool wrong = false;
		for( int j=0; j < firePrediction.size(); j++ )
		{
			if( firePrediction[j] != outputs[elimNum][j])
			{
				wrong = true;
				cout << ", *";
				numWrong ++;
				break;
			}
		}

		if(wrong)
		{
			printf("%9c%0.3f)\n",'(',errorProp);
		}
		else
		{
			printf("%12c%0.3f)\n",'(',errorProp);
		}
	}

	cout << "\nOverall accuracy: " << float(nSamples - numWrong)/nSamples *100
			<< " %" << endl;
	return 0;
}
