
/*
	Program: XORNet.cpp
*/
#include "NeuralNet.cpp"
#include <iostream>

using namespace std;

int main(int argc, char const *argv[])
{
	// Will be read in from a file in the final program
	int nLayers = 3;
	float errorThreshold = 0.01;
	float epochThreshold = 10000;

	float nSamples = 4;
	float inSize = 2;
	float outSize = 1;

	vector<int> nPerLayer(nLayers);
	nPerLayer[0] = inSize;
	nPerLayer[1] = 5;
	nPerLayer[2] = outSize;
	
	NeuralNet net( nLayers, nPerLayer );


    vector<vector< float > > inputs(nSamples, vector<float>(inSize) );
	vector<vector< float > > outputs(nSamples, vector<float>(outSize) );
	inputs[0][0] = 0;
	inputs[0][1] = 0;
	inputs[1][0] = 0;
	inputs[1][1] = 1;
	inputs[2][0] = 1;
	inputs[2][1] = 0;
	inputs[3][0] = 1;
	inputs[3][1] = 1;

	outputs[0][0] = 0;
	outputs[1][0] = 1;
	outputs[2][0] = 1;
	outputs[3][0] = 0;


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