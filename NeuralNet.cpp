
#include "NeuralNet.h"
#include <cmath>

NeuralNet::NeuralNet( int nLayers, vector<int> nPerLayer )
{
	// Seed our random number generator
    srand( time(NULL) );
	initRange = 1.0;
	steepness = 1.0;
	learningRate = 0.4;
	momentum = 0.8;

	numLayers = nLayers;
	nodesPerLayer = nPerLayer;

	// Initialize the input layer
	inputLayer.resize( nPerLayer[0] );


	// Create the appropriate number of layers
	//networkWeights = new vector<vector<vector<float> > > () ;

	// Create the correct number of layer weights
	networkWeights.resize( numLayers-1 );

	// For each layer beyond the input layer...
	for( int layer = 1; layer < numLayers; layer++ )
	{
		// Resize the layer for the correct number of nodes.
		networkWeights[ layer-1 ].resize( nPerLayer[layer] + 1 );

		// Initialize for each node in that layer
		int node;
		for( node = 0; node < nodesPerLayer[layer]; node++ )
		{
			// Initialize the correct number of input weights, plus 1 for the bias input
			networkWeights[ layer-1 ][ node ].resize( nodesPerLayer[layer-1] + 1 );

			// For each node in the layer, initialize the input weights
			for( int wNum = 0; wNum <= nodesPerLayer[layer-1]; wNum++ )
			{
				// Initalize to small random values
				networkWeights[ layer-1 ][ node ][ wNum ] = initWeight();
			}
		}
	}

}

NeuralNet::~NeuralNet()
{

}

void NeuralNet::printNetwork()
{

	cout << "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&" << endl;

	cout << "N Layers: " << numLayers << endl;
	cout << "Nodes per layer: " << endl;
	for( int i=0; i < numLayers; i++ ){
		cout << "Layer " << i << ": " << nodesPerLayer[i] << endl;
	}


	for( int i=0; i < numLayers-1; i++ )
	{
		//cout << "*";
		for( int j=0; j < nodesPerLayer[i+1]; j++ )
		{
			//cout << "#";
			for( int k=0; k < networkWeights[i][j].size(); k++ )
			{
				//cout << "&";
				cout << networkWeights[i][j][k] << " ";
			}
			cout << endl;
		}

		cout << "-------------------------------------------" << endl;
	}

}

// Initialize a weight to a small value
float NeuralNet::initWeight()
{
	float randVal = float(rand())/RAND_MAX - 0.5;
	randVal = randVal * initRange;
	return randVal;
}

vector< vector<float> > NeuralNet::evaluateNet( vector< float > inputs, vector< float > outputs )
{

	// Make our input vector the old output vector, for consistency in the loop 
	vector<float> tempOut = inputs;

	// For each layer
	for (int i = 0; i < numLayers-1; ++i)
	{
		// Create temporary output values
		vector<float> tempIn = tempOut;
		tempIn.push_back(1.0);

		tempOut.resize( nodesPerLayer[i+1], 0.0 );

		// cout << "Generated " << nodesPerLayer[i+1] << " nodes" << endl;

		// For each node in that layer
		for( int j=0; j < nodesPerLayer[i+1]; j++ )
		{
			// Evaluate the summation from the previous layer's outputs
			for( int w=0; w < nodesPerLayer[i]; w++ )
			{
				tempOut[j] += networkWeights[i][j][w]*tempIn[w];
			}

			tempOut[j] = activationFunction( tempOut[j] );
		}
		// cout << "SIZE: " << tempOut.size() << endl;

	}

	// cout << "SIZE: " << tempOut.size() << endl;
	return tempOut;
}

float NeuralNet::activationFunction(float x)
{
	return 1.0/(1.0+exp(-1.0 * steepness * x));
}

void NeuralNet::trainNetwork(vector<float> errors, vector<float> results)
{
	// Create a vector for weight changes for each layer
	vector< vector< vector<float> > > outputDeltas(numLayers-1);
	vector< vector< vector<float> > > updateDeltas(numLayers-1);

	// resize to the correct number of layers
	for( int i=0; i < numLayers-1; i++ )
	{
		// Resize to the correct number of nodes in the layer
		outputDeltas[i].resize( nodesPerLayer[i+1] );
		updateDeltas[i].resize( nodesPerLayer[i+1] );

		// For each node in the current layer
		for( int j=0; j < nodesPerLayer[i+1]; j++ )
		{
			// Resize to the correct number of input weights from
			//	the previous layer
			outputDeltas[i][j].resize(nodesPerLayer[i], 0.0);
			updateDeltas[i][j].resize(nodesPerLayer[i], 0.0);
		}
	}


	// Compute initial weight update values
	//	w_jk


	// Back propogate errors through the net
	//  to update all w_ij
	for (int i = numLayers-1; i > 0; i--)
	{
		
	}
}

void NeuralNet::readParameters( string filename )
{
	ifstream fin;

	fin.open( filename.c_str() );

	//read all lines into vector
	string line;
	while( getline( fin, line ))
	{
		lines.push_back( line );
	}

	//remove useless lines
	lines.erase(lines.begin(),lines.begin()+21);
	lines.erase(lines.begin()+7,lines.begin()+13);
	lines.erase(lines.begin()+8,lines.begin()+16);
	lines.erase(lines.begin()+11,lines.begin()+17);
	lines.erase(lines.begin()+12,lines.begin()+18);
	lines.erase(lines.begin()+14,lines.end());

	//remove end comments on line
	lines[0]=lines[0].substr(0,lines[0].find_first_of(" "));
	lines[1]=lines[1].substr(0,lines[1].find_first_of(" "));
	lines[2]=lines[2].substr(0,lines[2].find_first_of(" "));
	lines[3]=lines[3].substr(0,lines[3].find_first_of(" "));
	lines[4]=lines[4].substr(0,lines[4].find_first_of(" "));
	lines[5]=lines[5].substr(0,lines[5].find_first_of(" "));
	lines[6].substr(0,lines[6].find_first_of("#"));
	lines[6].substr(0,lines[6].find_last_not_of(" \t")+1);
	lines[8]=lines[8].substr(0,lines[8].find_first_of(" "));
	lines[9]=lines[9].substr(0,lines[9].find_first_of(" "));
	lines[10]=lines[10].substr(0,lines[10].find_first_of(" "));
	lines[11]=lines[11].substr(0,lines[11].find_first_of(" "));

	//initialize related values
	weightFilename=lines[0];
	epochs=atoi(lines[1].c_str());
	learningRate=atof(lines[2].c_str());
	momentum=atof(lines[3].c_str());
	threshold=atof(lines[4].c_str());
	numLayers=atoi(lines[5].c_str());
	//lines[6] nodes per layer handle
	for(int i=0;i<numLayers+1;i++)
	{
		string temp;
		if(i == numLayers-1)
		{
			nodesPerLayer.push_back(atoi(lines[6].c_str()));
		}
		else
		{
			temp=lines[6].substr(0,lines[6].find_first_of(" "));
			nodesPerLayer.push_back(atoi(temp.c_str()));
			lines[6]=lines[6].substr(lines[6].find_first_of(" ")+1);
		}
	}
	trainingFilename=lines[7];
	yearsBurned=atoi(lines[8].c_str());
	monthsData=atoi(lines[9].c_str());
	endMonth=atoi(lines[10].c_str());
	numOutputClasses=atoi(lines[11].c_str());
	mediumCutoff=atoi(lines[12].c_str());
	highCutoff=atoi(lines[13].c_str());

	//handle reading data file for only required data
}









