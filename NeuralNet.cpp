
#include "NeuralNet.h"
#include <cmath>

NeuralNet::NeuralNet( int nLayers, vector<int> nPerLayer )
{
	// Seed our random number generator
    srand( time(NULL) );
	initRange = 0.2;
	steepness = 1.0;
	learningRate = 0.4;
	momentum = 0.8;

	numLayers = nLayers;
	nodesPerLayer = nPerLayer;

	// Initialize the input layer
	inputLayer.resize( nPerLayer[0] );

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


	for( int layer=0; layer < numLayers-1; layer++ )
	{
		//cout << "*";
		for( int to=0; to < nodesPerLayer[layer+1]; to++ )
		{
			//cout << "#";
			for( int from=0; from < networkWeights[layer][to].size(); from++ )
			{
				//cout << "&";
				cout << networkWeights[layer][to][from] << " ";
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

	// return 0.1;
}

vector< vector<float> > NeuralNet::evaluateNet( vector< float > inputs, vector< float > outputs )
{

	// Make our input vector the old output vector (think of this as output from the input layer) 
	//	for consistency in the loop 
	vector<float> tempOut = inputs;
	vector< vector<float> > perceptronOutputs;

	// For each layer
	for (int layer = 0; layer < numLayers-1; ++layer)
	{
		
		// Create temporary output values
		vector<float> tempIn = tempOut;
		tempIn.push_back(1.0);

		// cout << "Inputs from layer " << layer << endl;
		// for( int tIn = 0; tIn < tempIn.size(); tIn++ )
		// {
		// 	cout << " " << tempIn[tIn];
		// }
		// cout << endl << endl;

		// Store perceptron output from previous layers for training
		perceptronOutputs.push_back( tempIn );

		// Resize to store the right number of output values for the current layer
		tempOut.resize( nodesPerLayer[layer+1], 0.0 );
		for( int tt = 0; tt < tempOut.size(); tt++ )
		{
			tempOut[tt] = 0.0;
		}

		// cout << "Generated " << nodesPerLayer[i+1] << " nodes" << endl;

		// For each node in that layer
		for( int to=0; to < nodesPerLayer[layer+1]; to++ )
		{
			// Evaluate the summation from the previous layer's outputs
			for( int from=0; from < nodesPerLayer[layer]+1; from++ )
			{
				tempOut[to] += networkWeights[layer][to][from]*tempIn[from];
			}

			tempOut[to] = activationFunction( tempOut[to] );
		}
		// cout << "SIZE: " << tempOut.size() << endl;


	}

	// Push the final 
	perceptronOutputs.push_back( tempOut );

	// cout << "SIZE: " << tempOut.size() << endl;
	return perceptronOutputs;
}



void NeuralNet::trainNetwork(vector<float> errors, vector< vector<float> > results)
{
	// Create a vector for weight changes for each layer
	vector< vector< vector<float> > > outputDeltas(numLayers-1); // Difference between desired and actual output
	vector< vector< vector<float> > > updateDeltas(numLayers-1); // Amount to update weight

	// cout << "Allocating space" << endl;
	// resize to the correct number of nodes in each layer
	for( int layer=0; layer < numLayers-1; layer++ )
	{
		// Resize to the correct number of nodes in the layer
		outputDeltas[layer].resize( nodesPerLayer[layer+1] );
		updateDeltas[layer].resize( nodesPerLayer[layer+1] );

		// For each node providing input to the current layer
		int numFrom = nodesPerLayer[layer] + 1;

		for( int to=0; to < nodesPerLayer[layer+1]; to++ )
		{
			// Resize to the correct number of input weights from
			//	the previous layer
			outputDeltas[layer][to].resize(numFrom, 0.0);
			updateDeltas[layer][to].resize(numFrom, 0.0);
		}
	}



	// Compute initial weight update values
	//	w_jk

	// Store delta values for back propogation
	vector< float > deltaK( nodesPerLayer[numLayers-1] );


	// cout << "Computing output layer weight updates" << endl;
	// For each node in the output layer...
	// k = to
	for( int to=0; to < nodesPerLayer[numLayers-1]; to++ )
	{
		// cout << "to = " << to << endl;
		// Assign output to node k from node j as yK for readable code
		float yK = results[numLayers-1][to];

		// Compute deltaK
		deltaK[to] = yK * (1-yK) * errors[to];

		// For each node into the output layer...
		// j = from
		for( int from=0; from < nodesPerLayer[numLayers-2] + 1; from++ )
		{
			// cout << "Updating weight from " << from << " to " << to << endl;
			float yJ = results[numLayers-2][from];

			// Store update delta to update weights later
			updateDeltas[numLayers-2][to][from] = learningRate * yJ * deltaK[to];

		}
		// cout << "Updated deltas" << endl;
	}

	// Back propogate errors through the net
	//  to update all w_ij
	// For each layer of weights to update...
	for (int layer = numLayers-2; layer > 0; layer--)
	{
		// Hold temporary delta values for back prop
		vector< float > deltaJ( nodesPerLayer[layer], 0.0 );

		// For each node in the current layer
		//j=to
		for( int to=0; to < nodesPerLayer[layer]; to++ )
		{
			// Compute the summation
			float sigma = 0;

			// Evaluate 
			// cout << "Summing ";
			for( int k=0; k < nodesPerLayer[layer+1]; k++ )
			{
				// cout << "layer: " << layer << " to: " << to << " k: " << k << endl;
				float wJK = networkWeights[layer][k][to];

				// cout << " " << wJK << "+" << deltaK[k] << "   +  ";
				// cout << "FUUUUUCCCCKKK CODE" << endl;
				sigma += wJK * deltaK[k];
			}
			// cout << endl << "Sigma = " << sigma << endl;

			float yJ = results[layer][to];
			// cout << "yJ is " << yJ << endl;

			deltaJ[to] = yJ * (1-yJ) * sigma;
			// cout << "delJ = " << deltaJ[to] << endl;

			// Update weights into node j
			// i = from
			for( int from=0; from < nodesPerLayer[layer-1]+1; from++ )
			{
				// cout << "Updating layer " << layer << " from " << from << " to " << to << endl;
				float yI = results[layer-1][from];
				// cout << "yI = " << yI << endl;
				// cout << "Code" << endl;
				// Store updated deltas to update weights later
				float tempDelta = deltaJ[to];
				// cout << " is " << endl;
				float tempCalc = learningRate * yI * tempDelta;
				// cout << "effing" << endl;
				updateDeltas[layer-1][to][from] = tempCalc;
				// cout << "updateDeltas to " << to << " from " << from << " = " << tempCalc << endl;
				// cout << "dumb" << endl;
			}

			// Move the stored delta values back one layer to continue
			//	back propogation
			deltaK = deltaJ;

		}
	}

	for( int layer = 0; layer < networkWeights.size(); layer++ )
		for( int to = 0; to < networkWeights[layer].size(); to++ )
			for( int from = 0; from < networkWeights[layer][to].size(); from++ )
			{
				// cout << "Updating " << updateDeltas[layer][to][from] << endl;
				networkWeights[layer][to][from] += updateDeltas[layer][to][from];
			}
}

void NeuralNet::readParameters( string filename )
{
	ifstream fin;

	fin.open( filename.c_str() );

	//read all lines into vector
	string line;
	while( getline( fin, line, '\n' ))
	{
		string hash;
		hash.assign(line,0,1);
		if(!line.empty() && line != "\n" && hash != "#")
		{
			lines.push_back( line );
		}
	}

	fin.close();

	//remove end comments on line
	lines[0]=lines[0].substr(0,lines[0].find_first_of(" \t"));
	lines[1]=lines[1].substr(0,lines[1].find_first_of(" \t"));
	lines[2]=lines[2].substr(0,lines[2].find_first_of(" \t"));
	lines[3]=lines[3].substr(0,lines[3].find_first_of(" \t"));
	lines[4]=lines[4].substr(0,lines[4].find_first_of(" \t"));
	lines[5]=lines[5].substr(0,lines[5].find_first_of(" \t"));
	lines[6]=lines[6].substr(0,lines[6].find_last_of("0123456789")+1);
	lines[8]=lines[8].substr(0,lines[8].find_first_of(" \t"));
	lines[9]=lines[9].substr(0,lines[9].find_first_of(" \t"));
	lines[10]=lines[10].substr(0,lines[10].find_first_of(" \t"));
	lines[11]=lines[11].substr(0,lines[11].find_first_of(" \t"));

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
		if(i == numLayers)
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

vector< vector<float> > NeuralNet::readDataFile(string dataFilename)
{
	ifstream fin;

	fin.open(trainingFilename.c_str());

	vector<float> values;
	vector<string> ind;
	string month;

	getline(fin, month);
	getline(fin, month);

	while( getline(fin, month) )
	{
		ind=split(month, ',');
		for(int i=0;i<ind.size();i++)
		{
			values.push_back(atof(ind[i].c_str()));
		}
		data.push_back(values);
		values.clear();
	}

	fin.close();

	for(int i=0;i<data.size();i++)
	{
		data[i].erase(data[i].begin());
		data[i].insert(data[i].begin()+1,1);
	}

	return data;
}

vector<std::string> NeuralNet::split(const string &text, char sep)
{
	vector<std::string> tokens;
	size_t start = 0, end = 0;
	while ((end = text.find(sep, start)) != string::npos)
	{
	tokens.push_back(text.substr(start, end - start));
	start = end + 1;
	}
	tokens.push_back(text.substr(start));
	return tokens;
}

float NeuralNet::activationFunction(float x)
{
	float f = 1.0 / (1.0 + exp(-1.0 * steepness * x));
	// cout << "Activation function for x = " << x << " is " << f << endl;
	return f;
}

int main(int argc, char** argv)
{
	return 0;
}



