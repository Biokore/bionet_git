#include "main.hpp"
#include "fsys.hpp"
#include "brain.hpp"






int main(/*int argc, char **argv*/)
{
	if(std::system("clear")) exit(0);
	
	srand(time(NULL));
	
	std::cout << "Welcome in Brain++ V1.2 ! " << std::endl;
	std::cout << "\n\n";
	
	
	IDX_class trFile("database/train-images.idx3-ubyte", "database/train-labels.idx1-ubyte");
	IDX_class teFile("database/t10k-images.idx3-ubyte", "database/t10k-labels.idx1-ubyte");
	
	
	
	brain obrain(784);
	
	obrain.addLayer(RELU, 150, -0.5);
	obrain.addLayer(HTAN, 50, -0.5);
	obrain.addLayer(SIGMOID, 10, +0.5);
	
	std::cout << obrain.listBrain() << '\n';
	
	std::cout << "base error = " << obrain.checkError(teFile.getInputsArray(), teFile.getOutputsArray()) << std::endl;
	
	std::cout << "\n\n";
	
	int tmp(0);
	for(int it = 0; it < 20; it++)
	{
		for(int i = 0; i < 60000; i++)
		{
			obrain.brainPropagate(trFile.getInputsArray(i));
			obrain.brainRetroPropagate(trFile.getOutputsArray(i));
			obrain.refitWeights(trFile.getInputsArray(i));
		}
		
		tmp = rand()%10000;
		
		teFile.printPic(tmp);
		obrain.brainPropagate(teFile.getInputsArray(tmp));
		std::cout << obrain.getMax(obrain.getBrainOutput()) << std::endl;
		std::cout << "error = " << obrain.checkError(teFile.getInputsArray(), teFile.getOutputsArray()) << std::endl;
		
		
	}
	
	
	std::cout << "\n\n";
	return 0;
};






// END
