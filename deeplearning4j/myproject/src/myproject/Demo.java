package example;

/*
 * a demo of network using dl4j library 
 * author: wuyifan
 */

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;

public class example_network {
	
	public static void main(String args[]) throws Exception{
		// the parameters for the net conf
		int seed = 100;
		double learn_rate = 0.01;
		int batch_size = 50;
		int nEpoch = 30;
		int num_node_input = 2; 
		//the number of features
		
		int num_node_output = 1;
		// the number of node output from network
		
		// let's build the network configuration
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(seed)
				.iterations(2)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.learningRate(learn_rate)
				.updater(Updater.NESTEROVS)
				.list()
				.layer(0, new DenseLayer.Builder()
								.nIn(num_node_input)
								.nOut(5)
								.weightInit(WeightInit.XAVIER)
								.activation(Activation.RELU)
								.build()
								)
				.layer(1, new DenseLayer.Builder()
						.nIn(5)
						.nOut(num_node_output)
						.weightInit(WeightInit.XAVIER)
						.activation(Activation.RELU)
						.build()
						)
				.pretrain(false).backprop(true).build();
		/*
		 * it is a very sample network 
		 * 2 nodes -> hidden 5 nodes -> 1 nodes
		 */
		
		/* when we have the conf of the network , we can use it to build a real 
		 * netwoek model 
		 */
		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		
		model.setListeners(new ScoreIterationListener(10));
		//output the score of model every 10 iterations of training
		System.out.println("the model is built !");
		
		
		//all right, let us read the data from location 
		//final String filenameTest = new ClassPathResource("e:\\linear_data_eval.csv").getFile().getPath();
		//final String filenameTrain = new ClassPathResource("e:\\linear_data_train.csv").getFile().getPath();
		
		//load the data from the file 
		RecordReader rr = new CSVRecordReader();
		rr.initialize(new FileSplit(new File("d:\\linear_data_eval.csv")));
		DataSetIterator train = new RecordReaderDataSetIterator(rr, batch_size, 0,2);
		
		RecordReader rt = new CSVRecordReader();
		rt.initialize(new FileSplit(new File("d:\\linear_data_train.csv")));
		DataSetIterator test = new RecordReaderDataSetIterator(rt, batch_size, 0,2);
		System.out.println("we get the load location");
		
		for (int n = 0; n < nEpoch ; n++){
			System.out.println("it is the "+n+" train step==========");
			model.fit(train);
		}
		
		System.out.println("the model has been trained !");
		
		
		// let us evaluation the model 
		Evaluation eval = new Evaluation(num_node_output);
		
		while (test.hasNext()){
			DataSet t = test.next();
			INDArray fea = t.getFeatureMatrix();
			INDArray labels = t.getLabels();
			INDArray pred = model.output(fea, false);
			
			eval.eval(labels, pred);
			
		}
		
		System.out.println(eval.stats());
		
		System.out.println("the program is over");
		
	}
}
