package com.didi.dl.network;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;

//import org.nd4j.linalg.;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;
import java.io.IOException;
/**
 * Hello world!
 *
 */
public class App 
{
    public static void main( String[] args ) throws Exception
    {
        System.out.println( "Hello World!" );
    	int seed = 100;
		double learn_rate = 0.01;
		int batch_size = 50;
		int nEpoch = 1;
		int num_node_input = 2; 
		//the number of features
		
		int num_node_output = 2;
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
								.activation("relu")
								.build()
								)
				.layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
                        .weightInit(WeightInit.XAVIER)
                        .activation("softmax")
                        .nIn(5).nOut(num_node_output).build())
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
        
        RecordReader rr = new CSVRecordReader();
		rr.initialize(new FileSplit(new File("f:\\test.csv")));
		DataSetIterator train = new RecordReaderDataSetIterator(rr, batch_size, 0,2);
		
		RecordReader rt = new CSVRecordReader();
		rt.initialize(new FileSplit(new File("f:\\train.csv")));
		DataSetIterator test = new RecordReaderDataSetIterator(rt, batch_size, 0,2);
		
        
		
		
		
		System.out.println("we get the load location");
		
		for ( int n = 0; n < nEpoch; n++) {
			System.out.println(n);
            model.fit( test );
        }
		
		System.out.println("Evaluate model....");
        Evaluation eval = new Evaluation(num_node_output);
        while(train.hasNext()){
            DataSet t = train.next();
            INDArray features = t.getFeatureMatrix();
            INDArray labels = t.getLabels();
            INDArray predicted = model.output(features,false);
            System.out.println(predicted);
            System.out.println(labels);
            System.out.println("=============================");
            eval.eval(labels, predicted);

        }
        //Print the evaluation statistics
        System.out.println(eval.stats());
        
        
    }
}
