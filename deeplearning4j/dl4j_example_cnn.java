package com.meituan.deeplearning4j;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class LenetMnistExample {

	public static void main(String[] args) throws IOException {

		int nChannels = 1;
		int outputNum = 10;
		int batchSize = 64;
		int nEpochs = 1;
		int iterations = 1;
		int seed = 123;
		System.out.println("load data");
		DataSetIterator mnisTrain = new MnistDataSetIterator(batchSize, true,
				12345);
		DataSetIterator mnisTest = new MnistDataSetIterator(batchSize, false,
				12345);
		System.out.println("Builder model..");
		Map<Integer, Double> lrSchedule = new HashMap<Integer, Double>();
		System.out.println("build model....");
		MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
				.seed(seed)
				.iterations(iterations)
				.regularization(true)
				.l2(0.0005)
				.learningRate(0.01)
				.weightInit(WeightInit.XAVIER)
				.optimizationAlgo(
						OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.updater(Updater.NESTEROVS)
				.momentum(0.9)
				.list()
				.layer(0,
						new ConvolutionLayer.Builder(5, 5).nIn(nChannels)
								.stride(1, 1).nOut(20)
								.activation(Activation.IDENTITY).build())
				.layer(1,
						new SubsamplingLayer.Builder(
								SubsamplingLayer.PoolingType.MAX)
								.kernelSize(2, 2).stride(2, 2).build())
				.layer(2,
						new ConvolutionLayer.Builder(5, 5)
								// Note that nIn need not be specified in later
								// layers
								.stride(1, 1).nOut(50)
								.activation(Activation.IDENTITY).build())
				.layer(3,
						new SubsamplingLayer.Builder(
								SubsamplingLayer.PoolingType.MAX)
								.kernelSize(2, 2).stride(2, 2).build())
				.layer(4,
						new DenseLayer.Builder().activation(Activation.RELU)
								.nOut(500).build())
				.layer(5,
						new OutputLayer.Builder(
								LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
								.nOut(outputNum).activation(Activation.SOFTMAX)
								.build())
				.setInputType(InputType.convolutionalFlat(28, 28, 1)) // See
																		// note
																		// below
				.backprop(true).pretrain(false);
		
		MultiLayerConfiguration conf=builder.build();
		MultiLayerNetwork  model=new MultiLayerNetwork(conf);
		model.init();
		System.out.println("train model is start....");
		for(int i=0;i<4;i++){
			model.fit(mnisTrain);
			System.out.println(" Completed epoch is :" + i);

	            System.out.println("Evaluate model....");
	            Evaluation eval = new Evaluation( );
	            while(mnisTest.hasNext()){
	                DataSet ds = mnisTest.next();
	                INDArray output = model.output(ds.getFeatureMatrix(), false);
	                eval.eval(ds.getLabels(), output);
	            }
	            System.out.println(eval.stats());
	            mnisTest.reset();
			
		}
		System.out.println("model finish");

	}

}



//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////

coding in the maven :
                             
<properties>  
  <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>  
  <nd4j.version>0.7.1</nd4j.version>  
  <dl4j.version>0.7.1</dl4j.version>  
  <datavec.version>0.7.1</datavec.version>  
  <scala.binary.version>2.10</scala.binary.version>  
</properties>  
<dependencies>  
<dependency>  
    <groupId>org.nd4j</groupId>  
    <artifactId>nd4j-native</artifactId>   
    <version>${nd4j.version}</version>  
</dependency>  
<dependency>  
    <groupId>org.deeplearning4j</groupId>  
    <artifactId>dl4j-spark_2.11</artifactId>  
    <version>${dl4j.version}</version>  
</dependency>  
     <dependency>  
          <groupId>org.datavec</groupId>  
          <artifactId>datavec-spark_${scala.binary.version}</artifactId>  
          <version>${datavec.version}</version>  
    </dependency>  
      <dependency>  
   <groupId>org.deeplearning4j</groupId>  
   <artifactId>deeplearning4j-core</artifactId>  
   <version>${dl4j.version}</version>  
</dependency>  
</dependencies>  




int nChannels = 1;      //black & white picture, 3 if color image  
int outputNum = 10;     //number of classification  
int batchSize = 64;     //mini batch size for sgd  
int nEpochs = 10;       //total rounds of training  
int iterations = 1;     //number of iteration in each traning round  
int seed = 123;         //random seed for initialize weights  
  
log.info("Load data....");  
DataSetIterator mnistTrain = null;  
DataSetIterator mnistTest = null;  
  
mnistTrain = new MnistDataSetIterator(batchSize, true, 12345);  
mnistTest = new MnistDataSetIterator(batchSize, false, 12345);  

MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()  
        .seed(seed)  
        .iterations(iterations)  
        .regularization(true).l2(0.0005)  
        .learningRate(0.01)//.biasLearningRate(0.02)  
        //.learningRateDecayPolicy(LearningRatePolicy.Inverse).lrPolicyDecayRate(0.001).lrPolicyPower(0.75)  
        .weightInit(WeightInit.XAVIER)  
        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)  
        .updater(Updater.NESTEROVS).momentum(0.9)  
        .list()  
        .layer(0, new ConvolutionLayer.Builder(5, 5)  
                //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied  
                .nIn(nChannels)  
                .stride(1, 1)  
                .nOut(20)  
                .activation("identity")  
                .build())  
        .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)  
                .kernelSize(2,2)  
                .stride(2,2)  
                .build())  
        .layer(2, new ConvolutionLayer.Builder(5, 5)  
                //Note that nIn need not be specified in later layers  
                .stride(1, 1)  
                .nOut(50)  
                .activation("identity")  
                .build())  
        .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)  
                .kernelSize(2,2)  
                .stride(2,2)  
                .build())  
        .layer(4, new DenseLayer.Builder().activation("relu")  
                .nOut(500).build())  
        .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)  
                .nOut(outputNum)  
                .activation("softmax")  
                .build())  
        .backprop(true).pretrain(false)  
        .cnnInputSize(28, 28, 1);  
// The builder needs the dimensions of the image along with the number of channels. these are 28x28 images in one channel  
//new ConvolutionLayerSetup(builder,28,28,1);  
  
MultiLayerConfiguration conf = builder.build();  
MultiLayerNetwork model = new MultiLayerNetwork(conf);  
model.init();          
model.setListeners(new ScoreIterationListener(1));         // a listener which can print loss function score after each iteration  



for( int i = 0; i < nEpochs; ++i ) {  
    model.fit(mnistTrain);  
    log.info("*** Completed epoch " + i + "***");  
  
    log.info("Evaluate model....");  
    Evaluation eval = new Evaluation(outputNum);  
    while(mnistTest.hasNext()){  
        DataSet ds = mnistTest.next();            
        INDArray output = model.output(ds.getFeatureMatrix(), false);  
        eval.eval(ds.getLabels(), output);  
    }  
    log.info(eval.stats());  
    mnistTest.reset();  
}  


 

Examples labeled as 0 classified by model as 0: 974 times  
Examples labeled as 0 classified by model as 6: 2 times  
Examples labeled as 0 classified by model as 7: 2 times  
Examples labeled as 0 classified by model as 8: 1 times  
Examples labeled as 0 classified by model as 9: 1 times  
Examples labeled as 1 classified by model as 0: 1 times  
Examples labeled as 1 classified by model as 1: 1128 times  
Examples labeled as 1 classified by model as 2: 1 times  
Examples labeled as 1 classified by model as 3: 2 times  
Examples labeled as 1 classified by model as 5: 1 times  
Examples labeled as 1 classified by model as 6: 2 times  
Examples labeled as 2 classified by model as 2: 1026 times  
Examples labeled as 2 classified by model as 4: 1 times  
Examples labeled as 2 classified by model as 6: 1 times  
Examples labeled as 2 classified by model as 7: 3 times  
Examples labeled as 2 classified by model as 8: 1 times  
Examples labeled as 3 classified by model as 0: 1 times  
Examples labeled as 3 classified by model as 1: 1 times  
Examples labeled as 3 classified by model as 2: 1 times  
Examples labeled as 3 classified by model as 3: 998 times  
Examples labeled as 3 classified by model as 5: 3 times  
Examples labeled as 3 classified by model as 7: 1 times  
Examples labeled as 3 classified by model as 8: 4 times  
Examples labeled as 3 classified by model as 9: 1 times  
Examples labeled as 4 classified by model as 2: 1 times  
Examples labeled as 4 classified by model as 4: 973 times  
Examples labeled as 4 classified by model as 6: 2 times  
Examples labeled as 4 classified by model as 7: 1 times  
Examples labeled as 4 classified by model as 9: 5 times  
Examples labeled as 5 classified by model as 0: 2 times  
Examples labeled as 5 classified by model as 3: 4 times  
Examples labeled as 5 classified by model as 5: 882 times  
Examples labeled as 5 classified by model as 6: 1 times  
Examples labeled as 5 classified by model as 7: 1 times  
Examples labeled as 5 classified by model as 8: 2 times  
Examples labeled as 6 classified by model as 0: 4 times  
Examples labeled as 6 classified by model as 1: 2 times  
Examples labeled as 6 classified by model as 4: 1 times  
Examples labeled as 6 classified by model as 5: 4 times  
Examples labeled as 6 classified by model as 6: 945 times  
Examples labeled as 6 classified by model as 8: 2 times  
Examples labeled as 7 classified by model as 1: 5 times  
Examples labeled as 7 classified by model as 2: 3 times  
Examples labeled as 7 classified by model as 3: 1 times  
Examples labeled as 7 classified by model as 7: 1016 times  
Examples labeled as 7 classified by model as 8: 1 times  
Examples labeled as 7 classified by model as 9: 2 times  
Examples labeled as 8 classified by model as 0: 1 times  
Examples labeled as 8 classified by model as 3: 1 times  
Examples labeled as 8 classified by model as 5: 2 times  
Examples labeled as 8 classified by model as 7: 2 times  
Examples labeled as 8 classified by model as 8: 966 times  
Examples labeled as 8 classified by model as 9: 2 times  
Examples labeled as 9 classified by model as 3: 1 times  
Examples labeled as 9 classified by model as 4: 2 times  
Examples labeled as 9 classified by model as 5: 4 times  
Examples labeled as 9 classified by model as 6: 1 times  
Examples labeled as 9 classified by model as 7: 5 times  
Examples labeled as 9 classified by model as 8: 3 times  
Examples labeled as 9 classified by model as 9: 993 times  
  
  
==========================Scores========================================  
 Accuracy:        0.9901  
 Precision:       0.99  
 Recall:          0.99  
 F1 Score:        0.99  
========================================================================  
[main] INFO cv.LenetMnistExample - ****************Example finished********************  
