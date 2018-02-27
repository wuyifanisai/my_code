
/////////////////////////////////// EditAtBottleneckOthersFrozen.java //////////////////////////////
package org.deeplearning4j.examples.transferlearning.vgg16;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.examples.transferlearning.vgg16.dataHelpers.FlowerDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.trainedmodels.TrainedModelHelper;
import org.deeplearning4j.nn.modelimport.keras.trainedmodels.TrainedModels;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;

import java.io.File;

/**
 * @author susaneraly on 3/1/17.
 *
 * IMPORTANT:
 * 1. The forward pass on VGG16 is time consuming. Refer to "FeaturizedPreSave" and "FitFromFeaturized" for how to use presaved datasets
 * 2. RAM at the very least 16G, set JVM mx heap space accordingly
 *
 * We use the transfer learning API to construct a new model based of org.deeplearning4j.transferlearning.vgg16.
 * We keep block5_pool and below frozen
 *      and modify/add dense layers to form
 *          block5_pool -> flatten -> fc1 -> fc2 -> fc3 -> newpredictions (5 classes)
 *       from
 *          block5_pool -> flatten -> fc1 -> fc2 -> predictions (1000 classes)
 *
 * Note that we could presave the output out block5_pool like we do in FeaturizedPreSave + FitFromFeaturized
 * Refer to those two classes for more detail
 */
public class EditAtBottleneckOthersFrozen {
    // private为只有子类可以访问， protected为外部类不能访问，static final 就可以定义全局不变常量
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(EditAtBottleneckOthersFrozen.class);

    protected static final int numClasses = 5;

    protected static final long seed = 12345;
    private static final int trainPerc = 80;
    private static final int batchSize = 15;
    private static final String featureExtractionLayer = "block5_pool";

    public static void main(String [] args) throws Exception {

        //Import vgg
        //Note that the model imported does not have an output layer (check printed summary)
        //  nor any training related configs (model from keras was imported with only weights and json)
        log.info("\n\nLoading org.deeplearning4j.transferlearning.vgg16...\n\n");

        ZooModel zooModel = new VGG16();//向上转型，使得zooModel 可以调用VGG16的方法

        ComputationGraph vgg16 = (ComputationGraph) zooModel.initPretrained();
        //向上转型，使得zooModel调用方法后获取的实例对象向下转型为compatationgraph类型，得到一个vgg16


        log.info(vgg16.summary());

        //Decide on a fine tune configuration to use.
        //In cases where there already exists a setting the fine tune setting will
        //  override the setting for all layers that are not "frozen".
        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
            .activation(Activation.LEAKYRELU)
            .weightInit(WeightInit.RELU)
            .learningRate(5e-5)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(Updater.NESTEROVS)
            .dropOut(0.5)
            .seed(seed)
            .build();

        //Construct a new model with the intended architecture and print summary
        //  Note: This architecture is constructed with the primary intent of demonstrating use of the transfer learning API,
        //        secondary to what might give better results
        ComputationGraph vgg16Transfer = new TransferLearning.GraphBuilder(vgg16)
            .fineTuneConfiguration(fineTuneConf)
            .setFeatureExtractor(featureExtractionLayer) //"block5_pool" and below are frozen
            .nOutReplace("fc2",1024, WeightInit.XAVIER) //modify nOut of the "fc2" vertex
            .removeVertexAndConnections("predictions") //remove the final vertex and it's connections
            .addLayer("fc3",new DenseLayer.Builder().activation(Activation.TANH).nIn(1024).nOut(256).build(),"fc2") //add in a new dense layer
            .addLayer("newpredictions",new OutputLayer
                                        .Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                        .activation(Activation.SOFTMAX)
                                        .nIn(256)
                                        .nOut(numClasses)
                                        .build(),"fc3") //add in a final output dense layer,
                                                        // note that learning related configurations applied on a new layer here will be honored
                                                        // In other words - these will override the finetune confs.
                                                        // For eg. activation function will be softmax not RELU
            .setOutputs("newpredictions") //since we removed the output vertex and it's connections we need to specify outputs for the graph
            .build();
        log.info(vgg16Transfer.summary());

        //Dataset iterators
        FlowerDataSetIterator.setup(batchSize,trainPerc);
        DataSetIterator trainIter = FlowerDataSetIterator.trainIterator();
        DataSetIterator testIter = FlowerDataSetIterator.testIterator();

        Evaluation eval;
        eval = vgg16Transfer.evaluate(testIter);
        log.info("Eval stats BEFORE fit.....");
        log.info(eval.stats() + "\n");
        testIter.reset();

        int iter = 0;
        while(trainIter.hasNext()) {
            vgg16Transfer.fit(trainIter.next());
            if (iter % 10 == 0) {
                log.info("Evaluate model at iter "+iter +" ....");
                eval = vgg16Transfer.evaluate(testIter);
                log.info(eval.stats());
                testIter.reset();
            }
            iter++;
        }
        log.info("Model build complete");

        //Save the model
        //Note that the saved model will not know which layers were frozen during training.
        //Frozen models always have to specified before training.
        // Models with frozen layers can be constructed in the following two ways:
        //      1. .setFeatureExtractor in the transfer learning API which will always a return a new model (as seen in this example)
        //      2. in place with the TransferLearningHelper constructor which will take a model, and a specific vertexname
        //              and freeze it and the vertices on the path from an input to it (as seen in the FeaturizePreSave class)
        //The saved model can be "fine-tuned" further as in the class "FitFromFeaturized"
        File locationToSave = new File("MyComputationGraph.zip");
        boolean saveUpdater = false;
        ModelSerializer.writeModel(vgg16Transfer, locationToSave, saveUpdater);

        log.info("Model saved");
    }
}





//////////////////////////// EditLastLayerOthersFrozen.java ///////////////////////////////////

package org.deeplearning4j.examples.transferlearning.vgg16;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.examples.transferlearning.vgg16.dataHelpers.FlowerDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;

import java.io.IOException;

/**
 * @author susaneraly on 3/9/17.
 *
 * We use the transfer learning API to construct a new model based of org.deeplearning4j.transferlearning.vgg16
 * We will hold all layers but the very last one frozen and change the number of outputs in the last layer to
 * match our classification task.
 * In other words we go from where fc2 and predictions are vertex names in org.deeplearning4j.transferlearning.vgg16
 *  fc2 -> predictions (1000 classes)
 *  to
 *  fc2 -> predictions (5 classes)
 * The class "FitFromFeaturized" attempts to train this same architecture the difference being the outputs from the last
 * frozen layer is presaved and the fit is carried out on this featurized dataset.
 * When running multiple epochs this can save on computation time.
 */
public class EditLastLayerOthersFrozen {
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(EditLastLayerOthersFrozen.class);

    protected static final int numClasses = 5;
    protected static final long seed = 12345;

    private static final int trainPerc = 80;
    private static final int batchSize = 15;
    private static final String featureExtractionLayer = "fc2";

    public static void main(String [] args) throws UnsupportedKerasConfigurationException, IOException, InvalidKerasConfigurationException {

        //Import vgg
        //Note that the model imported does not have an output layer (check printed summary)
        //  nor any training related configs (model from keras was imported with only weights and json)
        log.info("\n\nLoading org.deeplearning4j.transferlearning.vgg16...\n\n");
        ZooModel zooModel = new VGG16();
        ComputationGraph vgg16 = (ComputationGraph) zooModel.initPretrained();
        log.info(vgg16.summary());

        //Decide on a fine tune configuration to use.
        //In cases where there already exists a setting the fine tune setting will
        //  override the setting for all layers that are not "frozen".
        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
            .learningRate(5e-5)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(Updater.NESTEROVS)
            .seed(seed)
            .build();

        //Construct a new model with the intended architecture and print summary
        ComputationGraph vgg16Transfer = new TransferLearning.GraphBuilder(vgg16)
            .fineTuneConfiguration(fineTuneConf)
            .setFeatureExtractor(featureExtractionLayer) //the specified layer and below are "frozen"
            .removeVertexKeepConnections("predictions") //replace the functionality of the final vertex
            .addLayer("predictions",
                new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                    .nIn(4096).nOut(numClasses)
                    .weightInit(WeightInit.DISTRIBUTION)
                    .dist(new NormalDistribution(0,0.2*(2.0/(4096+numClasses)))) //This weight init dist gave better results than Xavier
                    .activation(Activation.SOFTMAX).build(),
                "fc2")
            .build();
        log.info(vgg16Transfer.summary());

        //Dataset iterators
        FlowerDataSetIterator.setup(batchSize,trainPerc);
        DataSetIterator trainIter = FlowerDataSetIterator.trainIterator();
        DataSetIterator testIter = FlowerDataSetIterator.testIterator();

        Evaluation eval;
        eval = vgg16Transfer.evaluate(testIter);
        log.info("Eval stats BEFORE fit.....");
        log.info(eval.stats() + "\n");
        testIter.reset();

        int iter = 0;
        while(trainIter.hasNext()) {
            vgg16Transfer.fit(trainIter.next());
            if (iter % 10 == 0) {
                log.info("Evaluate model at iter "+iter +" ....");
                eval = vgg16Transfer.evaluate(testIter);
                log.info(eval.stats());
                testIter.reset();
            }
            iter++;
        }

        log.info("Model build complete");
    }
}

////////////////////////// FineTuneFromBlockFour.java //////////////////////////////////
package org.deeplearning4j.examples.transferlearning.vgg16;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.examples.transferlearning.vgg16.dataHelpers.FlowerDataSetIterator;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;

import java.io.File;
import java.io.IOException;

/**
 * Important:
 * 1. Either run "EditAtBottleneckOthersFrozen" first or save a model named "MyComputationGraph.zip" based on org.deeplearning4j.transferlearning.vgg16 with block4_pool and below intact
 * 2. You will need a LOT of RAM, at the very least 16G. Set max JVM heap space accordingly
 *
 * Here we read in an already saved model based off on org.deeplearning4j.transferlearning.vgg16 from one of our earlier runs and "finetune"
 * Since we already have reasonable results with our saved off model we can be assured that there will not be any
 * large disruptive gradients flowing back to wreck the carefully trained weights in the lower layers in vgg.
 *
 * Finetuning like this is usually done with a low learning rate and a simple SGD optimizer
 * @author susaneraly on 3/6/17.
 */
public class FineTuneFromBlockFour {
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(FineTuneFromBlockFour.class);

    protected static final int numClasses = 5;
    protected static final long seed = 12345;

    private static final String featureExtractionLayer = "block4_pool";
    private static final int trainPerc = 80;
    private static final int batchSize = 15;

    public static void main(String [] args) throws IOException {

        //Import the saved model
        File locationToSave = new File("MyComputationGraph.zip");
        log.info("\n\nRestoring saved model...\n\n");
        ComputationGraph vgg16Transfer = ModelSerializer.restoreComputationGraph(locationToSave);

        //Decide on a fine tune configuration to use.
        //In cases where there already exists a setting the fine tune setting will
        //  override the setting for all layers that are not "frozen".
        //  For eg. We override the learning rate and updater
        //          But our optimization algorithm remains unchanged (already sgd)
        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
            .learningRate(1e-5)
            .updater(Updater.SGD)
            .seed(seed)
            .build();
        ComputationGraph vgg16FineTune = new TransferLearning.GraphBuilder(vgg16Transfer)
            .fineTuneConfiguration(fineTuneConf)
            .setFeatureExtractor(featureExtractionLayer)
            .build();
        log.info(vgg16FineTune.summary());

        //Dataset iterators
        FlowerDataSetIterator.setup(batchSize,trainPerc);
        DataSetIterator trainIter = FlowerDataSetIterator.trainIterator();
        DataSetIterator testIter = FlowerDataSetIterator.testIterator();

        Evaluation eval;
        eval = vgg16FineTune.evaluate(testIter);
        log.info("Eval stats BEFORE fit.....");
        log.info(eval.stats() + "\n");
        testIter.reset();

        int iter = 0;
        while(trainIter.hasNext()) {
            vgg16FineTune.fit(trainIter.next());
            if (iter % 10 == 0) {
                log.info("Evaluate model at iter "+iter +" ....");
                eval = vgg16FineTune.evaluate(testIter);
                log.info(eval.stats());
                testIter.reset();
            }
            iter++;
        }

        log.info("Model build complete");
        //Save the model
        File locationToSaveFineTune = new File("MyComputationGraphFineTune.zip");
        boolean saveUpdater = false;
        ModelSerializer.writeModel(vgg16FineTune, locationToSaveFineTune, saveUpdater);
        log.info("Model saved");

    }
}