package org.deeplearning4j.examples.rl4j;

import java.io.IOException;
import org.deeplearning4j.rl4j.learning.HistoryProcessor;
import org.deeplearning4j.rl4j.learning.async.a3c.discrete.A3CDiscrete;
import org.deeplearning4j.rl4j.learning.async.a3c.discrete.A3CDiscreteConv;
import org.deeplearning4j.rl4j.mdp.ale.ALEMDP;
import org.deeplearning4j.rl4j.network.ac.ActorCriticFactoryCompGraphStdConv;
import org.deeplearning4j.rl4j.util.DataManager;

/**
 * @author saudet
 *
 * Main example for A3C with The Arcade Learning Environment (ALE)
 *
 */
public class A3CALE {

    public static HistoryProcessor.Configuration ALE_HP =
            new HistoryProcessor.Configuration(
                    4,       //History length
                    84,      //resize width
                    110,     //resize height
                    84,      //crop width
                    84,      //crop height
                    0,       //cropping x offset
                    0,       //cropping y offset
                    4        //skip mod (one frame is picked every x
            );

    public static A3CDiscrete.A3CConfiguration ALE_A3C =
            new A3CDiscrete.A3CConfiguration(
                    123,            //Random seed
                    10000,          //Max step By epoch
                    8000000,        //Max step
                    8,              //Number of threads
                    32,             //t_max
                    500,            //num step noop warmup
                    0.1,            //reward scaling
                    0.99,           //gamma
                    10.0            //td-error clipping
            );

    public static final ActorCriticFactoryCompGraphStdConv.Configuration ALE_NET_A3C =
            new ActorCriticFactoryCompGraphStdConv.Configuration(
                    0.00025, //learning rate
                    0.000,   //l2 regularization
                    null, null, false
            );

    public static void main(String[] args) throws IOException {

        //record the training data in rl4j-data in a new folder
        DataManager manager = new DataManager(true);

        //setup the emulation environment through ALE, you will need a ROM file
        ALEMDP mdp = null;
        try {
            mdp = new ALEMDP("pong.bin");
        } catch (UnsatisfiedLinkError e) {
            System.out.println("To run this example, uncomment the \"ale-platform\" dependency in the pom.xml file.");
        }

        //setup the training
        A3CDiscreteConv<ALEMDP.GameScreen> a3c = new A3CDiscreteConv(mdp, ALE_NET_A3C, ALE_HP, ALE_A3C, manager);

        //start the training
        a3c.train();

        //save the model at the end
        a3c.getPolicy().save("ale-a3c.model");

        //close the ALE env
        mdp.close();
    }
}


////////////////another example CartPole-v0 using A3c Reforce learning//////////////////////////////////////////////

  <properties>
      <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding> 
      <nd4j.version>0.8.0</nd4j.version>  
      <dl4j.version>0.8.0</dl4j.version>  
      <datavec.version>0.8.0</datavec.version>  
      <rl4j.version>0.8.0</rl4j.version>
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
        <artifactId>deeplearning4j-core</artifactId>  
        <version>${dl4j.version}</version>  
    </dependency>  
        <dependency>
            <groupId>org.deeplearning4j</groupId>
            <artifactId>rl4j-core</artifactId>
            <version>${rl4j.version}</version>
        </dependency>
        <dependency>
            <groupId>org.deeplearning4j</groupId>
            <artifactId>rl4j-gym</artifactId>
            <version>${rl4j.version}</version>
        </dependency>

  </dependencies>

  public static QLearning.QLConfiguration CARTPOLE_QL =
            new QLearning.QLConfiguration(
                    123,    //Random seed
                    200,    //Max step By epoch
                    150000, //Max step
                    150000, //Max size of experience replay
                    32,     //size of batches
                    500,    //target update (hard)
                    10,     //num step noop warmup
                    0.01,   //reward scaling
                    0.99,   //gamma
                    1.0,    //td-error clipping
                    0.1f,   //min epsilon
                    1000,   //num step for eps greedy anneal
                    true    //double DQN
            );

    public static DQNFactoryStdDense.Configuration CARTPOLE_NET = 
            new DQNFactoryStdDense.Configuration.builder()  
                                    .l2(0.001)                                                                                  
                                    .learningRate(0.0005)
                                    .numHiddenNodes(16)
                                    .numLayer(3)
                                    .build();

/*
第一部分中定义Q-learning的参数，包括每一轮的训练的可采取的action的步数，
最大步数以及存储过往action的最大步数等。除此以外，DQNFactoryStdDense用来定义基于MLP的DQN网络结构，
包括网络深度等常见参数。这里的代码定义的是一个三层（只有一层隐藏层）的全连接神经网络。

接下来，定义两个方法分别用于训练和测试。catpole方法用于训练DQN，而loadCartpole则用于测试。

训练：
*/
    public static void() {

        //record the training data in rl4j-data in a new folder (save)
        DataManager manager = new DataManager(true);

        //define the mdp from gym (name, render)
        GymEnv<Box, Integer, DiscreteSpace> mdp = null;
        try {
            mdp = new GymEnv<Box, Integer, DiscreteSpace>("CartPole-v0", false, false);
        } catch (RuntimeException e){
            System.out.print("To run this example, download and start the gym-http-api repo found at https://github.com/openai/gym-http-api.");
        }
        //define the training
        QLearningDiscreteDense<Box> dql = new QLearningDiscreteDense<Box>(mdp, CARTPOLE_NET, CARTPOLE_QL, manager);

        //train
        dql.train();

        //get the final policy
        DQNPolicy<Box> pol = dql.getPolicy();

        //serialize and save (serialization showcase, but not required)
        pol.save("/tmp/pol1");

        //close the mdp (close http)
        mdp.close();

    }

//测试：

    public static void loadCartpole(){

        //showcase serialization by using the trained agent on a new similar mdp (but render it this time)

        //define the mdp from gym (name, render)
        GymEnv<Box, Integer, DiscreteSpace> mdp2 = new GymEnv<Box, Integer, DiscreteSpace>("CartPole-v0", true, false);

        //load the previous agent
        DQNPolicy<Box> pol2 = DQNPolicy.load("/tmp/pol1");

        //evaluate the agent
        double rewards = 0;
        for (int i = 0; i < 1000; i++) {
            mdp2.reset();
            double reward = pol2.play(mdp2);
            rewards += reward;
            Logger.getAnonymousLogger().info("Reward: " + reward);
        }

        Logger.getAnonymousLogger().info("average: " + rewards/1000);
        
        mdp2.close();

    }
/*
在训练模型的方法中，包含了第二、三步的内容。首先需要定义gym数据读取对象，即代码中的GymEnv<Box, Integer,
 DiscreteSpace> mdp。它会通过gym-http-api的接口读取训练数据。
 接着，将第一步中定义的Q-learning的相关参数，神经网络结构作为参数传入DQN训练的包装类中。
 其中DataManager的作用是用来管理训练数据。

测试部分的代码实现了之前说的第四步，即加载策略模型并进行测试的过程。
在测试的过程中，将每次action的reward打上log，并最后求取平均的reward。
*/