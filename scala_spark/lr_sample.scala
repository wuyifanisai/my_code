/*
Spark-MLlib实例——逻辑回归，应用于二元分类的情况，这里以垃圾邮件分类为例，即是否为垃圾邮件两种情况。
1、垃圾邮件分类，使用Spark-MLlib中的两个函数： 
1）HashingTF： 从文本数据构建词频（term frequency）特征向量
2）LogisticRegressionWithSGD： 使用随机梯度下降法（Stochastic Gradient Descent）,实现逻辑回归。
2、训练原数据集
垃圾邮件例子 spam.txt
[plain] view plain copy
Dear sir, I am a Prince in a far kingdom you have not heard of.  I want to send you money via wire transfer so please ...  
Get Viagra real cheap!  Send money right away to ...  
Oh my gosh you can be really strong too with these drugs found in the rainforest. Get them cheap right now ...  
YOUR COMPUTER HAS BEEN INFECTED!  YOU MUST RESET YOUR PASSWORD.  Reply to this email with your password and SSN ...  
THIS IS NOT A SCAM!  Send money and get access to awesome stuff really cheap and never have to ...  
非垃圾邮件例子 normal.txt
[plain] view plain copy
Dear Spark Learner, Thanks so much for attending the Spark Summit 2014!  Check out videos of talks from the summit at ...  
Hi Mom, Apologies for being late about emailing and forgetting to send you the package.  I hope you and bro have been ...  
Wow, hey Fred, just heard about the Spark petabyte sort.  I think we need to take time to try it out immediately ...  
Hi Spark user list, This is my first question to this list, so thanks in advance for your help!  I tried running ...  
Thanks Tom for your email.  I need to refer you to Alice for this one.  I haven't yet figured out that part either ...  
Good job yesterday!  I was attending your talk, and really enjoyed it.  I want to try out GraphX ...  
Summit demo got whoops from audience!  Had to let you know. --Joe  

然后，根据词频把每个文件中的文本转换为特征向量，然后训练出一个可以把两类消息分开的逻辑回归模型。

3、垃圾邮件分类器

[java] view plain copy
*/

import org.apache.spark.{ SparkConf, SparkContext }  
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD  
import org.apache.spark.mllib.feature.HashingTF  
import org.apache.spark.mllib.regression.LabeledPoint  
import org.apache.log4j.Level  
import org.apache.log4j.Logger  
  
object MLlib {  
  
  def main(args: Array[String]) {  
    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR);  
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.ERROR);  
  
    val conf = new SparkConf().setAppName("MLlib")  
  
    val sc = new SparkContext(conf)  
  
    val spam = sc.textFile("spam.txt")  
    val ham = sc.textFile("Dham.txt")  
  
    //创建一个HashingTF实例来把邮件文本映射为包含25000特征的向量  
    val tf = new HashingTF(numFeatures = 25000)  
     
    //各邮件都被切分为单词，每个单词被映射为一个特征  
    val spamFeatures = spam.map(email => tf.transform(email.split(" ")))  
    val hamFeatures = ham.map(email => tf.transform(email.split(" ")))  
  
    //创建LabeledPoint数据集分别存放垃圾邮件(spam)和正常邮件(ham)的例子  
    spamFeatures.collect().foreach { x => print(x + " ,") }  
    hamFeatures.collect().foreach { x => print(x + " ,") }  
  
    // Create LabeledPoint datasets for positive (spam) and negative (ham) examples.  
    val positiveExamples = spamFeatures.map(features => LabeledPoint(1, features))  
    val negativeExamples = hamFeatures.map(features => LabeledPoint(0, features))  
    val trainingData = positiveExamples.union(negativeExamples)  
    trainingData.cache() // 逻辑回归是迭代算法，所以缓存训练数据的RDD  
  
      
    //使用SGD算法运行逻辑回归  
    val lrLearner = new LogisticRegressionWithSGD()  
    val model = lrLearner.run(trainingData)  
  
    //以垃圾邮件和正常邮件的例子分别进行测试。  
    val posTestExample = tf.transform("O M G GET cheap stuff by sending money to ...".split(" "))  
    val negTestExample = tf.transform("Hi Dad, I started studying Spark the other ...".split(" "))  
  
    val posTest1Example = tf.transform("I really wish well to all my friends.".split(" "))  
    val posTest2Example = tf.transform("He stretched into his pocket for some money.".split(" "))  
    val posTest3Example = tf.transform("He entrusted his money to me.".split(" "))  
    val posTest4Example = tf.transform("Where do you keep your money?".split(" "))  
    val posTest5Example = tf.transform("She borrowed some money of me.".split(" "))  
  
    //首先使用，一样的HashingTF特征来得到特征向量，然后对该向量应用得到的模型  
    println(s"Prediction for positive test example: ${model.predict(posTestExample)}")  
    println(s"Prediction for negative test example: ${model.predict(negTestExample)}")  
  
    println(s"posTest1Example for negative test example: ${model.predict(posTest1Example)}")  
    println(s"posTest2Example for negative test example: ${model.predict(posTest2Example)}")  
    println(s"posTest3Example for negative test example: ${model.predict(posTest3Example)}")  
    println(s"posTest4Example for negative test example: ${model.predict(posTest4Example)}")  
    println(s"posTest5Example for negative test example: ${model.predict(posTest5Example)}")  
  
    sc.stop()  
  }  
} 

/*
分析结果：
[plain] view plain copy
Prediction for positive test example: 1.0  
Prediction for negative test example: 0.0  
posTest1Example for negative test example: 0.0  
posTest2Example for negative test example: 0.0  
posTest3Example for negative test example: 1.0  
posTest4Example for negative test example: 0.0  
posTest5Example for negative test example: 1.0  

1 即为 垃圾邮件， 0 为正常邮件。
*/
