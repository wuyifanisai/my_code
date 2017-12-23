// solve a regression task based on bike-sharing dataset
// based on spark and scala
// 2017.12.18
// wrote by wuyifan


import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.storage.StorageLevel
import org.apache.spark.rdd.RDD
import org.apache.spark.{ SparkConf, SparkContext }
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.evaluation._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.feature.StandardScaler
//import org.apache.spark.mllib.tree.DecisionTree
//import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
//import org.joda.time.DateTime
//import org.joda.time.Duration
import org.jfree.data.category.DefaultCategoryDataset

object RunRFRegression {
  
def main(args: Array[String]): Unit = {

    SetLogger()

    val sc = new SparkContext(new SparkConf().setAppName("App").setMaster("local[4]"))
    // define SparkContext

    println("RunRFRegression")
    println("==========資料準備階段===============")

    // return dataset
    val (trainData, validationData, testData) = PrepareData(sc)

    // persist in computer memory
    trainData.persist(); validationData.persist(); testData.persist()

    println("==========訓練評估階段===============")
    println()
    print("是否需要進行參數調校 (Y:是  N:否) ? ")
    if (readLine() == "Y") {
      val model = parametersTunning(trainData, validationData) // tunning and training
      println("==========測試階段===============")

      val rmse = evaluateModel(model, testData)
      println("使用testata測試最佳模型,結果 rmse:" + rmse)

      println("==========預測資料===============")
      PredictData(sc, model)
    } else {
      val model = trainEvaluate(trainData, validationData)

      println("==========測試階段===============")
      val rmse = evaluateModel(model, testData)
      println("使用testata測試最佳模型,結果 rmse:" + rmse)

      println("==========預測資料===============")
      PredictData(sc, model)
    }

    //unpersist data from computer memory 
    trainData.unpersist(); validationData.unpersist(); testData.unpersist()
  }




  //#################################################################
  //#################################################################
  def PrepareData(sc: SparkContext): (RDD[LabeledPoint], 
                                      RDD[LabeledPoint],
                                      RDD[LabeledPoint]) = {

    //========================= 1. read and transform the data=================
    //将读入的数据初步转换，为了后续转换后成labelpoint格式

    println("begin to read data...")
    val rawDataWithHeader = sc.textFile("file:/home/wuyifanhadoop/workspace/Regression/data/hour.csv")

    val rawData = rawDataWithHeader.mapPartitionsWithIndex{(idx, iter) =>
      if (idx==0) iter.drop(1) else iter} //delete the header  

    println("number of data is "+ rawData.count().toString())

    //======================== 2. transform the data to RDD[labelpoint]==================
    val records = rawData.map(line => line.split(','))
  
    val data = records.map{
      fields=>
        val label = fields(fields.size - 1).toInt

        val feature_season = fields.slice(4, fields.size - 3).map(d=>d.toDouble)

        val features = fields.slice(4,fields.size -3).map(d=>d.toDouble)

        LabeledPoint(label, Vectors.dense(feature_season ++ features))
    } 


    // ======================3. 将 RDD数据分成三丰 ===========================
    val Array(trainData, validationData, testData) = data.randomSplit(Array(0.8, 0.1, 0.1))                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
    println("number of three data:")
    println("trainData:" + trainData.count() + "validationData:" + validationData.count()
      + "testData:" + testData.count())
    
    trainData.persist()
    validationData.persist()
    testData.persist()

    return (trainData, validationData, testData)

  }



  //######################################################################
  //######################################################################
  def PredictData(sc:SparkContext, 
                  model:RandomForestModel): Unit={
    //======= =========1.read data ====================
    val rawDataWithHeader = sc.textFile("file:/home/wuyifanhadoop/workspace/Regression/data/hour.csv")
    val rawData = rawDataWithHeader.mapPartitionsWithIndex { (idx, iter) => 
      if (idx == 0) iter.drop(1) else iter }

    println("number of test data is" + rawData.count.toString())

    //================2. transform to RDD[Labeledpoint] ==============
      val records = rawData.map(line => line.split(','))

    val dataRDD = records.take(100).map{  
      fields =>
        val label = fields(fields.size - 1).toInt

        val feature_season = fields.slice(4, fields.size - 3).map(d=>d.toDouble)

        val features = fields.slice(4,fields.size -3).map(d=>d.toDouble)

        val fea = Vectors.dense(feature_season ++ features)

        val predict = model.predict(fea)

       // val result = (if(label == predict) "correct" else "wrong" )

        val error = (math.abs((label - predict)/label).toString())

        println( "prediction error==> "+ error.toString())
        

   
    }

  }




  //###################################################################
  //###################################################################
  // train and return a model and evaluate it to see AUC
  def trainEvaluate(trainData:RDD[LabeledPoint],   
                    validationData:RDD[LabeledPoint]):RandomForestModel={  
    println("begin to train..")
    val (model) = trainModel(trainData,10,"variance", 15, 50)
    val rmse= evaluateModel(model, validationData)
    println("評估結果rmse=" + rmse)
    return (model)
  }


  //######################################################
  //######################################################
  // train a model with trainData
  def trainModel(
                  trainData:RDD[LabeledPoint],
                  num_tree:Int,
                  impurity:String,
                  maxDepth:Int,
                  maxBins:Int
                ):(RandomForestModel)={ //return a model and time

    //val startTime = new DateTime()
    val model = RandomForest.trainRegressor(trainData,
                                            Map[Int, Int](),  // for what ??
                                            num_tree,
                                            "auto",
                                            impurity,
                                            maxDepth,
                                            maxBins
                                            )
    //val endTime = new DateTime()
    //val duration = new Duration(startTime, endTime)

    return (model)
  }  




  //##################################################
  //##################################################
  // To evaluate a model to see rmse
  def evaluateModel(model:RandomForestModel, validationData:RDD[LabeledPoint]):Double={
    val predict_labels = validationData.map{
      data =>
        var predict = model.predict(data.features)
        (predict, data.label)
    }

    val Metrics = new RegressionMetrics(predict_labels)
    val rmse = Metrics.rootMeanSquaredError

    return rmse
  }


  //###################################################
  //###################################################
  // tuning parameters and return model
  def parametersTunning(trainData: RDD[LabeledPoint], 
                      validationData:RDD[LabeledPoint]): RandomForestModel={

  println("-----evaluate model by different num_tree--------------")
  evaluateParameter(trainData, 
                    validationData, 
                    "num_tree", 
                    Array(5,10,20,50,100),
                    Array("variance"), 
                    Array(15), 
                    Array(10))

  println("-----evaluate model by different maxDepth--------------")
  evaluateParameter(trainData, 
                    validationData, 
                    "maxDepth", 
                    Array(10),
                    Array("variance"), 
                    Array(3, 5, 10, 15, 20, 25), 
                    Array(10))

  println("-----evaluate model by different maxBins--------------")
  evaluateParameter(trainData, 
                    validationData, 
                    "maxBins", 
                    Array(10),
                    Array("variance"), 
                    Array(15), 
                    Array(3, 5, 10, 50, 100, 200))
  val bestModel = evaluateAllParameter(
                                        trainData,
                                        validationData,
                                        Array(5,10,15,20,50,100),
                                        Array("variance"),
                                        Array(3,5,10,15,20,30),
                                        Array(3,5,10,50,100,150,200))
  return bestModel
  }



  //#################################################################
  //##############################################################
  //evaluate single parameter every time
  def evaluateParameter(
                      trainData:RDD[LabeledPoint], 
                      validationData:RDD[LabeledPoint],
                      evaluateParameter: String,
                      num_treeArray:Array[Int],
                      impurityArray:Array[String],
                      maxDepthArray:Array[Int],
                      maxBinsArray:Array[Int]
                    )={

   // var dataBarChart = new DefaultCategoryDataset()
    //var dataLineChart = new DefaultCategoryDataset()

    for ( impurity<-impurityArray; num_tree <- num_treeArray; maxDepth <- maxDepthArray; maxBins <- maxBinsArray ) {
        val (model) = trainModel(trainData, num_tree, impurity, maxDepth, maxBins)
        val rmse = evaluateModel(model , validationData)
        println("num_tree:" +num_tree.toString() +"impurity:"+impurity.toString()+" maxDepth:"+maxDepth.toString()+" maxBins:"+maxBins.toString()+" rmse:"+rmse.toString())

        val parameterData =   // for plotting
          evaluateParameter match {
            case "num_tree" => num_tree;
            case "impurity" => impurity;
            case "maxDepth" => maxDepth;
            case "maxBins"  => maxBins
          }
        //dataBarChart.addValue(auc, evaluateParameter, parameterData.toString())
       //dataLineChart.addValue(time, "Time", parameterData.toString())

        //Chart.plotBarLineChart("DecisionTree evaluations " + evaluateParameter,
           //                     evaluateParameter, 
              //                  "AUC", 0.58, 0.7, "Time", 
                 //               dataBarChart, 
                    //            dataLineChart)
    }
  }
  //#########################################################
  //##########################################################
  // evaluate all the parameters at once
  def evaluateAllParameter(
                            trainData: RDD[LabeledPoint], 
                            validationData: RDD[LabeledPoint], 
                            num_treeArray: Array[Int],
                            impurityArray: Array[String], 
                            maxdepthArray: Array[Int], 
                            maxBinsArray: Array[Int]):RandomForestModel={

    // 通过for循环以及yield，每一次循环返还一组数
    val evaluationsArray =
        for (num_tree<-num_treeArray; impurity <- impurityArray; maxDepth <- maxdepthArray; maxBins <- maxBinsArray) yield {
          val (model) = trainModel(trainData, num_tree,impurity, maxDepth, maxBins)
          val rmse = evaluateModel(model, validationData)
          println("=== best para === num_tree, impurity, maxDepth, maxBins==>",num_tree,impurity, maxDepth, maxBins, rmse)
          (num_tree,impurity, maxDepth, maxBins, rmse)
        } 

    val bestEval = (evaluationsArray.sortBy(_._4))
    val bestEval_para = bestEval(0)
    println(bestEval_para)

    //println("best para:")
    //println("impurity" + bestEval_para._1)
    //println("maxDepth:"+bestEval_para._2)
    //println("maxBins:"+bestEval_para._3)
    //println("rmse of best model:"+ bestEval_para._4)

    val (bestModel ) = trainModel(
                                        trainData.union(validationData),
                                        bestEval_para._1,
                                        bestEval_para._2,
                                        bestEval_para._3,
                                        bestEval_para._4
                                        )
    return bestModel
  } 


  //####################################
  //#####################################
  def SetLogger() = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("com").setLevel(Level.OFF)
    System.setProperty("spark.ui.showConsoleProgress", "false")
    Logger.getRootLogger().setLevel(Level.OFF);
  }


}
