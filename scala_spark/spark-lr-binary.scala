// solve a BinaryClassificaion task based on stumble dataset
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
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.spark.mllib.classification.LogisticRegressionModel
//import org.joda.time.DateTime
//import org.joda.time.Duration
import org.jfree.data.category.DefaultCategoryDataset

object RunLogisticBinary {
  
def main(args: Array[String]): Unit = {

    SetLogger()

    val sc = new SparkContext(new SparkConf().setAppName("App").setMaster("local[4]"))
    // define SparkContext

    println("RunDecisionTreeBinary")
    println("==========資料準備階段===============")

    // return dataset
    val (trainData, validationData, testData, categoriesMap) = PrepareData(sc)

    // persist in computer memory
    trainData.persist(); validationData.persist(); testData.persist()

    println("==========訓練評估階段===============")
    println()
    print("是否需要進行參數調校 (Y:是  N:否) ? ")
    if (readLine() == "Y") {
      val model = parametersTunning(trainData, validationData) // tunning and training
      println("==========測試階段===============")

      val auc = evaluateModel(model, testData)
      println("使用testata測試最佳模型,結果 AUC:" + auc)

      println("==========預測資料===============")
      PredictData(sc, model, categoriesMap)
    } else {
      val model = trainEvaluate(trainData, validationData)

      println("==========測試階段===============")
      val auc = evaluateModel(model, testData)
      println("使用testata測試最佳模型,結果 AUC:" + auc)

      println("==========預測資料===============")
      PredictData(sc, model, categoriesMap)
    }

    //unpersist data from computer memory 
    trainData.unpersist(); validationData.unpersist(); testData.unpersist()
  }




  //#################################################################
  //#################################################################
  def PrepareData(sc: SparkContext): (RDD[LabeledPoint], 
                                      RDD[LabeledPoint],
                                      RDD[LabeledPoint],
                                      Map[String, Int]) ={

    //========================= 1. read and transform the data=================
    //将读入的数据初步转换，为了后续转换后成labelpoint格式

    println("begin to read data...")
    val rawDataWithHeader = sc.textFile("file:/home/wuyifanhadoop/workspace/Classification/data/train.tsv")

    val rawData = rawDataWithHeader.mapPartitionsWithIndex{(idx, iter) =>
      if (idx==0) iter.drop(1) else iter} //delete the header 

    val lines = rawData.map(_.split("\t")) 

    println("number of data is "+ lines.count().toString())

    //======================== 2. transform the data to RDD[labelpoint]==================
    val categoriesMap = lines.map(fields => fields(3))
      .distinct.collect.zipWithIndex.toMap
    //从lines中提取出每个网页与编号的对照表格,zipWithIndex ! 

    val labeledpointRDD = lines.map{ //
      fields =>  //下面的各种转换提取就基于 fields

        val trFields = fields.map(_.replaceAll("\"","")) 
      //除去各个字段中的空格

        val categoryFeaturesArray = Array.ofDim[Double](categoriesMap.size)
      //构建一个数组，用于数据的 one-hot 转换

        val categoryIdx = categoriesMap(fields(3)) 
      //提取fields中的网页，并提取出网页编号，也就是转换成one-hot之前的数值
        categoryFeaturesArray(categoryIdx) = 1 //形成one-hot形式

        val numericalFeatures = trFields.slice(4, fields.size - 1) 
          .map(d => if (d == "?") 0.0 else d.toDouble)
      // 提取数值型的特征，并从中处理缺失值 就是从4到最后第二个特征,闭开区间

        val label = trFields(fields.size - 1).toInt
      // 提取label

        LabeledPoint(label, Vectors.dense(categoryFeaturesArray ++ numericalFeatures))
      // 最后形成的labeledpoint，就是这map最终形成的结果
    }
    // ====================== 3. data standard =======================
    val featuresData = labeledpointRDD.map(labelpoint => labelpoint.features)
    val stdScaler = new StandardScaler(withMean = true, withStd = true).fit(featuresData)
    val scaledRDD =   labeledpointRDD.map(labelpoint => LabeledPoint(labelpoint.label, stdScaler.transform(labelpoint.features)))
    
    
    // ======================4. 将 RDD数据分成三丰 ===========================
    val Array(trainData, validationData, testData) = scaledRDD.randomSplit(Array(8, 1, 1))                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
    println("number of three data:")
    println("trainData:" + trainData.count() + "validationData:" + validationData.count()
      + "testData:" + testData.count())

    return (trainData, validationData, testData, categoriesMap)

  }




  //######################################################################
  //######################################################################
  def PredictData(sc:SparkContext, 
                  model:LogisticRegressionModel, 
                  categoriesMap:Map[String, Int]): Unit={
    //======= =========1.read data ====================
    val rawDataWithHeader = sc.textFile("file:/home/wuyifanhadoop/workspace/Classification/data/test.tsv")
    val rawData = rawDataWithHeader.mapPartitionsWithIndex { (idx, iter) => 
      if (idx == 0) iter.drop(1) else iter }

    val lines = rawData.map(_.split("\t"))

    println("number of test data is" + lines.count.toString())

    //================2. transform to RDD[Labeledpoint] ==============
    val dataRDD = lines.take(20).map{  
      fields =>
        val trFields = fields.map(_.replaceAll("\"", ""))
        val categoryFeaturesArray = Array.ofDim[Double](categoriesMap.size)
        val categoryIdx = categoriesMap(fields(3))    //categoryFeaturesArray is from the 4th of origin features
        categoryFeaturesArray(categoryIdx) = 1
        val numericalFeatures = trFields.slice(4, fields.size)  // numericalFeatures is from 5th to last origin features
          .map(d => if (d == "?") 0.0 else d.toDouble)
        //get the categoryFeaturesArray and numericalFeatures

    // =================3.predict =====================
        val url = trFields(0) //the name of the website
        val Features = Vectors.dense(categoryFeaturesArray ++ numericalFeatures)

        val predict = model.predict(Features).toInt //predict

        var predictDesc = { 
          predict match { case 0 => "暫時性網頁(ephemeral)"; case 1 => "長青網頁(evergreen)"; } 
        }
        println(" 網址：  " + url + "==>預測:" + predictDesc) 
        //output the predict information when define the dataRDD  
    }

  }




  //###################################################################
  //###################################################################
  // train and return a model and evaluate it to see AUC
  def trainEvaluate(trainData:RDD[LabeledPoint],   
                    validationData:RDD[LabeledPoint]):LogisticRegressionModel={  
    println("begin to train..")
    val (model) = trainModel(trainData,100,100,0.3)
    val AUC = evaluateModel(model, validationData)
    println("評估結果AUC=" + AUC)
    return (model)
  }


  //######################################################
  //######################################################
  // train a model with trainData
  def trainModel(
                  trainData:RDD[LabeledPoint],
                  numiter:Int,
                  stepsize:Int,
                  batch_size:Double
                ):(LogisticRegressionModel)={ //return a model and time

    //val startTime = new DateTime()
    val model = LogisticRegressionWithSGD.train(trainData,
                                                  numiter,
                                                  stepsize,
                                                  batch_size
                                                  )
    //val endTime = new DateTime()
    //val duration = new Duration(startTime, endTime)

    return (model)
  }  




  //##################################################
  //##################################################
  // To evaluate a model to see AUC
  def evaluateModel(model:LogisticRegressionModel, validationData:RDD[LabeledPoint]):Double={
    val predict_labels = validationData.map{
      data =>
        var predict = model.predict(data.features)
        (predict, data.label)
    }

    val Metrics = new BinaryClassificationMetrics(predict_labels)
    val auc = Metrics.areaUnderROC

    return auc
  }


  //###################################################
  //###################################################
  // tuning parameters and return model
  def parametersTunning(trainData: RDD[LabeledPoint], 
                      validationData:RDD[LabeledPoint]):LogisticRegressionModel={
  println("-----evaluate model by different numiter--------------")
  evaluateParameter(
                    trainData, 
                    validationData, 
                    "numiter", 
                    Array(5,15,20,60,100), 
                    Array(100), 
                    Array(1.0)
                  )

  println("-----evaluate model by different stepsize--------------")
  evaluateParameter(trainData, 
                    validationData, 
                    "stepsize", 
                    Array(100), 
                    Array(10,50,10,200), 
                    Array(1.0))

  println("-----evaluate model by different batch_size--------------")
  evaluateParameter(trainData, 
                    validationData, 
                    "batch_size", 
                    Array(100), 
                    Array(100), 
                    Array(0.3,0.5,0.7,0.9))
  val bestModel = evaluateAllParameter(
                                        trainData,
                                        validationData,
                                        Array(5,10,20,50,100), 
                                        Array(10,30,50,70,100,130), 
                                        Array(0.3,0.5,0.7,0.9))
  return bestModel
  }



  //#################################################################
  //##############################################################
  //evaluate single parameter every time
  def evaluateParameter(trainData:RDD[LabeledPoint], 
                      validationData:RDD[LabeledPoint],
                      evaluateParameter: String,
                      numiterArray:Array[Int],
                      stepsizeArray:Array[Int],
                      batch_sizeArray:Array[Double]
                    )={

   // var dataBarChart = new DefaultCategoryDataset()
    //var dataLineChart = new DefaultCategoryDataset()

    for ( numiter <- numiterArray; stepsize <- stepsizeArray; batch_size <- batch_sizeArray ) {
        val (model) = trainModel(trainData, numiter, stepsize, batch_size)
        val auc = evaluateModel(model , validationData)
        println("numiter:"+numiter.toString()+" stepsize:"+stepsize.toString()+" batch_size:"+batch_size.toString()+" auc:"+auc.toString())

        val parameterData =   // for plotting
          evaluateParameter match {
            case "numiter" => numiter;
            case "stepsize" => stepsize;
            case "batch_size"  => batch_size
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
                            numiterArray: Array[Int], 
                            stepsizeArray: Array[Int], 
                            batch_sizeArray: Array[Double]):LogisticRegressionModel = {

    // 通过for循环以及yield，每一次循环返还一组数
    val evaluationsArray =
        for ( numiter <- numiterArray; stepsize <- stepsizeArray; batch_size <- batch_sizeArray ) yield {
          val (model) = trainModel(trainData, numiter, stepsize, batch_size)
          val auc = evaluateModel(model, validationData)
          (numiter, stepsize, batch_size, auc)
        } 

    val bestEval_para = (evaluationsArray.sortBy(_._4).reverse)(0)
    println("best para:")
    println("numiter" + bestEval_para._1)
    println("stepsize:"+bestEval_para._2)
    println("batch_size:"+bestEval_para._3)
    println("auc of best model:"+ bestEval_para._4)

    val (bestModel ) = trainModel(
                                        trainData.union(validationData),
                                        bestEval_para._1,
                                        bestEval_para._2,
                                        bestEval_para._3
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
