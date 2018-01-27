// this is a simple: using spark mllib pipeline to solve a regression problem
// coder:wuyifan

import org.apache.log4j.{Logger}
//core and SparkSQL
import org.apache.log4j.Level
import org.apache.spark.{SparkConf, SparkContext}
// Spark config

import org.apache.spark.sql.hive.HiveContext
// hive tool

import org.apache.spark.sql.SQLContext
//sql tool

import org.apache.spark.sql.DataFrame
// hive DataFrame for storing Data

import org.apache.spark.mllib.linalg.Vectors

import org.apache.spark.ml.feature.{PCA, StringIndexer, VectorAssembler, OneHotEncoder, StandardScaler, Normalizer, Binarizer, Bucketizer}
// ML Feature Creation

import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}
//import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
// there is no TrainValidationSplit ??
// tuning hyper parameters

import org.apache.spark.ml.evaluation.{RegressionEvaluator}
// evaluation for regression model

import org.apache.spark.ml.regression.{LinearRegression, DecisionTreeRegressor, RandomForestRegressor, GBTRegressor}
// regression model
// we will use some different models and compare them with each other ,finally a best model with best parameters would be appeared ! 

import org.apache.spark.ml.Pipeline
// pipeline

import org.apache.spark.mllib.evaluation.RegressionMetrics
// evaluation for regression model



/*
show some of the train_store data
+-----+---------+----------+-------------------+-------------------------+------------------------+---------------+---------------+------+-------+----+---------+-------------+
|Store|StoreType|Assortment|CompetitionDistance|CompetitionOpenSinceMonth|CompetitionOpenSinceYear|Promo2SinceWeek|Promo2SinceYear|Promo2|  label|Open|DayOfWeek|SchoolHoliday|
+-----+---------+----------+-------------------+-------------------------+------------------------+---------------+---------------+------+-------+----+---------+-------------+
|   31|        d|         c|               9800|                        7|                    2012|              0|           2010|     0| 7248.0|   1|      5.0|            1|
|  231|        d|         c|               3840|                       10|                    2008|             39|           2010|     1| 8353.0|   1|      5.0|            0|
|  431|        d|         c|               4520|                        0|                    2010|              0|           2010|     0|12369.0|   1|      5.0|            1|
|  631|        d|         c|               2870|                        0|                    2010|             35|           2012|     1| 7428.0|   1|      5.0|            1|
|  831|        a|         a|                800|                        6|                    2007|              0|           2010|     0|15152.0|   1|      5.0|            1|
| 1031|        d|         a|                590|                        5|                    2001|              0|           2010|     0| 6014.0|   1|      5.0|            1|
+-----+---------+----------+-------------------+-------------------------+------------------------+---------------+---------------+------+-------+----+---------+-------------+
show some of the test_store data
+-----+---------+----------+-------------------+-------------------------+------------------------+---------------+---------------+------+-------+----+---------+-------------+
|Store|StoreType|Assortment|CompetitionDistance|CompetitionOpenSinceMonth|CompetitionOpenSinceYear|Promo2SinceWeek|Promo2SinceYear|Promo2|  label|Open|DayOfWeek|SchoolHoliday|
+-----+---------+----------+-------------------+-------------------------+------------------------+---------------+---------------+------+-------+----+---------+-------------+
|  891|        a|         c|                350|                        0|                    2010|             31|           2013|     1|10821.0|   1|      1.0|            0|
|  892|        a|         a|              19370|                        4|                    2002|              0|           2010|     0|13186.0|   1|      1.0|            0|
|  893|        a|         a|                130|                        0|                    2010|              1|           2013|     1| 8121.0|   1|      1.0|            0|
|  894|        a|         a|                190|                       11|                    2012|              0|           2010|     0|13713.0|   1|      1.0|            0|
|  895|        a|         c|               4150|                        0|                    2010|              0|           2010|     0|11425.0|   1|      1.0|            0|
|  896|        a|         c|                170|                        9|                    2012|              0|           2010|     0| 7813.0|   1|      1.0|            0|
+-----+---------+----------+-------------------+-------------------------+------------------------+---------------+---------------+------+-------+----+---------+-------------+*/

//=================================================================================================================================================================================

/*show some data of store_train_data_sql_04:
+-----+---------+----------+-------------------+-------------------------+------------------------+---------------+---------------+------+-------+----+---------+-------------+--------------------------+---------------+-----------------------------+------------------------------+-------------------------------------+
|Store|StoreType|Assortment|CompetitionDistance|CompetitionOpenSinceMonth|CompetitionOpenSinceYear|Promo2SinceWeek|Promo2SinceYear|Promo2|  label|Open|DayOfWeek|SchoolHoliday|CompetitionDistance*Promo2|CompetitionTime|CompetitionDistance_Binarized|CompetitionDistance_Bucketized|CompetitionDistance*Promo2_Bucketized|
+-----+---------+----------+-------------------+-------------------------+------------------------+---------------+---------------+------+-------+----+---------+-------------+--------------------------+---------------+-----------------------------+------------------------------+-------------------------------------+
|   31|        d|         c|             9800.0|                      7.0|                  2012.0|            0.0|         2010.0|   0.0| 7248.0| 1.0|      5.0|            1|                       0.0|            6.0|                          1.0|                           4.0|                                  1.0|
|  231|        d|         c|             3840.0|                     10.0|                  2008.0|           39.0|         2010.0|   1.0| 8353.0| 1.0|      5.0|            0|                    3840.0|           10.0|                          0.0|                           2.0|                                  2.0|
|  431|        d|         c|             4520.0|                      0.0|                  2010.0|            0.0|         2010.0|   0.0|12369.0| 1.0|      5.0|            1|                       0.0|            8.0|                          0.0|                           2.0|                                  1.0|
|  631|        d|         c|             2870.0|                      0.0|                  2010.0|           35.0|         2012.0|   1.0| 7428.0| 1.0|      5.0|            1|                    2870.0|            8.0|                          0.0|                           1.0|                                  1.0|
|  831|        a|         a|              800.0|                      6.0|                  2007.0|            0.0|         2010.0|   0.0|15152.0| 1.0|      5.0|            1|                       0.0|           11.0|                          0.0|                           1.0|                                  1.0|
+-----+---------+----------+-------------------+-------------------------+------------------------+---------------+---------------+------+-------+----+---------+-------------+--------------------------+---------------+-----------------------------+------------------------------+-------------------------------------+

show some data of store_test_data_sql_04:
+-----+---------+----------+-------------------+-------------------------+------------------------+---------------+---------------+------+-------+----+---------+-------------+--------------------------+---------------+-----------------------------+------------------------------+-------------------------------------+
|Store|StoreType|Assortment|CompetitionDistance|CompetitionOpenSinceMonth|CompetitionOpenSinceYear|Promo2SinceWeek|Promo2SinceYear|Promo2|  label|Open|DayOfWeek|SchoolHoliday|CompetitionDistance*Promo2|CompetitionTime|CompetitionDistance_Binarized|CompetitionDistance_Bucketized|CompetitionDistance*Promo2_Bucketized|
+-----+---------+----------+-------------------+-------------------------+------------------------+---------------+---------------+------+-------+----+---------+-------------+--------------------------+---------------+-----------------------------+------------------------------+-------------------------------------+
|  891|        a|         c|              350.0|                      0.0|                  2010.0|           31.0|         2013.0|   1.0|10821.0| 1.0|      1.0|            0|                     350.0|            8.0|                          0.0|                           1.0|                                  1.0|
|  892|        a|         a|            19370.0|                      4.0|                  2002.0|            0.0|         2010.0|   0.0|13186.0| 1.0|      1.0|            0|                       0.0|           16.0|                          1.0|                           7.0|                                  1.0|
|  893|        a|         a|              130.0|                      0.0|                  2010.0|            1.0|         2013.0|   1.0| 8121.0| 1.0|      1.0|            0|                     130.0|            8.0|                          0.0|                           1.0|                                  1.0|
|  894|        a|         a|              190.0|                     11.0|                  2012.0|            0.0|         2010.0|   0.0|13713.0| 1.0|      1.0|            0|                       0.0|            6.0|                          0.0|                           1.0|                                  1.0|
|  895|        a|         c|             4150.0|                      0.0|                  2010.0|            0.0|         2010.0|   0.0|11425.0| 1.0|      1.0|            0|                       0.0|            8.0|                          0.0|                           2.0|                                  1.0|
+-----+---------+----------+-------------------+-------------------------+------------------------+---------------+---------------+------+-------+----+---------+-------------+--------------------------+---------------+-----------------------------+------------------------------+-------------------------------------+*/


/* 
methods to transform features should be noticed in some aspects as below:
1. such as StringIndexer, when it is fitted in a column of train data, and when it is used in a column of test data
   if a string which is not seen by the StringIndexer before , a problem would occur , shown as below:
==== fit in train data ===
 id | category | categoryIndex  
----|----------|---------------  
 0  | a        | 0.0  
 1  | b        | 2.0  
 2  | c        | 1.0  
 3  | a        | 0.0  
 4  | a        | 0.0  
 5  | c        | 1.0  

==== use in test data ===
 id | category | categoryIndex  
----|----------|---------------  
 0  | a        | 0.0  
 1  | b        | 2.0  
 2  | d        | ？  
 3  | e        | ？  
 4  | a        | 0.0  
 5  | c        | 1.0 

Spark give two way to solve the problem：
for example:  
val labelIndexerModel=new StringIndexer().  
                setInputCol("label")  
                .setOutputCol("indexedLabel")  
                .setHandleInvalid("error")  //or .setHandleInvalid("skip") 
                .fit(rawData); 

 （1）.setHandleInvalid("error")：print error  
 org.apache.spark.SparkException: Unseen label: d，e  
 （2）.setHandleInvalid("skip") ignore the data contain the d,e

2.such as one-hot encoder, so one-hot result dimension of train data and test data may be different
  because of some value of the feature (which is taken to one-hot) in train data does not exists in test data
  so the dimension of test data feature would be less than train data 

3.some encoder need to be fitted ,some do not !
need to be fitted: VectorIndexer, StringIndexer, QuantileDiscretizer ...
not to be fitted: Bucketizer, oneHotEncoder ...


*/

object pipeline {

//=========================  StoreType dealing  
// StoreType : string => Index
val StoreTypeIndexer = new StringIndexer()
	.setInputCol("StoreType")
	.setOutputCol("StoreTypeIndex")

// StoreTypeIndex : one-hot
val StoreTypeIndexEncoder = new OneHotEncoder()
  .setInputCol("StoreTypeIndex")
  .setOutputCol("StoreTypeVec")

//=========================  Assortment dealing
val AssortmentIndexer = new StringIndexer()
	.setInputCol("Assortment")
	.setOutputCol("AssortmentIndex")

// AssortmentIndex : one-hot
val AssortmentIndexEncoder = new OneHotEncoder()
  .setInputCol("AssortmentIndex")
  .setOutputCol("AssortmentVec")

//=========================  CompetitionDistance dealing 
// StandardScaler for CompetitionDistance
val CompetitionDistanceScaler = new StandardScaler()
  .setInputCol("CompetitionDistance")
  .setOutputCol("CompetitionDistancescaled")
  .setWithStd(true)
  .setWithMean(false)

val CompetitionDistanceNormalizer = new Normalizer()
  .setInputCol("CompetitionDistance")
  .setOutputCol("CompetitionDistancescaled")


//=========================  CompetitionOpenSinceMonth dealing
//one-hot for CompetitionOpenSinceMonth
val CompetitionOpenSinceMonthEncoder = new OneHotEncoder()
  .setInputCol("CompetitionOpenSinceMonth")
  .setOutputCol("CompetitionOpenSinceMonthVec")

//=========================  CompetitionOpenSinceYear dealing 
// one-hot for it
val CompetitionOpenSinceYearEncoder = new OneHotEncoder()
  .setInputCol("CompetitionOpenSinceYear")
  .setOutputCol("CompetitionOpenSinceYearVec")	

//=========================  Promo2SinceWeek dealing
// USE ORIGINAL DATA

//=========================  Promo2SinceYear dealing 
//one-hot for Promo2SinceYear
val Promo2SinceYearEncoder = new OneHotEncoder()
  .setInputCol("Promo2SinceYear")
  .setOutputCol("Promo2SinceYearVec")	

//=========================  Promo2 dealing 
// USE ORIGINAL DATA

//=========================  Open dealing
// open : one-hot
val OpenEncoder = new OneHotEncoder()
  .setInputCol("Open")
  .setOutputCol("OpenVec")

//=========================  schoolHoliday dealing 
// schoolHoliday : string => Index 
val SchoolHolidayIndexer = new StringIndexer()
  .setInputCol("SchoolHoliday")
  .setOutputCol("SchoolHolidayIndex")

//=========================  SchoolHolidayIndex : one-hot
val SchoolHolidayEncoder = new OneHotEncoder()
  .setInputCol("SchoolHolidayIndex")
  .setOutputCol("SchoolHolidayVec")


//=========================  DayOfWeek dealing 
// DayOfWeek: one-hot
val DayOfWeekEncoder = new OneHotEncoder()
  .setInputCol("DayOfWeek")
  .setOutputCol("DayOfWeekVec")

//=========================  store dealing 
// seems it is usefulless

//=========================  CompetitionTime dealing 
// one-hot for it 
val CompetitionTimeEncoder = new OneHotEncoder()
  .setInputCol("CompetitionTime")
  .setOutputCol("CompetitionTimeVec")

//=========================  CompetitionDistance_Bucketized dealing
// one-hot for it
val CompetitionDistance_BucketizedEncoder = new OneHotEncoder()
  .setInputCol("CompetitionDistance_Bucketized")
  .setOutputCol("CompetitionDistance_BucketizedVec")

//=========================  CompetitionDistance*Promo2_Bucketized 
// one-hot for it
val CompetitionDistance_Promo2_BucketizedEncoder = new OneHotEncoder()
  .setInputCol("CompetitionDistance*Promo2_Bucketized")
  .setOutputCol("CompetitionDistance*Promo2_BucketizedVec")



//=========================  assemble all the features =========================================
val Assembler = new VectorAssembler()
  .setInputCols(Array(
  						"StoreTypeVec",
						"AssortmentVec",
						//"CompetitionDistancescaled",
						"CompetitionOpenSinceMonthVec",
						"CompetitionOpenSinceYearVec",
						"Promo2SinceYearVec",
						"Promo2",
						"OpenVec",
						"SchoolHolidayVec",
						"DayOfWeekVec",
						"CompetitionTimeVec",
						"CompetitionDistance_BucketizedVec",
						"CompetitionDistance*Promo2_BucketizedVec",
						"CompetitionDistance_Binarized"
            		)
        )
  .setOutputCol("features")

//=========================  PCA all the features ==============================================
val PcaEncoder = new PCA()
  .setInputCol("features")
  .setOutputCol("pcaFeatures")
  .setK(10)



//------------------------------------ Creating the Pipelines -------------------------------------
/*Much like the DAG that we saw in the the core concepts of the Pipeline:
Transformer: 
A Transformer is an algorithm which can transform one DataFrame into another DataFrame.
E.g., an ML model is a Transformer which transforms DataFrame with features into a DataFrame with predictions.
Estimator: 
An Estimator is an algorithm which can be fit on a DataFrame to produce a Transformer. 
E.g., a learning algorithm is an Estimator which trains on a DataFrame and produces a model.
Pipeline: 
A Pipeline chains multiple Transformers and Estimators together to specify an ML workflow.
*/

/*
Once you see the code, you'll notice that in our train test split we're also setting an Evaluatorthat 
will judge how well our model is doing and automatically select the best parameter for us to use based 
on that metric. This means that we get to train lots and lots of different models to see which one is best. 
Super simple! Let's walk through the creation for each model.
*/

// this is a pipeline which can cerate new features and train the model and tune the parameters
// finally , this pipeline would give us a model with best parameters 
def preppedLRPipeline():CrossValidator = {

  println("========== add ml model to pipeline ===============")
  println()
  println("which model do you want to use? ")
  println("0 ---> LinearRegression")
  println("1 ---> DecisionTreeRegressor")
  println("2 ---> RandomForestRegressor")
  println("3---> GBTRegressor")

  if (readLine() == "0") {
  val m = new LinearRegression()
  val paramGrid = new ParamGridBuilder()  // parameters to tune
    .addGrid(m.regParam, Array(0.1, 0.01))
    .addGrid(m.elasticNetParam, Array(0.0, 0.25, 0.5, 0.75, 1.0))
    .build()
  }
  else if(readLine() == "1") {
  val m = new DecisionTreeRegressor()
  val paramGrid = new ParamGridBuilder()  // parameters to tune
    .addGrid(m.impurity, Array("variance"))
    .addGrid(m.maxDepth, Array(3,5,10,20))
    .addGrid(m.maxBins, Array(8,16,32,64))
    .build()
  }
  else if(readLine() == "2") {
  val m = new RandomForestRegressor()
  val paramGrid = new ParamGridBuilder()  // parameters to tune
    .addGrid(m.numTrees, Array(10,30,50,100))
    .addGrid(m.maxDepth, Array(3,5,10,20))
    .addGrid(m.maxBins, Array(8,16,32,64))
    .addGrid(m.impurity, Array("variance"))
    .build()
  }
  else{
  val m = new GBTRegressor()
  val paramGrid= new ParamGridBuilder()  // parameters to tune
    .addGrid(m.numIterations, Array(10,30,50,100))
    .addGrid(m.maxDepth, Array(3,5,10,20))
    .build()
  }



  val pipeline = new Pipeline()
    .setStages(Array(                   // put all the preprocessing before and model into the pipeline
            		StoreTypeIndexer,
					StoreTypeIndexEncoder,
					AssortmentIndexer,
					AssortmentIndexEncoder,
					//CompetitionDistanceNormalizer,
					//CompetitionDistanceScaler, // some methods including Normalizer and PCA need its input something like vector column??
					CompetitionOpenSinceMonthEncoder,
					CompetitionOpenSinceYearEncoder,
					Promo2SinceYearEncoder,
					OpenEncoder,
					SchoolHolidayIndexer,
					SchoolHolidayEncoder,
					DayOfWeekEncoder,
					CompetitionTimeEncoder,
					CompetitionDistance_BucketizedEncoder,
					CompetitionDistance_Promo2_BucketizedEncoder,
              		Assembler, 
              		PcaEncoder,
              		m  // the last thing is the model
                    )
            )

  val tvs = new CrossValidator()  // put the [preprocessing, model], Evaluator, tuning, paramgrid together
    .setEstimator(pipeline) // here is something can be fit
    .setEvaluator(new RegressionEvaluator) //here is a model Evaluator
    .setEstimatorParamMaps(paramGrid) // here is the hyper parameters to tune
    .setNumFolds(4) 

  tvs  // return the model with best parameters
}


// -------------------- save predictions --------------------------------------
def savePredictions(predictions:DataFrame, testRaw:DataFrame) = {
  val tdOut = testRaw
    .select("Id") // select the colunms of "Id"
    //.distinct()
    .join(predictions, testRaw("Id") === predictions("PredId"), "outer") // combine acorrding to id
    .select("Id", "Sales")
    .na.fill(0:Double) // some of our inputs were null so we have to
                       // fill these with something
  tdOut.rdd.saveAsTextFile("/home/wuyifanhadoop/workspace/kaggle_rossmann/linear_regression_predictions_02.csv")
  tdOut.rdd.foreach(x=>println(x))
  
}


// ------Fitting, Testing, and Using The Model-------------------------
/*
Now we've brought in our data, created our pipeline, 
we are now ready to train up our models and see how they perform. 
This will take some time to run because we are exploring a 
hyperparameter space for each model. It takes time to try out all 
the permutations in our parameter grid as well as create a training 
set for each tree so be patient!
*/
def fitModel(tvs:CrossValidator, data:DataFrame) = {
  val Array(training, test) = data.randomSplit(Array(0.8, 0.2), seed = 12345)
  // get the train data and test data

  println("Fitting data")
  val model = tvs.fit(training) // fit the model using tvs(a pipeline)(including tunning hyperparameters !)

  println("Now performing test on hold out set")
  val holdout = model.transform(test).select("prediction","label") // result of test

  // have to do a type conversion for RegressionMetrics
  val rm = new RegressionMetrics(holdout.rdd.map(x =>
    (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double])))

  println("Test Metrics")
  println("Test Explained Variance:")
  println(rm.explainedVariance)
  println("Test R^2 Coef:")
  println(rm.r2)
  println("Test MSE:")
  println(rm.meanSquaredError)
  println("Test RMSE:")
  println(rm.rootMeanSquaredError)

  model //return the model with best parameters 
}

def SetLogger() = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("com").setLevel(Level.OFF)
    System.setProperty("spark.ui.showConsoleProgress", "false")
    Logger.getRootLogger().setLevel(Level.OFF);
  }




//############################################################ main function #####################################################################
def main(args: Array[String]): Unit = {

  SetLogger()
  
  
//val conf = new SparkConf().setAppName("pipeline_regression")
val sc = new SparkContext(new SparkConf().setAppName("App").setMaster("local[8]"))
val sqlContext = new SQLContext(sc)

//====================================   reading data =========================================

// reading train data----------------------------------------------------------------------
val dt1 = sc.textFile("/home/wuyifanhadoop/workspace/kaggle_rossmann/train.csv", 1)
val trainRaw = dt1.map(line => line.split(","))
val train = trainRaw.map(x => R_train(
										x(0).toInt,
										x(1).toInt,
										x(2).toString,
										x(3).toDouble,
										x(4).toInt,
										x(5).toInt,
										x(6).toInt,
										//x(7).toString,
										x(8).toString
										)
)
//val trainRaw = dt1.map(s => Vectors.dense(s.split(',').map(_.toDouble)))


// reading test data---------------------------------------------------------------------------
val dt2= sc.textFile("/home/wuyifanhadoop/workspace/kaggle_rossmann/test.csv", 1)
val testRaw = dt2.map(line => line.split(","))
val test = testRaw.map(x => R_train(
										x(0).toInt,
										x(1).toInt,
										x(2).toString,
										x(3).toDouble,
										x(4).toInt,
										x(5).toInt,
										x(6).toInt,
										//x(7).toString,
										x(8).toString
										)
)


//reading store data -------------------------------------------------------------------------
val dt3= sc.textFile("/home/wuyifanhadoop/workspace/kaggle_rossmann/store.csv", 1)
val storeRaw = dt3.map(line => line.split(","))
val store = storeRaw.map(x => store_info(
										x(0).toInt,
										x(1).toString,
										x(2).toString,
										x(3).toInt,
										x(4).toInt,
										x(5).toInt,
										x(6).toInt,
										x(7).toInt,
										x(8).toInt						
))

// transfrom RDD into dataFrame-----------------------------------------------------------------
import sqlContext.implicits._

val train_data = train.toDF()
val test_data = test.toDF()
val store_data = store.toDF()

train_data.registerTempTable("table_train")
println("train data has been registered into TempTable ==> <table_train> ")

test_data.registerTempTable("table_test")
println("test data has been registered into TempTable ==> <table_test> ")

store_data.registerTempTable("store_table")
println("store data has been registered into TempTable ==> <table_store> ")

// using sql to get the data we want from table--------------------------------------------------

val train_data_sql = sqlContext.sql("""
			SELECT 
                Store,
                Sales label, 
                Open Open, 
                DayOfWeek DayOfWeek,
                SchoolHoliday
            FROM table_train
            Limit 100
          """).na.drop()
println("show some of the train data")
train_data_sql.show(3)


val test_data_sql = sqlContext.sql("""
			SELECT 
                Store,
                Sales actual_sale, 
                Open Open, 
                DayOfWeek DayOfWeek,
                SchoolHoliday
            FROM table_test
          """).na.drop()
println("show some of the test data")
test_data_sql.show(3)


val store_data_sql = sqlContext.sql("""
			SELECT *
            FROM store_table
          """).na.drop()
println("show some of the store data")
store_data_sql.show(3)

//========================================= try to combine train data and store data together =============================

val store_train_data_sql = sqlContext.sql("""
		SELECT 
			s.Store,
			s.StoreType,
			s.Assortment,
			if(s.CompetitionDistance = 999999.0, 0.0 , s.CompetitionDistance) CompetitionDistance,
			if(s.CompetitionOpenSinceMonth = 999999.0, 0.0 , s.CompetitionOpenSinceMonth) CompetitionOpenSinceMonth,
			if(s.CompetitionOpenSinceYear=999999.0, 2010.0, s.CompetitionOpenSinceYear) CompetitionOpenSinceYear,
			if(s.Promo2SinceWeek = 999999.0, 0.0, s.Promo2SinceWeek ) Promo2SinceWeek,
			if(s.Promo2SinceYear = 999999.0, 2010.0, s.Promo2SinceYear) Promo2SinceYear,
			s.Promo2,
			t.Sales label,
			t.Open,
			t.DayOfWeek,
			t.SchoolHoliday
        FROM table_train t inner join store_table s on s.Store =t.Store
        Limit 100000
          """).na.drop()
println("show some of the train_store data")
store_train_data_sql.show(5)

//================================================ try to combine test data and store data together ============================

val store_test_data_sql = sqlContext.sql("""
		SELECT 
			s.Store,
			s.StoreType,
			s.Assortment,
			if(s.CompetitionDistance = 999999.0, 0.0, s.CompetitionDistance) CompetitionDistance,
			if(s.CompetitionOpenSinceMonth = 999999.0, 0.0 , s.CompetitionOpenSinceMonth) CompetitionOpenSinceMonth,
			if(s.CompetitionOpenSinceYear=999999.0, 2010.0, s.CompetitionOpenSinceYear) CompetitionOpenSinceYear,
			if(s.Promo2SinceWeek = 999999.0, 0.0, s.Promo2SinceWeek ) Promo2SinceWeek,
			if(s.Promo2SinceYear = 999999.0, 2010.0, s.Promo2SinceYear) Promo2SinceYear,
			s.Promo2,
			t.Sales label,
			t.Open,
			t.DayOfWeek,
			t.SchoolHoliday
        FROM table_test t inner join store_table s on s.Store =t.Store
       
          """).na.drop()
println("show some of the test_store data")

// ===================================== do some data exploration ==============================================
println("do some data exploration:")
store_train_data_sql.describe("CompetitionDistance","CompetitionOpenSinceMonth","Promo2SinceWeek","label").show()
store_test_data_sql.describe("CompetitionDistance","CompetitionOpenSinceMonth","Promo2SinceWeek","label").show()
/*
do some data exploration:
+-------+-------------------+-------------------------+-----------------+------------------+
|summary|CompetitionDistance|CompetitionOpenSinceMonth|  Promo2SinceWeek|             label|
+-------+-------------------+-------------------------+-----------------+------------------+
|  count|             100000|                   100000|           100000|            100000|
|   mean|          4978.7178|                  4.99416|         13.83461|        5681.19931|
| stddev|  7002.943309063922|        4.310244296371147|17.56382749140688|3816.8442560373783|
|    min|                 60|                      0.0|              0.0|               0.0|
|    max|              30030|                     12.0|             50.0|           35154.0|
+-------+-------------------+-------------------------+-----------------+------------------+

+-------+-------------------+-------------------------+-----------------+-----------------+
|summary|CompetitionDistance|CompetitionOpenSinceMonth|  Promo2SinceWeek|            label|
+-------+-------------------+-------------------------+-----------------+-----------------+
|  count|                 45|                       45|               45|               45|
|   mean|  4583.777777777777|                      3.6|9.955555555555556|           9742.8|
| stddev|   6081.30186679314|        3.523886743040669|15.41565662356951|3247.693086210924|
|    min|                 70|                      0.0|              0.0|           3033.0|
|    max|              22350|                     11.0|             45.0|          16862.0|
+-------+-------------------+-------------------------+-----------------+-----------------+
*/

// ======================================= do some dealing to data before pipeline =========================================
// add some other column-------------------------
val store_train_data_sql_01 = store_train_data_sql.withColumn("CompetitionDistance*Promo2",store_train_data_sql("CompetitionDistance")*store_train_data_sql("Promo2"))
val store_test_data_sql_01 = store_test_data_sql.withColumn("CompetitionDistance*Promo2",store_test_data_sql("CompetitionDistance")*store_test_data_sql("Promo2"))

val store_train_data_sql_02 = store_train_data_sql_01.withColumn("CompetitionTime", store_train_data_sql("CompetitionOpenSinceYear")*(-1)+2018)
val store_test_data_sql_02 = store_test_data_sql_01.withColumn("CompetitionTime", store_test_data_sql("CompetitionOpenSinceYear")*(-1)+2018)

println("show the store_train_data_sql schema:")
store_train_data_sql_02.printSchema()
println("show the store_test_data_sql schema:")
store_test_data_sql_02.printSchema()

println("show some data of store_train_data_sql_02:")
store_train_data_sql_02.show(5)
println("show some data of store_test_data_sql_02:")
store_test_data_sql_02.show(5)


// do some binarizer for features(CompetitionDistance, CompetitionOpenSinceMonth) as new columns to add as the final feature -------------------------
val Binarizer_CompetitionDistance = new Binarizer()
  .setInputCol("CompetitionDistance")
  .setOutputCol("CompetitionDistance_Binarized")
  .setThreshold(5000)

val store_train_data_sql_03 = Binarizer_CompetitionDistance.transform(store_train_data_sql_02)
val store_test_data_sql_03 = Binarizer_CompetitionDistance.transform(store_test_data_sql_02)

println("show the store_train_data_sql schema:")
store_train_data_sql_03.printSchema()
println("show the store_test_data_sql schema:")
store_test_data_sql_03.printSchema()

println("show some data of store_train_data_sql_03:")
store_train_data_sql_03.show(5)
println("show some data of store_test_data_sql_03:")
store_test_data_sql_03.show(5)


// do some buckets for features(CompetitionDistance, CompetitionDistance*Promo2) as new columns to add as the final feature -------------------------
val splits_CompetitionDistance = Array(Double.NegativeInfinity,
					0.0,
					3000.0,
					6000.0,
					9000.0,
					12000.0,
					15000.0,
					18000.0,
					21000.0,
					24000.0,
					27000.0,
					30000.0,
					Double.PositiveInfinity)

val splits_CompetitionDistance_Promo2 = Array(Double.NegativeInfinity,
					0.0,
					3000.0,
					6000.0,
					9000.0,
					12000.0,
					15000.0,
					18000.0,
					21000.0,
					24000.0,
					27000.0,
					30000.0,
					Double.PositiveInfinity)

val bucketizer_CompetitionDistance = new Bucketizer()
  .setInputCol("CompetitionDistance")
  .setOutputCol("CompetitionDistance_Bucketized")
  .setSplits(splits_CompetitionDistance)

val bucketizer_CompetitionDistance_Promo2 = new Bucketizer()
  .setInputCol("CompetitionDistance*Promo2")
  .setOutputCol("CompetitionDistance*Promo2_Bucketized")
  .setSplits(splits_CompetitionDistance_Promo2)

val store_train_data_sql_04 = bucketizer_CompetitionDistance_Promo2.transform(bucketizer_CompetitionDistance.transform(store_train_data_sql_03))
val store_test_data_sql_04 = bucketizer_CompetitionDistance_Promo2.transform(bucketizer_CompetitionDistance.transform(store_test_data_sql_03))

println("show the store_train_data_sql schema:")
store_train_data_sql_04.printSchema()
println("show the store_test_data_sql schema:")
store_test_data_sql_04.printSchema()

println("show some data of store_train_data_sql_04:")
store_train_data_sql_04.show(5)
println("show some data of store_test_data_sql_04:")
store_test_data_sql_04.show(5)




// ================================================ train of main step ======================================================
// The linear Regression Pipeline ----------------------------------------
val linearTvs = preppedLRPipeline()
val lrModel = fitModel(linearTvs, store_train_data_sql_04)
println("finish building pipeline !")
println("finish training fitting !")
println("begin Generating kaggle predictions")

val lrOut = lrModel.transform(store_test_data_sql_04) // transform is predicting in the sklearn
// there is a error maybe the dimension of train data input and test data input are different !!!!
  .withColumnRenamed("prediction","predict_Sales") //rename the column

lrOut.select("label","predict_Sales").rdd.foreach(row => println(row))

println("Saving kaggle predictions")
//savePredictions(lrOut, test_data_sql)
lrOut.rdd.saveAsTextFile("/home/wuyifanhadoop/workspace/kaggle_rossmann/linear_regression_predictions_03.csv")
println("saving is done !")


}
}


// =====  the table form for change data from RDD into sql_table
case class R_train(
Store:Int,
DayOfWeek:Double,
Date:String,
Sales:Double,
Customers:Int,
Open:Double,
Promo:Int,
//StateHoliday:String,
SchoolHoliday:String
)   

case class store_info(
Store:Int,
StoreType:String,	
Assortment:String,
CompetitionDistance: Double,
CompetitionOpenSinceMonth:Double,	
CompetitionOpenSinceYear:Double,
Promo2:Double,
Promo2SinceWeek:Double,
Promo2SinceYear:Double
)
