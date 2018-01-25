// this is a simple: using spark mllib pipeline to solve a regression problem
//

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

import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler, OneHotEncoder, StandardScaler, Normalizer}
// ML Feature Creation

import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}
//import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
// there is no TrainValidationSplit ??
// tuning hyper parameters

import org.apache.spark.ml.evaluation.{RegressionEvaluator}
// evaluation for regression model

import org.apache.spark.ml.regression.{LinearRegression}
// regression model

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



object pipeline {

//=========================  StoreType dealing  =====================================
// StoreType : string => Index
val StoreTypeIndexer = new StringIndexer()
	.setInputCol("StoreType")
	.setOutputCol("StoreTypeIndex")

// StoreTypeIndex : one-hot
val StoreTypeIndexEncoder = new OneHotEncoder()
  .setInputCol("StoreTypeIndex")
  .setOutputCol("StoreTypeVec")

// =========================  Assortment dealing  =====================================
val AssortmentIndexer = new StringIndexer()
	.setInputCol("Assortment")
	.setOutputCol("AssortmentIndex")

// AssortmentIndex : one-hot
val AssortmentIndexEncoder = new OneHotEncoder()
  .setInputCol("AssortmentIndex")
  .setOutputCol("AssortmentVec")

//================================ CompetitionDistance dealing =========================
// StandardScaler for CompetitionDistance
val CompetitionDistanceScaler = new StandardScaler()
  .setInputCol("CompetitionDistance")
  .setOutputCol("CompetitionDistancescaled")
  .setWithStd(true)
  .setWithMean(false)

val CompetitionDistanceNormalizer = new Normalizer()
  .setInputCol("CompetitionDistance")
  .setOutputCol("CompetitionDistancescaled")

// ================================ CompetitionOpenSinceMonth dealing ===============
//one-hot for CompetitionOpenSinceMonth
val CompetitionOpenSinceMonthEncoder = new OneHotEncoder()
  .setInputCol("CompetitionOpenSinceMonth")
  .setOutputCol("CompetitionOpenSinceMonthVec")

// ============================ CompetitionOpenSinceYear dealing ===============
//one-hot for CompetitionOpenSinceYear
val CompetitionOpenSinceYearEncoder = new OneHotEncoder()
  .setInputCol("CompetitionOpenSinceYear")
  .setOutputCol("CompetitionOpenSinceYearVec")	

//============================ Promo2SinceWeek dealing ===================
// USE ORIGINAL DATA

// ========================== Promo2SinceYear dealing ==============
//one-hot for Promo2SinceYear
val Promo2SinceYearEncoder = new OneHotEncoder()
  .setInputCol("Promo2SinceYear")
  .setOutputCol("Promo2SinceYearVec")	

// ============================ Promo2 dealing =========================
// USE ORIGINAL DATA

// =============== Open dealing ================
// open : one-hot
val OpenEncoder = new OneHotEncoder()
  .setInputCol("Open")
  .setOutputCol("OpenVec")

//============================ schoolHoliday dealing ======================
// schoolHoliday : string => Index 
val SchoolHolidayIndexer = new StringIndexer()
  .setInputCol("SchoolHoliday")
  .setOutputCol("SchoolHolidayIndex")

// SchoolHolidayIndex : one-hot
val SchoolHolidayEncoder = new OneHotEncoder()
  .setInputCol("SchoolHolidayIndex")
  .setOutputCol("SchoolHolidayVec")


// ================================ DayOfWeek dealing ======================
// DayOfWeek: one-hot
val DayOfWeekEncoder = new OneHotEncoder()
  .setInputCol("DayOfWeek")
  .setOutputCol("DayOfWeekVec")

// ===============================store dealing =========================
// seems it is usefulless


//=============================== assemble all the features =========================================
val Assembler = new VectorAssembler()
  .setInputCols(Array(
  						"StoreTypeVec",
						"AssortmentVec",
						"CompetitionDistancescaled",
						"CompetitionOpenSinceMonthVec",
						"CompetitionOpenSinceYearVec",
						"Promo2SinceYearVec",
						"OpenVec",
						"SchoolHolidayVec",
						"DayOfWeekVec"
            		)
        )
  .setOutputCol("features")



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
  val lr = new LinearRegression()

  val paramGrid = new ParamGridBuilder()  // parameters to tune
    .addGrid(lr.regParam, Array(0.1, 0.01))
    //.addGrid(lr.fitIntercept)
    .addGrid(lr.elasticNetParam, Array(0.0, 0.25, 0.5, 0.75, 1.0))
    .build()

  val pipeline = new Pipeline()
    .setStages(Array(                   // put all the preprocessing before and model into the pipeline
            		StoreTypeIndexer,
					StoreTypeIndexEncoder,
					AssortmentIndexer,
					AssortmentIndexEncoder,
					CompetitionDistanceNormalizer,
					//CompetitionDistanceScaler,
					CompetitionOpenSinceMonthEncoder,
					CompetitionOpenSinceYearEncoder,
					Promo2SinceYearEncoder,
					OpenEncoder,
					SchoolHolidayIndexer,
					SchoolHolidayEncoder,
					DayOfWeekEncoder,
              		Assembler, 
              		lr  // the last thing is the model
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
val sc = new SparkContext(new SparkConf().setAppName("App").setMaster("local[4]"))
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
        Limit 10000
          """).na.drop()
println("show some of the train_store data")
store_train_data_sql.show(5)

//================================================ try to combine test data and store data together ============================

val store_test_data_sql = sqlContext.sql("""
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
        FROM table_test t inner join store_table s on s.Store =t.Store
       
          """).na.drop()
println("show some of the test_store data")
store_test_data_sql.show(5)



// ================================================ train of main step ======================================================
// The linear Regression Pipeline ----------------------------------------
val linearTvs = preppedLRPipeline()
val lrModel = fitModel(linearTvs, store_train_data_sql)
println("finish building pipeline !")
println("finish training fitting !")
println("begin Generating kaggle predictions")
val lrOut = lrModel.transform(store_train_data_sql) // transform is predicting in the sklearn
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
CompetitionDistance: Int,
CompetitionOpenSinceMonth:Double,	
CompetitionOpenSinceYear:Double,
Promo2:Double,
Promo2SinceWeek:Double,
Promo2SinceYear:Double
)
